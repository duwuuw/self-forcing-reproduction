from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "scripts" / "looped_self_forcing_pipeline" / "source"


def _load_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_modules(monkeypatch):
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = [str(SOURCE_ROOT / "utils")]
    monkeypatch.setitem(sys.modules, "utils", utils_package)

    temporal_loop = _load_module(
        monkeypatch,
        "utils.temporal_loop",
        SOURCE_ROOT / "utils" / "temporal_loop.py",
    )
    cache = _load_module(
        monkeypatch,
        "test_causal_cache",
        SOURCE_ROOT / "wan" / "modules" / "causal_cache.py",
    )

    # The pipeline only uses these names as annotations. Avoid importing the
    # heavyweight Wan model modules in this CPU-only orchestration test.
    wrapper = types.ModuleType("utils.wan_wrapper")
    wrapper.WanDiffusionWrapper = object
    wrapper.SchedulerInterface = object
    monkeypatch.setitem(sys.modules, "utils.wan_wrapper", wrapper)

    pipeline_module = _load_module(
        monkeypatch,
        "test_self_forcing_training_pipeline",
        SOURCE_ROOT / "pipeline" / "self_forcing_training.py",
    )
    return types.SimpleNamespace(
        cache=cache,
        pipeline=pipeline_module.SelfForcingTrainingPipeline,
        temporal_loop=temporal_loop,
    )


class TinyBlock(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, hidden, key, value):
        context = (key + value).mean(dim=(1, 2)).unsqueeze(1)
        return hidden + self.weight * (hidden + context)


class TinyCausalModel:
    patch_size = (1, 1, 1)
    local_attn_size = -1

    def __init__(self, cache_module, num_layers: int):
        self.blocks = nn.ModuleList(
            [TinyBlock(0.1 * (index + 1)) for index in range(num_layers)]
        )
        self.cache_module = cache_module

    def create_kv_cache(
        self, *, batch_size, x, frame_seqlen, num_frames, dtype, device
    ):
        return [
            self.cache_module.create_kv_cache(
                batch_size,
                1,
                1,
                frame_seqlen=frame_seqlen,
                num_frames=num_frames,
                dtype=dtype,
                device=device,
            )
            for _ in self.blocks
        ]

    def create_crossattn_cache(self, *, batch_size, context, dtype, device):
        return [
            {
                "k": None,
                "v": None,
                "is_init": False,
                "context_len": context.shape[1],
            }
            for _ in self.blocks
        ]


class TinyGenerator:
    def __init__(self, source_modules, num_layers: int = 4):
        self.model = TinyCausalModel(source_modules.cache, num_layers)
        self.source_modules = source_modules
        self.calls = []

    def __call__(
        self,
        *,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache,
        crossattn_cache,
        current_start,
        temporal_loop_plan=None,
    ):
        batch_size, _, _, height, width = noisy_image_or_video.shape
        frame_seqlen = height * width
        plan = temporal_loop_plan or self.source_modules.temporal_loop.TemporalLoopPlan()
        trace = self.source_modules.temporal_loop.LoopExecutionTrace()
        prefix_before = tuple(
            entry["k"][:, :current_start].detach().clone() for entry in kv_cache
        )
        end_before = tuple(
            int(entry["global_end_index"].item()) for entry in kv_cache
        )

        hidden = noisy_image_or_video.reshape(batch_size, -1, 1)

        def call_block(block_index, _repeat_index, block, current):
            entry = kv_cache[block_index]
            key = current.unsqueeze(-1)
            value = 2 * key
            transaction = self.source_modules.cache.KVCacheTransaction(
                entry,
                current_start=current_start,
                num_tokens=current.shape[1],
                cache_start=current_start,
                max_attention_size=entry["k"].shape[1],
                frame_seqlen=frame_seqlen,
            )
            attention_key, attention_value = transaction.stage(key, value)
            output = block(current, attention_key, attention_value)
            transaction.commit()
            return output

        hidden = self.source_modules.temporal_loop.TemporalLoopExecutor.execute(
            self.model.blocks,
            hidden,
            plan,
            call_block,
            trace=trace,
        )
        flow_prediction = hidden.reshape_as(noisy_image_or_video)
        self.calls.append(
            {
                "input": noisy_image_or_video.detach().clone(),
                "timestep": timestep.detach().clone(),
                "current_start": current_start,
                "supplied_plan": temporal_loop_plan,
                "trace": tuple(trace.records),
                "end_before": end_before,
                "end_after": tuple(
                    int(entry["global_end_index"].item()) for entry in kv_cache
                ),
                "prefix_before": prefix_before,
                "cache_after": tuple(
                    entry["k"].detach().clone() for entry in kv_cache
                ),
                "prediction": flow_prediction.detach().clone(),
            }
        )
        return flow_prediction, torch.zeros_like(flow_prediction)


def _make_pipeline(
    source_modules,
    *,
    patch_size=(1, 1, 1),
    total_ar_blocks=None,
    independent_first_frame=False,
    loop_settings=None,
):
    generator = TinyGenerator(source_modules)
    generator.model.patch_size = patch_size
    loop_settings = dict(loop_settings or {})
    training_enabled = loop_settings.pop("training_enabled", True)
    temporal_loop_settings = {
        "enabled": True,
        "training_enabled": True,
        "k_min": 3,
        "k_max": 3,
        "strength": 1.0,
        "layer_start": 1,
        "layer_end": 2,
        "mode": "layer",
        "stop_grad_early": False,
        "schedule": "fixed",
    }
    temporal_loop_settings.update(loop_settings)
    if training_enabled is None:
        temporal_loop_settings.pop("training_enabled")
    else:
        temporal_loop_settings["training_enabled"] = training_enabled
    loop = source_modules.temporal_loop.TemporalLoopConfig.from_config(
        {
            "temporal_loop": temporal_loop_settings,
        },
        runtime_num_layers=len(generator.model.blocks),
    )
    pipeline = source_modules.pipeline(
        denoising_step_list=[1000, 0],
        scheduler=object(),
        generator=generator,
        num_frame_per_block=2,
        independent_first_frame=independent_first_frame,
        num_training_frames=4,
        total_ar_blocks=total_ar_blocks,
        temporal_loop=loop,
    )
    return pipeline, generator


def _teacher_forced_prediction(pipeline, *args):
    method = getattr(pipeline, "teacher_forced_flow_prediction", None)
    assert callable(method), "pipeline is missing teacher_forced_flow_prediction"
    return method(*args)


def test_teacher_forcing_loops_noisy_targets_and_replays_clean_context(source_modules):
    pipeline, generator = _make_pipeline(source_modules)
    clean_latent = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 1, 2, 2)
    noisy_latent = clean_latent + 100
    timestep = torch.tensor([[11, 11, 23, 23]])
    conditional_dict = {"prompt_embeds": torch.ones(1, 3, 2)}

    flow_prediction = _teacher_forced_prediction(
        pipeline, clean_latent, noisy_latent, timestep, conditional_dict
    )

    assert flow_prediction.shape == (1, 4, 1, 2, 2)
    assert len(generator.calls) == 4
    expected_prediction = torch.cat(
        [generator.calls[0]["prediction"], generator.calls[2]["prediction"]], dim=1
    )
    torch.testing.assert_close(flow_prediction.detach(), expected_prediction)

    for block_index in range(2):
        start = block_index * 2
        start_token = start * 4
        end_token = (start + 2) * 4
        target_call = generator.calls[block_index * 2]
        clean_call = generator.calls[block_index * 2 + 1]

        torch.testing.assert_close(target_call["input"], noisy_latent[:, start:start + 2])
        torch.testing.assert_close(target_call["timestep"], timestep[:, start:start + 2])
        assert target_call["current_start"] == start_token
        assert target_call["supplied_plan"].block_idx == block_index
        assert target_call["supplied_plan"].total_ar_blocks == 2
        assert target_call["supplied_plan"].mode == "layer"
        assert target_call["supplied_plan"].layer_start == 1
        assert target_call["supplied_plan"].layer_end == 2
        assert target_call["supplied_plan"].loop_count == 3
        assert target_call["supplied_plan"].stop_grad_early is False
        assert target_call["end_before"] == (start_token,) * 4
        assert target_call["end_after"] == (end_token,) * 4

        expected_trace = (
            (0, 0),
            (1, 0), (2, 0),
            (1, 1), (2, 1),
            (1, 2), (2, 2),
            (3, 0),
        )
        assert tuple(
            (record.block_index, record.repeat_index)
            for record in target_call["trace"]
        ) == expected_trace
        assert all(record.grad_enabled for record in target_call["trace"])
        assert all(not record.output_detached for record in target_call["trace"])

        torch.testing.assert_close(clean_call["input"], clean_latent[:, start:start + 2])
        torch.testing.assert_close(clean_call["timestep"], torch.zeros_like(timestep[:, start:start + 2]))
        assert clean_call["current_start"] == start_token
        assert clean_call["supplied_plan"] is None
        assert clean_call["end_before"] == (end_token,) * 4
        assert clean_call["end_after"] == (end_token,) * 4
        assert all(not record.grad_enabled for record in clean_call["trace"])
        assert tuple(record.repeat_index for record in clean_call["trace"]) == (0, 0, 0, 0)

        if block_index:
            previous_clean_call = generator.calls[block_index * 2 - 1]
            for layer_index in range(4):
                torch.testing.assert_close(
                    target_call["prefix_before"][layer_index],
                    previous_clean_call["cache_after"][layer_index][:, :start_token],
                )

        current_clean_tokens = clean_latent[:, start:start + 2].reshape(1, -1)
        torch.testing.assert_close(
            clean_call["cache_after"][0][:, start_token:end_token, 0, 0],
            current_clean_tokens,
        )

    flow_prediction.square().mean().backward()
    assert generator.model.blocks[1].weight.grad is not None
    assert generator.model.blocks[1].weight.grad.abs().item() > 0
    assert generator.model.blocks[2].weight.grad is not None
    assert generator.model.blocks[2].weight.grad.abs().item() > 0


def test_teacher_forcing_rejects_invalid_shapes_and_block_counts(source_modules):
    pipeline, _ = _make_pipeline(source_modules, total_ar_blocks=3)
    clean_latent = torch.zeros(1, 4, 1, 2, 2)
    noisy_latent = torch.ones_like(clean_latent)
    timestep = torch.ones(1, 4, dtype=torch.long)
    conditional_dict = {"prompt_embeds": torch.ones(1, 3, 2)}

    with pytest.raises(ValueError, match="matching"):
        _teacher_forced_prediction(
            pipeline, clean_latent, noisy_latent[:, :, :, :, :1], timestep, conditional_dict
        )

    with pytest.raises(ValueError, match="timestep"):
        _teacher_forced_prediction(
            pipeline, clean_latent, noisy_latent, timestep[:, :3], conditional_dict
        )

    with pytest.raises(ValueError, match="divisible"):
        _teacher_forced_prediction(
            pipeline,
            clean_latent[:, :3],
            noisy_latent[:, :3],
            timestep[:, :3],
            conditional_dict,
        )

    with pytest.raises(ValueError, match="total_ar_blocks"):
        _teacher_forced_prediction(
            pipeline, clean_latent, noisy_latent, timestep, conditional_dict
        )


def test_temporal_loop_config_retains_and_validates_training_enabled(source_modules):
    loop_settings = {
        "enabled": True,
        "mode": "layer",
        "training_enabled": True,
    }
    loop = source_modules.temporal_loop.TemporalLoopConfig.from_config(
        {"temporal_loop": loop_settings}, runtime_num_layers=4
    )
    assert getattr(loop, "training_enabled", None) is True

    loop_settings.pop("training_enabled")
    defaulted_loop = source_modules.temporal_loop.TemporalLoopConfig.from_config(
        {"temporal_loop": loop_settings}, runtime_num_layers=4
    )
    assert getattr(defaulted_loop, "training_enabled", None) is False

    with pytest.raises(ValueError, match="training_enabled"):
        source_modules.temporal_loop.TemporalLoopConfig.from_config(
            {
                "temporal_loop": {
                    **loop_settings,
                    "training_enabled": "yes",
                }
            },
            runtime_num_layers=4,
        )


@pytest.mark.parametrize(
    ("loop_settings", "message"),
    [
        ({"enabled": False}, "temporal_loop.enabled"),
        ({"training_enabled": False}, "training_enabled"),
        ({"training_enabled": None}, "training_enabled"),
        ({"mode": "block"}, "mode='layer'"),
        ({"stop_grad_early": True}, "stop_grad_early=false"),
    ],
)
def test_teacher_forcing_rejects_loop_contract_before_cache_setup(
    source_modules, loop_settings, message
):
    pipeline, generator = _make_pipeline(source_modules, loop_settings=loop_settings)
    latent = torch.zeros(1, 4, 1, 2, 2)

    with pytest.raises(ValueError, match=message):
        _teacher_forced_prediction(
            pipeline,
            latent,
            latent + 1,
            torch.ones(1, 4, dtype=torch.long),
            {"prompt_embeds": torch.ones(1, 3, 2)},
        )

    assert generator.calls == []
    assert pipeline.kv_cache1 is None


def test_teacher_forcing_rejects_independent_first_frame_before_cache_setup(
    source_modules,
):
    pipeline, generator = _make_pipeline(
        source_modules, independent_first_frame=True
    )
    latent = torch.zeros(1, 4, 1, 2, 2)

    with pytest.raises(ValueError, match="independent_first_frame"):
        _teacher_forced_prediction(
            pipeline,
            latent,
            latent + 1,
            torch.ones(1, 4, dtype=torch.long),
            {"prompt_embeds": torch.ones(1, 3, 2)},
        )

    assert generator.calls == []
    assert pipeline.kv_cache1 is None


def test_teacher_forcing_requires_unit_temporal_patch_size(source_modules):
    pipeline, _ = _make_pipeline(source_modules, patch_size=(2, 1, 1))
    latent = torch.zeros(1, 4, 1, 2, 2)

    with pytest.raises(ValueError, match="temporal patch size"):
        _teacher_forced_prediction(
            pipeline,
            latent,
            latent,
            torch.ones(1, 4, dtype=torch.long),
            {"prompt_embeds": torch.ones(1, 3, 2)},
        )


def test_cache_stage_keeps_live_gradient_but_persistent_state_is_detached(source_modules):
    cache = source_modules.cache.create_kv_cache(
        1, 1, 1, frame_seqlen=2, num_frames=2, dtype=torch.float32, device="cpu"
    )
    current = torch.tensor([[[2.0], [3.0]]], requires_grad=True)
    key = current.unsqueeze(-1)
    value = 2 * key
    transaction = source_modules.cache.KVCacheTransaction(
        cache, current_start=0, num_tokens=2, cache_start=0, frame_seqlen=2
    )

    attention_key, attention_value = transaction.stage(key, value)
    objective = (attention_key + attention_value).sum()
    transaction.commit()
    objective.backward()

    torch.testing.assert_close(current.grad, torch.full_like(current, 3.0))
    assert not cache["k"].requires_grad
    assert cache["k"].grad_fn is None
    assert not cache["v"].requires_grad
    assert cache["v"].grad_fn is None

    previous_end = int(cache["global_end_index"].item())
    clean_key = torch.full_like(key, 7.0)
    clean_value = torch.full_like(value, 9.0)
    replay = source_modules.cache.KVCacheTransaction(
        cache, current_start=0, num_tokens=2, cache_start=0, frame_seqlen=2
    )
    assert replay.is_replay
    replay.stage(clean_key, clean_value)
    replay.commit()

    assert int(cache["global_end_index"].item()) == previous_end
    torch.testing.assert_close(cache["k"][:, :2], clean_key)
    torch.testing.assert_close(cache["v"][:, :2], clean_value)
