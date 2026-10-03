from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


class TinyScheduler:
    num_train_timesteps = 4

    def __init__(self):
        self.timesteps = torch.arange(4, dtype=torch.float32)
        self.sigmas = torch.tensor([0.0, 0.25, 0.5, 0.75])
        self.weights = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def add_noise(self, clean, noise, timestep):
        indices = timestep.long()
        sigma = self.sigmas.to(clean.device)[indices].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * clean + sigma * noise).type_as(noise)

    def training_target(self, clean, noise, timestep):
        return noise - clean

    def training_weight(self, timestep):
        return self.weights.to(timestep.device)[timestep.long().flatten()]


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(1, 1, bias=False)
        self.v = nn.Linear(1, 1, bias=False)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_weight = nn.Parameter(torch.tensor(1.0))
        self.self_attn = TinyAttention()


class TinyCausalModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(num_layers)])
        self.patch_size = (1, 1, 1)
        self.num_frame_per_block = 1

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


class TinyWanDiffusionWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TinyCausalModel()
        self.scheduler = TinyScheduler()

    def get_scheduler(self):
        return self.scheduler

    def enable_gradient_checkpointing(self):
        self.model.enable_gradient_checkpointing()

    def clip_grad_norm_(self, max_norm):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


class TinyTextEncoder(nn.Module):
    def forward(self, text_prompts):
        return {"prompt_embeds": torch.ones(len(text_prompts), 2, 1)}


class TinyVAE(nn.Module):
    def encode_to_latent(self, frames):
        return frames


class RecordingTrainingPipeline:
    def __init__(self, *, generator, **kwargs):
        self.generator = generator
        self.kwargs = kwargs
        self.calls = []

    def teacher_forced_flow_prediction(
        self, clean_latent, noisy_latent, timestep, conditional_dict
    ):
        self.calls.append((clean_latent, noisy_latent, timestep, conditional_dict))
        adapters = [
            parameter
            for name, parameter in self.generator.named_parameters()
            if "lora_" in name
        ]
        if adapters:
            flow_value = torch.stack([parameter.mean() for parameter in adapters]).sum()
        else:
            flow_value = noisy_latent.new_tensor(0.5)
        return torch.ones_like(noisy_latent) * flow_value


class TinyFSDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.wrapped = module

    def forward(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["wrapped"], name)

    def clip_grad_norm_(self, max_norm):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


class TinyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"prompts": "synthetic prompt"}


class TinySampler:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


@pytest.fixture
def source_modules(monkeypatch):
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = [str(SOURCE_ROOT / "utils")]
    model_package = types.ModuleType("model")
    model_package.__path__ = [str(SOURCE_ROOT / "model")]
    pipeline_package = types.ModuleType("pipeline")
    trainer_package = types.ModuleType("trainer")
    trainer_package.__path__ = [str(SOURCE_ROOT / "trainer")]
    for name, package in (
        ("utils", utils_package),
        ("model", model_package),
        ("pipeline", pipeline_package),
        ("trainer", trainer_package),
    ):
        monkeypatch.setitem(sys.modules, name, package)

    temporal_loop = _load_module(
        monkeypatch,
        "utils.temporal_loop",
        SOURCE_ROOT / "utils" / "temporal_loop.py",
    )
    training_config = _load_module(
        monkeypatch,
        "utils.training_config",
        SOURCE_ROOT / "utils" / "training_config.py",
    )
    lora = _load_module(
        monkeypatch,
        "model.lora",
        SOURCE_ROOT / "model" / "lora.py",
    )

    wrapper_module = types.ModuleType("utils.wan_wrapper")
    wrapper_module.WanDiffusionWrapper = TinyWanDiffusionWrapper
    wrapper_module.WanTextEncoder = TinyTextEncoder
    wrapper_module.WanVAEWrapper = TinyVAE
    monkeypatch.setitem(sys.modules, "utils.wan_wrapper", wrapper_module)
    pipeline_package.SelfForcingTrainingPipeline = RecordingTrainingPipeline

    loss_module = types.ModuleType("utils.loss")
    loss_module.get_denoising_loss = lambda _name: object
    monkeypatch.setitem(sys.modules, "utils.loss", loss_module)

    einops_module = types.ModuleType("einops")
    einops_module.rearrange = lambda value, *_args, **_kwargs: value
    monkeypatch.setitem(sys.modules, "einops", einops_module)

    base_model = _load_module(
        monkeypatch,
        "model.base",
        SOURCE_ROOT / "model" / "base.py",
    )
    diffusion_model = _load_module(
        monkeypatch,
        "model.diffusion",
        SOURCE_ROOT / "model" / "diffusion.py",
    )
    model_package.CausalDiffusion = diffusion_model.CausalDiffusion

    dataset_module = types.ModuleType("utils.dataset")
    dataset_module.ShardingLMDBDataset = lambda *_args, **_kwargs: TinyDataset()
    dataset_module.cycle = lambda _dataloader: iter(())
    monkeypatch.setitem(sys.modules, "utils.dataset", dataset_module)

    misc_module = types.ModuleType("utils.misc")
    misc_module.set_seed = lambda _seed: None
    monkeypatch.setitem(sys.modules, "utils.misc", misc_module)

    distributed_module = types.ModuleType("utils.distributed")
    distributed_module.EMA_FSDP = lambda *_args, **_kwargs: None
    distributed_module.barrier = lambda: None
    distributed_module.fsdp_wrap = lambda module, **_kwargs: TinyFSDP(module)
    distributed_module.fsdp_state_dict = lambda _module: {}
    distributed_module.launch_distributed_job = lambda: None
    monkeypatch.setitem(sys.modules, "utils.distributed", distributed_module)

    wandb_module = types.ModuleType("wandb")
    wandb_module.login = lambda **_kwargs: None
    wandb_module.init = lambda **_kwargs: None
    wandb_module.log = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    trainer = _load_module(
        monkeypatch,
        "trainer.diffusion",
        SOURCE_ROOT / "trainer" / "diffusion.py",
    )
    return SimpleNamespace(
        base_model=base_model,
        dataset=dataset_module,
        diffusion_model=diffusion_model,
        distributed=distributed_module,
        lora=lora,
        temporal_loop=temporal_loop,
        trainer=trainer,
        training_config=training_config,
        wrapper=wrapper_module,
    )


def _loop_config(**overrides):
    config = {
        "enabled": True,
        "training_enabled": True,
        "mode": "layer",
        "layer_start": 1,
        "layer_end": 2,
        "k_min": 2,
        "k_max": 2,
        "strength": 1.0,
        "stop_grad_early": False,
        "schedule": "fixed",
    }
    config.update(overrides)
    return config


def _lora_config(enabled=True):
    return {
        "enabled": enabled,
        "rank": 2,
        "alpha": 2,
        "dropout": 0.0,
        "target_modules": ["self_attn.q", "self_attn.v"],
    }


def _training_config(generator_ckpt, **overrides):
    values = {
        "trainer": "diffusion",
        "generator_ckpt": str(generator_ckpt),
        "model_kwargs": {"model_name": "Wan2.1-T2V-1.3B"},
        "temporal_loop": _loop_config(),
        "lora": _lora_config(),
        "teacher_forcing": True,
        "noise_augmentation_max_timestep": 0,
        "independent_first_frame": False,
        "num_frame_per_block": 2,
        "num_training_frames": 4,
        "min_training_frames": 4,
        "training_gradient_window_frames": 4,
        "total_ar_blocks": 2,
        "denoising_step_list": [3, 2, 1, 0],
        "warp_denoising_step": False,
        "mixed_precision": False,
        "gradient_checkpointing": False,
        "num_train_timestep": 4,
        "guidance_scale": 1.0,
        "timestep_shift": 1.0,
        "i2v": False,
        "causal": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_base_checkpoint(path):
    wrapper = TinyWanDiffusionWrapper()
    torch.save({"generator": wrapper.state_dict()}, path)


def test_supervised_preflight_accepts_teacherless_layerwise_profile(
    source_modules, tmp_path
):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base checkpoint")
    config = _training_config(checkpoint)

    source_modules.training_config.validate_supervised_training_preflight(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora": _lora_config(False)}, "lora.enabled"),
        ({"temporal_loop": _loop_config(mode="block")}, "mode"),
        ({"temporal_loop": _loop_config(training_enabled=False)}, "training_enabled"),
        ({"temporal_loop": _loop_config(stop_grad_early=True)}, "stop_grad_early"),
        ({"independent_first_frame": True}, "independent_first_frame"),
        ({"teacher_forcing": False}, "teacher_forcing"),
    ],
)
def test_supervised_preflight_rejects_non_supervised_contract(
    source_modules, tmp_path, overrides, message
):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base checkpoint")
    config = _training_config(checkpoint, **overrides)

    with pytest.raises(ValueError, match=message):
        source_modules.training_config.validate_supervised_training_preflight(config)


def test_causal_diffusion_init_builds_teacherless_loop_student(
    source_modules, tmp_path
):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base checkpoint")
    config = _training_config(checkpoint)

    model = source_modules.diffusion_model.CausalDiffusion(
        config, device=torch.device("cpu")
    )

    assert model.temporal_loop.mode == "layer"
    assert model.temporal_loop.stop_grad_early is False
    assert model.lora_enabled is True
    assert hasattr(model, "scheduler")
    assert model.scheduler.timesteps.device.type == "cpu"
    assert not hasattr(model, "fake_score")
    assert not hasattr(model, "real_score")
    assert all(not parameter.requires_grad for parameter in model.generator.model.parameters())


def test_generator_loss_uses_cached_teacher_forcing_and_weighted_flow_loss(
    source_modules, monkeypatch, tmp_path
):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base checkpoint")
    model = source_modules.diffusion_model.CausalDiffusion(
        _training_config(checkpoint), device=torch.device("cpu")
    )
    fixed_noise = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2) / 10
    clean_latent = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2) / 20
    fixed_timestep = torch.tensor([[1, 1, 2, 2]])
    monkeypatch.setattr(torch, "randn_like", lambda _value: fixed_noise.clone())
    monkeypatch.setattr(model, "_get_timestep", lambda *_args, **_kwargs: fixed_timestep)
    conditional_dict = {"prompt_embeds": torch.ones(1, 2, 1)}

    loss, log_dict = model.generator_loss(clean_latent, conditional_dict)

    pipeline = model.inference_pipeline
    assert len(pipeline.calls) == 1
    called_clean, noisy, timestep, called_condition = pipeline.calls[0]
    torch.testing.assert_close(called_clean, clean_latent)
    torch.testing.assert_close(timestep, fixed_timestep.float())
    assert called_condition is conditional_dict

    flat_noisy = model.scheduler.add_noise(
        clean_latent.flatten(0, 1),
        fixed_noise.flatten(0, 1),
        fixed_timestep.flatten(0, 1).float(),
    ).unflatten(0, (1, 4))
    torch.testing.assert_close(noisy, flat_noisy)
    target = fixed_noise - clean_latent
    prediction = torch.full_like(clean_latent, 0.5)
    per_frame = (prediction - target).square().mean(dim=(2, 3, 4))
    weights = model.scheduler.training_weight(fixed_timestep.float()).unflatten(0, (1, 4))
    torch.testing.assert_close(loss, (per_frame * weights).mean())
    assert "x0" in log_dict


def test_diffusion_trainer_injects_adapters_before_fsdp_and_updates_only_selected(
    source_modules, monkeypatch, tmp_path
):
    checkpoint = tmp_path / "base.pt"
    _write_base_checkpoint(checkpoint)
    config = _training_config(
        checkpoint,
        seed=1,
        mixed_precision=False,
        disable_wandb=True,
        logdir=str(tmp_path),
        wandb_host="",
        wandb_key="",
        wandb_entity="",
        wandb_project="local",
        wandb_save_dir=str(tmp_path),
        causal=True,
        sharding_strategy="full",
        generator_fsdp_wrap_strategy="size",
        text_encoder_fsdp_wrap_strategy="size",
        no_visualize=True,
        load_raw_video=False,
        batch_size=1,
        data_path="synthetic-only",
        ema_weight=0.0,
        ema_start_step=10,
        beta1=0.0,
        beta2=0.99,
        weight_decay=0.0,
        lr=0.05,
        gc_interval=100,
        negative_prompt="unused",
    )
    events = []
    monkeypatch.setattr(source_modules.trainer, "launch_distributed_job", lambda: None)
    monkeypatch.setattr(source_modules.trainer.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(source_modules.trainer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        source_modules.trainer.dist,
        "broadcast",
        lambda tensor, src: events.append(("broadcast", tensor.shape, src)),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(source_modules.trainer, "set_seed", lambda _seed: None)

    original_load = source_modules.diffusion_model.CausalDiffusion.load_generator_base_checkpoint

    def tracked_load(self, path):
        events.append(("base_load", Path(path)))
        return original_load(self, path)

    monkeypatch.setattr(
        source_modules.diffusion_model.CausalDiffusion,
        "load_generator_base_checkpoint",
        tracked_load,
    )

    def inject_test_adapters(model, temporal_loop, lora_config):
        events.append(("lora", temporal_loop.layer_start, temporal_loop.layer_end))
        model.requires_grad_(False)
        names = []
        for layer_index in range(temporal_loop.layer_start, temporal_loop.layer_end + 1):
            for target in lora_config.target_modules:
                parameter_name = "lora_" + target.rsplit(".", 1)[-1]
                model.blocks[layer_index].register_parameter(
                    parameter_name, nn.Parameter(torch.tensor(0.25))
                )
                names.append(f"blocks.{layer_index}.{parameter_name}")
        return tuple(names)

    monkeypatch.setattr(
        source_modules.trainer, "prepare_generator_for_lora", inject_test_adapters
    )

    wrap_calls = []

    def wrap(module, **kwargs):
        events.append(("fsdp", type(module).__name__))
        wrap_calls.append(kwargs)
        return TinyFSDP(module)

    monkeypatch.setattr(source_modules.trainer, "fsdp_wrap", wrap)
    monkeypatch.setattr(
        torch.utils.data.distributed,
        "DistributedSampler",
        lambda dataset, **_kwargs: TinySampler(dataset),
    )

    trainer = source_modules.trainer.Trainer(config)

    kinds = [event[0] for event in events]
    assert kinds.index("base_load") < kinds.index("lora")
    assert kinds.index("lora") < kinds.index("broadcast")
    assert kinds.index("broadcast") < kinds.index("fsdp")
    assert wrap_calls[0]["min_num_params"] == 40_000_000
    assert not hasattr(trainer.model, "fake_score")
    assert not hasattr(trainer, "critic_optimizer")

    inner_generator = trainer.model.generator.model
    trainable = {
        name: parameter
        for name, parameter in inner_generator.named_parameters()
        if parameter.requires_grad
    }
    assert set(trainable) == set(trainer.model.lora_parameter_names)
    assert all("blocks.1.lora_" in name or "blocks.2.lora_" in name for name in trainable)
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in inner_generator.named_parameters()
        if not parameter.requires_grad
    }
    adapter_before = {name: parameter.detach().clone() for name, parameter in trainable.items()}

    batch = {
        "prompts": ["a small synthetic video"],
        "clean_latent": torch.ones(1, 4, 1, 2, 2),
    }
    trainer.train_one_step(batch)

    assert trainer.step == 1
    assert trainer.model.inference_pipeline.calls
    for name, parameter in inner_generator.named_parameters():
        if name in adapter_before:
            assert not torch.equal(parameter.detach(), adapter_before[name])
        else:
            torch.testing.assert_close(parameter.detach(), base_before[name])
