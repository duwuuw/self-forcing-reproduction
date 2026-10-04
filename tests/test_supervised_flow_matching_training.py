from __future__ import annotations

import importlib.util
import hashlib
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


class TinyEMA:
    instances = []

    def __init__(self, module, decay, parameter_filter=None):
        self.decay = decay
        self.parameter_filter = parameter_filter
        self.shadow = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self._parameters(module)
        }
        self.update_count = 0
        self.load_count = 0
        self.instances.append(self)

    def _parameters(self, module):
        module = getattr(module, "module", getattr(module, "wrapped", module))
        module = getattr(module, "model", module)
        for name, parameter in module.named_parameters():
            if self.parameter_filter is None or self.parameter_filter(name, parameter):
                yield name, parameter

    def update(self, module):
        for name, parameter in self._parameters(module):
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach().float().cpu(), alpha=1.0 - self.decay
            )
        self.update_count += 1

    def state_dict(self):
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state):
        self.load_count += 1
        self.shadow = {name: value.clone() for name, value in state.items()}


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
    distributed_module.fsdp_lora_state_dict = lambda *_args, **_kwargs: {}
    distributed_module.optimizer_state_dict_for_checkpoint = (
        lambda _model, optimizer: optimizer.state_dict()
    )
    distributed_module.load_fsdp_lora_state_dict = lambda *_args, **_kwargs: None
    distributed_module.load_optimizer_state_dict_for_checkpoint = (
        lambda _model, optimizer, state: optimizer.load_state_dict(state)
    )
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


def _inject_tiny_lora(model, temporal_loop, lora_config):
    model.requires_grad_(False)
    names = []
    for layer_index in range(temporal_loop.layer_start, temporal_loop.layer_end + 1):
        for target in lora_config.target_modules:
            module_name = target.rsplit(".", 1)[-1]
            linear = getattr(model.blocks[layer_index].self_attn, module_name)
            for side in ("A", "B"):
                adapter = nn.Linear(1, 1, bias=False)
                adapter.weight.data.fill_(0.25)
                setattr(linear, f"lora_{side}", nn.ModuleDict({"default": adapter}))
                names.append(
                    f"blocks.{layer_index}.self_attn.{module_name}."
                    f"lora_{side}.default.weight"
                )
    model.peft_config = {"default": object()}
    return tuple(names)


def _make_trainer(source_modules, monkeypatch, tmp_path, *, load_events=None, **overrides):
    checkpoint = Path(overrides.pop("generator_ckpt", tmp_path / "base.pt"))
    if not checkpoint.exists():
        _write_base_checkpoint(checkpoint)
    values = {
        "seed": 1,
        "mixed_precision": False,
        "disable_wandb": True,
        "logdir": str(tmp_path),
        "wandb_host": "",
        "wandb_key": "",
        "wandb_entity": "",
        "wandb_project": "local",
        "wandb_save_dir": str(tmp_path),
        "causal": True,
        "sharding_strategy": "full",
        "generator_fsdp_wrap_strategy": "size",
        "text_encoder_fsdp_wrap_strategy": "size",
        "no_visualize": True,
        "load_raw_video": False,
        "batch_size": 1,
        "data_path": "synthetic-only",
        "ema_weight": 0.0,
        "ema_start_step": 0,
        "beta1": 0.0,
        "beta2": 0.99,
        "weight_decay": 0.0,
        "lr": 0.05,
        "gc_interval": 100,
        "negative_prompt": "unused",
        "config_name": "synthetic",
        "no_save": False,
        "log_iters": 100,
        "resume_checkpoint": None,
    }
    values.update(overrides)
    config = _training_config(checkpoint, **values)

    monkeypatch.setattr(source_modules.trainer, "launch_distributed_job", lambda: None)
    monkeypatch.setattr(source_modules.trainer.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(source_modules.trainer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        source_modules.trainer.dist,
        "broadcast",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(source_modules.trainer, "set_seed", lambda _seed: None)
    monkeypatch.setattr(
        source_modules.trainer,
        "prepare_generator_for_lora",
        _inject_tiny_lora,
    )
    monkeypatch.setattr(
        source_modules.trainer,
        "fsdp_wrap",
        lambda module, **_kwargs: TinyFSDP(module),
    )
    monkeypatch.setattr(
        source_modules.trainer,
        "fsdp_lora_state_dict",
        lambda module, expected_names: source_modules.lora.extract_lora_state_dict(
            module.wrapped.model, expected_names
        ),
        raising=False,
    )

    def load_lora(module, state):
        if load_events is not None:
            load_events["adapter"] += 1
        return source_modules.lora.load_lora_state_dict(module.wrapped.model, state)

    monkeypatch.setattr(
        source_modules.trainer,
        "load_fsdp_lora_state_dict",
        load_lora,
        raising=False,
    )

    def load_optimizer(model, optimizer, state):
        if load_events is not None:
            load_events["optimizer"] += 1
        return optimizer.load_state_dict(state)

    monkeypatch.setattr(
        source_modules.trainer,
        "load_optimizer_state_dict_for_checkpoint",
        load_optimizer,
        raising=False,
    )
    monkeypatch.setattr(
        torch.utils.data.distributed,
        "DistributedSampler",
        lambda dataset, **_kwargs: TinySampler(dataset),
    )
    return source_modules.trainer.Trainer(config)


def _install_tiny_peft_loader(source_modules, monkeypatch):
    peft = types.ModuleType("peft")

    def set_peft_model_state_dict(model, state):
        parameters = {
            source_modules.lora._portable_lora_name(name): parameter
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }
        unexpected = sorted(set(state) - set(parameters))
        missing = sorted(set(parameters) - set(state))
        with torch.no_grad():
            for name, value in state.items():
                if name in parameters:
                    parameters[name].copy_(value)
        return SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)

    peft.set_peft_model_state_dict = set_peft_model_state_dict
    monkeypatch.setitem(sys.modules, "peft", peft)


def _make_resume_checkpoint(source_modules, monkeypatch, tmp_path):
    monkeypatch.setattr(source_modules.trainer, "EMA_FSDP", TinyEMA)
    _install_tiny_peft_loader(source_modules, monkeypatch)
    checkpoint = tmp_path / "base.pt"
    _write_base_checkpoint(checkpoint)
    original = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path,
        generator_ckpt=checkpoint,
        ema_weight=0.5,
        ema_start_step=0,
    )
    original.train_one_step(
        {
            "prompts": ["a small synthetic video"],
            "clean_latent": torch.ones(1, 4, 1, 2, 2),
        }
    )
    original.save()
    resume_path = tmp_path / "checkpoint_model_000001" / "model.pt"
    return checkpoint, resume_path


def _set_nested(mapping, dotted_path, value):
    parts = dotted_path.split(".")
    current = mapping
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


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


def test_diffusion_trainer_saves_only_adapters_and_optimizer_state(
    source_modules, monkeypatch, tmp_path
):
    trainer = _make_trainer(source_modules, monkeypatch, tmp_path)
    trainer.train_one_step(
        {
            "prompts": ["a small synthetic video"],
            "clean_latent": torch.ones(1, 4, 1, 2, 2),
        }
    )

    trainer.save()
    checkpoint_path = tmp_path / "checkpoint_model_000001" / "model.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = source_modules.lora.extract_lora_state_dict(
        trainer.model.generator.wrapped.model,
        trainer.model.lora_parameter_names,
    )

    assert payload["generator_format"] == "lora_adapter"
    assert set(payload["generator"]) == set(expected)
    assert all(
        name.endswith((".lora_A.weight", ".lora_B.weight"))
        for name in payload["generator"]
    )
    assert all(
        name.startswith(("blocks.1.", "blocks.2."))
        for name in payload["generator"]
    )
    assert payload["critic"] is None
    assert payload["critic_optimizer"] is None
    assert payload["generator_optimizer"]["state"]
    assert payload["metadata"]["training_objective"] == "supervised_flow_matching"
    assert payload["metadata"]["step"] == 1
    assert payload["metadata"]["temporal_loop"] == {
        "enabled": True,
        "k_min": 2,
        "k_max": 2,
        "strength": 1.0,
        "layer_start": 1,
        "layer_end": 2,
        "mode": "layer",
        "stop_grad_early": False,
        "schedule": "fixed",
        "runtime_num_layers": 4,
        "training_enabled": True,
    }
    assert payload["metadata"]["lora"] == trainer.config.lora
    assert payload["metadata"]["ema"] == {
        "enabled": False,
        "decay": 0.0,
        "start_step": 0,
    }


def test_diffusion_trainer_ema_starts_at_threshold_and_updates_lora_only(
    source_modules, monkeypatch, tmp_path
):
    monkeypatch.setattr(source_modules.trainer, "EMA_FSDP", TinyEMA)
    trainer = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path,
        ema_weight=0.5,
        ema_start_step=2,
    )
    batch = {
        "prompts": ["a small synthetic video"],
        "clean_latent": torch.ones(1, 4, 1, 2, 2),
    }

    assert trainer.generator_ema is None
    trainer.train_one_step(batch)
    assert trainer.generator_ema is None
    trainer.train_one_step(batch)

    ema = trainer.generator_ema
    assert isinstance(ema, TinyEMA)
    assert ema.update_count == 0
    assert ema.shadow
    assert all("lora_" in name for name in ema.shadow)
    assert all(name.startswith(("blocks.1.", "blocks.2.")) for name in ema.shadow)
    before_update = ema.state_dict()

    trainer.train_one_step(batch)

    assert trainer.step == 3
    assert ema.update_count == 1
    assert any(
        not torch.equal(before_update[name], ema.shadow[name]) for name in before_update
    )
    trainer.save()
    payload = torch.load(
        tmp_path / "checkpoint_model_000003" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert payload["generator_ema"]


def test_diffusion_trainer_resume_restores_adapters_optimizer_ema_and_step(
    source_modules, monkeypatch, tmp_path
):
    monkeypatch.setattr(source_modules.trainer, "EMA_FSDP", TinyEMA)
    _install_tiny_peft_loader(source_modules, monkeypatch)
    checkpoint = tmp_path / "base.pt"
    _write_base_checkpoint(checkpoint)
    original = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path,
        generator_ckpt=checkpoint,
        ema_weight=0.5,
        ema_start_step=0,
    )
    original.train_one_step(
        {
            "prompts": ["a small synthetic video"],
            "clean_latent": torch.ones(1, 4, 1, 2, 2),
        }
    )
    assert original.generator_ema.update_count == 1
    original.save()
    resume_path = tmp_path / "checkpoint_model_000001" / "model.pt"
    original_adapters = source_modules.lora.extract_lora_state_dict(
        original.model.generator.wrapped.model,
        original.model.lora_parameter_names,
    )
    original_ema = original.generator_ema.state_dict()
    original_optimizer = original.generator_optimizer.state_dict()

    resumed = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path / "resumed",
        generator_ckpt=checkpoint,
        ema_weight=0.5,
        ema_start_step=0,
        resume_checkpoint=str(resume_path),
    )
    resumed_adapters = source_modules.lora.extract_lora_state_dict(
        resumed.model.generator.wrapped.model,
        resumed.model.lora_parameter_names,
    )

    assert resumed.step == original.step == 1
    assert resumed.generator_optimizer.state
    resumed_optimizer = resumed.generator_optimizer.state_dict()
    assert resumed_optimizer["param_groups"] == original_optimizer["param_groups"]
    assert resumed_optimizer["state"].keys() == original_optimizer["state"].keys()
    for parameter_id, original_state in original_optimizer["state"].items():
        for name, value in original_state.items():
            resumed_value = resumed_optimizer["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(resumed_value, value)
            else:
                assert resumed_value == value
    assert set(resumed_adapters) == set(original_adapters)
    for name in original_adapters:
        torch.testing.assert_close(resumed_adapters[name], original_adapters[name])
    assert resumed.generator_ema.state_dict().keys() == original_ema.keys()
    for name in original_ema:
        torch.testing.assert_close(
            resumed.generator_ema.state_dict()[name], original_ema[name]
        )


def test_diffusion_trainer_resume_canonicalizes_relative_base_checkpoint(
    source_modules, monkeypatch, tmp_path
):
    monkeypatch.setattr(source_modules.trainer, "EMA_FSDP", TinyEMA)
    _install_tiny_peft_loader(source_modules, monkeypatch)
    monkeypatch.chdir(tmp_path)
    checkpoint = Path("base.pt")
    _write_base_checkpoint(checkpoint)
    original = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path,
        generator_ckpt=checkpoint,
        ema_weight=0.5,
        ema_start_step=0,
    )
    original.train_one_step(
        {
            "prompts": ["a small synthetic video"],
            "clean_latent": torch.ones(1, 4, 1, 2, 2),
        }
    )
    original.save()
    resume_path = tmp_path / "checkpoint_model_000001" / "model.pt"
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)

    assert payload["metadata"]["base_checkpoint"] == str(checkpoint.resolve())

    resumed = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path / "resumed",
        generator_ckpt=checkpoint,
        ema_weight=0.5,
        ema_start_step=0,
        resume_checkpoint=str(resume_path),
    )

    assert resumed.step == original.step == 1


@pytest.mark.parametrize(
    ("metadata_path", "replacement"),
    [
        ("training_objective", "dmd"),
        ("student_model", "another-model"),
        ("base_checkpoint", "/different/base.pt"),
        ("temporal_loop.enabled", False),
        ("temporal_loop.runtime_num_layers", 29),
        ("temporal_loop.k_max", 3),
        ("lora.rank", 4),
        ("lora.target_modules", ["self_attn.k"]),
        ("ema.decay", 0.9),
        ("ema.start_step", 5),
    ],
)
def test_resume_rejects_incompatible_metadata_before_loading_state(
    source_modules, monkeypatch, tmp_path, metadata_path, replacement
):
    checkpoint, resume_path = _make_resume_checkpoint(
        source_modules, monkeypatch, tmp_path
    )
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    _set_nested(payload["metadata"], metadata_path, replacement)
    bad_resume = tmp_path / "metadata_mismatch.pt"
    torch.save(payload, bad_resume)
    load_events = {"adapter": 0, "optimizer": 0}
    previous_ema_count = len(TinyEMA.instances)

    with pytest.raises(ValueError):
        _make_trainer(
            source_modules,
            monkeypatch,
            tmp_path / "resumed",
            generator_ckpt=checkpoint,
            ema_weight=0.5,
            ema_start_step=0,
            resume_checkpoint=str(bad_resume),
            load_events=load_events,
        )

    assert load_events == {"adapter": 0, "optimizer": 0}
    assert all(
        ema.load_count == 0 for ema in TinyEMA.instances[previous_ema_count:]
    )


def test_resume_uses_sue_portable_base_reference_and_hash(
    source_modules, monkeypatch, tmp_path
):
    checkpoint, resume_path = _make_resume_checkpoint(
        source_modules, monkeypatch, tmp_path
    )
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload["metadata"]["base_checkpoint"] = "base.pt"
    payload["metadata"]["base_checkpoint_sha256"] = digest
    sue_resume = tmp_path / "sue_resume.pt"
    torch.save(payload, sue_resume)
    monkeypatch.setenv("SUE_ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_RELATIVE_PATH", "base.pt")
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_SHA256", digest)

    resumed = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path / "resumed",
        generator_ckpt=checkpoint,
        ema_weight=0.5,
        ema_start_step=0,
        resume_checkpoint=str(sue_resume),
    )

    assert resumed.step == 1


def test_resume_rejects_sue_base_hash_mismatch_before_loading_state(
    source_modules, monkeypatch, tmp_path
):
    checkpoint, resume_path = _make_resume_checkpoint(
        source_modules, monkeypatch, tmp_path
    )
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    payload["metadata"]["base_checkpoint"] = "base.pt"
    payload["metadata"]["base_checkpoint_sha256"] = "0" * 64
    bad_resume = tmp_path / "base_hash_mismatch.pt"
    torch.save(payload, bad_resume)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv("SUE_ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_RELATIVE_PATH", "base.pt")
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_SHA256", digest)
    load_events = {"adapter": 0, "optimizer": 0}
    previous_ema_count = len(TinyEMA.instances)

    with pytest.raises(ValueError):
        _make_trainer(
            source_modules,
            monkeypatch,
            tmp_path / "resumed",
            generator_ckpt=checkpoint,
            ema_weight=0.5,
            ema_start_step=0,
            resume_checkpoint=str(bad_resume),
            load_events=load_events,
        )

    assert load_events == {"adapter": 0, "optimizer": 0}
    assert all(
        ema.load_count == 0 for ema in TinyEMA.instances[previous_ema_count:]
    )


def test_sue_diffusion_loop_checks_each_step_and_stops_after_final_barrier(
    source_modules, monkeypatch
):
    trainer_class = source_modules.trainer.Trainer
    events = []

    trainer = object.__new__(trainer_class)
    trainer.config = SimpleNamespace(no_save=False, log_iters=2)
    trainer.dataloader = iter(("first", "final"))
    trainer.step = 0
    trainer.is_main_process = True
    trainer.disable_wandb = True
    trainer.previous_time = None

    def train_one_step(self, batch):
        self.step += 1
        events.append(("optimizer", self.step, batch))

    def save(self):
        events.append(("native_log_iters_save", self.step))

    def checkpoint_callback(self):
        events.append(("sue_due_check", self.step))
        if self.step == 2:
            events.append(("final_checkpoint", self.step))
            return True
        return False

    trainer.train_one_step = types.MethodType(train_one_step, trainer)
    trainer.save = types.MethodType(save, trainer)
    monkeypatch.setattr(
        trainer_class,
        "_sue_checkpoint_callback",
        staticmethod(checkpoint_callback),
        raising=False,
    )
    monkeypatch.setattr(source_modules.trainer, "barrier", lambda: events.append(("barrier", trainer.step)))
    times = iter((1.0, 2.0))
    monkeypatch.setattr(source_modules.trainer.time, "time", lambda: next(times))

    try:
        trainer.train()
    except StopIteration:
        events.append(("uncapped", trainer.step))

    assert events == [
        ("optimizer", 1, "first"),
        ("sue_due_check", 1),
        ("barrier", 1),
        ("optimizer", 2, "final"),
        ("sue_due_check", 2),
        ("final_checkpoint", 2),
        ("barrier", 2),
    ]


def test_non_sue_diffusion_loop_keeps_native_log_iters_save_path(
    source_modules, monkeypatch
):
    trainer_class = source_modules.trainer.Trainer
    monkeypatch.delattr(trainer_class, "_sue_checkpoint_callback", raising=False)
    events = []

    class StopLoop(Exception):
        pass

    trainer = object.__new__(trainer_class)
    trainer.config = SimpleNamespace(no_save=False, log_iters=2)
    trainer.dataloader = iter(("one", "two", "three"))
    trainer.step = 0
    trainer.is_main_process = True
    trainer.disable_wandb = True
    trainer.previous_time = None

    def train_one_step(self, batch):
        self.step += 1
        events.append(("optimizer", self.step, batch))
        if self.step == 3:
            raise StopLoop

    def save(self):
        events.append(("native_save", self.step))

    trainer.train_one_step = types.MethodType(train_one_step, trainer)
    trainer.save = types.MethodType(save, trainer)
    monkeypatch.setattr(source_modules.trainer, "barrier", lambda: events.append(("barrier", trainer.step)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty_cache", trainer.step)))
    times = iter((1.0, 2.0))
    monkeypatch.setattr(source_modules.trainer.time, "time", lambda: next(times))

    with pytest.raises(StopLoop):
        trainer.train()

    assert events == [
        ("optimizer", 1, "one"),
        ("barrier", 1),
        ("optimizer", 2, "two"),
        ("empty_cache", 2),
        ("native_save", 2),
        ("empty_cache", 2),
        ("barrier", 2),
        ("optimizer", 3, "three"),
    ]


def test_sue_checkpoint_error_raises_before_the_existing_barrier(
    source_modules, monkeypatch
):
    trainer_class = source_modules.trainer.Trainer
    events = []
    trainer = object.__new__(trainer_class)
    trainer.config = SimpleNamespace(no_save=False, log_iters=1)
    trainer.dataloader = iter(("one",))
    trainer.step = 0
    trainer.is_main_process = False
    trainer.disable_wandb = True
    trainer.previous_time = None

    def train_one_step(self, batch):
        self.step += 1
        events.append(("optimizer", self.step, batch))

    def checkpoint_callback(self):
        events.append(("save_status_failure", self.step))
        raise RuntimeError("SUE checkpoint failed on all ranks")

    trainer.train_one_step = types.MethodType(train_one_step, trainer)
    monkeypatch.setattr(
        trainer_class,
        "_sue_checkpoint_callback",
        staticmethod(checkpoint_callback),
        raising=False,
    )
    monkeypatch.setattr(
        source_modules.trainer,
        "barrier",
        lambda: events.append(("barrier", trainer.step)),
    )

    with pytest.raises(RuntimeError, match="failed on all ranks"):
        trainer.train()

    assert events == [
        ("optimizer", 1, "one"),
        ("save_status_failure", 1),
    ]


@pytest.mark.parametrize(("resume_step", "error"), [(3, None), (4, "exceeds")])
def test_sue_resume_cap_never_consumes_an_extra_batch(
    source_modules, monkeypatch, resume_step, error
):
    trainer_class = source_modules.trainer.Trainer
    events = []

    class NoBatchAvailable:
        def __iter__(self):
            return self

        def __next__(self):
            events.append(("data_read",))
            raise AssertionError("resume cap check must precede data consumption")

    trainer = object.__new__(trainer_class)
    trainer.config = SimpleNamespace(no_save=False, log_iters=2)
    trainer.dataloader = NoBatchAvailable()
    trainer.step = resume_step
    trainer.is_main_process = False
    trainer.disable_wandb = True
    trainer.previous_time = None

    def final_checkpoint(self):
        events.append(("final_checkpoint", self.step))
        return True

    monkeypatch.setattr(trainer_class, "_sue_max_steps", 3, raising=False)
    monkeypatch.setattr(
        trainer_class,
        "_sue_checkpoint_callback",
        staticmethod(final_checkpoint),
        raising=False,
    )
    monkeypatch.setattr(
        source_modules.trainer,
        "barrier",
        lambda: events.append(("barrier", trainer.step)),
    )

    if error:
        with pytest.raises(ValueError, match=error):
            trainer.train()
        assert events == []
    else:
        trainer.train()
        assert events == [("final_checkpoint", 3), ("barrier", 3)]


@pytest.mark.parametrize("corruption", ["shape", "type"])
def test_resume_rejects_corrupt_same_key_ema_before_loading_state(
    source_modules, monkeypatch, tmp_path, corruption
):
    checkpoint, resume_path = _make_resume_checkpoint(
        source_modules, monkeypatch, tmp_path
    )
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    key = next(iter(payload["generator_ema"]))
    value = payload["generator_ema"][key]
    payload["generator_ema"][key] = (
        torch.zeros((*value.shape, 1)) if corruption == "shape" else "not a tensor"
    )
    bad_resume = tmp_path / "corrupt_ema.pt"
    torch.save(payload, bad_resume)
    load_events = {"adapter": 0, "optimizer": 0}
    previous_ema_count = len(TinyEMA.instances)

    with pytest.raises(ValueError, match="EMA"):
        _make_trainer(
            source_modules,
            monkeypatch,
            tmp_path / "resumed",
            generator_ckpt=checkpoint,
            ema_weight=0.5,
            ema_start_step=0,
            resume_checkpoint=str(bad_resume),
            load_events=load_events,
        )

    assert load_events == {"adapter": 0, "optimizer": 0}
    assert all(
        ema.load_count == 0 for ema in TinyEMA.instances[previous_ema_count:]
    )


def test_diffusion_trainer_resume_rejects_missing_active_ema(
    source_modules, monkeypatch, tmp_path
):
    monkeypatch.setattr(source_modules.trainer, "EMA_FSDP", TinyEMA)
    _install_tiny_peft_loader(source_modules, monkeypatch)
    checkpoint = tmp_path / "base.pt"
    _write_base_checkpoint(checkpoint)
    original = _make_trainer(
        source_modules,
        monkeypatch,
        tmp_path,
        generator_ckpt=checkpoint,
        ema_weight=0.5,
        ema_start_step=0,
    )
    original.train_one_step(
        {
            "prompts": ["a small synthetic video"],
            "clean_latent": torch.ones(1, 4, 1, 2, 2),
        }
    )
    original.save()
    payload = torch.load(
        tmp_path / "checkpoint_model_000001" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    payload["generator_ema"] = None
    bad_resume = tmp_path / "missing_ema.pt"
    torch.save(payload, bad_resume)

    with pytest.raises(ValueError, match="EMA"):
        _make_trainer(
            source_modules,
            monkeypatch,
            tmp_path / "resumed",
            generator_ckpt=checkpoint,
            ema_weight=0.5,
            ema_start_step=0,
            resume_checkpoint=str(bad_resume),
        )
