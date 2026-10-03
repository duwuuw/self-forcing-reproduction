"""Pure configuration validation used before model construction."""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _get(namespace: Any, name: str, default: Any = None) -> Any:
    if isinstance(namespace, Mapping):
        return namespace.get(name, default)
    return getattr(namespace, name, default)


def resolve_gradient_accumulation_steps(config: Any) -> int:
    """Return the configured number of micro-batches per optimizer update."""

    steps = _get(config, "gradient_accumulation_steps", 1)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    return steps


def validate_training_preflight(config: Any) -> None:
    """Fail closed for the selected-block DMD/LoRA training contract."""

    resolve_gradient_accumulation_steps(config)
    lora = _get(config, "lora", None)
    if not _get(lora, "enabled", False):
        return

    teacher_checkpoint = _get(config, "teacher_checkpoint", None)
    if not teacher_checkpoint:
        raise ValueError(
            "LoRA DMD training requires an explicit teacher checkpoint path"
        )
    teacher_path = Path(str(teacher_checkpoint)).expanduser()
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"teacher checkpoint does not exist: {teacher_path}"
        )

    if _get(config, "real_name", None) != "Wan2.1-T2V-14B":
        raise ValueError("real_name must be Wan2.1-T2V-14B for LoRA DMD training")
    if _get(config, "fake_name", "Wan2.1-T2V-1.3B") != "Wan2.1-T2V-1.3B":
        raise ValueError("fake_name must be Wan2.1-T2V-1.3B for LoRA DMD training")

    model_kwargs = _get(config, "model_kwargs", {})
    if _get(model_kwargs, "model_name", "Wan2.1-T2V-1.3B") != "Wan2.1-T2V-1.3B":
        raise ValueError(
            "model_kwargs.model_name must be Wan2.1-T2V-1.3B for the causal student"
        )

    temporal_loop = _get(config, "temporal_loop", None)
    if not _get(temporal_loop, "enabled", False):
        raise ValueError("LoRA DMD training requires temporal_loop.enabled=true")
    if not _get(temporal_loop, "training_enabled", False):
        raise ValueError(
            "LoRA DMD training requires temporal_loop.training_enabled=true"
        )


def validate_supervised_training_preflight(config: Any) -> None:
    """Validate the critic-free layerwise supervised flow-matching contract."""

    resolve_gradient_accumulation_steps(config)
    if _get(config, "trainer", None) != "diffusion":
        raise ValueError("supervised flow-matching training requires trainer=diffusion")

    lora = _get(config, "lora", None)
    if _get(lora, "enabled", False) is not True:
        raise ValueError("supervised flow-matching training requires lora.enabled=true")

    checkpoint = _get(config, "generator_ckpt", None)
    if not checkpoint:
        raise ValueError("supervised LoRA training requires generator_ckpt")
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"generator base checkpoint does not exist: {checkpoint_path}"
        )

    if _get(config, "teacher_forcing", False) is not True:
        raise ValueError("supervised flow-matching training requires teacher_forcing=true")
    if _get(config, "independent_first_frame", False) is not False:
        raise ValueError(
            "supervised flow-matching training requires independent_first_frame=false"
        )
    noise_augmentation = _get(config, "noise_augmentation_max_timestep", 0)
    if noise_augmentation != 0:
        raise ValueError(
            "supervised flow-matching training requires "
            "noise_augmentation_max_timestep=0"
        )

    denoising_step_list = _get(config, "denoising_step_list", None)
    if (
        isinstance(denoising_step_list, (str, bytes))
        or not isinstance(denoising_step_list, Sequence)
        or not denoising_step_list
    ):
        raise ValueError(
            "supervised training pipeline requires a non-empty denoising_step_list"
        )

    temporal_loop = _get(config, "temporal_loop", None)
    if _get(temporal_loop, "enabled", False) is not True:
        raise ValueError("supervised flow-matching training requires temporal_loop.enabled=true")
    if _get(temporal_loop, "training_enabled", False) is not True:
        raise ValueError(
            "supervised flow-matching training requires "
            "temporal_loop.training_enabled=true"
        )
    if _get(temporal_loop, "mode", None) != "layer":
        raise ValueError("supervised flow-matching training requires temporal_loop.mode='layer'")
    if _get(temporal_loop, "stop_grad_early", True) is not False:
        raise ValueError(
            "supervised flow-matching training requires temporal_loop.stop_grad_early=false"
        )
