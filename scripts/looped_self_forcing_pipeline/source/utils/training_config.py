"""Pure configuration validation used before model construction."""

from __future__ import annotations

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
