"""Selected-block LoRA configuration and adapter lifecycle helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


_MISSING = object()


def _value(namespace: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(namespace, Mapping):
        result = namespace.get(name, _MISSING)
    else:
        result = getattr(namespace, name, _MISSING)
    if result is _MISSING:
        if default is _MISSING:
            raise AttributeError(f"missing LoRA setting: {name}")
        return default
    return result


@dataclass(frozen=True)
class LoraTrainingConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("self_attn.q", "self_attn.v")

    @classmethod
    def from_config(cls, config: Any = None) -> "LoraTrainingConfig":
        namespace = _value(config, "lora", None) if config is not None else None
        if namespace is None:
            return cls()

        enabled = _value(namespace, "enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("lora.enabled must be a boolean")
        if not enabled:
            return cls(enabled=False)

        rank = _value(namespace, "rank", 8)
        alpha = _value(namespace, "alpha", 16.0)
        dropout = _value(namespace, "dropout", 0.0)
        target_modules = _value(
            namespace, "target_modules", ("self_attn.q", "self_attn.v")
        )

        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("lora.rank must be a positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("lora.alpha must be a positive number")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0.0 <= float(dropout) <= 1.0
        ):
            raise ValueError("lora.dropout must be in [0, 1]")
        if isinstance(target_modules, str) or not isinstance(target_modules, Sequence):
            raise ValueError("lora.target_modules must be a non-empty sequence")
        if not target_modules:
            raise ValueError("lora.target_modules must be non-empty")

        normalized = tuple(str(target) for target in target_modules)
        if any(
            not target
            or "." not in target
            or target.startswith(".")
            or target.endswith(".")
            for target in normalized
        ):
            raise ValueError(
                "lora target modules must be fully qualified paths such as "
                "self_attn.q"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("lora.target_modules must not contain duplicates")

        return cls(
            enabled=True,
            rank=rank,
            alpha=float(alpha),
            dropout=float(dropout),
            target_modules=normalized,
        )


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for segment in path.split("."):
        if segment.isdigit():
            try:
                current = current[int(segment)]
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError(f"LoRA target module does not exist: {path}") from exc
        else:
            if not hasattr(current, segment):
                raise ValueError(f"LoRA target module does not exist: {path}")
            current = getattr(current, segment)
    if not isinstance(current, nn.Linear):
        raise TypeError(f"LoRA target must resolve to nn.Linear: {path}")
    return current


def _selected_target_paths(model: nn.Module, temporal_loop_config, lora_config):
    if not temporal_loop_config.enabled:
        raise ValueError("LoRA training requires an enabled temporal loop")

    start = int(temporal_loop_config.layer_start)
    end = int(temporal_loop_config.layer_end)
    paths = tuple(
        f"blocks.{block_index}.{target}"
        for block_index in range(start, end + 1)
        for target in lora_config.target_modules
    )
    for path in paths:
        _resolve_module(model, path)
    return paths


def _expected_adapter_names(target_paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        name
        for path in target_paths
        for name in (
            f"{path}.lora_A.default.weight",
            f"{path}.lora_B.default.weight",
        )
    )


def inject_selected_block_lora(
    model: nn.Module,
    temporal_loop_config,
    lora_config: LoraTrainingConfig,
) -> tuple[str, ...]:
    """Inject PEFT adapters into the exact selected causal-block paths."""

    if not lora_config.enabled:
        return ()
    if hasattr(model, "peft_config"):
        raise RuntimeError("LoRA adapters are already injected into this model")

    target_paths = _selected_target_paths(model, temporal_loop_config, lora_config)
    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import inject_adapter_in_model
    except ImportError as exc:  # pragma: no cover - dependency is tested in CI
        raise RuntimeError("LoRA training requires the 'peft' package") from exc

    peft_config = PeftLoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(target_paths),
        bias="none",
        task_type=None,
    )
    inject_adapter_in_model(peft_config, model)
    expected_names = _expected_adapter_names(target_paths)
    assert_lora_trainable_scope(model, expected_names)
    return expected_names


def prepare_generator_for_lora(
    model: nn.Module,
    temporal_loop_config,
    lora_config: LoraTrainingConfig,
) -> tuple[str, ...]:
    """Freeze a student base and inject its selected-block adapters."""

    if not lora_config.enabled:
        return ()
    model.requires_grad_(False)
    return inject_selected_block_lora(model, temporal_loop_config, lora_config)


def assert_lora_trainable_scope(
    model: nn.Module, selected_names: Sequence[str]
) -> None:
    expected = set(selected_names)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != expected:
        raise AssertionError(
            "unexpected LoRA trainable parameter set: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    if not actual or any("lora_A" not in name and "lora_B" not in name for name in actual):
        raise AssertionError("only LoRA A/B parameters may be trainable")


def _portable_lora_name(name: str) -> str:
    wrappers = {
        "_fsdp_wrapped_module",
        "_checkpoint_wrapped_module",
        "_orig_mod",
    }
    segments = [segment for segment in str(name).split(".") if segment not in wrappers]
    if segments and segments[0] == "model":
        segments.pop(0)
    if len(segments) >= 2 and segments[-2] == "default":
        segments.pop(-2)
    return ".".join(segments)


def extract_lora_state_dict(
    model: nn.Module,
    expected_names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    if not hasattr(model, "peft_config"):
        return {}

    state = {
        _portable_lora_name(name): parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (".lora_A." in f".{name}." or ".lora_B." in f".{name}.")
    }
    if expected_names is not None:
        expected = {_portable_lora_name(name) for name in expected_names}
        if set(state) != expected:
            raise ValueError(
                "generator adapter parameter set does not match injected scope: "
                f"expected={sorted(expected)}, actual={sorted(state)}"
            )
    return state


def load_lora_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    if not hasattr(model, "peft_config"):
        raise ValueError("cannot load LoRA state into a model without adapters")
    from peft import set_peft_model_state_dict

    normalized_state = {}
    for name, value in state_dict.items():
        normalized = str(name)
        for prefix in (
            "_fsdp_wrapped_module.",
            "_checkpoint_wrapped_module.",
            "_orig_mod.",
            "model.",
        ):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        for wrapper in (
            "._fsdp_wrapped_module.",
            "._checkpoint_wrapped_module.",
            "._orig_mod.",
        ):
            normalized = normalized.replace(wrapper, ".")
        normalized = normalized.replace(".default.weight", ".weight")
        normalized_state[normalized] = value

    result = set_peft_model_state_dict(model, normalized_state)
    missing_adapter_keys = [
        key for key in getattr(result, "missing_keys", []) if "lora_" in key
    ]
    if missing_adapter_keys:
        raise ValueError(f"missing LoRA checkpoint keys: {missing_adapter_keys}")
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"unexpected LoRA checkpoint keys: {result.unexpected_keys}")


def build_lora_checkpoint_payload(
    *,
    generator: Mapping[str, torch.Tensor],
    critic: Mapping[str, Any],
    generator_ema: Mapping[str, torch.Tensor] | None,
    generator_optimizer: Mapping[str, Any] | None,
    critic_optimizer: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the adapter-only checkpoint envelope used by the trainer."""

    generator_names = set(generator)
    if not generator_names:
        raise ValueError("generator adapter mapping must not be empty")
    out_of_scope = sorted(
        name
        for name in generator_names
        if not str(name).endswith((".lora_A.weight", ".lora_B.weight"))
    )
    if out_of_scope:
        raise ValueError(
            "generator adapter mapping contains out-of-scope keys: "
            f"{out_of_scope}"
        )

    return {
        "checkpoint_version": 1,
        "generator_format": "lora_adapter",
        "generator": dict(generator),
        "critic": dict(critic),
        "generator_ema": None if generator_ema is None else dict(generator_ema),
        "generator_optimizer": (
            None if generator_optimizer is None else dict(generator_optimizer)
        ),
        "critic_optimizer": None if critic_optimizer is None else dict(critic_optimizer),
        "metadata": dict(metadata),
    }


def load_lora_checkpoint(model: nn.Module, payload: Mapping[str, Any]) -> None:
    if payload.get("generator_format") != "lora_adapter":
        raise ValueError("checkpoint does not contain an adapter-only generator")
    load_lora_state_dict(model, payload["generator"])
