"""Dependency-light planning and execution for temporal loops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


_MISSING = object()


def _get_value(namespace: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(namespace, Mapping):
        value = namespace.get(name, _MISSING)
    else:
        value = getattr(namespace, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise AttributeError(f"missing temporal loop setting: {name}")
        return default
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


@dataclass(frozen=True)
class TemporalLoopPlan:
    """The immutable plan for one autoregressive chunk."""

    loop_count: int = 1
    strength: float = 1.0
    layer_start: int = 0
    layer_end: int = -1
    mode: str = "block"
    stop_grad_early: bool = True
    enabled: bool = False
    block_idx: int = 0
    total_ar_blocks: int = 1

    def repeats_block(self, block_index: int) -> bool:
        return self.enabled and self.layer_start <= block_index <= self.layer_end


@dataclass(frozen=True)
class TemporalLoopConfig:
    """Validated temporal-loop settings with a baseline-safe disabled state."""

    enabled: bool = False
    k_min: int = 1
    k_max: int = 1
    strength: float = 1.0
    layer_start: int = 0
    layer_end: int = -1
    mode: str = "block"
    stop_grad_early: bool = True
    schedule: str = "fixed"
    runtime_num_layers: int = 0

    @classmethod
    def from_config(cls, config: Any, runtime_num_layers: int) -> "TemporalLoopConfig":
        runtime_num_layers = _require_int(runtime_num_layers, "runtime_num_layers")
        if runtime_num_layers < 1:
            raise ValueError("runtime_num_layers must be positive")

        namespace = _get_value(config, "temporal_loop", None) if config is not None else None
        if namespace is None:
            return cls(runtime_num_layers=runtime_num_layers)

        enabled = _get_value(namespace, "enabled", False)
        if enabled is False or enabled is None:
            return cls(runtime_num_layers=runtime_num_layers)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")

        k_min = _require_int(_get_value(namespace, "k_min", 1), "k_min")
        k_max = _require_int(_get_value(namespace, "k_max", k_min), "k_max")
        strength = _require_number(_get_value(namespace, "strength", 1.0), "strength")
        layer_start = _require_int(_get_value(namespace, "layer_start", 0), "layer_start")
        layer_end = _require_int(
            _get_value(namespace, "layer_end", runtime_num_layers - 1), "layer_end"
        )
        mode = _get_value(namespace, "mode", "block")
        schedule = _get_value(namespace, "schedule", "fixed")
        stop_grad_early = _get_value(namespace, "stop_grad_early", True)

        if not 1 <= k_min <= k_max:
            raise ValueError("loop bounds must satisfy 1 <= k_min <= k_max")
        if not 0 <= layer_start <= layer_end < runtime_num_layers:
            raise ValueError(
                "layer range must satisfy 0 <= layer_start <= layer_end < runtime_num_layers"
            )
        if not isinstance(mode, str) or mode not in {"block", "layer"}:
            raise ValueError("mode must be 'block' or 'layer'")
        if not isinstance(schedule, str) or schedule not in {"fixed", "temporal_uniform"}:
            raise ValueError("schedule must be 'fixed' or 'temporal_uniform'")
        if schedule == "fixed" and k_min != k_max:
            raise ValueError("the fixed schedule requires k_min == k_max")
        if not isinstance(stop_grad_early, bool):
            raise ValueError("stop_grad_early must be a boolean")

        return cls(
            enabled=True,
            k_min=k_min,
            k_max=k_max,
            strength=strength,
            layer_start=layer_start,
            layer_end=layer_end,
            mode=mode,
            stop_grad_early=stop_grad_early,
            schedule=schedule,
            runtime_num_layers=runtime_num_layers,
        )

    def resolve_loop_count(self, block_idx: int, total_ar_blocks: int) -> int:
        """Resolve the loop count for an autoregressive chunk.

        ``fixed`` uses the constant ``k_min``. ``temporal_uniform`` ramps
        ``k_min`` up to ``k_max`` across the AR chunks: chunk ``block_idx`` of
        ``total_ar_blocks`` gets ``k_min`` plus
        ``floor((k_max - k_min + 1) * block_idx / (total_ar_blocks - 1))``, and
        the result is clamped into ``[k_min, k_max]``. A single-chunk rollout
        has no progress to interpolate and resolves to ``k_min``.
        """

        if not self.enabled:
            return 1
        if self.schedule == "fixed" or total_ar_blocks <= 1:
            return self.k_min
        ramp = (self.k_max - self.k_min + 1) * block_idx // (total_ar_blocks - 1)
        return self.k_min + min(self.k_max - self.k_min, max(0, ramp))

    def plan_for_block(self, block_idx: int, total_ar_blocks: int) -> TemporalLoopPlan:
        return TemporalLoopPlan(
            loop_count=self.resolve_loop_count(block_idx, total_ar_blocks),
            strength=self.strength,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            mode=self.mode,
            stop_grad_early=self.stop_grad_early,
            enabled=self.enabled,
            block_idx=block_idx,
            total_ar_blocks=total_ar_blocks,
        )


@dataclass(frozen=True)
class LoopExecutionRecord:
    """Trace data for one block invocation."""

    block_index: int
    repeat_index: int
    grad_enabled: bool
    output_detached: bool


@dataclass
class LoopExecutionTrace:
    """Mutable per-execution trace populated by :class:`TemporalLoopExecutor`."""

    records: list[LoopExecutionRecord] = field(default_factory=list)

    def record(
        self,
        block_index: int,
        repeat_index: int,
        grad_enabled: bool,
        output_detached: bool,
    ) -> None:
        self.records.append(
            LoopExecutionRecord(
                block_index=block_index,
                repeat_index=repeat_index,
                grad_enabled=grad_enabled,
                output_detached=output_detached,
            )
        )

    @property
    def calls(self) -> list[LoopExecutionRecord]:
        return self.records


BlockCallback = Callable[[int, int, Any, torch.Tensor], torch.Tensor]


def _blend_residual(
    hidden: torch.Tensor, residual: torch.Tensor, weight: float | None
) -> torch.Tensor:
    """Return ``hidden + weight * (residual - hidden)`` for one loop iteration.

    Blocks outside the selected range have no blend weight and pass their output
    through unchanged. A weight of exactly 1 keeps the residual output itself,
    which makes a single full-strength iteration identical to a plain block call.
    """

    if weight is None or weight == 1.0:
        return residual
    return hidden + weight * (residual - hidden)


class TemporalLoopExecutor:
    """Execute an immutable temporal-loop plan without changing grad mode."""

    @staticmethod
    def execute(
        blocks: Sequence[Any],
        x: torch.Tensor,
        plan: TemporalLoopPlan,
        call_block: BlockCallback,
        trace: LoopExecutionTrace | None = None,
    ) -> torch.Tensor:
        if plan.mode not in {"block", "layer"}:
            raise ValueError(f"unsupported temporal loop mode: {plan.mode}")

        def call_once(
            block_index: int,
            repeat_index: int,
            current: torch.Tensor,
            output_detached: bool,
        ) -> torch.Tensor:
            grad_enabled = torch.is_grad_enabled()
            next_x = call_block(block_index, repeat_index, blocks[block_index], current)
            if trace is not None:
                trace.record(block_index, repeat_index, grad_enabled, output_detached)
            return next_x

        if plan.mode == "layer" and plan.enabled:
            for block_index in range(plan.layer_start):
                x = call_once(block_index, 0, x, output_detached=False)

            blend_weight = plan.strength / plan.loop_count
            if plan.stop_grad_early and plan.loop_count > 1:
                # The original Self-Forcing loop treats the selected stack as
                # one loop body and does not backpropagate from its early
                # rollout passes into the prefix.
                x = x.detach()

            for repeat_index in range(plan.loop_count):
                early_repeat = plan.stop_grad_early and repeat_index < plan.loop_count - 1
                stack_input = x
                if early_repeat:
                    with torch.no_grad():
                        for block_index in range(plan.layer_start, plan.layer_end + 1):
                            x = call_once(block_index, repeat_index, x, output_detached=True)
                        x = _blend_residual(stack_input, x, blend_weight)
                    x = x.detach()
                else:
                    for block_index in range(plan.layer_start, plan.layer_end + 1):
                        x = call_once(block_index, repeat_index, x, output_detached=False)
                    x = _blend_residual(stack_input, x, blend_weight)

            for block_index in range(plan.layer_end + 1, len(blocks)):
                x = call_once(block_index, 0, x, output_detached=False)
            return x

        for block_index, block in enumerate(blocks):
            selected = plan.repeats_block(block_index)
            repeat_count = plan.loop_count if selected else 1
            blend_weight = plan.strength / repeat_count if selected else None
            for repeat_index in range(repeat_count):
                early_repeat = selected and plan.stop_grad_early and repeat_index < repeat_count - 1
                if early_repeat:
                    with torch.no_grad():
                        grad_enabled = torch.is_grad_enabled()
                        next_x = _blend_residual(
                            x, call_block(block_index, repeat_index, block, x), blend_weight
                        )
                    next_x = next_x.detach()
                    output_detached = True
                else:
                    grad_enabled = torch.is_grad_enabled()
                    next_x = _blend_residual(
                        x, call_block(block_index, repeat_index, block, x), blend_weight
                    )
                    output_detached = False
                if trace is not None:
                    trace.record(block_index, repeat_index, grad_enabled, output_detached)
                x = next_x
        return x
