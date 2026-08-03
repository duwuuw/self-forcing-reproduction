from __future__ import annotations

from functools import wraps
from typing import Callable

from ..loop_config import LoopConfig
from ..loop_guidance import blend_loop_guidance

from .patch import (
    clear_raev2_step_metadata,
    resolve_raev2_backbone,
    set_raev2_step_metadata,
)


def wrap_raev2_model_fn(
    model_fn: Callable,
    model,
    total_steps: int,
    config: LoopConfig | None = None,
) -> Callable:
    """Track one RAEv2 ODE model call per Euler sampling step.

    The official sampler invokes the supplied model function once for each ODE
    transition, including CFG/IG modes. Call ``reset_raev2_model_fn`` before
    reusing the wrapper for another generated batch.
    """

    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")

    @wraps(model_fn)
    def tracked(*args, **kwargs):
        step_index = tracked._raev2_step_index
        if step_index >= total_steps:
            raise RuntimeError(
                "RAEv2 tracked model_fn exceeded total_steps; "
                "call reset_raev2_model_fn before the next sample"
            )
        set_raev2_step_metadata(model, step_index, total_steps)
        tracked._raev2_step_index += 1
        if (
            config is not None
            and config.loopguidance_enabled
            and config.resolve_k(step_index, total_steps) > 0
        ):
            backbone = resolve_raev2_backbone(model)
            stats = getattr(backbone, "_raev2_loop_stats", None)
            backbone._raev2_loopguidance_reference_mode = True
            try:
                reference = model_fn(*args, **kwargs)
            finally:
                backbone._raev2_loopguidance_reference_mode = False
            if stats is not None:
                stats.loopguidance_reference_predictions += 1
            looped = model_fn(*args, **kwargs)
            if stats is not None:
                stats.loopguidance_loop_predictions += 1
            return blend_loop_guidance(reference, looped, config.loopguidance_weight)
        return model_fn(*args, **kwargs)

    tracked._raev2_step_index = 0
    tracked._raev2_total_steps = int(total_steps)
    tracked._raev2_model = model
    return tracked

def reset_raev2_model_fn(model_fn: Callable) -> None:
    if not hasattr(model_fn, "_raev2_step_index"):
        raise TypeError("model_fn was not created by wrap_raev2_model_fn")
    model_fn._raev2_step_index = 0
    clear_raev2_step_metadata(model_fn._raev2_model)
