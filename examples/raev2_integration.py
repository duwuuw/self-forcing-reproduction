"""Minimal RAEv2 integration for an already loaded public checkpoint."""

from __future__ import annotations

from collections.abc import Callable

from diffusion_loop.config_resolution import ResolvedRunConfig
from diffusion_loop.raev2 import (
    apply_raev2_loop_patch,
    reset_raev2_model_fn,
    wrap_raev2_model_fn,
)


def configure_raev2(
    model,
    official_model_fn: Callable,
    resolved: ResolvedRunConfig,
) -> tuple[Callable, object]:
    """Patch blocks and return the model function passed to the official ODE sampler."""
    config = resolved.loop_config
    stats = apply_raev2_loop_patch(model, config)
    tracked_model_fn = wrap_raev2_model_fn(
        official_model_fn,
        model,
        int(resolved.generation["num_inference_steps"]),
        config,
    )
    return tracked_model_fn, stats


def reset_before_next_image(tracked_model_fn: Callable) -> None:
    """Reset the tracked outer-step index before every sampler invocation."""
    reset_raev2_model_fn(tracked_model_fn)
