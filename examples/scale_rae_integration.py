"""Minimal Scale-RAE integration for an already loaded public checkpoint.

Loading remains delegated to the checkpoint's public repository because its
wrapper differs between releases. This file shows every method-specific hook
that must be installed before invoking the checkpoint's ordinary sampler.
"""

from __future__ import annotations

from diffusion_loop.config_resolution import ResolvedRunConfig
from diffusion_loop.scale_rae import (
    apply_scale_rae_loop_patch,
    apply_scale_rae_sampler_step_hook,
    reset_scale_rae_sampler_step_hook,
    set_scale_rae_loopguidance_config,
    set_scale_rae_sampler_step_count,
)


def configure_scale_rae(model, resolved: ResolvedRunConfig):
    """Install the selected Dense/Sparse loop and optional Loop Guidance."""
    config = resolved.loop_config
    stats = apply_scale_rae_loop_patch(model, config)
    set_scale_rae_sampler_step_count(
        model, int(resolved.generation["num_inference_steps"])
    )
    apply_scale_rae_sampler_step_hook(model)
    set_scale_rae_loopguidance_config(model, config)
    return stats


def reset_before_next_image(model) -> None:
    """Reset the tracked outer-step index before every sampler invocation."""
    reset_scale_rae_sampler_step_hook(model)
