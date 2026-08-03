from __future__ import annotations

from typing import Any, Callable

import torch

from ..loop_config import LoopConfig
from ..loop_guidance import blend_loop_guidance


def _loopguidance_active(backbone, config: LoopConfig) -> bool:
    if not config.loopguidance_enabled:
        return False
    step_index = getattr(backbone, "_loop_step_index", None)
    total_steps = getattr(backbone, "_loop_total_steps", None)
    if step_index is None or total_steps is None:
        return True
    return config.resolve_k(int(step_index), int(total_steps)) > 0


def _model_prediction(
    model: Callable[..., torch.Tensor],
    x_t: torch.Tensor,
    t: torch.Tensor,
    denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    model_kwargs: dict[str, Any] | None,
) -> torch.Tensor:
    pred = model(x_t, t, **(model_kwargs or {}))
    if denoised_fn is not None:
        pred = denoised_fn(pred)
    return pred


def _set_reference_mode(backbone, enabled: bool) -> Any:
    previous = getattr(backbone, "_scale_rae_loopguidance_reference_mode", None)
    backbone._scale_rae_loopguidance_reference_mode = enabled
    return previous


def _restore_reference_mode(backbone, previous: Any) -> None:
    if previous is None:
        if hasattr(backbone, "_scale_rae_loopguidance_reference_mode"):
            delattr(backbone, "_scale_rae_loopguidance_reference_mode")
    else:
        backbone._scale_rae_loopguidance_reference_mode = previous


def loopguidance_prediction(
    *,
    backbone,
    config: LoopConfig,
    model: Callable[..., torch.Tensor],
    x_t: torch.Tensor,
    t: torch.Tensor,
    denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    model_kwargs: dict[str, Any] | None,
) -> torch.Tensor:
    """Return D_ref + w * (D_loop - D_ref) for one denoiser input."""
    stats = getattr(backbone, "_scale_rae_loop_stats", None)
    previous = _set_reference_mode(backbone, True)
    try:
        reference_pred = _model_prediction(model, x_t, t, denoised_fn, model_kwargs)
    finally:
        _restore_reference_mode(backbone, previous)
    if stats is not None:
        stats.loopguidance_reference_predictions += 1

    loop_pred = _model_prediction(model, x_t, t, denoised_fn, model_kwargs)
    if stats is not None:
        stats.loopguidance_loop_predictions += 1

    weight = float(config.loopguidance_weight)
    return blend_loop_guidance(reference_pred, loop_pred, weight)


def euler_sample_with_loopguidance(
    *,
    flow,
    backbone,
    config: LoopConfig,
    original_euler_sample: Callable[..., torch.Tensor],
    model,
    x_t: torch.Tensor,
    u_t: torch.Tensor,
    u_s: torch.Tensor,
    clip_denoised: bool,
    denoised_fn,
    cond_fn,
    model_kwargs: dict[str, Any] | None,
) -> torch.Tensor:
    if not _loopguidance_active(backbone, config):
        return original_euler_sample(model, x_t, u_t, u_s, clip_denoised, denoised_fn, cond_fn, model_kwargs)

    sigma_t = flow.get_sigmas(u_t)
    sigma_s = flow.get_sigmas(u_s)
    model_pred = loopguidance_prediction(
        backbone=backbone,
        config=config,
        model=model,
        x_t=x_t,
        t=sigma_t,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
    )
    delta_t = (sigma_s - sigma_t).view(-1, 1, 1, 1).to(x_t.device)
    return x_t + delta_t * model_pred


def euler_forward_with_loopguidance(
    *,
    backbone,
    config: LoopConfig,
    original_euler_forward: Callable[..., tuple[torch.Tensor]],
    model,
    x_t: torch.Tensor,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    denoised_fn,
    model_kwargs: dict[str, Any] | None,
) -> tuple[torch.Tensor]:
    if not _loopguidance_active(backbone, config):
        return original_euler_forward(model, x_t, t_curr, t_next, denoised_fn, model_kwargs)

    model_pred = loopguidance_prediction(
        backbone=backbone,
        config=config,
        model=model,
        x_t=x_t,
        t=t_curr,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
    )
    delta_t = (t_next - t_curr).view(-1, 1, 1, 1).to(x_t.device)
    return (x_t + delta_t * model_pred,)
