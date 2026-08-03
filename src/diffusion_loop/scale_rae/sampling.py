from __future__ import annotations

from .loopguidance import euler_forward_with_loopguidance, euler_sample_with_loopguidance
from .patch import resolve_scale_rae_backbone


# Scale-RAE's diffusion head treats cfg_interval as a CFG bypass interval.
# Equal endpoints make that open interval empty, so every sampler step uses the
# requested guidance scale instead of silently falling back to cond_eps.
CFG_ALWAYS_ENABLED_INTERVAL = (-10_000.0, -10_000.0)


def _normalize_cfg_interval(value) -> tuple[float, float]:
    try:
        lower, upper = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Scale-RAE cfg_interval must contain exactly two values, got {value!r}") from exc
    return float(lower), float(upper)


def set_scale_rae_cfg_config(
    model,
    guidance_level: float,
    *,
    cfg_sweep_enabled: bool = False,
) -> dict[str, object]:
    """Make Scale-RAE's requested CFG scale effective and return provenance.

    The vendored diffusion head defaults cfg_interval to [-1e4, 1e4]. Its
    forward_with_cfg implementation bypasses CFG inside that interval, which
    covers every normal sampler timestep. For guidance levels above one, use an
    empty bypass interval so cfg_scale participates in every denoising step.
    """

    level = float(guidance_level)
    if level <= 0.0:
        raise ValueError(f"guidance_level must be > 0, got {level}")
    if cfg_sweep_enabled and level <= 1.0:
        raise ValueError(f"CFG sweep variants must use guidance_level > 1, got {level}")

    diff_head = model.diff_head
    if not hasattr(diff_head, "_scale_rae_original_cfg_interval"):
        original = _normalize_cfg_interval(getattr(diff_head, "cfg_interval", (-10_000.0, 10_000.0)))
        diff_head._scale_rae_original_cfg_interval = original
    original = _normalize_cfg_interval(diff_head._scale_rae_original_cfg_interval)

    cfg_enabled = level > 1.0
    effective = CFG_ALWAYS_ENABLED_INTERVAL if cfg_enabled else original
    diff_head.cfg_interval = list(effective)
    return {
        "guidance_level": level,
        "cfg_enabled": cfg_enabled,
        "cfg_sweep_enabled": bool(cfg_sweep_enabled),
        "cfg_interval": list(effective),
        "original_cfg_interval": list(original),
    }


def set_scale_rae_sampler_step_count(model, step_count: int | None) -> dict[str, int | list[int] | None]:
    """Optionally reduce the official Rectified Flow sampler step count.

    ``None`` restores the checkpoint's default ``used_timesteps``. Otherwise,
    select ``step_count + 1`` evenly spaced points from that schedule and
    return provenance for the resulting outer-sampling protocol.
    """

    flow = model.diff_head.inference_flow
    if not hasattr(flow, "_scale_rae_original_used_timesteps"):
        flow._scale_rae_original_used_timesteps = list(flow.used_timesteps)
    original = list(flow._scale_rae_original_used_timesteps)
    default_transitions = max(0, len(original) - 1)
    if step_count is None:
        flow.used_timesteps = list(original)
        return {
            "requested_step_count": None,
            "observed_step_count": default_transitions,
            "default_step_count": default_transitions,
            "used_timesteps": list(flow.used_timesteps),
        }
    if step_count < 1:
        raise ValueError(f"step_count must be >= 1, got {step_count}")
    if step_count > default_transitions:
        raise ValueError(f"step_count {step_count} exceeds default transitions {default_transitions}")
    last = len(original) - 1
    indices = [round(i * last / step_count) for i in range(step_count + 1)]
    if len(set(indices)) != len(indices):
        raise ValueError(f"step_count {step_count} cannot be represented with unique timesteps from {default_transitions}")
    flow.used_timesteps = [original[index] for index in indices]
    return {
        "requested_step_count": int(step_count),
        "observed_step_count": max(0, len(flow.used_timesteps) - 1),
        "default_step_count": default_transitions,
        "used_timesteps": list(flow.used_timesteps),
    }


def apply_scale_rae_sampler_step_hook(model) -> None:
    """Expose outer-sampler progress to patched DiT blocks.

    Before each official denoising step, the hook records
    ``_loop_step_index`` and ``_loop_total_steps``. The original sampler still
    performs the numerical integration.
    """

    backbone = resolve_scale_rae_backbone(model)
    flow = model.diff_head.inference_flow
    if hasattr(flow, "_scale_rae_original_step_fn"):
        return
    flow._scale_rae_original_step_fn = flow.step_fn
    flow._scale_rae_original_euler_forward = flow.euler_forward
    if hasattr(flow, "euler_sample"):
        flow._scale_rae_original_euler_sample = flow.euler_sample
    flow._scale_rae_step_index = 0

    def set_step_metadata():
        total = max(0, len(flow.used_timesteps) - 1)
        backbone._loop_step_index = flow._scale_rae_step_index
        backbone._loop_total_steps = total
        flow._scale_rae_step_index += 1

    def hooked_step_fn(*args, **kwargs):
        set_step_metadata()
        config = getattr(flow, "_scale_rae_loopguidance_config", None)
        original_name = getattr(flow._scale_rae_original_step_fn, "__name__", "")
        if config is not None and getattr(config, "loopguidance_enabled", False) and original_name == "euler_forward":
            return euler_forward_with_loopguidance(
                backbone=backbone,
                config=config,
                original_euler_forward=flow._scale_rae_original_step_fn,
                model=args[0],
                x_t=args[1],
                t_curr=args[2],
                t_next=args[3],
                denoised_fn=args[4],
                model_kwargs=args[5],
            )
        return flow._scale_rae_original_step_fn(*args, **kwargs)

    def hooked_euler_forward(*args, **kwargs):
        set_step_metadata()
        config = getattr(flow, "_scale_rae_loopguidance_config", None)
        return euler_forward_with_loopguidance(
            backbone=backbone,
            config=config,
            original_euler_forward=flow._scale_rae_original_euler_forward,
            model=args[0],
            x_t=args[1],
            t_curr=args[2],
            t_next=args[3],
            denoised_fn=args[4],
            model_kwargs=args[5],
        ) if config is not None else flow._scale_rae_original_euler_forward(*args, **kwargs)

    def hooked_euler_sample(*args, **kwargs):
        set_step_metadata()
        config = getattr(flow, "_scale_rae_loopguidance_config", None)
        return euler_sample_with_loopguidance(
            flow=flow,
            backbone=backbone,
            config=config,
            original_euler_sample=flow._scale_rae_original_euler_sample,
            model=args[0],
            x_t=args[1],
            u_t=args[2],
            u_s=args[3],
            clip_denoised=args[4],
            denoised_fn=args[5],
            cond_fn=args[6],
            model_kwargs=args[7],
        ) if config is not None else flow._scale_rae_original_euler_sample(*args, **kwargs)

    flow.step_fn = hooked_step_fn
    flow.euler_forward = hooked_euler_forward
    if hasattr(flow, "_scale_rae_original_euler_sample"):
        flow.euler_sample = hooked_euler_sample


def set_scale_rae_loopguidance_config(model, config) -> None:
    """Expose the current Loop Guidance configuration to the sampler hook."""
    flow = model.diff_head.inference_flow
    flow._scale_rae_loopguidance_config = config


def reset_scale_rae_sampler_step_hook(model) -> None:
    """Reset the tracked sampler index before generating another image."""
    flow = model.diff_head.inference_flow
    if hasattr(flow, "_scale_rae_step_index"):
        flow._scale_rae_step_index = 0


def restore_scale_rae_sampler_step_hook(model) -> None:
    """Restore original sampler functions and clear hook metadata."""
    flow = model.diff_head.inference_flow
    original_step_fn = getattr(flow, "_scale_rae_original_step_fn", None)
    if original_step_fn is not None:
        flow.step_fn = original_step_fn
        del flow._scale_rae_original_step_fn
    original_euler_forward = getattr(flow, "_scale_rae_original_euler_forward", None)
    if original_euler_forward is not None:
        flow.euler_forward = original_euler_forward
        del flow._scale_rae_original_euler_forward
    original_euler_sample = getattr(flow, "_scale_rae_original_euler_sample", None)
    if original_euler_sample is not None:
        flow.euler_sample = original_euler_sample
        del flow._scale_rae_original_euler_sample
    if hasattr(flow, "_scale_rae_step_index"):
        del flow._scale_rae_step_index
    if hasattr(flow, "_scale_rae_loopguidance_config"):
        del flow._scale_rae_loopguidance_config
