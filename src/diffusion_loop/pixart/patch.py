from __future__ import annotations

from typing import Any

import torch

from ..loop_config import LoopConfig
from ..loop_guidance import blend_loop_guidance
from ..loop_stats import LoopStats
from ..loop_strategies import OPERATORS, calls_per_loop

PIXART_DEPTH = 28
SUPPORTED_OPERATORS = ("euler",)


def resolve_pixart_transformer(pipeline_or_transformer):
    """Return the PixArtTransformer2DModel from a PixArtAlphaPipeline or module."""
    transformer = getattr(pipeline_or_transformer, "transformer", pipeline_or_transformer)
    if transformer is None or getattr(transformer, "transformer_blocks", None) is None:
        raise TypeError(
            "PixArt adapter requires PixArtAlphaPipeline.transformer or a transformer with .transformer_blocks"
        )
    return transformer


def _validate_step(step_index: int, total_steps: int) -> None:
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if not 0 <= step_index < total_steps:
        raise ValueError(f"step_index must be in [0, {total_steps}), got {step_index}")


def set_pixart_step_metadata(
    pipeline_or_transformer,
    *,
    total_steps: int,
    step_index: int | None = None,
) -> None:
    """Bind the sampler step count so block loops can resolve the t-window.

    PixArt's diffusers pipeline does not pass an explicit step index into the
    transformer forward, so the wrapped forward counts calls: the Nth forward
    call after this binding is treated as sampler step N (or ``step_index + N``
    when resuming a shard mid-schedule).
    """
    transformer = resolve_pixart_transformer(pipeline_or_transformer)
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if step_index is not None:
        _validate_step(int(step_index), int(total_steps))
    transformer._pixart_loop_total_steps = int(total_steps)
    transformer._pixart_loop_step_index = None if step_index is None else int(step_index)
    transformer._pixart_loop_forward_calls = 0 if step_index is None else int(step_index)


def clear_pixart_step_metadata(pipeline_or_transformer) -> None:
    transformer = resolve_pixart_transformer(pipeline_or_transformer)
    for attr in (
        "_pixart_loop_total_steps",
        "_pixart_loop_step_index",
        "_pixart_loop_forward_calls",
    ):
        if hasattr(transformer, attr):
            delattr(transformer, attr)


def _first_tensor(output: Any) -> torch.Tensor:
    """Return the prediction tensor of a PixArt transformer output.

    diffusers PixArtTransformer2DModel returns Transformer2DModelOutput with
    the prediction in ``.sample``; stay defensive for tuple/plain-tensor
    outputs so tests and diffusers upgrades keep working.
    """
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    return output


def _with_first_tensor(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    if hasattr(output, "sample"):
        output.sample = value
        return output
    return value


def _set_reference_mode(transformer, enabled: bool) -> Any:
    previous = getattr(transformer, "_pixart_loopguidance_reference_mode", None)
    transformer._pixart_loopguidance_reference_mode = enabled
    return previous


def _restore_reference_mode(transformer, previous: Any) -> None:
    if previous is None:
        if hasattr(transformer, "_pixart_loopguidance_reference_mode"):
            delattr(transformer, "_pixart_loopguidance_reference_mode")
    else:
        transformer._pixart_loopguidance_reference_mode = previous


def _loopguidance_forward(transformer, args, kwargs, config: LoopConfig, stats: LoopStats):
    """LoopGuidance two-pass forward: extrapolate loop vs no-loop predictions.

    The reference pass runs the original forward with the reference-mode flag
    ON so patched blocks skip looping; the loop pass runs with loops active.
    Both passes happen inside one wrapped forward call, so the step-metadata
    counter advances exactly once per sampler step (same as flux2).
    """
    previous = _set_reference_mode(transformer, True)
    try:
        reference_output = transformer._pixart_original_forward(*args, **kwargs)
    finally:
        _restore_reference_mode(transformer, previous)
    stats.loopguidance_reference_predictions += 1

    loop_output = transformer._pixart_original_forward(*args, **kwargs)
    stats.loopguidance_loop_predictions += 1

    reference_pred = _first_tensor(reference_output)
    loop_pred = _first_tensor(loop_output)
    guided = blend_loop_guidance(reference_pred, loop_pred, config.loopguidance_weight)
    return _with_first_tensor(loop_output, guided)


def _wrap_transformer_forward(transformer) -> None:
    if hasattr(transformer, "_pixart_original_forward"):
        return
    transformer._pixart_original_forward = transformer.forward

    def forward_with_step_metadata(*args, _transformer=transformer, **kwargs):
        total_steps = getattr(_transformer, "_pixart_loop_total_steps", None)
        if total_steps is None:
            return _transformer._pixart_original_forward(*args, **kwargs)
        step_index = int(getattr(_transformer, "_pixart_loop_forward_calls", 0))
        _transformer._pixart_loop_forward_calls = step_index + 1
        _validate_step(step_index, int(total_steps))
        _transformer._pixart_loop_step_index = step_index
        try:
            config = getattr(_transformer, "_pixart_loop_config", None)
            stats = getattr(_transformer, "_pixart_loop_stats", None)
            if (
                config is not None
                and stats is not None
                and getattr(config, "loopguidance_enabled", False)
                and config.resolve_k(step_index, int(total_steps)) > 0
            ):
                return _loopguidance_forward(_transformer, args, kwargs, config, stats)
            return _transformer._pixart_original_forward(*args, **kwargs)
        finally:
            _transformer._pixart_loop_step_index = None

    transformer.forward = forward_with_step_metadata


def _block_target_tensor(output: Any, reference: torch.Tensor) -> torch.Tensor:
    """Extract the hidden-state tensor from a PixArt block output.

    diffusers BasicTransformerBlock (as used by PixArtTransformer2DModel)
    returns a single tensor; stay defensive against tuple/list outputs so a
    diffusers upgrade fails loudly instead of silently looping the wrong value.
    """
    if torch.is_tensor(output):
        return output
    values = output if isinstance(output, (tuple, list)) else ()
    for value in values:
        if torch.is_tensor(value) and tuple(value.shape) == tuple(reference.shape):
            return value
    raise TypeError("PixArt block output does not contain a tensor matching the loop target")


def _with_block_target_tensor(output: Any, reference: torch.Tensor, value: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return value
    if isinstance(output, tuple):
        return tuple(
            value if torch.is_tensor(item) and tuple(item.shape) == tuple(reference.shape) else item
            for item in output
        )
    if isinstance(output, list):
        return [
            value if torch.is_tensor(item) and tuple(item.shape) == tuple(reference.shape) else item
            for item in output
        ]
    raise TypeError("PixArt block output does not contain a tensor matching the loop target")


def _run_dense_token_loop(
    original_forward,
    hidden_states: torch.Tensor,
    args,
    kwargs,
    operator,
    k_t: int,
    lambda_value: float,
) -> Any:
    """Apply Dense Token Loop to one PixArt block.

    The block is re-called with the same conditioning kwargs it received from
    PixArtTransformer2DModel (``attention_mask``, ``encoder_hidden_states``,
    ``encoder_attention_mask``, ``timestep``, ``cross_attention_kwargs``,
    ``class_labels``); they are captured positionally/by keyword and passed
    through unchanged on every loop pass.
    """
    reference = hidden_states
    captured: dict[str, Any] = {"output": None}

    def block_forward(state, *inner_args, **inner_kwargs):
        output = original_forward(state, *inner_args, **inner_kwargs)
        captured["output"] = output
        return _block_target_tensor(output, reference)

    looped = operator(block_forward, hidden_states, args, kwargs, k_t, lambda_value)
    return _with_block_target_tensor(captured["output"], reference, looped)


def apply_pixart_loop_patch(pipeline_or_transformer, config: LoopConfig) -> LoopStats:
    """Patch PixArt-alpha blocks with inference-only Dense Token Loop."""

    if config.token_operator != "dense":
        raise ValueError("The PixArt adapter supports Dense Token Loop only")
    if config.loop_granularity != "layerwise":
        raise ValueError("The PixArt adapter supports layerwise loops only")
    if config.operator not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"PixArt v1 supports operators {SUPPORTED_OPERATORS} only, got {config.operator!r}"
        )
    if config.loopguidance_enabled and config.loopguidance_branch != "base":
        raise ValueError(
            f"PixArt LoopGuidance supports branch='base' only, got {config.loopguidance_branch!r}"
        )
    transformer = resolve_pixart_transformer(pipeline_or_transformer)
    restore_pixart_loop_patch(transformer)
    blocks = transformer.transformer_blocks
    depth = len(blocks)
    for index in config.block_indices or []:
        if not 0 <= index < depth:
            raise IndexError(f"block index {index} out of range for PixArt depth {depth}")

    stats = LoopStats()
    if not config.enabled:
        return stats
    stats.patched_block_count = len(config.block_indices or [])
    operator = OPERATORS[config.operator]
    per_loop_calls = calls_per_loop(config.operator)
    _wrap_transformer_forward(transformer)

    for block_index in config.block_indices or []:
        block = blocks[block_index]
        block._pixart_original_forward = block.forward
        original_forward = block._pixart_original_forward

        def looped_forward(
            hidden_states,
            *args,
            _original=original_forward,
            **kwargs,
        ):
            stats.patched_block_invocations += 1
            if getattr(transformer, "_pixart_loopguidance_reference_mode", False):
                # LoopGuidance reference pass: plain single block call, no loop.
                stats.loopguidance_reference_block_calls += 1
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            step_index = getattr(transformer, "_pixart_loop_step_index", None)
            total_steps = getattr(transformer, "_pixart_loop_total_steps", None)
            if step_index is None or total_steps is None:
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            k_t = config.resolve_k(int(step_index), int(total_steps))
            stats.k_t_sum += k_t
            stats.k_t_observations += 1
            if k_t <= 0:
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            lambda_value = config.lambda_for_k(k_t)
            if config.operator == "euler" and k_t == 1 and lambda_value == 1.0:
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            stats.active_block_invocations += 1
            actual_calls = per_loop_calls * k_t
            stats.actual_block_calls += actual_calls
            stats.extra_block_calls += actual_calls - 1
            return _run_dense_token_loop(
                _original,
                hidden_states,
                args,
                kwargs,
                operator,
                k_t,
                lambda_value,
            )

        block.forward = looped_forward

    transformer._pixart_loop_config = config
    transformer._pixart_loop_stats = stats
    return stats


def restore_pixart_loop_patch(pipeline_or_transformer) -> None:
    transformer = resolve_pixart_transformer(pipeline_or_transformer)
    for block in transformer.transformer_blocks:
        original = getattr(block, "_pixart_original_forward", None)
        if original is not None:
            block.forward = original
            delattr(block, "_pixart_original_forward")
    original_transformer = getattr(transformer, "_pixart_original_forward", None)
    if original_transformer is not None:
        transformer.forward = original_transformer
        delattr(transformer, "_pixart_original_forward")
    for attr in ("_pixart_loop_config", "_pixart_loop_stats", "_pixart_loopguidance_reference_mode"):
        if hasattr(transformer, attr):
            delattr(transformer, attr)
    clear_pixart_step_metadata(transformer)
