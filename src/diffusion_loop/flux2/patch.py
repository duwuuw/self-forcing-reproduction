from __future__ import annotations

from typing import Any

import torch

from ..loop_config import LoopConfig
from ..loop_guidance import blend_loop_guidance
from ..loop_stats import LoopStats
from ..loop_strategies import OPERATORS, calls_per_loop


def resolve_flux2_transformer(pipeline_or_transformer):
    transformer = getattr(pipeline_or_transformer, "transformer", pipeline_or_transformer)
    if transformer is None:
        raise TypeError("FLUX.2 adapter requires a pipeline with .transformer or a transformer module")
    return transformer


def resolve_flux2_block_groups(transformer) -> list[tuple[str, Any, int]]:
    groups: list[tuple[str, Any, int]] = []
    double_blocks = getattr(transformer, "transformer_blocks", None)
    if double_blocks is not None:
        groups.append(("double", double_blocks, 0))
    single_blocks = getattr(transformer, "single_transformer_blocks", None)
    if single_blocks is not None:
        groups.append(("single", single_blocks, len(double_blocks or [])))
    if not groups:
        raise TypeError(
            "FLUX.2 adapter requires .transformer_blocks and/or .single_transformer_blocks"
        )
    return groups


def _first_tensor(output: Any) -> torch.Tensor:
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


def _block_target_tensor(output: Any, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    values = output if isinstance(output, (tuple, list)) else ()
    for value in reversed(values):
        if torch.is_tensor(value) and tuple(value.shape) == tuple(reference.shape):
            return value
    sample = getattr(output, "sample", None)
    if torch.is_tensor(sample) and tuple(sample.shape) == tuple(reference.shape):
        return sample
    raise TypeError("FLUX.2 block output does not contain the loop target tensor")


def _with_block_target_tensor(output: Any, reference: torch.Tensor, value: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return value
    if isinstance(output, tuple):
        items = list(output)
        for index in reversed(range(len(items))):
            if torch.is_tensor(items[index]) and tuple(items[index].shape) == tuple(reference.shape):
                items[index] = value
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for index in reversed(range(len(items))):
            if torch.is_tensor(items[index]) and tuple(items[index].shape) == tuple(reference.shape):
                items[index] = value
                return items
    sample = getattr(output, "sample", None)
    if torch.is_tensor(sample) and tuple(sample.shape) == tuple(reference.shape):
        output.sample = value
        return output
    raise TypeError("FLUX.2 block output does not contain the loop target tensor")


def _split_stream_output_tensors(
    output: Any,
    text_reference: torch.Tensor,
    image_reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = output if isinstance(output, (tuple, list)) else ()
    if len(values) >= 2:
        text_value, image_value = values[0], values[1]
        if (
            torch.is_tensor(text_value)
            and torch.is_tensor(image_value)
            and tuple(text_value.shape) == tuple(text_reference.shape)
            and tuple(image_value.shape) == tuple(image_reference.shape)
        ):
            return text_value, image_value
    text_value = None
    image_value = None
    for value in values:
        if torch.is_tensor(value) and text_value is None and tuple(value.shape) == tuple(text_reference.shape):
            text_value = value
        elif torch.is_tensor(value) and image_value is None and tuple(value.shape) == tuple(image_reference.shape):
            image_value = value
    if text_value is None or image_value is None:
        raise TypeError("FLUX.2 split-stream output does not contain text and image tensors")
    return text_value, image_value


def _with_split_stream_output_tensors(
    output: Any,
    text_reference: torch.Tensor,
    image_reference: torch.Tensor,
    text_value: torch.Tensor,
    image_value: torch.Tensor,
) -> Any:
    if not isinstance(output, (tuple, list)):
        raise TypeError("FLUX.2 split-stream output must be a tuple or list")
    items = list(output)
    text_done = False
    image_done = False
    for index, item in enumerate(items):
        if torch.is_tensor(item) and not text_done and tuple(item.shape) == tuple(text_reference.shape):
            items[index] = text_value
            text_done = True
        elif torch.is_tensor(item) and not image_done and tuple(item.shape) == tuple(image_reference.shape):
            items[index] = image_value
            image_done = True
    if not text_done or not image_done:
        raise TypeError("FLUX.2 split-stream output does not contain text and image tensors")
    return tuple(items) if isinstance(output, tuple) else items


def _encoder_hidden_states_arg(args, kwargs: dict[str, Any]):
    if args:
        return args[0]
    return kwargs.get("encoder_hidden_states")


def _with_encoder_hidden_states_arg(
    args,
    kwargs: dict[str, Any],
    value: torch.Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if new_args:
        new_args[0] = value
    else:
        new_kwargs["encoder_hidden_states"] = value
    return tuple(new_args), new_kwargs


def _run_dense_loop(
    original_forward,
    hidden_states: torch.Tensor,
    args,
    kwargs,
    operator,
    k_t: int,
    step_size: float,
) -> tuple[torch.Tensor, Any]:
    latest_output: Any = None

    def tensor_forward(state, *inner_args, **inner_kwargs):
        nonlocal latest_output
        latest_output = original_forward(state, *inner_args, **inner_kwargs)
        return _block_target_tensor(latest_output, state)

    looped = operator(tensor_forward, hidden_states, args, kwargs, k_t, step_size)
    return looped, latest_output


def _run_split_stream_dense_loop(
    original_forward,
    hidden_states: torch.Tensor,
    args,
    kwargs,
    operator,
    k_t: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    encoder_hidden_states = _encoder_hidden_states_arg(args, kwargs)
    if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim < 2:
        raise ValueError("FLUX.2 split-stream looping requires encoder_hidden_states")
    text_tokens = int(encoder_hidden_states.shape[1])
    joint_state = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    latest_output: Any = None

    def tensor_forward(state, *inner_args, **inner_kwargs):
        nonlocal latest_output
        text_state = state[:, :text_tokens, :]
        image_state = state[:, text_tokens:, :]
        next_args, next_kwargs = _with_encoder_hidden_states_arg(args, kwargs, text_state)
        latest_output = original_forward(image_state, *next_args, **next_kwargs)
        text_next, image_next = _split_stream_output_tensors(
            latest_output,
            text_state,
            image_state,
        )
        return torch.cat([text_next, image_next], dim=1)

    looped = operator(tensor_forward, joint_state, (), {}, k_t, step_size)
    return looped[:, :text_tokens, :], looped[:, text_tokens:, :], latest_output


def _set_reference_mode(transformer, enabled: bool) -> Any:
    previous = getattr(transformer, "_flux2_loopguidance_reference_mode", None)
    transformer._flux2_loopguidance_reference_mode = enabled
    return previous


def _restore_reference_mode(transformer, previous: Any) -> None:
    if previous is None:
        if hasattr(transformer, "_flux2_loopguidance_reference_mode"):
            delattr(transformer, "_flux2_loopguidance_reference_mode")
    else:
        transformer._flux2_loopguidance_reference_mode = previous


def _loopguidance_forward(
    transformer,
    args,
    kwargs,
    config: LoopConfig,
    stats: LoopStats,
):
    previous = _set_reference_mode(transformer, True)
    try:
        reference_output = transformer._flux2_original_forward(*args, **kwargs)
    finally:
        _restore_reference_mode(transformer, previous)
    stats.loopguidance_reference_predictions += 1

    loop_output = transformer._flux2_original_forward(*args, **kwargs)
    stats.loopguidance_loop_predictions += 1
    reference_prediction = _first_tensor(reference_output)
    loop_prediction = _first_tensor(loop_output)
    guided = blend_loop_guidance(
        reference_prediction,
        loop_prediction,
        config.loopguidance_weight,
    )
    return _with_first_tensor(loop_output, guided)


def _wrap_transformer_forward(transformer) -> None:
    if hasattr(transformer, "_flux2_original_forward"):
        return
    transformer._flux2_original_forward = transformer.forward

    def forward_with_step_metadata(*args, _transformer=transformer, **kwargs):
        total_steps = getattr(_transformer, "_flux2_loop_total_steps", None)
        if total_steps is not None:
            call_index = int(getattr(_transformer, "_flux2_loop_forward_calls", 0))
            _transformer._flux2_loop_forward_calls = call_index + 1
            _transformer._flux2_loop_step_index = min(call_index, int(total_steps) - 1)
        try:
            config = getattr(_transformer, "_flux2_loop_config", None)
            stats = getattr(_transformer, "_flux2_loop_stats", None)
            if (
                config is not None
                and stats is not None
                and config.loopguidance_enabled
                and total_steps is not None
                and config.resolve_k(
                    int(getattr(_transformer, "_flux2_loop_step_index", 0)),
                    int(total_steps),
                )
                > 0
            ):
                return _loopguidance_forward(_transformer, args, kwargs, config, stats)
            return _transformer._flux2_original_forward(*args, **kwargs)
        finally:
            _transformer._flux2_loop_step_index = None

    transformer.forward = forward_with_step_metadata


def apply_flux2_loop_patch(pipeline_or_transformer, config: LoopConfig) -> LoopStats:
    """Install the paper's layerwise Dense Token Loop for FLUX.2."""
    transformer = resolve_flux2_transformer(pipeline_or_transformer)
    restore_flux2_loop_patch(transformer)
    stats = LoopStats()
    if not config.enabled:
        return stats
    if config.token_operator != "dense":
        raise ValueError("The FLUX.2 adapter supports Dense Token Loop only")
    if config.loop_granularity != "layerwise":
        raise ValueError("The FLUX.2 adapter supports layerwise loops only")

    operator = OPERATORS[config.operator]
    per_loop_calls = calls_per_loop(config.operator)
    groups = resolve_flux2_block_groups(transformer)
    depth = sum(len(blocks) for _, blocks, _ in groups)
    selected_indices = set(config.block_indices or [])
    for index in selected_indices:
        if not 0 <= index < depth:
            raise IndexError(f"block index {index} out of range for FLUX.2 depth {depth}")

    _wrap_transformer_forward(transformer)
    transformer._flux2_loop_config = config
    transformer._flux2_loop_stats = stats

    for stream_kind, blocks, offset in groups:
        for local_index, block in enumerate(blocks):
            block_index = offset + local_index
            if block_index not in selected_indices:
                continue
            if not hasattr(block, "_flux2_original_forward"):
                block._flux2_original_forward = block.forward
            original_forward = block._flux2_original_forward
            stats.patched_block_count += 1

            def looped_forward(
                hidden_states,
                *args,
                _original=original_forward,
                _transformer=transformer,
                _stream_kind=stream_kind,
                **kwargs,
            ):
                stats.patched_block_invocations += 1
                if getattr(_transformer, "_flux2_loopguidance_reference_mode", False):
                    stats.loopguidance_reference_block_calls += 1
                    stats.actual_block_calls += 1
                    return _original(hidden_states, *args, **kwargs)

                step_index = getattr(_transformer, "_flux2_loop_step_index", None)
                total_steps = getattr(_transformer, "_flux2_loop_total_steps", None)
                if step_index is None or total_steps is None:
                    stats.actual_block_calls += 1
                    return _original(hidden_states, *args, **kwargs)

                k_t = config.resolve_k(int(step_index), int(total_steps))
                stats.k_t_sum += k_t
                stats.k_t_observations += 1
                if k_t <= 0:
                    stats.actual_block_calls += 1
                    return _original(hidden_states, *args, **kwargs)

                step_size = config.lambda_for_k(k_t)
                if k_t == 1 and step_size == 1.0:
                    stats.actual_block_calls += 1
                    return _original(hidden_states, *args, **kwargs)

                stats.active_block_invocations += 1
                encoder_hidden_states = _encoder_hidden_states_arg(args, kwargs)
                if _stream_kind == "double":
                    if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim < 3:
                        raise ValueError(
                            "FLUX.2 double-stream looping requires rank-3 "
                            "encoder_hidden_states"
                        )
                    looped_text, looped_image, latest_output = _run_split_stream_dense_loop(
                        _original,
                        hidden_states,
                        args,
                        kwargs,
                        operator,
                        k_t,
                        step_size,
                    )
                else:
                    # Official FLUX.2 concatenates text and image tokens before
                    # entering single-stream blocks; their hidden_states tensor
                    # is therefore the complete joint sequence already.
                    looped_text = None
                    looped_image, latest_output = _run_dense_loop(
                        _original,
                        hidden_states,
                        args,
                        kwargs,
                        operator,
                        k_t,
                        step_size,
                    )

                actual_calls = per_loop_calls * k_t
                stats.actual_block_calls += actual_calls
                stats.extra_block_calls += actual_calls - 1
                if latest_output is None:
                    return looped_image
                if looped_text is not None:
                    return _with_split_stream_output_tensors(
                        latest_output,
                        encoder_hidden_states,
                        hidden_states,
                        looped_text,
                        looped_image,
                    )
                return _with_block_target_tensor(latest_output, hidden_states, looped_image)

            block.forward = looped_forward

    return stats


def set_flux2_step_metadata(
    pipeline_or_transformer,
    *,
    step_index: int | None = None,
    total_steps: int,
) -> None:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if step_index is not None and not 0 <= int(step_index) < int(total_steps):
        raise ValueError("step_index must be within the sampling schedule")
    transformer = resolve_flux2_transformer(pipeline_or_transformer)
    transformer._flux2_loop_total_steps = int(total_steps)
    transformer._flux2_loop_forward_calls = 0 if step_index is None else int(step_index)
    transformer._flux2_loop_step_index = step_index


def clear_flux2_step_metadata(pipeline_or_transformer) -> None:
    transformer = resolve_flux2_transformer(pipeline_or_transformer)
    for attr in (
        "_flux2_loop_step_index",
        "_flux2_loop_total_steps",
        "_flux2_loop_forward_calls",
    ):
        if hasattr(transformer, attr):
            delattr(transformer, attr)


def restore_flux2_loop_patch(pipeline_or_transformer) -> None:
    transformer = resolve_flux2_transformer(pipeline_or_transformer)
    for _, blocks, _ in resolve_flux2_block_groups(transformer):
        for block in blocks:
            original = getattr(block, "_flux2_original_forward", None)
            if original is not None:
                block.forward = original
                delattr(block, "_flux2_original_forward")
    original_transformer_forward = getattr(transformer, "_flux2_original_forward", None)
    if original_transformer_forward is not None:
        transformer.forward = original_transformer_forward
        delattr(transformer, "_flux2_original_forward")
    for attr in (
        "_flux2_loop_config",
        "_flux2_loop_stats",
        "_flux2_loopguidance_reference_mode",
    ):
        if hasattr(transformer, attr):
            delattr(transformer, attr)
    clear_flux2_step_metadata(transformer)
