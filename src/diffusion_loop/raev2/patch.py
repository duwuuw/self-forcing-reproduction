from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..loop_config import LoopConfig
from ..loop_stats import LoopStats
from ..loop_strategies import OPERATORS, calls_per_loop
from ..token_loop import sparse_loop_call_count, sparse_token_loop
from ..token_selection import random_score, score_to_topk_mask


def resolve_raev2_backbone(model):
    """Resolve the official RAEv2 stage-2 DiT from common wrappers."""

    candidate = model
    seen: set[int] = set()
    while getattr(candidate, "blocks", None) is None:
        if id(candidate) in seen:
            break
        seen.add(id(candidate))
        for attr in ("module", "model", "stage_2", "diffusion_model"):
            nested = getattr(candidate, attr, None)
            if nested is not None:
                candidate = nested
                break
        else:
            break
    if getattr(candidate, "blocks", None) is None:
        raise TypeError("RAEv2 adapter requires a stage-2 backbone with .blocks")
    return candidate


def resolve_raev2_blocks(backbone):
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise TypeError("RAEv2 adapter requires a stage-2 backbone with .blocks")
    return blocks


def set_raev2_step_metadata(model, step_index: int, total_steps: int) -> None:
    """Expose the official ODE sampler progress to patched RAEv2 blocks."""

    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if not 0 <= step_index < total_steps:
        raise ValueError(
            f"step_index must be in [0, {total_steps}), got {step_index}"
        )
    backbone = resolve_raev2_backbone(model)
    backbone._loop_step_index = int(step_index)
    backbone._loop_total_steps = int(total_steps)


def clear_raev2_step_metadata(model) -> None:
    backbone = resolve_raev2_backbone(model)
    for attr in ("_loop_step_index", "_loop_total_steps"):
        if hasattr(backbone, attr):
            delattr(backbone, attr)


def _validate_window_indices(config: LoopConfig, depth: int) -> list[int]:
    indices = [int(index) for index in (config.block_indices or [])]
    if not indices:
        raise ValueError("RAEv2 range-wise loop requires a non-empty block range")
    if any(not 0 <= index < depth for index in indices):
        raise IndexError(f"block index out of range for RAEv2 depth {depth}: {indices}")
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            "RAEv2 range-wise loop requires contiguous, ascending block_indices, "
            f"got {config.block_indices!r}"
        )
    return indices


def _run_window_once(
    window_indices: list[int],
    original_forwards: dict[int, Any],
    hidden_states,
    args,
    kwargs,
):
    state = hidden_states
    for block_index in window_indices:
        state = original_forwards[block_index](state, *args, **kwargs)
    return state


def _mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    selected = mask.squeeze(-1).to(dtype=torch.bool)
    counts = selected.sum(dim=1)
    if counts.numel() == 0 or int(counts.min().item()) <= 0:
        raise ValueError("RAEv2 token selector must select at least one image token")
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError(
            "RAEv2 selected-query attention requires equal token counts per batch"
        )
    if selected.shape[0] == 1:
        return selected[0].nonzero(as_tuple=False).flatten().unsqueeze(0)
    return selected.to(dtype=torch.int64).topk(
        k=int(counts[0].item()), dim=1
    ).indices


def _gather_tokens(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.shape[0] == 1:
        return value[0].index_select(0, indices[0]).unsqueeze(0)
    return value.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, value.shape[-1])
    )


def _scatter_tokens(
    base: torch.Tensor, indices: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    output = base.clone()
    if base.shape[0] == 1:
        output[0].index_copy_(0, indices[0], selected[0])
        return output
    return output.scatter(
        1, indices.unsqueeze(-1).expand(-1, -1, base.shape[-1]), selected
    )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    paired = value.reshape(*value.shape[:-1], value.shape[-1] // 2, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_selected_rope(
    value: torch.Tensor, rope, indices: torch.Tensor
) -> torch.Tensor:
    if not all(hasattr(rope, attr) for attr in ("freqs_cos", "freqs_sin")):
        raise TypeError("RAEv2 selected-query attention requires indexed RoPE buffers")
    cosine = rope.freqs_cos.to(device=value.device, dtype=value.dtype)
    sine = rope.freqs_sin.to(device=value.device, dtype=value.dtype)
    if indices.shape[0] == 1:
        cosine = cosine.index_select(0, indices[0]).unsqueeze(0)
        sine = sine.index_select(0, indices[0]).unsqueeze(0)
    else:
        gather_index = indices.unsqueeze(-1).expand(-1, -1, cosine.shape[-1])
        cosine = cosine.unsqueeze(0).expand(indices.shape[0], -1, -1).gather(
            1, gather_index
        )
        sine = sine.unsqueeze(0).expand(indices.shape[0], -1, -1).gather(
            1, gather_index
        )
    cosine = cosine[:, None, :, :]
    sine = sine[:, None, :, :]
    return value * cosine + _rotate_half(value) * sine


def _selected_image_forward(
    block,
    hidden_states: torch.Tensor,
    args,
    kwargs,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Run selected image queries against full image/time/text key-value context."""

    if not all(hasattr(block, attr) for attr in ("attn", "mlp", "norm1", "norm2")):
        return None
    attention = block.attn
    if not all(
        hasattr(attention, attr)
        for attr in (
            "q",
            "k",
            "v",
            "proj",
            "q_norm",
            "k_norm",
            "num_heads",
            "head_dim",
        )
    ):
        return None
    rope = args[0] if args else kwargs.get("rope")
    if rope is None:
        return None
    attention_mask = args[1] if len(args) >= 2 else kwargs.get("attn_mask")

    indices = _mask_to_indices(mask)
    normalized = block.norm1(hidden_states)
    selected_normalized = _gather_tokens(normalized, indices)
    batch, selected_count, _ = selected_normalized.shape
    total_tokens = normalized.shape[1]

    query = attention.q(selected_normalized).reshape(
        batch, selected_count, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    key = attention.k(normalized).reshape(
        batch, total_tokens, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    value = attention.v(normalized).reshape(
        batch, total_tokens, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    query = _apply_selected_rope(attention.q_norm(query), rope, indices)
    key = rope(attention.k_norm(key))
    selected_attention = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask
    )
    selected_attention = selected_attention.permute(0, 2, 1, 3).reshape(
        batch, selected_count, -1
    )
    selected_hidden = _gather_tokens(hidden_states, indices)
    selected_hidden = selected_hidden + attention.proj(selected_attention)
    selected_hidden = selected_hidden + block.mlp(block.norm2(selected_hidden))
    return _scatter_tokens(hidden_states, indices, selected_hidden)


def _token_masks(
    backbone,
    config: LoopConfig,
    hidden_states: torch.Tensor,
    block_index: int,
    step_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_domain = str(config.token_domain)
    if token_domain == "backend_default":
        token_domain = "image_prefix_only"
    image_tokens = int(
        getattr(getattr(backbone, "s_embedder", None), "num_patches", 0)
    )
    if not 0 < image_tokens < hidden_states.shape[1]:
        raise ValueError(
            "RAEv2 token-selective loop cannot resolve image and condition tokens"
        )
    if token_domain == "image_vs_condition":
        selected = torch.zeros(
            (*hidden_states.shape[:2], 1),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        selected[:, :image_tokens] = True
        return selected, ~selected
    if token_domain == "all_tokens":
        candidate_mask = torch.ones(
            (*hidden_states.shape[:2], 1),
            dtype=torch.bool,
            device=hidden_states.device,
        )
    elif token_domain == "image_prefix_only":
        candidate_mask = torch.zeros(
            (*hidden_states.shape[:2], 1),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        candidate_mask[:, :image_tokens] = True
    else:
        raise ValueError(f"unsupported RAEv2 token_domain: {config.token_domain!r}")
    score = random_score(
        hidden_states,
        config.selector_seed,
        block_index,
        step_index,
        config.selector,
    )
    selected = score_to_topk_mask(
        score, config.selector_ratio, candidate_mask=candidate_mask
    )
    complement = candidate_mask & ~selected
    if not selected.any() or not complement.any():
        raise ValueError(
            "Sparse Token Loop requires both selected and complement tokens"
        )
    return selected, complement


def _run_sparse_token_loop(
    *,
    block,
    original_forward,
    hidden_states: torch.Tensor,
    args,
    kwargs,
    selected_mask: torch.Tensor,
    complement_mask: torch.Tensor,
    config: LoopConfig,
    k_t: int,
    lambda_value: float,
) -> torch.Tensor:
    masks = {
        "selected": selected_mask.to(device=hidden_states.device, dtype=torch.bool),
        "complement": complement_mask.to(device=hidden_states.device, dtype=torch.bool),
    }
    masks["all"] = masks["selected"] | masks["complement"]

    def selected_forward(state: torch.Tensor, group: str) -> torch.Tensor:
        mask = masks[group]
        selected = _selected_image_forward(block, state, args, kwargs, mask)
        if selected is not None:
            return selected
        fallback = original_forward(state, *args, **kwargs)
        return torch.where(mask, fallback, state)

    return sparse_token_loop(
        selected_forward,
        hidden_states,
        masks,
        operator=config.operator,
        step_size=lambda_value,
        num_loops=k_t,
    )


def _apply_raev2_window_patch(
    backbone,
    blocks,
    config: LoopConfig,
    stats: LoopStats,
    operator,
    per_loop_calls: int,
) -> None:
    window_indices = _validate_window_indices(config, len(blocks))
    window_start = window_indices[0]
    window_followers = set(window_indices[1:])
    original_forwards: dict[int, Any] = {}
    for block_index in window_indices:
        block = blocks[block_index]
        if not hasattr(block, "_raev2_original_forward"):
            block._raev2_original_forward = block.forward
        original_forwards[block_index] = block._raev2_original_forward

    for block_index in window_indices:
        block = blocks[block_index]
        original_forward = original_forwards[block_index]
        if block_index == window_start:

            def window_start_forward(
                hidden_states,
                *args,
                _original=original_forward,
                **kwargs,
            ):
                stats.patched_block_invocations += 1
                if getattr(backbone, "_raev2_loopguidance_reference_mode", False):
                    stats.loopguidance_reference_block_calls += 1
                    stats.actual_block_calls += 1
                    return _original(hidden_states, *args, **kwargs)

                step_index = getattr(backbone, "_loop_step_index", None)
                total_steps = getattr(backbone, "_loop_total_steps", None)
                if not config.enabled or step_index is None or total_steps is None:
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

                def window_forward(state, *inner_args, **inner_kwargs):
                    return _run_window_once(window_indices, original_forwards, state, args, kwargs)

                looped = operator(window_forward, hidden_states, (), {}, k_t, lambda_value)
                window_size = len(window_indices)
                actual_calls = per_loop_calls * k_t * window_size
                stats.active_block_invocations += 1
                stats.actual_block_calls += actual_calls
                stats.extra_block_calls += actual_calls - window_size
                backbone._raev2_window_skip_blocks = set(window_followers)
                return looped

            block.forward = window_start_forward
            continue

        def window_follower_forward(
            hidden_states,
            *args,
            _original=original_forward,
            _block_index=block_index,
            **kwargs,
        ):
            stats.patched_block_invocations += 1
            skip_blocks = getattr(backbone, "_raev2_window_skip_blocks", None)
            if isinstance(skip_blocks, set) and _block_index in skip_blocks:
                skip_blocks.remove(_block_index)
                if not skip_blocks:
                    delattr(backbone, "_raev2_window_skip_blocks")
                return hidden_states
            if getattr(backbone, "_raev2_loopguidance_reference_mode", False):
                stats.loopguidance_reference_block_calls += 1
            stats.actual_block_calls += 1
            return _original(hidden_states, *args, **kwargs)

        block.forward = window_follower_forward


def apply_raev2_loop_patch(model, config: LoopConfig) -> LoopStats:
    """Patch official RAEv2 blocks with Dense or Sparse Token Loop."""
    backbone = resolve_raev2_backbone(model)
    blocks = resolve_raev2_blocks(backbone)
    depth = len(blocks)
    for index in config.block_indices or []:
        if not 0 <= index < depth:
            raise IndexError(f"block index {index} out of range for depth {depth}")

    if hasattr(backbone, "_raev2_loop_config"):
        restore_raev2_loop_patch(backbone)

    stats = LoopStats(patched_block_count=len(config.block_indices or []))
    operator = OPERATORS[config.operator]
    per_loop_calls = calls_per_loop(config.operator)

    if config.token_operator == "sparse":
        if config.loop_granularity != "layerwise":
            raise ValueError(
                "RAEv2 Sparse Token Loop requires layerwise granularity"
            )
        if config.selector not in {
            "random",
            "image_condition_split",
        }:
            raise ValueError(
                "RAEv2 Sparse Token Loop supports random routing or the "
                "image/condition split"
            )
        encoder_depth = int(getattr(backbone, "num_enc_blocks", 0))
        if encoder_depth <= 0 or any(
            index >= encoder_depth for index in (config.block_indices or [])
        ):
            raise ValueError(
                "RAEv2 Sparse Token Loop requires encoder-only block indices"
            )

    if config.loop_granularity == "rangewise":
        if config.token_operator != "dense":
            raise ValueError("RAEv2 range-wise looping supports Dense Token Loop only")
        _apply_raev2_window_patch(backbone, blocks, config, stats, operator, per_loop_calls)
        backbone._raev2_loop_config = config
        backbone._raev2_loop_stats = stats
        return stats

    for block_index in config.block_indices or []:
        block = blocks[block_index]
        if not hasattr(block, "_raev2_original_forward"):
            block._raev2_original_forward = block.forward
        original_forward = block._raev2_original_forward

        def looped_forward(
            hidden_states,
            *args,
            _original=original_forward,
            _block=block,
            _block_index=block_index,
            **kwargs,
        ):
            stats.patched_block_invocations += 1
            if getattr(backbone, "_raev2_loopguidance_reference_mode", False):
                stats.loopguidance_reference_block_calls += 1
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)
            step_index = getattr(backbone, "_loop_step_index", None)
            total_steps = getattr(backbone, "_loop_total_steps", None)
            if (
                not config.enabled
                or step_index is None
                or total_steps is None
            ):
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            k_t = config.resolve_k(step_index, total_steps)
            stats.k_t_sum += k_t
            stats.k_t_observations += 1
            if k_t <= 0:
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            lambda_value = config.lambda_for_k(k_t)
            if (
                config.token_operator == "dense"
                and config.operator == "euler"
                and k_t == 1
                and lambda_value == 1.0
            ):
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            stats.active_block_invocations += 1
            if config.token_operator == "sparse":
                selected_mask, complement_mask = _token_masks(
                    backbone,
                    config,
                    hidden_states,
                    _block_index,
                    int(step_index),
                )
                stats.sparse_observations += 1
                stats.sparse_selected_tokens += int(selected_mask.sum().item())
                stats.sparse_complement_tokens += int(complement_mask.sum().item())
                stats.sparse_total_tokens += int(
                    selected_mask.sum().item() + complement_mask.sum().item()
                )
                actual_calls = sparse_loop_call_count(config.operator, k_t)
                stats.actual_block_calls += actual_calls
                stats.extra_block_calls += actual_calls - 1
                return _run_sparse_token_loop(
                    block=_block,
                    original_forward=_original,
                    hidden_states=hidden_states,
                    args=args,
                    kwargs=kwargs,
                    selected_mask=selected_mask,
                    complement_mask=complement_mask,
                    config=config,
                    k_t=k_t,
                    lambda_value=lambda_value,
                )

            actual_calls = per_loop_calls * k_t
            stats.actual_block_calls += actual_calls
            stats.extra_block_calls += actual_calls - 1
            return operator(
                _original,
                hidden_states,
                args,
                kwargs,
                k_t,
                lambda_value,
            )

        block.forward = looped_forward

    backbone._raev2_loop_config = config
    backbone._raev2_loop_stats = stats
    return stats


def restore_raev2_loop_patch(model) -> None:
    backbone = resolve_raev2_backbone(model)
    for block in resolve_raev2_blocks(backbone):
        original = getattr(block, "_raev2_original_forward", None)
        if original is not None:
            block.forward = original
            del block._raev2_original_forward
    for attr in (
        "_raev2_loop_config",
        "_raev2_loop_stats",
        "_loop_step_index",
        "_loop_total_steps",
        "_raev2_loopguidance_reference_mode",
        "_raev2_window_skip_blocks",
    ):
        if hasattr(backbone, attr):
            delattr(backbone, attr)
