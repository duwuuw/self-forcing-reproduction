from __future__ import annotations

from typing import Any

import torch

from ..loop_config import LoopConfig
from ..loop_stats import LoopStats
from ..loop_strategies import OPERATORS, calls_per_loop
from ..token_loop import sparse_loop_call_count, sparse_token_loop
from ..token_selection import random_score, score_to_row_topk_mask, score_to_topk_mask


def resolve_scale_rae_backbone(model):
    """Resolve the DiT backbone from supported Scale-RAE wrappers."""
    candidate = model
    if hasattr(candidate, "diff_head"):
        candidate = candidate.diff_head
    if hasattr(candidate, "model"):
        candidate = candidate.model
    if getattr(candidate, "layers", None) is None and getattr(candidate, "dit_blocks", None) is None:
        raise TypeError("Scale-RAE adapter requires a diffusion backbone with .layers or .dit_blocks")
    return candidate


def resolve_scale_rae_blocks(backbone):
    """Resolve block containers used by supported Scale-RAE versions."""
    blocks = getattr(backbone, "layers", None)
    if blocks is None:
        blocks = getattr(backbone, "dit_blocks", None)
    if blocks is None:
        raise TypeError("Scale-RAE adapter requires a diffusion backbone with .layers or .dit_blocks")
    return blocks


def _condition_arg(args, kwargs):
    """Extract conditioning vector ``c`` from the DiT block arguments."""
    if args:
        return args[0]
    return kwargs.get("c")


def _feat_rope_arg(args, kwargs):
    """Extract rotary-position features from the DiT block arguments."""
    if len(args) >= 2:
        return args[1]
    return kwargs.get("feat_rope")


def _expand_token_condition(value, hidden_states: torch.Tensor):
    if value is None:
        return None
    if value.ndim < hidden_states.ndim:
        return value.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
    return value


def _block_modulation(block, c):
    """Call AdaLN modulation, including the supported no-shift variant."""
    if getattr(block, "wo_shift", False):
        scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(c).chunk(4, dim=-1)
        return None, scale_msa, gate_msa, None, scale_mlp, gate_mlp
    return block.adaLN_modulation(c).chunk(6, dim=-1)


def _modulate(x: torch.Tensor, shift, scale):
    if shift is not None and len(shift.shape) < len(x.shape):
        shift = shift.unsqueeze(1)
    if len(scale.shape) < len(x.shape):
        scale = scale.unsqueeze(1)
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, gate_value):
    if len(gate_value.shape) < len(x.shape):
        gate_value = gate_value.unsqueeze(1)
    return x * gate_value


def _adaln_condition_score(block, hidden_states: torch.Tensor, args, kwargs) -> torch.Tensor | None:
    """Estimate token importance from AdaLN gate/scale/shift magnitudes."""
    if not all(hasattr(block, attr) for attr in ("adaLN_modulation", "norm1", "norm2")):
        return None
    c = _condition_arg(args, kwargs)
    if c is None:
        return None
    with torch.no_grad():
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _block_modulation(block, c)
        parts = []
        for value, weight in (
            (gate_msa, 1.0),
            (gate_mlp, 1.0),
            (scale_msa, 0.25),
            (scale_mlp, 0.25),
            (shift_msa, 0.25),
            (shift_mlp, 0.25),
        ):
            expanded = _expand_token_condition(value, hidden_states)
            if expanded is not None:
                parts.append(float(weight) * expanded.detach().float().norm(dim=-1))
        if not parts:
            return None
        score = torch.stack(parts, dim=0).sum(dim=0)
        if score.shape[:2] != hidden_states.shape[:2]:
            return None
        return score


def _attention_output_score(block, hidden_states: torch.Tensor, args, kwargs) -> torch.Tensor | None:
    """Use gated attention-output norms as attention-aware token scores."""
    if not all(hasattr(block, attr) for attr in ("adaLN_modulation", "attn", "norm1")):
        return None
    c = _condition_arg(args, kwargs)
    if c is None:
        return None
    feat_rope = _feat_rope_arg(args, kwargs)
    with torch.no_grad():
        shift_msa, scale_msa, gate_msa, *_ = _block_modulation(block, c)
        attn_input = _modulate(block.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = block.attn(attn_input, rope=feat_rope)
        return _gate(attn_output, gate_msa).detach().float().norm(dim=-1)


def _selector_mask(
    block,
    config: LoopConfig,
    hidden_states: torch.Tensor,
    args,
    kwargs,
    block_index: int,
    step_index: int,
) -> torch.Tensor:
    """Build the selected-token mask for Sparse Token Loop."""
    ratio = config.selector_ratio
    if config.selector == "random":
        return score_to_topk_mask(
            random_score(hidden_states, config.selector_seed, block_index, step_index, "random"),
            ratio,
        )
    if config.selector == "condition_aware":
        score = _adaln_condition_score(block, hidden_states, args, kwargs)
        if score is None:
            score = random_score(hidden_states, config.selector_seed, block_index, step_index, "condition_aware")
        return score_to_topk_mask(score, ratio)
    if config.selector == "attention_aware":
        score = _attention_output_score(block, hidden_states, args, kwargs)
        if score is None:
            score = random_score(hidden_states, config.selector_seed, block_index, step_index, "attention_aware")
        return score_to_row_topk_mask(score, ratio)
    raise ValueError(f"unknown selector: {config.selector!r}")


def _mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    """Convert a [batch,tokens,1] mask to selected-query indices."""
    selected = mask.squeeze(-1).to(dtype=torch.bool)
    counts = selected.sum(dim=1)
    if counts.numel() == 0 or int(counts.min().item()) <= 0:
        raise ValueError("selected-query token mask must select at least one token per batch item")
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("selected-query token mask requires equal selected-token counts per batch item")
    if selected.shape[0] == 1:
        return selected[0].nonzero(as_tuple=False).flatten().unsqueeze(0)
    return selected.to(dtype=torch.int64).topk(k=int(counts[0].item()), dim=1).indices


def _gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        return x[0].index_select(0, indices[0]).unsqueeze(0)
    return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _scatter_tokens(base: torch.Tensor, indices: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    out = base.clone()
    if base.shape[0] == 1:
        out[0].index_copy_(0, indices[0], selected[0])
        return out
    return out.scatter(1, indices.unsqueeze(-1).expand(-1, -1, base.shape[-1]), selected)


def _select_condition_tokens(value: torch.Tensor | None, indices: torch.Tensor, total_tokens: int):
    if value is None:
        return None
    if value.ndim >= 3 and value.shape[1] == total_tokens:
        return _gather_tokens(value, indices)
    return value


def _qkv_weight_chunks(attn):
    weight = attn.qkv.weight
    bias = getattr(attn.qkv, "bias", None)
    channels = weight.shape[0] // 3
    q_weight = weight[:channels]
    k_weight = weight[channels : 2 * channels]
    v_weight = weight[2 * channels :]
    if bias is None:
        return q_weight, k_weight, v_weight, None, None, None
    return q_weight, k_weight, v_weight, bias[:channels], bias[channels : 2 * channels], bias[2 * channels :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = paired.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_selected(x: torch.Tensor, rope, indices: torch.Tensor) -> torch.Tensor:
    if not all(hasattr(rope, attr) for attr in ("freqs_cos", "freqs_sin")):
        raise NotImplementedError("selected-token RoPE requires freqs_cos/freqs_sin buffers")
    cos = rope.freqs_cos.to(device=x.device, dtype=x.dtype)
    sin = rope.freqs_sin.to(device=x.device, dtype=x.dtype)
    if indices.shape[0] == 1:
        cos_s = cos.index_select(0, indices[0]).unsqueeze(0)
        sin_s = sin.index_select(0, indices[0]).unsqueeze(0)
    else:
        gather_index = indices.unsqueeze(-1).expand(-1, -1, cos.shape[-1])
        cos_s = cos.unsqueeze(0).expand(indices.shape[0], -1, -1).gather(1, gather_index)
        sin_s = sin.unsqueeze(0).expand(indices.shape[0], -1, -1).gather(1, gather_index)
    cos_s = cos_s[:, None, :, :]
    sin_s = sin_s[:, None, :, :]
    return x * cos_s + _rotate_half(x) * sin_s


def _project_q_selected(attn, x: torch.Tensor, rope=None, indices: torch.Tensor | None = None):
    import torch.nn.functional as F

    q_weight, _, _, q_bias, _, _ = _qkv_weight_chunks(attn)
    q = F.linear(x, q_weight, q_bias)
    batch, tokens, _ = q.shape
    q = q.reshape(batch, tokens, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
    q = attn.q_norm(q)
    if rope is not None:
        if indices is None:
            q = rope(q).to(q.dtype)
        else:
            q = _apply_rope_selected(q, rope, indices).to(q.dtype)
    return q


def _project_kv_selected(attn, x: torch.Tensor, rope=None, indices: torch.Tensor | None = None):
    import torch.nn.functional as F

    channels = attn.qkv.weight.shape[0] // 3
    kv_weight = attn.qkv.weight[channels:]
    kv_bias = None if getattr(attn.qkv, "bias", None) is None else attn.qkv.bias[channels:]
    kv = F.linear(x, kv_weight, kv_bias)
    batch, tokens, _ = kv.shape
    kv = kv.reshape(batch, tokens, 2, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    k = attn.k_norm(k)
    if rope is not None:
        if indices is None:
            k = rope(k).to(k.dtype)
        else:
            k = _apply_rope_selected(k, rope, indices).to(k.dtype)
    return k, v


def _selected_q_fullkv_attention(attn, x: torch.Tensor, indices: torch.Tensor, rope=None):
    """Compute selected queries while retaining full key/value context."""
    import torch.nn.functional as F

    batch, _, channels = x.shape
    selected_input = _gather_tokens(x, indices)
    q = _project_q_selected(attn, selected_input, rope=rope, indices=indices if rope is not None else None)
    k, v = _project_kv_selected(attn, x, rope=rope)
    selected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=float(attn.scale),
    )
    selected = selected.transpose(1, 2).reshape(batch, indices.shape[1], channels)
    selected = attn.proj(selected)
    return attn.proj_drop(selected)


def _selected_query_fullkv_forward(block, hidden_states: torch.Tensor, args, kwargs, mask: torch.Tensor):
    """Run one block update for selected queries with full-token context.

    Attention queries and the MLP are restricted to selected tokens while keys
    and values use all tokens. Return ``None`` when the installed checkpoint's
    block structure is incompatible with this optimized path.
    """
    required_attrs = ("adaLN_modulation", "attn", "mlp", "norm1", "norm2")
    if not all(hasattr(block, attr) for attr in required_attrs):
        return None
    attn = block.attn
    if not all(hasattr(attn, attr) for attr in ("qkv", "q_norm", "k_norm", "proj", "proj_drop", "num_heads", "head_dim", "scale")):
        return None
    c = _condition_arg(args, kwargs)
    if c is None:
        return None

    feat_rope = _feat_rope_arg(args, kwargs)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _block_modulation(block, c)

    total_tokens = hidden_states.shape[1]
    indices = _mask_to_indices(mask)
    shift_msa_s = _select_condition_tokens(shift_msa, indices, total_tokens)
    scale_msa_s = _select_condition_tokens(scale_msa, indices, total_tokens)
    gate_msa_s = _select_condition_tokens(gate_msa, indices, total_tokens)
    shift_mlp_s = _select_condition_tokens(shift_mlp, indices, total_tokens)
    scale_mlp_s = _select_condition_tokens(scale_mlp, indices, total_tokens)
    gate_mlp_s = _select_condition_tokens(gate_mlp, indices, total_tokens)

    attn_input = _modulate(block.norm1(hidden_states), shift_msa, scale_msa)
    selected_attn = _selected_q_fullkv_attention(attn, attn_input, indices, rope=feat_rope)
    selected_x = _gather_tokens(hidden_states, indices) + _gate(selected_attn, gate_msa_s)

    selected_mlp_input = _modulate(block.norm2(selected_x), shift_mlp_s, scale_mlp_s)
    selected_x = selected_x + _gate(block.mlp(selected_mlp_input), gate_mlp_s)
    return _scatter_tokens(hidden_states, indices, selected_x)


def _prepare_selected_query_fullkv_context(block, hidden_states: torch.Tensor, args, kwargs, masks: dict[str, torch.Tensor]):
    """Prepare reusable modulation and token-index context for sparse replay."""
    required_attrs = ("adaLN_modulation", "attn", "mlp", "norm1", "norm2")
    if not all(hasattr(block, attr) for attr in required_attrs):
        return None
    attn = block.attn
    if not all(hasattr(attn, attr) for attr in ("qkv", "q_norm", "k_norm", "proj", "proj_drop", "num_heads", "head_dim", "scale")):
        return None
    c = _condition_arg(args, kwargs)
    if c is None:
        return None

    total_tokens = hidden_states.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _block_modulation(block, c)
    mask_data = {}
    for name, mask in masks.items():
        indices = _mask_to_indices(mask)
        mask_data[name] = {
            "indices": indices,
            "shift_msa": _select_condition_tokens(shift_msa, indices, total_tokens),
            "scale_msa": _select_condition_tokens(scale_msa, indices, total_tokens),
            "gate_msa": _select_condition_tokens(gate_msa, indices, total_tokens),
            "shift_mlp": _select_condition_tokens(shift_mlp, indices, total_tokens),
            "scale_mlp": _select_condition_tokens(scale_mlp, indices, total_tokens),
            "gate_mlp": _select_condition_tokens(gate_mlp, indices, total_tokens),
        }
    return {
        "feat_rope": _feat_rope_arg(args, kwargs),
        "shift_msa": shift_msa,
        "scale_msa": scale_msa,
        "mask_data": mask_data,
    }


def _selected_query_fullkv_forward_prepared(block, hidden_states: torch.Tensor, context, group: str):
    """Run selected-query/full-KV forward using precomputed group context."""
    data = context["mask_data"][group]
    indices = data["indices"]
    attn_input = _modulate(block.norm1(hidden_states), context["shift_msa"], context["scale_msa"])
    selected_attn = _selected_q_fullkv_attention(block.attn, attn_input, indices, rope=context["feat_rope"])
    selected_x = _gather_tokens(hidden_states, indices) + _gate(selected_attn, data["gate_msa"])

    selected_mlp_input = _modulate(block.norm2(selected_x), data["shift_mlp"], data["scale_mlp"])
    selected_x = selected_x + _gate(block.mlp(selected_mlp_input), data["gate_mlp"])
    return _scatter_tokens(hidden_states, indices, selected_x)


def _sparse_selected_query_fullkv_loop(
    block,
    fallback_block,
    hidden_states: torch.Tensor,
    args,
    kwargs,
    selected_mask: torch.Tensor,
    complement_mask: torch.Tensor,
    *,
    operator: str,
    step_size: float,
    num_loops: int,
) -> torch.Tensor:
    """Run Sparse Token Loop with selected-query/full-KV attention."""
    masks = {
        "selected": selected_mask.to(device=hidden_states.device, dtype=torch.bool),
        "complement": complement_mask.to(device=hidden_states.device, dtype=torch.bool),
    }
    masks["all"] = masks["selected"] | masks["complement"]
    prepared_context = _prepare_selected_query_fullkv_context(block, hidden_states, args, kwargs, masks)

    def selected_forward(state, group: str):
        mask = masks[group]
        if prepared_context is not None:
            selected = _selected_query_fullkv_forward_prepared(block, state, prepared_context, group)
        else:
            selected = _selected_query_fullkv_forward(block, state, args, kwargs, mask)
        if selected is not None:
            return selected
        fallback = fallback_block(state, *args, **kwargs)
        return torch.where(mask, fallback, state)

    return sparse_token_loop(
        selected_forward,
        hidden_states,
        masks,
        operator=operator,
        step_size=step_size,
        num_loops=num_loops,
    )


def _validate_window_indices(config: LoopConfig, depth: int) -> list[int]:
    indices = [int(index) for index in (config.block_indices or [])]
    if not indices:
        raise ValueError("Scale-RAE range-wise loop requires a non-empty block range")
    if any(not 0 <= index < depth for index in indices):
        raise IndexError(f"block index out of range for Scale-RAE depth {depth}: {indices}")
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            "Scale-RAE range-wise loop requires contiguous, ascending block_indices, "
            f"got {config.block_indices!r}"
        )
    return indices


def _run_window_once(
    window_indices: list[int],
    original_forwards: dict[int, Any],
    hidden_states: torch.Tensor,
    args,
    kwargs,
) -> torch.Tensor:
    state = hidden_states
    for block_index in window_indices:
        state = original_forwards[block_index](state, *args, **kwargs)
        if not torch.is_tensor(state):
            raise TypeError(
                "Scale-RAE range-wise loop requires every DiT block to return a tensor, "
                f"but block {block_index} returned {type(state).__name__}"
            )
    return state


def _apply_scale_rae_window_patch(
    backbone,
    blocks,
    config: LoopConfig,
    stats: LoopStats,
    operator,
    per_loop_calls: int,
) -> LoopStats:
    if config.token_operator != "dense":
        raise ValueError("Scale-RAE range-wise looping currently supports Dense Token Loop only")

    window_indices = _validate_window_indices(config, len(blocks))
    window_start = window_indices[0]
    window_followers = set(window_indices[1:])
    original_forwards: dict[int, Any] = {}
    for block_index in window_indices:
        block = blocks[block_index]
        if not hasattr(block, "_scale_rae_original_forward"):
            block._scale_rae_original_forward = block.forward
        original_forwards[block_index] = block._scale_rae_original_forward

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
                if getattr(backbone, "_scale_rae_loopguidance_reference_mode", False):
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
                backbone._scale_rae_window_skip_blocks = set(window_followers)
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
            skip_blocks = getattr(backbone, "_scale_rae_window_skip_blocks", None)
            if isinstance(skip_blocks, set) and _block_index in skip_blocks:
                skip_blocks.remove(_block_index)
                if not skip_blocks:
                    delattr(backbone, "_scale_rae_window_skip_blocks")
                return hidden_states
            if getattr(backbone, "_scale_rae_loopguidance_reference_mode", False):
                stats.loopguidance_reference_block_calls += 1
            stats.actual_block_calls += 1
            return _original(hidden_states, *args, **kwargs)

        block.forward = window_follower_forward

    return stats


def apply_scale_rae_loop_patch(model, config: LoopConfig) -> LoopStats:
    """Install the Scale-RAE DiT block loop patch.

    The patch replaces only the requested block forwards. Outside the
    Sampling-Progress Gating interval it preserves the original forward;
    inside the interval it applies Dense or Sparse Token Loop.
    """
    backbone = resolve_scale_rae_backbone(model)
    blocks = resolve_scale_rae_blocks(backbone)
    depth = len(blocks)
    for index in config.block_indices or []:
        if not 0 <= index < depth:
            raise IndexError(f"block index {index} out of range for depth {depth}")

    if hasattr(backbone, "_scale_rae_loop_config"):
        restore_scale_rae_loop_patch(backbone)

    stats = LoopStats(patched_block_count=len(config.block_indices or []))
    operator = OPERATORS[config.operator]
    per_loop_calls = calls_per_loop(config.operator)

    if config.loop_granularity == "rangewise":
        _apply_scale_rae_window_patch(backbone, blocks, config, stats, operator, per_loop_calls)
        backbone._scale_rae_loop_config = config
        backbone._scale_rae_loop_stats = stats
        return stats

    for block_index in config.block_indices or []:
        block = blocks[block_index]
        if not hasattr(block, "_scale_rae_original_forward"):
            block._scale_rae_original_forward = block.forward
        original_forward = block._scale_rae_original_forward

        def looped_forward(hidden_states, *args, _original=original_forward, _block_index=block_index, _block=block, **kwargs):
            stats.patched_block_invocations += 1
            if getattr(backbone, "_scale_rae_loopguidance_reference_mode", False):
                stats.loopguidance_reference_block_calls += 1
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            step_index = getattr(backbone, "_loop_step_index", None)
            total_steps = getattr(backbone, "_loop_total_steps", None)
            if not config.enabled or step_index is None or total_steps is None:
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            k_t = config.resolve_k(step_index, total_steps)
            stats.k_t_sum += k_t
            stats.k_t_observations += 1
            if k_t <= 0:
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            lambda_value = config.lambda_for_k(k_t)
            if config.operator == "euler" and k_t == 1 and lambda_value == 1.0:
                stats.actual_block_calls += 1
                return _original(hidden_states, *args, **kwargs)

            if config.token_operator == "dense":
                stats.active_block_invocations += 1
                actual_calls = per_loop_calls * k_t
                stats.actual_block_calls += actual_calls
                stats.extra_block_calls += actual_calls - 1
                return operator(_original, hidden_states, args, kwargs, k_t, lambda_value)

            mask = _selector_mask(_block, config, hidden_states, args, kwargs, _block_index, step_index)
            if config.token_operator == "sparse":
                stats.active_block_invocations += 1
                actual_calls = sparse_loop_call_count(config.operator, k_t)
                stats.actual_block_calls += actual_calls
                stats.extra_block_calls += actual_calls - 1
                complement_mask = ~mask.to(dtype=torch.bool)
                looped = _sparse_selected_query_fullkv_loop(
                    _block,
                    _original,
                    hidden_states,
                    args,
                    kwargs,
                    mask,
                    complement_mask,
                    operator=config.operator,
                    step_size=lambda_value,
                    num_loops=k_t,
                )
                stats.sparse_observations += 1
                stats.sparse_selected_tokens += int(mask.sum().item())
                stats.sparse_complement_tokens += int(complement_mask.sum().item())
                stats.sparse_total_tokens += int(mask.numel())
                return looped

            raise ValueError(f"unknown token_operator: {config.token_operator!r}")

        block.forward = looped_forward

    backbone._scale_rae_loop_config = config
    backbone._scale_rae_loop_stats = stats
    return stats


def restore_scale_rae_loop_patch(model) -> None:
    """Restore patched block forwards and clear loop state."""
    backbone = resolve_scale_rae_backbone(model)
    for block in resolve_scale_rae_blocks(backbone):
        original = getattr(block, "_scale_rae_original_forward", None)
        if original is not None:
            block.forward = original
            del block._scale_rae_original_forward
    for attr in (
        "_scale_rae_loop_config",
        "_scale_rae_loop_stats",
        "_scale_rae_loopguidance_reference_mode",
        "_scale_rae_window_skip_blocks",
        "_loop_step_index",
        "_loop_total_steps",
    ):
        if hasattr(backbone, attr):
            delattr(backbone, attr)
