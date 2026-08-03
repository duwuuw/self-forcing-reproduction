from __future__ import annotations

from typing import Callable

import torch

from .loop_strategies import calls_per_loop


SelectedForward = Callable[[torch.Tensor, str], torch.Tensor]


def mix_tokens(looped: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use looped values for selected tokens and baseline values elsewhere."""
    if looped.shape != baseline.shape:
        raise ValueError(
            "looped and baseline outputs must have the same shape, "
            f"got {looped.shape} and {baseline.shape}"
        )
    while mask.ndim < looped.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, looped, baseline)


def loop_step(
    selected_forward: SelectedForward,
    state: torch.Tensor,
    group: str,
    step_size: float,
    operator: str,
) -> torch.Tensor:
    """Run one residual loop update for one token group."""
    if operator != "euler":
        raise ValueError("the released implementation supports Euler loop updates only")
    layer_output = selected_forward(state, group)
    return state + float(step_size) * (layer_output - state)


def sparse_token_loop(
    selected_forward: SelectedForward,
    hidden_states: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    operator: str,
    step_size: float,
    num_loops: int,
) -> torch.Tensor:
    """Apply Sparse Token Loop with one cached complement residual.

    The first round evaluates all tokens once. Its complement-token update is
    cached. Later rounds freshly evaluate only selected tokens using the full
    sequence as attention context, while applying the cached complement update.
    This is Algorithm 2 in the supplementary material.
    """
    if num_loops < 1:
        raise ValueError("num_loops must be >= 1")
    selected_mask = masks["selected"].to(device=hidden_states.device, dtype=torch.bool)
    complement_mask = masks["complement"].to(device=hidden_states.device, dtype=torch.bool)
    if bool((selected_mask & complement_mask).any()):
        raise ValueError("selected and complement masks must not overlap")
    all_mask = selected_mask | complement_mask
    if not bool(all_mask.any()):
        raise ValueError("the looped token domain must not be empty")

    first = loop_step(selected_forward, hidden_states, "all", step_size, operator)
    cached_complement_step = torch.where(
        complement_mask,
        (first - hidden_states).detach(),
        torch.zeros_like(hidden_states),
    )
    state = first
    for _ in range(1, int(num_loops)):
        selected_updated = loop_step(
            selected_forward,
            state,
            "selected",
            step_size,
            operator,
        )
        state = torch.where(
            complement_mask,
            state + cached_complement_step,
            selected_updated,
        )
    return state


def sparse_loop_call_count(operator: str, num_loops: int) -> int:
    """Return the number of layer evaluations used by Sparse Token Loop."""
    if num_loops < 1:
        return 0
    return int(num_loops) * calls_per_loop(operator)
