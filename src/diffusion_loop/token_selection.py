from __future__ import annotations

import math

import torch


def random_seed(seed: int, block_index: int, step_index: int, mode: str) -> int:
    """Build a deterministic seed for one sampling step and selected layer."""
    return int(seed) + 1009 * int(step_index) + 9176 * int(block_index)


def random_score(hidden_states: torch.Tensor, seed: int, block_index: int, step_index: int, mode: str) -> torch.Tensor:
    """Generate token scores for random selector modes."""
    if hidden_states.ndim < 3:
        raise ValueError("token selector requires hidden_states shaped like [batch, tokens, channels]")
    generator = torch.Generator(device=hidden_states.device)
    generator.manual_seed(random_seed(seed, block_index, step_index, mode))
    return torch.rand(hidden_states.shape[:2], device=hidden_states.device, generator=generator)


def masked_score(score: torch.Tensor, candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mask out tokens that are not eligible for selection."""
    if candidate_mask is None:
        return score
    candidates = candidate_mask.squeeze(-1).to(device=score.device, dtype=torch.bool)
    return torch.where(candidates, score, torch.full_like(score, -torch.inf))


def score_to_topk_mask(
    score: torch.Tensor,
    ratio: float,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert per-token scores to a [batch, tokens, 1] top-k boolean mask."""
    candidates = (
        torch.ones_like(score, dtype=torch.bool)
        if candidate_mask is None
        else candidate_mask.squeeze(-1).to(device=score.device, dtype=torch.bool)
    )
    out = torch.zeros_like(candidates, dtype=torch.bool)
    if ratio <= 0.0:
        return out.unsqueeze(-1)
    for batch_index in range(score.shape[0]):
        candidate_indices = candidates[batch_index].nonzero(as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        count = (
            candidate_indices.numel()
            if ratio >= 1.0
            else max(
                1,
                min(
                    int(candidate_indices.numel()),
                    int(math.ceil(candidate_indices.numel() * float(ratio))),
                ),
            )
        )
        candidate_scores = score[batch_index].index_select(0, candidate_indices)
        selected = candidate_indices[torch.topk(candidate_scores, k=count, dim=0).indices]
        out[batch_index].scatter_(0, selected, True)
    return out.unsqueeze(-1)


def score_to_row_topk_mask(
    score: torch.Tensor,
    ratio: float,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select top-k tokens independently per latent row when token count is square."""
    candidates = (
        torch.ones_like(score, dtype=torch.bool)
        if candidate_mask is None
        else candidate_mask.squeeze(-1).to(device=score.device, dtype=torch.bool)
    )
    tokens = score.shape[1]
    width = int(tokens**0.5)
    if width * width != tokens:
        return score_to_topk_mask(score, ratio, candidate_mask)
    out = torch.zeros_like(candidates, dtype=torch.bool)
    if ratio <= 0.0:
        return out.unsqueeze(-1)
    for batch_index in range(score.shape[0]):
        for row in range(width):
            row_slice = slice(row * width, (row + 1) * width)
            candidate_indices = candidates[batch_index, row_slice].nonzero(as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            count = (
                candidate_indices.numel()
                if ratio >= 1.0
                else max(
                    1,
                    min(
                        int(candidate_indices.numel()),
                        int(math.ceil(candidate_indices.numel() * float(ratio))),
                    ),
                )
            )
            row_scores = score[batch_index, row_slice].index_select(0, candidate_indices)
            selected = candidate_indices[torch.topk(row_scores, k=count, dim=0).indices] + row * width
            out[batch_index].scatter_(0, selected, True)
    return out.unsqueeze(-1)
