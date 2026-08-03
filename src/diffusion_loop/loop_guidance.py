from __future__ import annotations

from typing import Any

import torch


def blend_loop_guidance(reference: Any, looped: Any, weight: float):
    """Apply ``reference + weight * (looped - reference)`` recursively.

    This prediction-space operation is independent of whether ``looped`` was
    produced by Dense Token Loop or Sparse Token Loop. Tensor, tuple, and list
    outputs are supported so backend adapters only need to produce matching
    ordinary and looped predictions.
    """
    if torch.is_tensor(reference) and torch.is_tensor(looped):
        if reference.shape != looped.shape:
            raise ValueError("Loop Guidance predictions must have matching shapes")
        return reference + float(weight) * (looped - reference)
    if isinstance(reference, tuple) and isinstance(looped, tuple):
        if len(reference) != len(looped):
            raise TypeError("Loop Guidance tuple outputs must have equal length")
        return tuple(
            blend_loop_guidance(base, with_loop, weight)
            for base, with_loop in zip(reference, looped)
        )
    if isinstance(reference, list) and isinstance(looped, list):
        if len(reference) != len(looped):
            raise TypeError("Loop Guidance list outputs must have equal length")
        return [
            blend_loop_guidance(base, with_loop, weight)
            for base, with_loop in zip(reference, looped)
        ]
    raise TypeError(
        "Loop Guidance requires matching tensor, tuple, or list predictions"
    )
