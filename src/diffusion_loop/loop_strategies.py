from __future__ import annotations

from typing import Any, Callable, Sequence

BlockForward = Callable[..., Any]


def call_block(block: BlockForward, hidden_states, args: Sequence[Any], kwargs: dict[str, Any]):
    """Call a backend block forward with the normalized loop signature."""
    return block(hidden_states, *args, **kwargs)


def dense_token_loop(block: BlockForward, hidden_states, args, kwargs, num_loops: int, lambda_value: float):
    """Apply the paper's Dense Token Loop using Euler residual updates."""
    x = hidden_states
    for _ in range(num_loops):
        y = call_block(block, x, args, kwargs)
        x = x + lambda_value * (y - x)
    return x


# Backward-compatible operator name used by the backend dispatch table.
euler_loop = dense_token_loop


OPERATORS = {
    "euler": euler_loop,
}


def calls_per_loop(operator: str) -> int:
    """Return the number of block calls for one Euler loop update."""
    if operator != "euler":
        raise ValueError(f"unknown operator: {operator!r}")
    return 1
