from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LoopStats:
    """Backend-neutral loop patch counters written into generation records."""

    patched_block_count: int = 0
    patched_block_invocations: int = 0
    active_block_invocations: int = 0
    actual_block_calls: int = 0
    extra_block_calls: int = 0
    k_t_sum: int = 0
    k_t_observations: int = 0
    sparse_observations: int = 0
    sparse_selected_tokens: int = 0
    sparse_complement_tokens: int = 0
    sparse_total_tokens: int = 0
    loopguidance_reference_predictions: int = 0
    loopguidance_loop_predictions: int = 0
    loopguidance_reference_block_calls: int = 0

    @property
    def average_k_t(self) -> float:
        if not self.k_t_observations:
            return 0.0
        return self.k_t_sum / float(self.k_t_observations)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["average_k_t"] = self.average_k_t
        return record
