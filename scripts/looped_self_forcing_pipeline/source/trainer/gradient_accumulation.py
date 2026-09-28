from dataclasses import dataclass


@dataclass(frozen=True)
class GradientAccumulation:
    """Describe one optimizer update split across several micro-batches."""

    steps: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")

    @property
    def loss_scale(self) -> float:
        """Scale each micro-batch loss so the update is an arithmetic mean."""
        return 1.0 / self.steps

    def is_last_micro_step(self, micro_step: int) -> bool:
        if micro_step < 0 or micro_step >= self.steps:
            raise IndexError("micro_step must be within the accumulation window")
        return micro_step == self.steps - 1
