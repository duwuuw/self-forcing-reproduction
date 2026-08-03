from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal


LoopOperator = Literal["euler"]
TokenOperator = Literal["dense", "sparse"]
TokenDomain = Literal[
    "backend_default",
    "all_tokens",
    "image_prefix_only",
    "image_vs_condition",
]
Selector = Literal[
    "all",
    "random",
    "condition_aware",
    "attention_aware",
    "image_condition_split",
]
LoopGranularity = Literal["layerwise", "rangewise"]
LoopGuidanceBranch = Literal["base"]
KSchedule = list[tuple[float, int]]

VALID_OPERATORS = ("euler",)
VALID_TOKEN_OPERATORS = ("dense", "sparse")
VALID_TOKEN_DOMAINS = (
    "backend_default",
    "all_tokens",
    "image_prefix_only",
    "image_vs_condition",
)
VALID_SELECTORS = (
    "all",
    "random",
    "condition_aware",
    "attention_aware",
    "image_condition_split",
)
VALID_LOOP_GRANULARITIES = ("layerwise", "rangewise")
LOOP_GUIDANCE_SCALE_MAX = 18.0


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def parse_selector(selector: str, selection_ratio: float | None = None) -> tuple[str, float]:
    """Validate a paper-facing sparsifier name and its selected-token ratio."""
    if selector == "all":
        return "all", 1.0
    if selector not in VALID_SELECTORS:
        raise ValueError(f"unknown selector: {selector!r}")
    if selector == "image_condition_split":
        if selection_ratio is not None:
            raise ValueError("image_condition_split forbids selection_ratio")
        return selector, 1.0
    if selection_ratio is None:
        raise ValueError(f"selector {selector!r} requires selection_ratio")
    ratio = float(selection_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"selection_ratio must be in (0, 1], got {selection_ratio!r}")
    return selector, ratio


@dataclass
class LoopConfig:
    """Backend-neutral configuration using the terminology from the paper.

    ``token_operator='dense'`` selects Dense Token Loop and
    ``token_operator='sparse'`` selects Sparse Token Loop. Sampling-Progress
    Gating is controlled by ``start_frac`` and ``end_frac``. Loop Guidance is
    enabled when ``loopguidance_enabled`` is true.
    """

    enabled: bool = True
    block_indices: list[int] | None = None
    operator: LoopOperator = "euler"
    num_loops: int = 4
    lambda_one_over_k: bool = False
    lambda_value: float | None = 1.0
    start_frac: float = 0.0
    end_frac: float = 0.5
    k_schedule: KSchedule | None = None
    token_operator: TokenOperator = "dense"
    token_domain: TokenDomain = "backend_default"
    selector: Selector = "all"
    selection_ratio: float | None = None
    selector_seed: int = 0
    loopguidance_enabled: bool = False
    loopguidance_branch: LoopGuidanceBranch | None = None
    loopguidance_weight: float = 1.0
    loopguidance_reference_ratio: float = 0.0
    loop_granularity: LoopGranularity = "layerwise"

    def __post_init__(self) -> None:
        if not self.enabled:
            if self.block_indices is None:
                self.block_indices = [0]
            return
        self._validate_blocks()
        self._validate_loop()
        self._validate_loop_granularity()
        self._validate_token_operator()
        self._validate_loopguidance()

    def _validate_blocks(self) -> None:
        if not self.block_indices:
            raise ValueError("block_indices must be a non-empty list")
        if len(set(self.block_indices)) != len(self.block_indices):
            raise ValueError("block_indices must not contain duplicates")
        if any(not _is_plain_int(index) for index in self.block_indices):
            raise ValueError("block_indices must contain integers")
        if any(index < 0 for index in self.block_indices):
            raise ValueError("block_indices must be non-negative")
        expected = list(range(self.block_indices[0], self.block_indices[-1] + 1))
        if self.block_indices != expected:
            raise ValueError(
                "block_indices must be one contiguous, ascending loop layer range"
            )

    def _validate_loop(self) -> None:
        if self.operator not in VALID_OPERATORS:
            raise ValueError(f"unknown operator: {self.operator!r}")
        if not _is_plain_int(self.num_loops) or self.num_loops < 1:
            raise ValueError(f"num_loops must be an integer >= 1, got {self.num_loops!r}")
        if not (0.0 <= self.start_frac <= self.end_frac <= 1.0):
            raise ValueError("window must satisfy 0 <= start_frac <= end_frac <= 1")
        if self.lambda_one_over_k:
            if self.lambda_value is not None:
                raise ValueError("lambda_value must be null when lambda_one_over_k is true")
        elif self.lambda_value is None or self.lambda_value <= 0:
            raise ValueError("lambda_value must be > 0 when lambda_one_over_k is false")
        if self.k_schedule is not None:
            self._validate_schedule()

    def _validate_loop_granularity(self) -> None:
        if self.loop_granularity not in VALID_LOOP_GRANULARITIES:
            raise ValueError(f"unknown loop_granularity: {self.loop_granularity!r}")

    def _validate_token_operator(self) -> None:
        if self.token_operator not in VALID_TOKEN_OPERATORS:
            raise ValueError(f"unknown token_operator: {self.token_operator!r}")
        if self.token_domain not in VALID_TOKEN_DOMAINS:
            raise ValueError(f"unknown token_domain: {self.token_domain!r}")
        if self.selector not in VALID_SELECTORS:
            raise ValueError(f"unknown selector: {self.selector!r}")
        if not _is_plain_int(self.selector_seed):
            raise ValueError("selector_seed must be an integer")

        if self.token_operator == "dense":
            if self.selector != "all":
                raise ValueError("Dense Token Loop requires selector='all'")
            if self.selection_ratio is not None:
                raise ValueError("Dense Token Loop forbids selection_ratio")
            return

        if self.selector == "all":
            raise ValueError("Sparse Token Loop requires a non-'all' selector")
        if self.selector == "image_condition_split":
            if self.token_domain != "image_vs_condition":
                raise ValueError(
                    "image_condition_split requires token_domain='image_vs_condition'"
                )
            parse_selector(self.selector, self.selection_ratio)
            return
        if self.token_domain == "image_vs_condition":
            raise ValueError(
                "token_domain='image_vs_condition' requires selector='image_condition_split'"
            )
        parse_selector(self.selector, self.selection_ratio)
        if self.operator != "euler":
            raise ValueError("Sparse Token Loop currently requires operator='euler'")

    def _validate_loopguidance(self) -> None:
        if not isinstance(self.loopguidance_enabled, bool):
            raise ValueError("loopguidance_enabled must be a boolean")
        if not self.loopguidance_enabled:
            if self.loopguidance_branch is not None:
                raise ValueError("loopguidance_branch must be null when Loop Guidance is disabled")
            if float(self.loopguidance_weight) != 1.0:
                raise ValueError("loopguidance_weight must be 1.0 when Loop Guidance is disabled")
            if float(self.loopguidance_reference_ratio) != 0.0:
                raise ValueError("loopguidance_reference_ratio must be 0.0 when Loop Guidance is disabled")
            return

        if self.loopguidance_branch != "base":
            raise ValueError("loopguidance_branch must be 'base'")
        if not (0.0 <= float(self.loopguidance_weight) <= LOOP_GUIDANCE_SCALE_MAX):
            raise ValueError(f"loopguidance_weight must be in [0, {LOOP_GUIDANCE_SCALE_MAX}]")
        if float(self.loopguidance_reference_ratio) != 0.0:
            raise ValueError("loopguidance_reference_ratio must be 0.0")

    def _validate_schedule(self) -> None:
        if not self.k_schedule:
            raise ValueError("k_schedule must be non-empty when set")
        previous = -1.0
        for index, (frac, k_t) in enumerate(self.k_schedule):
            _validate_unit_interval(f"k_schedule[{index}] frac", frac)
            if frac <= previous:
                raise ValueError("k_schedule fractions must be strictly increasing")
            if not _is_plain_int(k_t) or k_t < 0:
                raise ValueError(f"k_schedule[{index}] K_t must be an integer >= 0")
            previous = frac
        if abs(self.k_schedule[-1][0] - 1.0) > 1e-9:
            raise ValueError("last k_schedule fraction must be 1.0")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LoopConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown LoopConfig fields: {unknown}")
        values = dict(raw)
        schedule = values.get("k_schedule")
        if schedule is not None:
            values["k_schedule"] = [(float(frac), int(k_t)) for frac, k_t in schedule]
        return cls(**values)

    @property
    def selector_ratio(self) -> float:
        return parse_selector(self.selector, self.selection_ratio)[1]

    def resolve_k(self, step_index: int, total_steps: int) -> int:
        frac = step_fraction(step_index, total_steps)
        if self.k_schedule is not None:
            for upper_frac, k_t in self.k_schedule:
                if frac <= upper_frac:
                    return k_t
            raise AssertionError("validated k_schedule must cover frac=1.0")
        if self.start_frac <= frac <= self.end_frac:
            return self.num_loops
        return 0

    def lambda_for_k(self, k_t: int) -> float:
        if k_t <= 0:
            return 0.0
        if self.lambda_one_over_k:
            return 1.0 / float(k_t)
        assert self.lambda_value is not None
        return float(self.lambda_value) / float(k_t)

    def resolve_selector_ratio(self, step_index: int | None = None, total_steps: int | None = None) -> float:
        return self.selector_ratio

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if self.token_domain == "backend_default":
            record.pop("token_domain")
        return record

    def with_loop_guidance(self, weight: float) -> "LoopConfig":
        """Return the same Dense/Sparse Token Loop with Loop Guidance enabled.

        Loop Guidance is deliberately a modifier rather than a third token-loop
        implementation: the value of ``token_operator`` is preserved.  This
        makes both Dense Token Loop and Sparse Token Loop composable with the
        prediction-space guidance rule from the paper.
        """
        if not self.enabled:
            raise ValueError("Loop Guidance requires an enabled token loop")
        return replace(
            self,
            loopguidance_enabled=True,
            loopguidance_branch="base",
            loopguidance_weight=float(weight),
            loopguidance_reference_ratio=0.0,
        )

    def without_loop_guidance(self) -> "LoopConfig":
        """Return the same token-loop configuration without Loop Guidance."""
        return replace(
            self,
            loopguidance_enabled=False,
            loopguidance_branch=None,
            loopguidance_weight=1.0,
            loopguidance_reference_ratio=0.0,
        )


def step_fraction(step_index: int, total_steps: int) -> float:
    """Normalize the denoising-call index to sampling progress in [0, 1]."""
    if total_steps <= 1:
        return 0.0
    return step_index / float(total_steps - 1)
