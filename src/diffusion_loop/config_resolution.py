from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from .loop_config import LoopConfig, VALID_SELECTORS, VALID_TOKEN_DOMAINS


TokenLoopChoice = Literal["none", "dense", "sparse"]

BASE_METHOD_IDS = {
    "dense": "dense_token_loop",
    "sparse": "sparse_token_loop",
}
GUIDED_METHOD_IDS = {
    "dense": "loop_guidance_dense_token_loop",
    "sparse": "loop_guidance_sparse_token_loop",
}


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def inclusive_block_range(first: int, last: int) -> list[int]:
    """Expand the paper's inclusive layer range ``[first, last]``."""
    if not _is_plain_int(first) or not _is_plain_int(last):
        raise ValueError("loop layer endpoints must be integers")
    if first < 0 or last < first:
        raise ValueError("loop layer range must satisfy 0 <= first <= last")
    return list(range(first, last + 1))


def canonical_method_id(token_loop: TokenLoopChoice, loop_guidance: bool) -> str:
    """Return an unambiguous paper-facing method ID for a composition."""
    if token_loop == "none":
        if loop_guidance:
            raise ValueError("Loop Guidance requires Dense or Sparse Token Loop")
        return "without_loop"
    if token_loop not in BASE_METHOD_IDS:
        raise ValueError(f"unknown token loop: {token_loop!r}")
    table = GUIDED_METHOD_IDS if loop_guidance else BASE_METHOD_IDS
    return table[token_loop]


@dataclass(frozen=True)
class ResolvedRunConfig:
    """A paper preset after optional, explicit command-line/API overrides."""

    backend: str
    generation: dict[str, Any]
    loop_config: LoopConfig
    method_id: str
    preset_id: str
    paper_preset: bool
    overrides: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "method_id": self.method_id,
            "preset_id": self.preset_id,
            "paper_preset": self.paper_preset,
            "overrides": deepcopy(self.overrides),
            "generation": deepcopy(self.generation),
            "loop": self.loop_config.to_record(),
        }


def _capability_set(payload: dict[str, Any], key: str) -> set[str]:
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("config is missing the capabilities object")
    values = capabilities.get(key)
    if not isinstance(values, list) or any(value not in {"dense", "sparse"} for value in values):
        raise ValueError(f"capabilities.{key} must contain only 'dense'/'sparse'")
    return set(values)


def _sparse_selector_capabilities(payload: dict[str, Any]) -> set[str]:
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("config is missing the capabilities object")
    values = capabilities.get("sparse_selectors")
    valid = set(VALID_SELECTORS) - {"all"}
    if not isinstance(values, list) or any(value not in valid for value in values):
        raise ValueError(
            "capabilities.sparse_selectors must contain paper-facing sparse selectors"
        )
    return set(values)


def resolve_run_config(
    payload: dict[str, Any],
    *,
    token_loop: TokenLoopChoice,
    loop_guidance: bool = False,
    outer_steps: int | None = None,
    num_loops: int | None = None,
    lambda_loop: float | None = None,
    loop_active: tuple[float, float] | None = None,
    loop_layers: tuple[int, int] | None = None,
    loop_guidance_weight: float | None = None,
    selector: str | None = None,
    selection_ratio: float | None = None,
    selector_seed: int | None = None,
    token_domain: str | None = None,
) -> ResolvedRunConfig:
    """Resolve a token-loop composition and all paper hyperparameter knobs.

    The JSON file remains the immutable paper preset.  Supplied values override
    a copy, and ``paper_preset`` becomes false in the returned record.  When a
    backend supports Sparse Token Loop + Loop Guidance but the paper did not
    report that exact pair, the sparse preset is composed with an explicitly
    supplied guidance weight instead of inventing a paper default.
    """
    if not isinstance(loop_guidance, bool):
        raise ValueError("loop_guidance must be a boolean")
    method_id = canonical_method_id(token_loop, loop_guidance)
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("config is missing the methods object")

    generation = deepcopy(payload.get("generation", {}))
    overrides: dict[str, Any] = {}
    if outer_steps is not None:
        if not _is_plain_int(outer_steps) or outer_steps < 1:
            raise ValueError("outer_steps must be an integer >= 1")
        generation["num_inference_steps"] = outer_steps
        overrides["outer_steps"] = outer_steps

    loop_override_values = (
        num_loops,
        lambda_loop,
        loop_active,
        loop_layers,
        loop_guidance_weight,
        selector,
        selection_ratio,
        selector_seed,
        token_domain,
    )
    if token_loop == "none":
        if any(value is not None for value in loop_override_values):
            raise ValueError("loop-specific overrides require Dense or Sparse Token Loop")
        raw = methods.get("without_loop")
        if raw is None:
            raise KeyError("without_loop preset is absent from config")
        return ResolvedRunConfig(
            backend=str(payload.get("backend", "")),
            generation=generation,
            loop_config=LoopConfig.from_dict(raw),
            method_id=method_id,
            preset_id="without_loop",
            paper_preset=not overrides,
            overrides=overrides,
        )

    supported_loops = _capability_set(payload, "token_loops")
    if token_loop not in supported_loops:
        raise ValueError(
            f"backend {payload.get('backend')!r} does not support {token_loop} Token Loop"
        )
    if loop_guidance:
        guided_loops = _capability_set(payload, "loop_guidance_token_loops")
        if token_loop not in guided_loops:
            raise ValueError(
                f"backend {payload.get('backend')!r} does not support Loop Guidance "
                f"with {token_loop} Token Loop"
            )

    base_id = BASE_METHOD_IDS[token_loop]
    preset_id = method_id if method_id in methods else base_id
    raw = methods.get(preset_id)
    if raw is None:
        raise KeyError(f"{base_id} preset is absent from config")
    values = deepcopy(raw)

    # A supported but unreported guided pair is composed from its base preset.
    if loop_guidance and method_id not in methods:
        if loop_guidance_weight is None:
            raise ValueError(
                f"{method_id} was not reported for this model; provide "
                "loop_guidance_weight explicitly for a custom run"
            )
        values.update(
            loopguidance_enabled=True,
            loopguidance_branch="base",
            loopguidance_weight=float(loop_guidance_weight),
            loopguidance_reference_ratio=0.0,
        )
        overrides["loop_guidance_weight"] = float(loop_guidance_weight)

    if num_loops is not None:
        values["num_loops"] = num_loops
        overrides["num_loops"] = num_loops
    if lambda_loop is not None:
        values["lambda_one_over_k"] = False
        values["lambda_value"] = lambda_loop
        overrides["lambda_loop"] = lambda_loop
    if loop_active is not None:
        start, end = loop_active
        values["start_frac"] = start
        values["end_frac"] = end
        overrides["loop_active"] = [start, end]
    if loop_layers is not None:
        first, last = loop_layers
        values["block_indices"] = inclusive_block_range(first, last)
        overrides["loop_layers"] = [first, last]
    if loop_guidance_weight is not None and method_id in methods:
        if not loop_guidance:
            raise ValueError("loop_guidance_weight requires --loop-guidance")
        values["loopguidance_weight"] = loop_guidance_weight
        overrides["loop_guidance_weight"] = loop_guidance_weight
    if selector is not None:
        if selector not in VALID_SELECTORS:
            raise ValueError(f"unknown selector: {selector!r}")
        values["selector"] = selector
        overrides["selector"] = selector
        if selector == "image_condition_split" and selection_ratio is None:
            values.pop("selection_ratio", None)
    if selection_ratio is not None:
        values["selection_ratio"] = selection_ratio
        overrides["selection_ratio"] = selection_ratio
    if selector_seed is not None:
        values["selector_seed"] = selector_seed
        overrides["selector_seed"] = selector_seed
    if token_domain is not None:
        if token_domain not in VALID_TOKEN_DOMAINS:
            raise ValueError(f"unknown token_domain: {token_domain!r}")
        values["token_domain"] = token_domain
        overrides["token_domain"] = token_domain

    loop_config = LoopConfig.from_dict(values)
    if loop_config.token_operator != token_loop:
        raise ValueError(
            f"resolved preset uses {loop_config.token_operator!r}, expected {token_loop!r}"
        )
    if loop_config.loopguidance_enabled != loop_guidance:
        raise ValueError("resolved Loop Guidance state does not match the requested composition")
    if token_loop == "sparse":
        supported_selectors = _sparse_selector_capabilities(payload)
        if loop_config.selector not in supported_selectors:
            raise ValueError(
                f"backend {payload.get('backend')!r} does not support sparse selector "
                f"{loop_config.selector!r}"
            )

    return ResolvedRunConfig(
        backend=str(payload.get("backend", "")),
        generation=generation,
        loop_config=loop_config,
        method_id=method_id,
        preset_id=preset_id,
        paper_preset=not overrides and preset_id == method_id,
        overrides=overrides,
    )
