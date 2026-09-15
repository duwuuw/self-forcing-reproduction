#!/usr/bin/env python3
"""Select a diverse, metadata-audited VBench-Long prompt subset.

The run contract is exactly 128 prompts: 10 static prompts from VBench's
temporal_flickering dimension and 118 prompts from motion-oriented VBench
dimensions. Selection within each pool is deterministic and uses a
TF-IDF/cosine farthest-point pass to reduce semantic near-duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


STATIC_DIMENSIONS = {"temporal_flickering"}
DYNAMIC_DIMENSIONS = {
    "human_action",
    "dynamic_degree",
    "motion_smoothness",
    "temporal_style",
}
GENERIC_TOKENS = {
    "a", "an", "and", "at", "behind", "beside", "capture", "captured",
    "camera", "close", "composition", "day", "during", "emphasis",
    "emphasizing", "entire", "environment", "featuring", "focus", "focused",
    "foreground", "frame", "from", "has", "in", "including", "medium",
    "moment", "of", "on", "overall", "perspective", "scene", "scenes",
    "shot", "showcasing", "shows", "soft", "space", "surrounding", "the",
    "this", "throughout", "to", "view", "wide", "with", "within",
}
WORD = re.compile("[a-z][a-z']+")


@dataclass(frozen=True)
class Prompt:
    index: int
    text: str
    static_hits: int
    dynamic_hits: int
    dimensions: tuple[str, ...]

    @property
    def label(self) -> str:
        return "static" if self.static_hits else "dynamic"


def load_dimensions(metadata_path: Path, expected_count: int) -> list[tuple[str, ...]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or len(metadata) != expected_count:
        raise ValueError(
            f"VBench metadata must be a list with {expected_count} entries: {metadata_path}"
        )
    dimensions: list[tuple[str, ...]] = []
    for item in metadata:
        if not isinstance(item, dict) or not isinstance(item.get("dimension"), list):
            raise ValueError(f"invalid VBench metadata entry in {metadata_path}")
        dimensions.append(tuple(str(value) for value in item["dimension"]))
    return dimensions


def content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in WORD.findall(text.lower()):
        if len(token) < 3 or token in GENERIC_TOKENS:
            continue
        for suffix in ("ingly", "edly", "ing", "ers", "ies", "es", "ed", "ly", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        tokens.append(token)
    tokens.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return tokens


def vectors(prompts: list[Prompt]) -> list[dict[str, float]]:
    token_lists = [content_tokens(prompt.text) for prompt in prompts]
    document_frequency: dict[str, int] = {}
    for tokens in token_lists:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    total = len(prompts)
    output: list[dict[str, float]] = []
    for tokens in token_lists:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        vector = {
            token: (1.0 + math.log(count))
            * (math.log((1.0 + total) / (1.0 + document_frequency[token])) + 1.0)
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        output.append({token: value / norm for token, value in vector.items()})
    return output


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def tie_break(prompt: Prompt) -> float:
    seed_text = f"{prompt.index}{chr(0)}{prompt.text}"
    digest = hashlib.sha1(seed_text.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def farthest_select(prompts: list[Prompt], count: int) -> list[Prompt]:
    if count > len(prompts):
        raise ValueError(f"cannot select {count} prompts from {len(prompts)} candidates")
    if count == len(prompts):
        return list(prompts)

    vecs = vectors(prompts)
    selected: list[int] = []
    first = max(
        range(len(prompts)),
        key=lambda i: (prompts[i].dynamic_hits + prompts[i].static_hits, -prompts[i].index),
    )
    selected.append(first)
    while len(selected) < count:
        selected_set = set(selected)
        best_index = None
        best_score = -1.0
        for candidate in range(len(prompts)):
            if candidate in selected_set:
                continue
            max_similarity = max(cosine(vecs[candidate], vecs[item]) for item in selected)
            diversity = 1.0 - max_similarity
            index_spread = min(abs(prompts[candidate].index - prompts[item].index) for item in selected)
            spread = min(1.0, index_spread / 100.0)
            quality = min(1.0, (prompts[candidate].dynamic_hits + prompts[candidate].static_hits) / 4.0)
            score = 0.76 * diversity + 0.14 * spread + 0.08 * quality + 0.02 * tie_break(prompts[candidate])
            if score > best_score:
                best_score = score
                best_index = candidate
        assert best_index is not None
        selected.append(best_index)
    return [prompts[index] for index in selected]


def record(prompt: Prompt, rank: int) -> dict[str, object]:
    return {
        "selected_rank": rank,
        "source_index": prompt.index,
        "classification": prompt.label,
        "static_cue_count": prompt.static_hits,
        "dynamic_cue_count": prompt.dynamic_hits,
        "dimensions": list(prompt.dimensions),
        "prompt": prompt.text,
    }


def write_selection(input_path: Path, metadata_path: Path, output_dir: Path) -> None:
    lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines()]
    if any(not line for line in lines):
        raise ValueError(f"prompt file contains an empty line: {input_path}")
    dimensions = load_dimensions(metadata_path, len(lines))
    prompts = []
    for index, text in enumerate(lines):
        prompt_dimensions = dimensions[index]
        static_hits = int(bool(STATIC_DIMENSIONS.intersection(prompt_dimensions)))
        dynamic_hits = int(bool(DYNAMIC_DIMENSIONS.intersection(prompt_dimensions)))
        prompts.append(Prompt(index, text, static_hits, dynamic_hits, prompt_dimensions))

    static = [prompt for prompt in prompts if prompt.static_hits]
    dynamic = [prompt for prompt in prompts if prompt.dynamic_hits and not prompt.static_hits]
    if len(static) < 10 or len(dynamic) < 118:
        raise ValueError(f"insufficient classified prompts: static={len(static)} dynamic={len(dynamic)}")

    selected_static = farthest_select(static, 10)
    selected_dynamic = farthest_select(dynamic, 118)
    final = sorted(selected_static + selected_dynamic, key=lambda prompt: prompt.index)
    if len(final) != 128:
        raise AssertionError(len(final))
    if sum(prompt.static_hits > 0 for prompt in final) != 10:
        raise AssertionError("final selection must contain exactly 10 static prompts")
    if len({prompt.text for prompt in final}) != 128:
        raise AssertionError("final selection contains duplicate prompt text")

    output_dir.mkdir(parents=True, exist_ok=True)
    separator = chr(10)
    (output_dir / "prompts_candidate_128.txt").write_text(
        separator.join(prompt.text for prompt in final) + separator, encoding="utf-8"
    )
    (output_dir / "prompts_selected_128.txt").write_text(
        separator.join(prompt.text for prompt in final) + separator, encoding="utf-8"
    )
    payload = {
        "source": str(input_path),
        "source_count": len(prompts),
        "selection_contract": {
            "candidate_pool_count": 128,
            "candidate_static_count": 10,
            "candidate_dynamic_count": 118,
            "final_count": 128,
            "final_static_count": 10,
            "final_dynamic_count": 118,
            "gpu_count": 3,
            "per_config_gpu_count": 1,
            "note": "Each K variant runs the complete 128-prompt set on one H100; prompts are not split across GPUs.",
        },
        "classification": {
            "metadata_path": str(metadata_path),
            "static_dimensions": sorted(STATIC_DIMENSIONS),
            "dynamic_dimensions": sorted(DYNAMIC_DIMENSIONS),
            "note": "Static is the official VBench temporal_flickering dimension; dynamic is the union of motion-oriented VBench dimensions.",
        },
        "candidate_pool": [record(prompt, rank) for rank, prompt in enumerate(final)],
        "final_selection": [record(prompt, rank) for rank, prompt in enumerate(final)],
    }
    (output_dir / "prompt_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + separator, encoding="utf-8"
    )
    print(json.dumps({
        "source_count": len(prompts),
        "classified_static": len(static),
        "classified_dynamic": len(dynamic),
        "final_count": len(final),
        "final_static": sum(item.static_hits > 0 for item in final),
        "final_dynamic": sum(item.static_hits == 0 for item in final),
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_selection(args.input, args.metadata, args.output_dir)


if __name__ == "__main__":
    main()
