#!/usr/bin/env python3
"""Aggregate normalized per-item scores and compute baseline deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {"method_id", "benchmark", "prompt_id", "metric", "value"}


def aggregate_rows(rows: list[dict], baseline: str) -> list[dict]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        missing = REQUIRED_COLUMNS - row.keys()
        if missing:
            raise ValueError(f"score row is missing columns: {sorted(missing)}")
        key = (
            str(row["method_id"]),
            str(row["benchmark"]),
            str(row["prompt_id"]),
            str(row["metric"]),
        )
        if key in seen:
            raise ValueError(f"duplicate per-item score: {key}")
        seen.add(key)
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite score for {key}")
        values[(key[0], key[1], key[3])].append(value)

    means = {
        key: (sum(items) / len(items), len(items)) for key, items in values.items()
    }
    baseline_means = {
        (benchmark, metric): mean
        for (method, benchmark, metric), (mean, _) in means.items()
        if method == baseline
    }
    if not baseline_means:
        raise ValueError(f"baseline method {baseline!r} is absent")

    output = []
    for (method, benchmark, metric), (mean, count) in sorted(means.items()):
        ref_key = (benchmark, metric)
        if ref_key not in baseline_means:
            raise ValueError(f"missing baseline for benchmark/metric {ref_key}")
        reference = baseline_means[ref_key]
        delta = mean - reference
        delta_pct = None
        if metric.lower() != "imagereward" and reference != 0:
            delta_pct = delta / abs(reference) * 100.0
        output.append(
            {
                "method_id": method,
                "benchmark": benchmark,
                "metric": metric,
                "count": count,
                "value": mean,
                "baseline": baseline,
                "baseline_value": reference,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-item-csv", type=Path, required=True)
    parser.add_argument("--baseline", default="without_loop")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    with args.per_item_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = aggregate_rows(rows, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"aggregate_rows": len(output), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
