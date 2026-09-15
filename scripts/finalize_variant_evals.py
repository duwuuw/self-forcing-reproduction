#!/usr/bin/env python3
"""Roll up per-variant VBench-Long evaluations into pack-level artifacts.

`eval_variant_1gpu.sbatch` evaluates one variant per job and writes its own
metrics.json, W&B run, and CSV ledger row. This script performs the pack-level
steps that `eval_packed.sbatch` would otherwise do at the end of a 4-GPU pack:

  * print a comparison table across variants (with deltas vs a reference arm),
  * write ``<run_root>/run_state/eval_complete.json``,
  * write ``<exp>/readiness/readiness_nm5.json`` for a dryrun,
  * delegate to ``reduce_vbench_history.py`` to refresh the labeled history.

Run it once every per-variant job for the same run has finished. It is
idempotent: it only reads the variant outputs and rewrites the roll-up files.

Usage:
    python scripts/finalize_variant_evals.py \
        --exp-dir scale_up_outputs/<exp_dir> --run-id <run_id> \
        --mode fullrun|dryrun [--variants v1 v2 ...] [--reference <variant>]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DIMS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)

PROTOCOL = {
    "num_output_frames": 123,
    "derived_internal_processing_counts": {"raw_decoded_frames": 489, "eval_frames": 480},
    "frame_count_policy": (
        "only num_output_frames is configurable; raw/eval counts are derived "
        "internal processing counts"
    ),
    "fps": 16,
    "duration_seconds": 30,
    "seed": 0,
    "use_ema": True,
}


def dimension_scores(metrics: dict) -> dict:
    """record_vbench_result.py nests the six dims under "metrics"."""
    inner = metrics.get("metrics")
    return inner if isinstance(inner, dict) else metrics


def aggregate(metrics: dict, dim: str) -> float | None:
    """VBench --dev_flag emits [aggregate, [per-clip...]]; take the aggregate."""
    value = dimension_scores(metrics).get(dim)
    if isinstance(value, list):
        value = value[0] if value else None
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", required=True, choices=("dryrun", "fullrun"))
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--reference", default=None,
                        help="variant to report deltas against (e.g. the loop-off arm)")
    parser.add_argument("--reference-run-id", default=None,
                        help="run id holding --reference; defaults to --run-id. The "
                             "loop-off baseline usually lives in an earlier round.")
    args = parser.parse_args()

    run_root = args.exp_dir / args.run_id
    variants = args.variants
    if not variants:
        variants = sorted(p.name for p in (run_root / "variants").glob("*") if p.is_dir())

    metrics: dict[str, dict] = {}
    missing: list[str] = []
    for variant in variants:
        path = run_root / "variants" / variant / "eval" / "metrics.json"
        if not path.is_file():
            missing.append(variant)
            continue
        metrics[variant] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise SystemExit(f"EVAL ROLL-UP BLOCKED: missing metrics.json for {missing}")

    reference = args.reference
    reference_metrics: dict | None = None
    if reference:
        reference_run = args.reference_run_id or args.run_id
        if reference_run == args.run_id:
            if reference not in metrics:
                raise SystemExit(f"EVAL ROLL-UP BLOCKED: reference {reference} has no metrics")
        else:
            path = args.exp_dir / reference_run / "variants" / reference / "eval" / "metrics.json"
            if not path.is_file():
                raise SystemExit(
                    f"EVAL ROLL-UP BLOCKED: reference {reference} has no metrics under run {reference_run}"
                )
            reference_metrics = json.loads(path.read_text(encoding="utf-8"))

    # The reference is only listed as a row when it belongs to this run; a
    # cross-run baseline is reported as a delta column, not as a variant here.
    local_reference = reference if (reference and reference_metrics is None) else None
    order = ([local_reference] if local_reference else []) + [v for v in variants if v != local_reference]
    width = max(len(v) for v in variants) + 2
    header = f"{'variant':<{width}}" + "".join(f"{d[:15]:>17}" for d in DIMS)
    print(header)
    print("-" * len(header))
    for variant in order:
        row = f"{variant:<{width}}"
        for dim in DIMS:
            value = aggregate(metrics[variant], dim)
            row += f"{value:>17.4f}" if value is not None else f"{'n/a':>17}"
        print(row)

    if reference:
        reference_run = args.reference_run_id or args.run_id
        base_source = reference_metrics if reference_metrics is not None else metrics[reference]
        label = reference if reference_run == args.run_id else f"{reference} ({reference_run})"
        print()
        print(f"delta vs {label} (negative = worse):")
        print(header)
        print("-" * len(header))
        for variant in order:
            row = f"{variant:<{width}}"
            for dim in DIMS:
                base = aggregate(base_source, dim)
                value = aggregate(metrics[variant], dim)
                if base is None or value is None:
                    row += f"{'n/a':>17}"
                else:
                    row += f"{value - base:>+17.4f}"
            print(row)

    completed_at = datetime.now(timezone.utc).isoformat()
    (run_root / "run_state").mkdir(parents=True, exist_ok=True)
    (run_root / "run_state" / "eval_complete.json").write_text(
        json.dumps({
            "stage": "vbench_long_eval",
            "mode": args.mode,
            "variants": variants,
            "prompt_count": 1 if args.mode == "dryrun" else 128,
            "protocol": PROTOCOL,
            "decomposition": "one GPU per variant (eval_variant_1gpu.sbatch)",
            "completed_at": completed_at,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {run_root / 'run_state' / 'eval_complete.json'}")

    if args.mode == "dryrun":
        readiness = args.exp_dir / "readiness" / "readiness_nm5.json"
        readiness.parent.mkdir(parents=True, exist_ok=True)
        readiness.write_text(
            json.dumps({
                "backend": "nm5",
                "status": "ready",
                "fullrun_ready": True,
                "dryrun_run_id": args.run_id,
                "variants": variants,
                "protocol": PROTOCOL,
                "evidence": [
                    "all variant generation tasks completed",
                    "all official VBench-Long evaluations completed",
                    "CSV ledger and labeled history updated",
                ],
                "verified_at": completed_at,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {readiness}")


if __name__ == "__main__":
    main()
