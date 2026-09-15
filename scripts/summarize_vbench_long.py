#!/usr/bin/env python3
"""Summarize a VBench-Long eval_results.json and optionally append a ledger row.

Reads the `results_<timestamp>_eval_results.json` produced by
`VBench/vbench2_beta_long/eval_long.py`, prints a compact table, and can append
or update the workspace CSV ledger.

Usage:
    python scripts/summarize_vbench_long.py <eval_results.json> \
        [--ledger artifacts/experiment_results.csv] [--method NAME] \
        [--experiment NAME] [--slurm-job-id ID] [--video-dir PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

DIMS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

LEDGER_FIELDS = [
    "method", "experiment_name", "wandb_run_id", "backend", "status", "slurm_job_id",
    "config", "checkpoint", "seed", "latent_frames", "eval_frames", "fps",
    "prompt_count", *DIMS, "output_path", "timestamp", "notes",
]


def extract_scores(payload: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for dim in DIMS:
        if dim not in payload:
            continue
        value = payload[dim]
        # eval_long.py stores (average, detailed) tuples, but tolerate a scalar.
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        try:
            scores[dim] = round(float(value), 4)
        except (TypeError, ValueError):
            continue
    return scores


def upsert_ledger_row(ledger: Path, row: dict[str, str]) -> str:
    """Insert or replace the row matching (method, experiment_name, slurm_job_id)."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if ledger.exists():
        with ledger.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    key = (row["method"], row["experiment_name"], row["slurm_job_id"])
    replaced = False
    for index, existing in enumerate(rows):
        existing_key = (
            existing.get("method", ""),
            existing.get("experiment_name", ""),
            existing.get("slurm_job_id", ""),
        )
        if existing_key == key:
            rows[index] = {**existing, **row}
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for existing in rows:
            writer.writerow({field: existing.get(field, "") for field in LEDGER_FIELDS})
    return "updated" if replaced else "appended"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--method", default="looped-self-forcing")
    parser.add_argument("--experiment", default="looped-self-forcing-vbench-long")
    parser.add_argument("--slurm-job-id", default="")
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--config", default="configs/self_forcing_dmd_temporal_loop.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--latent-frames", default="162")
    parser.add_argument("--eval-frames", default="640")
    parser.add_argument("--fps", default="16")
    parser.add_argument("--prompt-count", default="946")
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the row without writing the ledger")
    args = parser.parse_args()

    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    scores = extract_scores(payload)

    missing = [dim for dim in DIMS if dim not in scores]
    print(f"results: {args.results_json}")
    for dim in DIMS:
        print(f"  {dim:<24} {scores.get(dim, 'MISSING')}")
    if missing:
        print(f"  WARNING missing dimensions: {missing}")

    row = {
        "method": args.method,
        "experiment_name": args.experiment,
        "wandb_run_id": "",
        "backend": "nm5",
        "status": "completed" if not missing else "partial",
        "slurm_job_id": args.slurm_job_id,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "latent_frames": args.latent_frames,
        "eval_frames": args.eval_frames,
        "fps": args.fps,
        "prompt_count": args.prompt_count,
        "output_path": args.video_dir,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "notes": args.notes or "VBench-Long --mode long_custom_input --dev_flag",
    }
    row.update({dim: str(scores.get(dim, "")) for dim in DIMS})

    if args.dry_run or args.ledger is None:
        print("ledger row (not written):")
        for field in LEDGER_FIELDS:
            print(f"  {field}={row.get(field, '')}")
        return
    action = upsert_ledger_row(args.ledger, row)
    print(f"ledger {action}: {args.ledger}")


if __name__ == "__main__":
    main()
