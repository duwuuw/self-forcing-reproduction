#!/usr/bin/env python3
"""Append labeled VBench-Long conditions to one durable Markdown history."""

from __future__ import annotations

import argparse
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)


def metric(value: object) -> float:
    if isinstance(value, list):
        value = value[0]
    if not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric metric: {value!r}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--mode", choices=("dryrun", "fullrun"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()

    records = []
    for variant in args.variants:
        path = args.run_dir / "variants" / variant / "eval" / "metrics.json"
        if not path.exists():
            raise SystemExit(f"HISTORY ERROR: missing metrics for {variant}: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = [name for name in DIMENSIONS if name not in data.get("metrics", {})]
        if missing:
            raise SystemExit(f"HISTORY ERROR: {variant} missing metrics: {','.join(missing)}")
        records.append(data)

    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(args.history_path) + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = args.history_path.read_text(encoding="utf-8") if args.history_path.exists() else ""
        additions: list[str] = []
        for data in sorted(records, key=lambda item: str(item["variant"])):
            marker = f"<!-- eval:{args.run_id}:{data['variant']} -->"
            if marker in existing:
                continue
            metrics = {name: metric(data["metrics"][name]) for name in DIMENSIONS}
            additions.extend([
                marker,
                f"## Condition {data['variant']} - {args.mode} - {args.run_id}",
                "",
                (
                    f"- K range: [{data['k_min']}, {data['k_max']}]; "
                    f"loop layers (inclusive, zero-based): {data['layer_start']}..{data['layer_end']}"
                ),
                "- Frozen video protocol: only num_output_frames=123 is configurable; raw decode=489 and VBench processing=480 are derived internal counts; fps=16 (30 seconds), seed 0, EMA enabled",
                f"- Prompts evaluated: {data['prompt_count']} / {data['expected_prompt_count']}",
                f"- W&B run: {data['wandb_run_id']}; Slurm job: {data['slurm_job_id']}",
                "",
                "| subject | background | motion | dynamic | aesthetic | imaging |",
                "|---:|---:|---:|---:|---:|---:|",
                (
                    f"| {metrics['subject_consistency']:.9f} | "
                    f"{metrics['background_consistency']:.9f} | "
                    f"{metrics['motion_smoothness']:.9f} | "
                    f"{metrics['dynamic_degree']:.9f} | "
                    f"{metrics['aesthetic_quality']:.9f} | "
                    f"{metrics['imaging_quality']:.9f} |"
                ),
                "",
            ])
        if additions:
            prefix = "" if not existing or existing.endswith("\n") else "\n"
            args.history_path.write_text(existing + prefix + "\n".join(additions), encoding="utf-8")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    summary = {
        "run_id": args.run_id,
        "mode": args.mode,
        "variants": [data["variant"] for data in records],
        "metric_dimensions": list(DIMENSIONS),
        "protocol": {
            "num_output_frames": 123,
            "derived_internal_processing_counts": {"raw_decoded_frames": 489, "eval_frames": 480},
            "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts",
            "fps": 16,
            "duration_seconds": 30,
            "seed": 0,
            "use_ema": True,
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = args.run_dir / "state" / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
