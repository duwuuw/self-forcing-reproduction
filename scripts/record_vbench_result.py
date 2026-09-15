#!/usr/bin/env python3
"""Record one official VBench-Long result in W&B, JSON, and the CSV ledger."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
import yaml

DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)

LEDGER_COLUMNS = (
    "timestamp", "workspace", "backend", "status", "method_name",
    "experiment_name", "variant", "wandb_project", "wandb_group",
    "wandb_run_name", "wandb_run_id", "step", "epoch", "fid",
    "glow_bpd", "raw_bpd", "bpd", "train_loss", "training_speed_it_s",
    "parameters", "checkpoint_path", "checkpoint_step", "examples_seen",
    "output_path", "notes", "experiment_version_id", "wandb_tags",
    "num_output_frames", "raw_decoded_frames", "eval_frames", "seed",
    "k_min", "k_max", "layer_start", "layer_end",
)


def scalar(value: Any, name: str) -> float:
    """Return the aggregate value for one VBench dimension.

    VBench emits a bare number, a single-element list, or -- with --dev_flag --
    ``[aggregate, [per-clip detail dicts]]``. The aggregate is always the
    leading element, so take value[0] and discard any per-clip breakdown. This
    matches the dimension guard in eval_packed.sbatch.
    """
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} is empty: {value!r}")
        value = value[0]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} is not numeric: {value!r}")
    return float(value)


def condition(config_path: Path) -> dict[str, int]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loop = data.get("temporal_loop", {}) or {}
    return {
        "num_output_frames": int(data["num_output_frames"]),
        "num_frame_per_block": int(data["num_frame_per_block"]),
        "k_min": int(loop["k_min"]),
        "k_max": int(loop["k_max"]),
        "layer_start": int(loop["layer_start"]),
        "layer_end": int(loop["layer_end"]),
    }


def append_ledger(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows: list[dict[str, str]] = []
        fieldnames = list(LEDGER_COLUMNS)
        if path.exists() and path.stat().st_size:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or fieldnames)
                for name in LEDGER_COLUMNS:
                    if name not in fieldnames:
                        fieldnames.append(name)
                rows = list(reader)
        rows.append({name: row.get(name, "") for name in fieldnames})
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--mode", choices=("dryrun", "fullrun"), required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--experiment-version-id", default="3")
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--expected-prompt-count", type=int, required=True)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""))
    args = parser.parse_args()

    if args.prompt_count != args.expected_prompt_count:
        raise SystemExit("EVAL ERROR: prompt count does not match expected count")
    result = json.loads(args.result_path.read_text(encoding="utf-8"))
    metrics = {name: scalar(result[name], name) for name in DIMENSIONS}
    cfg = condition(args.config_path)
    if cfg["num_output_frames"] != 123 or cfg["num_frame_per_block"] != 3:
        raise SystemExit("EVAL ERROR: frozen frame contract must be 123 latent frames and 3 frames/block")
    raw_frames = 4 * cfg["num_output_frames"] - 3
    eval_frames = 16 * (raw_frames // 16)
    if (raw_frames, eval_frames) != (489, 480):
        raise SystemExit("EVAL ERROR: fixed num_output_frames=123 must yield derived internal counts raw=489 and eval=480")

    project = args.wandb_project.strip()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not project:
        raise SystemExit("EVAL ERROR: WANDB_PROJECT is empty")
    if not entity:
        raise SystemExit("EVAL ERROR: WANDB_ENTITY is empty")
    if not api_key:
        raise SystemExit("EVAL ERROR: WANDB_API_KEY is empty")

    run_name = f"{project}-{args.run_id}-{args.variant}"
    tags = [
        "backend:nm5", "platform:cuda", "gpu:H100",
        "workflow:looped_self_forcing_vbench_long_search",
        "stage:vbench_long_eval", f"run_id:{args.run_id}",
        f"slurm:{args.job_id}", f"variant:{args.variant}",
        f"mode:{args.mode}", f"k:{cfg['k_min']}-{cfg['k_max']}",
        f"layers:{cfg['layer_start']}-{cfg['layer_end']}",
        "protocol:30s_fixed_num_output_frames_123",
        "derived_counts:raw489_eval480",
    ]
    wandb_dir = Path(os.environ["WANDB_DIR"])
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "group": os.environ.get("WANDB_GROUP", args.run_id),
        "name": run_name,
        "job_type": "vbench_long_eval",
        "tags": tags,
        "dir": str(wandb_dir),
        "mode": os.environ.get("WANDB_MODE", "offline"),
        "config": {
            "workspace": "looped-flow-matching",
            "backend": "nm5",
            "variant": args.variant,
            "run_id": args.run_id,
            "slurm_job_id": args.job_id,
            "prompt_count": args.prompt_count,
            "expected_prompt_count": args.expected_prompt_count,
            "seed": 0,
            "use_ema": True,
            **cfg,
            "raw_decoded_frames": raw_frames,
            "eval_frames": eval_frames,
            "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts",
            "fps": 16,
            "duration_seconds": 30,
            "experiment_version_id": args.experiment_version_id,
        },
    }
    run = wandb.init(**wandb_kwargs)
    run.log({f"vbench/{name}": value for name, value in metrics.items()}, step=0)
    run.summary.update(metrics)
    run.summary.update({
        "prompt_count": args.prompt_count,
        "mode": args.mode,
        "k_min": cfg["k_min"],
        "k_max": cfg["k_max"],
        "layer_start": cfg["layer_start"],
        "layer_end": cfg["layer_end"],
        "checkpoint_step": 0,
        "examples_seen": args.prompt_count,
    })
    run.finish()

    metrics_payload = {
        "stage": "vbench_long_eval",
        "run_id": args.run_id,
        "variant": args.variant,
        "condition_label": f"{args.variant} | K=[{cfg['k_min']},{cfg['k_max']}] | layers={cfg['layer_start']}-{cfg['layer_end']}",
        "mode": args.mode,
        "slurm_job_id": args.job_id,
        "prompt_count": args.prompt_count,
        "expected_prompt_count": args.expected_prompt_count,
        "seed": 0,
        "use_ema": True,
        "num_output_frames": cfg["num_output_frames"],
        "raw_decoded_frames": raw_frames,
        "eval_frames": eval_frames,
        "derived_internal_processing_counts": {"raw_decoded_frames": raw_frames, "eval_frames": eval_frames},
        "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts",
        "fps": 16,
        "k_min": cfg["k_min"],
        "k_max": cfg["k_max"],
        "layer_start": cfg["layer_start"],
        "layer_end": cfg["layer_end"],
        "metrics": metrics,
        "wandb_project": project,
        "wandb_entity": entity,
        "wandb_group": os.environ.get("WANDB_GROUP", args.run_id),
        "wandb_run_name": run_name,
        "wandb_run_id": run.id,
        "wandb_mode": os.environ.get("WANDB_MODE", "offline"),
        "wandb_tags": tags,
        "checkpoint_path": "Self-Forcing/checkpoints/self_forcing_dmd.pt",
        "checkpoint_step": 0,
        "examples_seen": args.prompt_count,
        "output_path": str(args.output_path),
        "vbench_result_path": str(args.result_path),
        "experiment_version_id": args.experiment_version_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )

    row = {name: "" for name in LEDGER_COLUMNS}
    row.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workspace": "looped-flow-matching",
        "backend": "nm5",
        "status": "completed",
        "method_name": "looped-self-forcing",
        "experiment_name": project,
        "variant": args.variant,
        "wandb_project": project,
        "wandb_group": os.environ.get("WANDB_GROUP", args.run_id),
        "wandb_run_name": run_name,
        "wandb_run_id": run.id,
        "step": "0",
        "epoch": "0",
        "checkpoint_path": "Self-Forcing/checkpoints/self_forcing_dmd.pt",
        "checkpoint_step": "0",
        "examples_seen": str(args.prompt_count),
        "output_path": str(args.output_path),
        "notes": (
            f"mode={args.mode}; prompts={args.prompt_count}; static_prompts=10; "
            f"dynamic_prompts=118; fixed num_output_frames=123; derived internal processing counts raw=489/eval=480; fps=16; duration=30s; "
            f"K=[{cfg['k_min']},{cfg['k_max']}]; layers={cfg['layer_start']}-{cfg['layer_end']}; "
            "metrics_source=official VBench-Long aggregate"
        ),
        "experiment_version_id": args.experiment_version_id,
        "wandb_tags": ",".join(tags),
        "num_output_frames": str(cfg["num_output_frames"]),
        "raw_decoded_frames": str(raw_frames),
        "eval_frames": str(eval_frames),
        "seed": "0",
        "k_min": str(cfg["k_min"]),
        "k_max": str(cfg["k_max"]),
        "layer_start": str(cfg["layer_start"]),
        "layer_end": str(cfg["layer_end"]),
    })
    append_ledger(args.ledger_path, row)
    print(json.dumps({
        "variant": args.variant,
        "wandb_run_id": run.id,
        "metrics_path": str(args.metrics_path),
        "ledger_path": str(args.ledger_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
