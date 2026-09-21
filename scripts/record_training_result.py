#!/usr/bin/env python
"""Append a training-run row to the workspace CSV results ledger.

Writes to both ledger surfaces so the SUE tooling and the workspace-level
review file stay in step:

* ``<exp_dir>/artifacts/experiment_results.csv``  -- SUE ``paths.ledger_csv``
* ``<workspace>/artifacts/experiment_results.csv`` -- workspace review ledger

Values are never invented. Any metric that was not actually measured must be
passed as None and is written as an empty field.

Usage:
    python scripts/record_training_result.py \
        --exp-dir scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd \
        --run-id smoke_r1_20260918 --mode smoke \
        --status completed --slurm-job-id 46100510 \
        --step 120 --train-loss 0.0421 --checkpoint-path ... --notes "..."
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE_LEDGER_COLUMNS = [
    "method", "experiment_name", "wandb_run_id", "backend", "status",
    "slurm_job_id", "config", "checkpoint", "seed", "latent_frames",
    "eval_frames", "fps", "prompt_count", "subject_consistency",
    "background_consistency", "motion_smoothness", "dynamic_degree",
    "aesthetic_quality", "imaging_quality", "output_path", "timestamp",
    "notes",
]

TRAINING_LEDGER_COLUMNS = [
    "timestamp", "workspace", "backend", "stage", "status", "method_name",
    "experiment_name", "variant", "wandb_project", "wandb_group",
    "wandb_run_name", "wandb_run_id", "step", "train_loss",
    "generator_grad_norm", "critic_loss", "critic_grad_norm",
    "training_speed_it_s", "steps_completed", "elapsed_seconds",
    "parameters_trainable", "base_checkpoint", "teacher_checkpoint",
    "slurm_job_id", "qos", "partition", "account", "checkpoint_path",
    "checkpoint_step", "output_path", "notes", "train_config",
    "latent_frames_train", "eval_protocol", "slurm_job_name",
]


def _blank(value):
    return "" if value is None else value


def append_row(path: Path, columns: list[str], row: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    unknown = set(row) - set(columns)
    if unknown:
        raise SystemExit(f"LEDGER ERROR: unknown columns for {path}: {sorted(unknown)}")
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({c: _blank(row.get(c)) for c in columns})
    with path.open(encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mode", required=True, choices=["smoke", "fullrun"])
    ap.add_argument("--status", required=True)
    ap.add_argument("--slurm-job-id", default=None)
    ap.add_argument("--slurm-job-name", default=None)
    ap.add_argument("--qos", default=None)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--step", default=None)
    ap.add_argument("--train-loss", default=None)
    ap.add_argument("--generator-grad-norm", default=None)
    ap.add_argument("--critic-loss", default=None)
    ap.add_argument("--critic-grad-norm", default=None)
    ap.add_argument("--training-speed-it-s", default=None)
    ap.add_argument("--steps-completed", default=None)
    ap.add_argument("--elapsed-seconds", default=None)
    ap.add_argument("--parameters-trainable", default=None)
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--teacher-checkpoint", default=None)
    ap.add_argument("--generator-checkpoint", default=None)
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument("--checkpoint-step", default=None)
    ap.add_argument("--output-path", default=None)
    ap.add_argument("--train-config", default=None)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-run-id", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    exp_dir = (ws / args.exp_dir).resolve()
    if not str(exp_dir).startswith(str(ws)):
        raise SystemExit("LEDGER ERROR: exp-dir escapes the workspace")

    timestamp = datetime.now(timezone.utc).isoformat()
    method = "blockwise-lora-dmd"
    experiment_name = f"{method}-{args.run_id}"

    common = {
        "timestamp": timestamp,
        "workspace": ws.name,
        "backend": "nm5",
        "status": args.status,
        "slurm_job_id": args.slurm_job_id,
        "output_path": args.output_path,
        "notes": args.notes,
        "checkpoint_path": args.checkpoint_path or args.generator_checkpoint,
        "wandb_run_id": args.wandb_run_id,
    }

    training_row = dict(common, **{
        "stage": "training",
        "method_name": method,
        "experiment_name": experiment_name,
        "variant": args.mode,
        "wandb_project": args.wandb_project,
        "wandb_group": args.run_id,
        "wandb_run_name": args.wandb_run_name,
        "step": args.step,
        "train_loss": args.train_loss,
        "generator_grad_norm": args.generator_grad_norm,
        "critic_loss": args.critic_loss,
        "critic_grad_norm": args.critic_grad_norm,
        "training_speed_it_s": args.training_speed_it_s,
        "steps_completed": args.steps_completed,
        "elapsed_seconds": args.elapsed_seconds,
        "parameters_trainable": args.parameters_trainable,
        "base_checkpoint": args.base_checkpoint,
        "teacher_checkpoint": args.teacher_checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "train_config": args.train_config,
        "latent_frames_train": 21,
        "eval_protocol": "training 21 latent frames; NOT comparable to the 123-latent VBench-Long protocol",
        "qos": args.qos,
        "partition": args.partition,
        "account": args.account,
        "slurm_job_name": args.slurm_job_name,
    })

    # The workspace review ledger has a different (eval-shaped) schema; build
    # its row from exactly its own columns.
    workspace_row = {
        "method": method,
        "experiment_name": experiment_name,
        "wandb_run_id": args.wandb_run_id,
        "backend": "nm5",
        "status": args.status,
        "slurm_job_id": args.slurm_job_id,
        "config": args.train_config,
        "checkpoint": args.checkpoint_path or args.generator_checkpoint,
        "seed": 0,
        "latent_frames": 21,
        "output_path": args.output_path,
        "timestamp": timestamp,
        "notes": args.notes,
    }

    exp_csv = exp_dir / "artifacts" / "experiment_results.csv"
    ws_csv = ws / "artifacts" / "experiment_results.csv"

    n_exp = append_row(exp_csv, TRAINING_LEDGER_COLUMNS, training_row)
    n_ws = append_row(ws_csv, WORKSPACE_LEDGER_COLUMNS, workspace_row)

    print(f"LEDGER_OK exp={exp_csv} rows={n_exp}")
    print(f"LEDGER_OK workspace={ws_csv} rows={n_ws}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
