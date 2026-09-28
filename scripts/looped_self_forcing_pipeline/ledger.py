"""Durable CSV result ledger for the looped Self-Forcing pipeline."""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIELDNAMES = (
    "method",
    "experiment_name",
    "wandb_run_id",
    "backend",
    "status",
    "execution_id",
    "slurm_job_id",
    "seed",
    "step",
    "checkpoint_step",
    "fid",
    "bpd",
    "training_speed",
    "parameter_count",
    "trainable_parameter_count",
    "timestamp",
    "checkpoint_or_output_path",
    "error",
)


def write_result(path: str | Path, row: dict[str, Any]) -> None:
    """Insert or update one experiment row, serializing concurrent writers."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = {field: row.get(field, "") for field in FIELDNAMES}
    normalized["timestamp"] = normalized["timestamp"] or datetime.now(timezone.utc).isoformat()
    identity = str(normalized["experiment_name"]).strip()
    if not identity:
        raise ValueError("experiment_name is required for ledger upserts")

    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_stream:
        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - pipeline targets Linux backends
            raise RuntimeError("fcntl is required for safe concurrent ledger writes") from error
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            rows: list[dict[str, Any]] = []
            if target.exists() and target.stat().st_size > 0:
                with target.open(newline="", encoding="utf-8") as existing:
                    reader = csv.DictReader(existing)
                    if tuple(reader.fieldnames or ()) != FIELDNAMES:
                        raise ValueError(f"ledger schema mismatch at {target}")
                    rows = list(reader)
            if rows:
                updated = False
                for index, previous in enumerate(rows):
                    if previous.get("experiment_name") == identity:
                        rows[index] = {**previous, **normalized}
                        updated = True
                        break
                if not updated:
                    rows.append(normalized)
            else:
                rows = [normalized]

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, extrasaction="raise")
                    writer.writeheader()
                    writer.writerows(rows)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
