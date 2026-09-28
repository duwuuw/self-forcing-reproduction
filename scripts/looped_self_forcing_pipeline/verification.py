"""Evidence checks for worker-produced checkpoints, videos, and paired runs."""

from __future__ import annotations

import json
import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


_PAIR_SMOKE_METHODS = (
    ("layerwise_l23_30_k3_lr5gen", "k3-lr5gen", 3, 4e-7, 4e-7),
    ("layerwise_l23_30_k2_lr5both", "k2-lr5both", 2, 4e-7, 8e-8),
)


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoke_artifact_hash(path: Path, label: str, profile: str) -> str:
    try:
        return _sha256_path(path)
    except OSError as error:
        raise RuntimeError(f"smoke {label} hash could not be read for {profile}") from error


def verify_smoke_pair_evidence(
    exp_dir: str | Path,
    pair_id: str,
    *,
    run_command: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Require completed scheduler, checkpoint, W&B, ledger, and four-rank probe evidence."""
    from looped_self_forcing_pipeline.config import load_runtime_config, resolve_runtime_paths

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,52}", str(pair_id)):
        raise ValueError("smoke pair ID must be 1–53 lowercase letters, digits, _ or -")
    bundle = Path(exp_dir).resolve()
    roots = resolve_runtime_paths(bundle)
    runtime = load_runtime_config(bundle)
    resources = runtime["sandbox_resources"]["nm5"]
    expected_gpu_type = str(resources["gpu_type"])
    expected_gpu_count = int(resources["train_gpus"])
    max_safe_reserved_gib = float(resources["gpu_usable_memory_gib"]) - float(
        resources["gpu_non_torch_reserve_gib"]
    )
    ledger_path = roots["ledger_csv"]
    try:
        with ledger_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise RuntimeError("smoke pair result ledger is unavailable") from error
    receipt_path = roots["artifacts_root"] / "pairs" / pair_id / "submission.json"
    try:
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("smoke pair submission receipt is missing or unreadable") from error
    receipt_jobs = receipt.get("jobs") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("pair_id") != pair_id
        or not isinstance(receipt_jobs, list)
        or len(receipt_jobs) != 2
        or set(receipt) != {"schema_version", "pair_id", "created_at", "jobs"}
    ):
        raise RuntimeError("smoke pair submission receipt is invalid")
    if any(
        set(item) != {"profile", "run_id", "job_id", "submitted_at"}
        for item in receipt_jobs
        if isinstance(item, dict)
    ):
        raise RuntimeError("smoke pair submission receipt contains unsupported fields")
    receipt_by_profile = {
        item.get("profile"): item for item in receipt_jobs if isinstance(item, dict)
    }

    runner = run_command or subprocess.run
    evidence: list[dict[str, Any]] = []
    latest_submission_at: datetime | None = None
    for profile, suffix, k, expected_lr, expected_lr_critic in _PAIR_SMOKE_METHODS:
        run_id = f"{pair_id}-{suffix}"
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_id):
            raise ValueError("smoke pair ID produces an invalid run ID")
        experiment_name = f"{profile}-{run_id}-train"
        matches = [row for row in rows if row.get("experiment_name") == experiment_name]
        if len(matches) != 1:
            raise RuntimeError(f"smoke pair ledger row is missing or ambiguous for {profile}")
        row = matches[0]
        if row.get("status") != "completed" or row.get("backend") != "nm5":
            raise RuntimeError(f"smoke pair ledger does not show a completed NM5 run for {profile}")
        try:
            submitted_at = datetime.fromisoformat(
                str(row.get("timestamp", "")).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise RuntimeError(f"smoke pair ledger timestamp is invalid for {profile}") from error
        if submitted_at.tzinfo is None:
            raise RuntimeError(f"smoke pair ledger timestamp lacks a timezone for {profile}")
        job_id = str(row.get("slurm_job_id", "")).strip()
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise RuntimeError(f"smoke pair ledger has no valid Slurm job ID for {profile}")
        receipt_job = receipt_by_profile.get(profile)
        if (
            not isinstance(receipt_job, dict)
            or receipt_job.get("run_id") != run_id
            or str(receipt_job.get("job_id", "")) != job_id
        ):
            raise RuntimeError(f"smoke pair submission receipt disagrees with its ledger row for {profile}")
        try:
            receipt_submitted_at = datetime.fromisoformat(
                str(receipt_job.get("submitted_at", "")).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise RuntimeError(f"smoke pair receipt timestamp is invalid for {profile}") from error
        if receipt_submitted_at.tzinfo is None:
            raise RuntimeError(f"smoke pair receipt timestamp lacks a timezone for {profile}")
        if (
            receipt_submitted_at.astimezone(timezone.utc)
            - datetime.now(timezone.utc)
        ).total_seconds() > 30:
            raise RuntimeError(f"smoke pair receipt timestamp is in the future for {profile}")
        if latest_submission_at is None or receipt_submitted_at > latest_submission_at:
            latest_submission_at = receipt_submitted_at

        scheduler = runner(
            ["sacct", "--noheader", "--parsable2", "-j", job_id, "--format=JobIDRaw,State"],
            capture_output=True,
            text=True,
            check=False,
        )
        if getattr(scheduler, "returncode", 1) != 0:
            raise RuntimeError(f"sacct could not verify smoke job {job_id}")
        exact_states = []
        for line in str(getattr(scheduler, "stdout", "")).splitlines():
            fields = [field.strip() for field in line.split("|")]
            if len(fields) >= 2 and fields[0] == job_id:
                exact_states.append(fields[1].split()[0].split("+", 1)[0])
        if exact_states != ["COMPLETED"]:
            raise RuntimeError(f"smoke Slurm job {job_id} is not exactly COMPLETED")

        run_root = roots["artifacts_root"] / run_id
        marker = run_root / "wandb_run_id.txt"
        try:
            wandb_run_id = marker.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(f"smoke W&B run identity is missing for {profile}") from error
        if not wandb_run_id or row.get("wandb_run_id", "").strip() != wandb_run_id:
            raise RuntimeError(f"smoke W&B identity does not match its ledger row for {profile}")
        wandb_dir = run_root / "wandb"
        if not wandb_dir.is_dir() or not any(wandb_dir.iterdir()):
            raise RuntimeError(f"smoke W&B artifacts are missing for {profile}")

        manifest_path = run_root / "config" / "train" / "manifest.json"
        try:
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"smoke training manifest is missing or unreadable for {profile}") from error
        config = manifest.get("config") if isinstance(manifest, dict) else None
        if (
            manifest.get("schema_version") != 1
            or manifest.get("stage") != "train"
            or not isinstance(config, dict)
            or not isinstance(config.get("method"), dict)
            or not isinstance(config.get("train"), dict)
            or not isinstance(config.get("assets"), dict)
        ):
            raise RuntimeError(f"smoke training manifest is invalid for {profile}")
        encoded_identity = json.dumps(
            {"stage": manifest["stage"], "config": config},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(encoded_identity).hexdigest() != manifest.get("inputs_sha256"):
            raise RuntimeError(f"smoke training manifest integrity check failed for {profile}")
        if config.get("backend") != "nm5":
            raise RuntimeError(f"smoke training manifest backend is not NM5 for {profile}")
        method = config["method"]
        temporal_loop = method.get("temporal_loop")
        lora = method.get("lora")
        train = config["train"]
        if (
            not isinstance(temporal_loop, dict)
            or temporal_loop.get("mode") != "layer"
            or temporal_loop.get("layer_start") != 22
            or temporal_loop.get("layer_end") != 29
            or temporal_loop.get("k_min") != k
            or temporal_loop.get("k_max") != k
            or not isinstance(lora, dict)
            or lora.get("enabled") is not True
            or lora.get("target_modules") != ["self_attn.q", "self_attn.v"]
            or float(method.get("lr", 0)) != expected_lr
            or float(method.get("lr_critic", 0)) != expected_lr_critic
        ):
            raise RuntimeError(f"smoke training manifest does not match the approved {profile} profile")
        max_steps = train.get("max_steps")
        seed = train.get("seed")
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed <= 0
        ):
            raise RuntimeError(f"smoke training manifest has an invalid step count or seed for {profile}")
        if row.get("seed", "") != str(seed) or row.get("step", "") != str(max_steps):
            raise RuntimeError(f"smoke ledger step or seed disagrees with the manifest for {profile}")
        if row.get("checkpoint_step", "") != str(max_steps):
            raise RuntimeError(f"smoke ledger checkpoint step disagrees with the manifest for {profile}")
        try:
            speed = float(row.get("training_speed", ""))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"smoke ledger has no valid training speed for {profile}") from error
        if not math.isfinite(speed) or speed <= 0:
            raise RuntimeError(f"smoke ledger has no positive finite training speed for {profile}")

        checkpoint = roots["ckpt_root"] / run_id / "latest.pt"
        try:
            checkpoint_evidence = verify_training_checkpoint(
                checkpoint, expected_step=max_steps, expected_seed=seed
            )
        except Exception as error:
            message = str(error).replace(str(checkpoint), "<checkpoint>")
            raise RuntimeError(
                f"smoke checkpoint evidence failed for {profile}: {type(error).__name__}: {message}"
            ) from error
        metadata = checkpoint_evidence["metadata"]
        if metadata.get("final") is not True:
            raise RuntimeError(f"smoke checkpoint is not final for {profile}")
        base_hash = config["assets"].get("generator_checkpoint_sha256")
        if metadata.get("base_checkpoint_sha256") != base_hash:
            raise RuntimeError(f"smoke checkpoint base hash disagrees with its manifest for {profile}")
        recorded_loop = metadata.get("temporal_loop")
        for key in (
            "mode",
            "layer_start",
            "layer_end",
            "k_min",
            "k_max",
            "training_enabled",
            "stop_grad_early",
        ):
            if not isinstance(recorded_loop, dict) or recorded_loop.get(key) != temporal_loop.get(key):
                raise RuntimeError(f"smoke checkpoint temporal-loop metadata is stale for {profile}")
        if not (roots["ckpt_root"] / run_id / "model.pt").is_file():
            raise RuntimeError(f"smoke lightweight final model is missing for {profile}")
        recorded_path = str(row.get("checkpoint_or_output_path", "")).strip()
        if Path(recorded_path).resolve() != checkpoint.resolve():
            raise RuntimeError(f"smoke ledger checkpoint path disagrees with the selected checkpoint for {profile}")

        log_stem = f"{run_id}_train_{job_id}"
        stdout_path = roots["logs_root"] / f"{log_stem}.out"
        stderr_path = roots["logs_root"] / f"{log_stem}.err"
        try:
            stdout_raw = stdout_path.read_bytes()
            stderr_raw = stderr_path.read_bytes()
            log_text = stdout_raw.decode("utf-8") + "\n" + stderr_raw.decode("utf-8")
        except OSError as error:
            raise RuntimeError(f"smoke Slurm logs are incomplete for {profile}") from error
        if re.search(r"MEMPROBE WARNING|(?:CUDA\s+)?out of memory|(?:^|[\s:])OOM(?:$|[\s:])", log_text, re.IGNORECASE):
            raise RuntimeError(f"smoke log contains memory-probe warnings or OOM evidence for {profile}")
        model_marker = f"SUE_GPU_MODEL_PREFLIGHT passed expected={expected_gpu_type} count={expected_gpu_count}"
        if model_marker not in log_text:
            raise RuntimeError(f"smoke log lacks the allocated {expected_gpu_count}-GPU model check for {profile}")
        deep_trace_ranks = {
            int(value)
            for value in re.findall(r"MEMPROBE rank=(\d+) deep trace at step 3\b", log_text)
        }
        memory_records = [
            (int(rank), float(reserved), float(frag), float(peak))
            for rank, reserved, frag, peak in re.findall(
                r"MEMPROBE step=3 rank=(\d+) [^\n]*?reserved=([0-9]+(?:\.[0-9]+)?)GiB "
                r"frag=([0-9]+(?:\.[0-9]+)?)GiB peak=([0-9]+(?:\.[0-9]+)?)GiB",
                log_text,
            )
        ]
        ranks = set(range(expected_gpu_count))
        if deep_trace_ranks != ranks or {item[0] for item in memory_records} != ranks:
            raise RuntimeError(
                f"smoke log lacks rank-labelled step-3 memory evidence for every GPU in {profile}"
            )
        rank_reserved = [max(item[1] for item in memory_records if item[0] == rank) for rank in range(expected_gpu_count)]
        rank_frag = [max(item[2] for item in memory_records if item[0] == rank) for rank in range(expected_gpu_count)]
        rank_peak = [max(item[3] for item in memory_records if item[0] == rank) for rank in range(expected_gpu_count)]
        if (
            any(not math.isfinite(value) or value <= 0 for sequence in (rank_reserved, rank_peak) for value in sequence)
            or any(not math.isfinite(value) or value < 0 for value in rank_frag)
        ):
            raise RuntimeError(f"smoke log has invalid step-3 memory measurements for {profile}")
        max_reserved = max(rank_reserved)
        if max_reserved > max_safe_reserved_gib:
            raise RuntimeError(
                f"smoke H100 reserved-memory peak {max_reserved:.2f}GiB exceeds the "
                f"{max_safe_reserved_gib:.2f}GiB fullrun safety threshold for {profile}"
            )
        evidence.append(
            {
                "profile": profile,
                "run_id": run_id,
                "job_id": job_id,
                "k": k,
                "max_steps": max_steps,
                "seed": seed,
                "wandb_run_id": wandb_run_id,
                "gpu_type": expected_gpu_type,
                "gpu_model_verified": True,
                "probe_peak_gib": rank_peak,
                "probe_reserved_gib": rank_reserved,
                "probe_fragmentation_gib": rank_frag,
                "reserved_memory_headroom_gib": float(resources["gpu_usable_memory_gib"]) - max_reserved,
                "reserved_memory_headroom_by_rank_gib": [
                    float(resources["gpu_usable_memory_gib"]) - value for value in rank_reserved
                ],
                "required_non_torch_reserve_gib": float(resources["gpu_non_torch_reserve_gib"]),
                "submitted_at": receipt_submitted_at.astimezone(timezone.utc).isoformat(),
                "receipt_submitted_at": receipt_submitted_at.astimezone(timezone.utc).isoformat(),
                "ledger_updated_at": submitted_at.astimezone(timezone.utc).isoformat(),
                "submission_receipt_path": receipt_path.relative_to(bundle).as_posix(),
                "submission_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                "checkpoint_path": checkpoint.relative_to(bundle).as_posix(),
                "checkpoint_sha256": _smoke_artifact_hash(checkpoint, "checkpoint", profile),
                "model_sha256": _smoke_artifact_hash(
                    roots["ckpt_root"] / run_id / "model.pt", "model", profile
                ),
                "manifest_path": manifest_path.relative_to(bundle).as_posix(),
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "ledger_row_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "wandb_marker_path": marker.relative_to(bundle).as_posix(),
                "wandb_marker_sha256": _smoke_artifact_hash(marker, "W&B marker", profile),
                "slurm_stdout_path": stdout_path.relative_to(bundle).as_posix(),
                "slurm_stdout_sha256": hashlib.sha256(stdout_raw).hexdigest(),
                "slurm_stderr_path": stderr_path.relative_to(bundle).as_posix(),
                "slurm_stderr_sha256": hashlib.sha256(stderr_raw).hexdigest(),
            }
        )
    if latest_submission_at is None:
        raise RuntimeError("smoke pair has no receipt submission timestamps")
    from looped_self_forcing_pipeline.preflight import verify_max_parallel_artifact

    max_parallel = verify_max_parallel_artifact(
        bundle,
        pair_id,
        partition=str(runtime["sandbox_resources"]["nm5"]["partition"]),
        now=latest_submission_at,
    )
    for item in evidence:
        item["max_parallel"] = max_parallel
    return evidence


def _smoke_readiness_document(
    pair_id: str,
    evidence: list[dict[str, Any]],
    *,
    created_at: str,
) -> dict[str, Any]:
    if len(evidence) != 2:
        raise ValueError("a paired smoke readiness stamp requires exactly two verified profiles")
    return {
        "schema_version": 1,
        "smoke_pair_id": pair_id,
        "created_at": created_at,
        "max_parallel": evidence[0]["max_parallel"],
        "submission_receipt_path": evidence[0]["submission_receipt_path"],
        "submission_receipt_sha256": evidence[0]["submission_receipt_sha256"],
        "profiles": [
            {
                key: item[key]
                for key in (
                    "profile",
                    "run_id",
                    "job_id",
                    "k",
                    "max_steps",
                    "seed",
                    "wandb_run_id",
                    "gpu_type",
                    "gpu_model_verified",
                    "probe_peak_gib",
                    "probe_reserved_gib",
                    "probe_fragmentation_gib",
                    "reserved_memory_headroom_gib",
                    "reserved_memory_headroom_by_rank_gib",
                    "required_non_torch_reserve_gib",
                    "submitted_at",
                    "receipt_submitted_at",
                    "ledger_updated_at",
                    "submission_receipt_path",
                    "submission_receipt_sha256",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "model_sha256",
                    "manifest_path",
                    "manifest_sha256",
                    "ledger_row_sha256",
                    "wandb_marker_path",
                    "wandb_marker_sha256",
                    "slurm_stdout_path",
                    "slurm_stdout_sha256",
                    "slurm_stderr_path",
                    "slurm_stderr_sha256",
                )
            }
            for item in evidence
        ],
    }


def write_smoke_readiness_stamp(
    exp_dir: str | Path,
    pair_id: str,
    evidence: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> Path:
    """Write one immutable readiness stamp for a fully verified smoke pair."""
    from looped_self_forcing_pipeline.config import resolve_runtime_paths

    roots = resolve_runtime_paths(exp_dir)
    target = roots["readiness_root"] / f"layerwise_23_30_{pair_id}.json"
    created_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    record = _smoke_readiness_document(pair_id, evidence, created_at=created_at)
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing smoke readiness stamp is unreadable") from error
        existing_without_time = dict(existing)
        existing_without_time.pop("created_at", None)
        candidate_without_time = dict(record)
        candidate_without_time.pop("created_at", None)
        if existing_without_time != candidate_without_time:
            raise FileExistsError("smoke readiness stamp already exists with different evidence")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return write_smoke_readiness_stamp(exp_dir, pair_id, evidence, now=now)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_smoke_readiness_stamp(
    exp_dir: str | Path,
    pair_id: str,
    *,
    run_command: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Consume a readiness stamp and rehash every referenced smoke artifact."""
    from looped_self_forcing_pipeline.config import resolve_runtime_paths

    roots = resolve_runtime_paths(exp_dir)
    stamp_path = roots["readiness_root"] / f"layerwise_23_30_{pair_id}.json"
    try:
        existing = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("smoke readiness stamp is missing or unreadable") from error
    if (
        not isinstance(existing, dict)
        or existing.get("schema_version") != 1
        or existing.get("smoke_pair_id") != pair_id
    ):
        raise RuntimeError("smoke readiness stamp does not match the selected pair")
    try:
        created_at = datetime.fromisoformat(str(existing.get("created_at", "")).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("smoke readiness stamp created_at is missing or invalid") from error
    if created_at.tzinfo is None:
        raise RuntimeError("smoke readiness stamp created_at must include a timezone")
    current = now or datetime.now(timezone.utc)
    if (created_at.astimezone(timezone.utc) - current.astimezone(timezone.utc)).total_seconds() > 30:
        raise RuntimeError("smoke readiness stamp timestamp is in the future")
    evidence = verify_smoke_pair_evidence(exp_dir, pair_id, run_command=run_command)
    expected = _smoke_readiness_document(
        pair_id,
        evidence,
        created_at=created_at.astimezone(timezone.utc).isoformat(),
    )
    if existing != expected:
        raise RuntimeError("smoke readiness stamp hashes or evidence no longer match the verified run")
    return stamp_path, evidence


def verify_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_step: int,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Check the final training checkpoint's step and floating-point tensors."""
    import torch

    path = Path(checkpoint_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"training checkpoint is missing or empty: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("training checkpoint must contain a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("training checkpoint is missing metadata")
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step != expected_step:
        raise ValueError(
            f"training checkpoint step {step!r} does not match expected step {expected_step}"
        )
    if expected_seed is not None:
        seed = metadata.get("seed")
        if (
            isinstance(expected_seed, bool)
            or not isinstance(expected_seed, int)
            or expected_seed <= 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed != expected_seed
        ):
            raise ValueError(
                f"training checkpoint seed {seed!r} does not match expected ledger seed {expected_seed}"
            )
    generator = payload.get("generator")
    if not isinstance(generator, dict) or not generator:
        raise ValueError("training checkpoint is missing generator weights")
    tensors = list(_iter_tensors(generator))
    if not tensors:
        raise ValueError("training checkpoint contains no generator tensors")
    if payload.get("generator_ema") is not None:
        ema_tensors = list(_iter_tensors(payload["generator_ema"]))
        for tensor in ema_tensors:
            _require_finite(tensor)
    else:
        ema_tensors = []
    for tensor in tensors:
        _require_finite(tensor)
    return {
        "step": step,
        "trainable_parameter_count": sum(int(tensor.numel()) for tensor in tensors),
        "checkpoint_path": str(path.resolve()),
        "metadata": metadata,
        "payload": payload,
    }


def _iter_tensors(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tensors(nested)


def _require_finite(tensor: Any) -> None:
    import torch

    if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
        raise ValueError("training checkpoint contains non-finite values")


def prompt_count(prompt_path: str | Path) -> int:
    """Count prompts with the same line semantics as the copied TextDataset."""
    path = Path(prompt_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"prompt file is missing or empty: {path}")
    return len(path.read_text(encoding="utf-8").splitlines())


def require_ffprobe(which: Callable[[str], str | None] = shutil.which) -> str:
    """Resolve ffprobe before inference allocates a GPU or writes outputs."""
    executable = which("ffprobe")
    if not executable:
        raise FileNotFoundError(
            "ffprobe is required to verify generated MP4 files; add it to PATH before inference"
        )
    return executable


def require_empty_results_dir(results_dir: str | Path) -> Path:
    """Refuse to mix any prior artifact into a new inference run."""
    path = Path(results_dir)
    if path.is_symlink():
        raise FileExistsError(f"inference output directory must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"inference output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"inference output directory is not empty: {path}; use a new run_id"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_fresh_training_outputs(
    checkpoint_dir: str | Path,
    wandb_dir: str | Path,
    final_model: str | Path,
    *,
    wandb_run_id_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Refuse stale training evidence before reusing an immutable run id."""
    directories = (Path(checkpoint_dir), Path(wandb_dir))
    for directory in directories:
        if directory.is_symlink():
            raise FileExistsError(f"training artifact directory must not be a symlink: {directory}")
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(f"training artifact path is not a directory: {directory}")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"training artifacts already exist under {directory}; use a new run_id"
            )

    files = [Path(final_model), Path(final_model).with_name(Path(final_model).name + ".tmp")]
    if wandb_run_id_path is not None:
        files.append(Path(wandb_run_id_path))
    for file_path in files:
        if file_path.exists() or file_path.is_symlink():
            raise FileExistsError(
                f"training artifacts already exist at {file_path}; use a new run_id"
            )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def require_fresh_inference_outputs(
    results_dir: str | Path,
    wandb_dir: str | Path,
    wandb_run_id_path: str | Path,
) -> tuple[Path, Path]:
    """Require new video and W&B output locations for one inference run."""
    results = require_empty_results_dir(results_dir)
    tracking = Path(wandb_dir)
    if tracking.is_symlink():
        raise FileExistsError(f"inference W&B directory must not be a symlink: {tracking}")
    if tracking.exists() and not tracking.is_dir():
        raise FileExistsError(f"inference W&B path is not a directory: {tracking}")
    if tracking.exists() and any(tracking.iterdir()):
        raise FileExistsError(
            f"inference W&B artifacts already exist under {tracking}; use a new run_id"
        )
    marker = Path(wandb_run_id_path)
    if marker.exists() or marker.is_symlink():
        raise FileExistsError(
            f"inference W&B run identity already exists at {marker}; use a new run_id"
        )
    tracking.mkdir(parents=True, exist_ok=True)
    return results, tracking


def verify_inference_outputs(
    results_dir: str | Path,
    *,
    expected_count: int,
    expected_frames: int,
    expected_fps: int,
    run_command: Callable[..., Any] | None = None,
) -> list[Path]:
    """Require exact video count, decoded frame count, and frame rate."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if expected_frames <= 0 or expected_fps <= 0:
        raise ValueError("expected video frames and fps must be positive")
    path = Path(results_dir)
    videos = sorted(path.glob("*.mp4"))
    if len(videos) != expected_count:
        raise RuntimeError(
            f"expected exactly {expected_count} inference videos, found {len(videos)} in {path}"
        )
    probe = run_command or subprocess.run
    if run_command is None:
        require_ffprobe()
    for video in videos:
        if video.stat().st_size == 0:
            raise RuntimeError(f"empty inference output: {video}")
        result = probe(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "format=duration:stream=codec_type,nb_read_frames,avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError(f"ffprobe could not decode inference output: {video}")
        try:
            info = json.loads(result.stdout)
            video_streams = [
                stream
                for stream in info.get("streams", [])
                if stream.get("codec_type") == "video"
            ]
            duration = float(info.get("format", {}).get("duration", 0))
            if not video_streams:
                raise RuntimeError(f"inference output has no decodable video stream: {video}")
            stream = video_streams[0]
            decoded_frames = int(stream.get("nb_read_frames", "0"))
            rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
            frame_rate = Fraction(rate_text)
        except (TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as error:
            raise RuntimeError(f"ffprobe returned invalid metadata for {video}") from error
        if duration <= 0:
            raise RuntimeError(f"inference output has no decodable video stream: {video}")
        if decoded_frames != expected_frames:
            raise RuntimeError(
                f"inference output frame count {decoded_frames} does not match expected "
                f"{expected_frames}: {video}"
            )
        if frame_rate != expected_fps:
            raise RuntimeError(
                f"inference output fps {float(frame_rate):g} does not match expected "
                f"{expected_fps}: {video}"
            )
    return videos
