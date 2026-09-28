"""Static preflight adapter for the paired NM5 layerwise comparison."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib
import json
import os
import re
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from looped_self_forcing_pipeline.config import (  # noqa: E402
    compose_config,
    load_runtime_config,
    resolve_runtime_paths,
)
from looped_self_forcing_pipeline.worker import (  # noqa: E402
    check_stage,
    method_identity,
    require_training_seed,
)

PAIR_METHODS = (
    ("layerwise_l23_30_k3_lr5gen", "k3-lr5gen", 3, 4e-7, 4e-7),
    ("layerwise_l23_30_k2_lr5both", "k2-lr5both", 2, 4e-7, 8e-8),
)
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_CACHE_TARGETS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "MODELSCOPE_CACHE",
    "TORCH_HOME",
    "MPLCONFIGDIR",
    "WANDB_CACHE_DIR",
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _check_private_cache_roots(runtime: Mapping[str, Any], environment: Mapping[str, str]) -> None:
    cache_mapping = runtime["environment"]["cache_env_sources"]
    home = Path(environment["HOME"]).resolve() if environment.get("HOME") else None
    for target in _CACHE_TARGETS:
        source = cache_mapping[target]
        value = environment.get(source, "").strip()
        if not value:
            raise RuntimeError(f"required NM5 cache environment variable is missing: {source}")
        try:
            path = Path(value).resolve(strict=True)
        except OSError as error:
            raise RuntimeError(f"required NM5 cache directory is unavailable: {source}") from error
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise RuntimeError(f"required NM5 cache directory is unavailable: {source}")
        if home is not None and _inside(path, home):
            raise RuntimeError(f"NM5 cache directory {source} resolves under HOME")


def _check_storage(output_roots: Mapping[str, Path]) -> dict[str, int]:
    free_bytes: dict[str, int] = {}
    for name in ("artifacts_root", "ckpt_root", "logs_root"):
        root = output_roots[name]
        root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".sue-preflight-", dir=root)
            os.close(descriptor)
            Path(temporary).unlink()
        except OSError as error:
            raise RuntimeError(f"SUE {name} is not writable") from error
        try:
            stats = os.statvfs(root)
        except OSError as error:
            raise RuntimeError(f"SUE {name} capacity could not be read") from error
        available = stats.f_bavail * stats.f_frsize
        if available <= 0:
            raise RuntimeError(f"SUE {name} has no free bytes")
        free_bytes[name] = available
    return free_bytes


def verify_max_parallel_artifact(
    exp_dir: str | Path,
    pair_id: str,
    *,
    partition: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Require the fresh standard SUE capacity record for this stage run ID."""
    path = Path(exp_dir) / pair_id / "max_parallel.json"
    try:
        raw = path.read_bytes()
        artifact = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("fresh SUE max_parallel.json is missing or unreadable for this pair") from error
    if not isinstance(artifact, dict):
        raise RuntimeError("SUE max_parallel.json must contain a JSON object")
    timestamp_value = artifact.get("timestamp")
    try:
        queried_at = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("SUE max_parallel.json timestamp is missing or invalid") from error
    if queried_at.tzinfo is None:
        raise RuntimeError("SUE max_parallel.json timestamp must include a timezone")
    current = now or datetime.now(timezone.utc)
    age_seconds = (current.astimezone(timezone.utc) - queried_at.astimezone(timezone.utc)).total_seconds()
    if age_seconds < -30 or age_seconds > 600:
        raise RuntimeError("SUE max_parallel.json must be no more than 10 minutes old")
    if artifact.get("sandbox") != "nm5" or artifact.get("partition") != partition:
        raise RuntimeError("SUE max_parallel.json sandbox or partition disagrees with runtime.yaml")
    max_parallel = artifact.get("max_parallel")
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int) or max_parallel < 1:
        raise RuntimeError("SUE max_parallel.json max_parallel must be a positive integer")
    if not isinstance(artifact.get("limiting_factor"), str) or not artifact["limiting_factor"].strip():
        raise RuntimeError("SUE max_parallel.json limiting_factor is missing")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, (dict, list, str)) or not evidence:
        raise RuntimeError("SUE max_parallel.json redacted evidence summary is missing")
    if isinstance(evidence, dict) and any(
        key.lower() in {"account", "host", "hostname", "raw_queue", "raw_output", "private_path"}
        for key in evidence
    ):
        raise RuntimeError("SUE max_parallel.json evidence contains unredacted private fields")
    return {
        "artifact_path": path.relative_to(Path(exp_dir)).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "timestamp": queried_at.astimezone(timezone.utc).isoformat(),
        "age_seconds": max(0, int(age_seconds)),
        "max_parallel": max_parallel,
        "limiting_factor": artifact["limiting_factor"],
    }


def _submit_limit(values: str, *, label: str) -> int | None:
    limits: list[int] = []
    for line in values.splitlines():
        for field in line.split("|"):
            normalized = field.strip().upper()
            if not normalized:
                continue
            if normalized in {"-1", "UNLIMITED", "INFINITE", "NONE"}:
                continue
            if not normalized.isdigit():
                raise RuntimeError(f"sacctmgr returned an invalid {label} limit")
            limits.append(int(normalized))
    if not limits:
        if values.strip():
            return None
        raise RuntimeError(f"sacctmgr returned no {label} limit evidence")
    if any(limit == 0 for limit in limits):
        raise RuntimeError(f"sacctmgr reports zero {label} submit capacity")
    return min(limits)


def query_pending_submit_headroom(
    runtime: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    run_command: Callable[..., Any] | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Query Slurm limits and active work without retaining raw queue/account output."""
    runner = run_command or subprocess.run
    resources = runtime["sandbox_resources"]["nm5"]
    account = environment.get("NM5_ACCOUNT", "").strip()
    qos = str(resources.get("qos", "")).strip()
    partition = str(resources.get("partition", "")).strip()
    user = username or getpass.getuser()
    if not account:
        raise RuntimeError("NM5_ACCOUNT is required for pending-submit capacity checks")
    if not qos or not partition or not user:
        raise RuntimeError("NM5 runtime QoS, partition, or scheduler user is unavailable")

    commands = (
        (
            "association submit limit",
            [
                "sacctmgr",
                "--noheader",
                "--parsable2",
                "show",
                "assoc",
                "where",
                f"user={user}",
                f"account={account}",
                "format=MaxSubmitJobs",
            ],
        ),
        (
            "QoS submit limit",
            [
                "sacctmgr",
                "--noheader",
                "--parsable2",
                "show",
                "qos",
                "where",
                f"name={qos}",
                "format=MaxSubmitJobsPerUser",
            ],
        ),
        (
            "active user jobs",
            [
                "squeue",
                "--noheader",
                "--user",
                user,
                "--states=PENDING,RUNNING",
                "--format=%T|%a|%q|%P",
            ],
        ),
    )
    results: list[Any] = []
    for label, command in commands:
        try:
            result = runner(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise RuntimeError(f"Slurm {label} query could not run") from error
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError(f"Slurm {label} query failed")
        results.append(result)
    association_limit = _submit_limit(str(results[0].stdout), label="association")
    qos_limit = _submit_limit(str(results[1].stdout), label="QoS")
    finite_limits = [limit for limit in (association_limit, qos_limit) if limit is not None]
    queue_rows = 0
    for line in str(results[2].stdout).splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and (fields[1] == account or fields[2] == qos):
            queue_rows += 1
    limit = min(finite_limits) if finite_limits else None
    headroom = None if limit is None else max(0, limit - queue_rows)
    if headroom is not None and headroom < 2:
        raise RuntimeError("Slurm pending-submit headroom is below the two-job dependency pair")
    return {
        "sandbox": "nm5",
        "partition": partition,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "active_jobs": queue_rows,
        "association_submit_limit": association_limit,
        "qos_submit_limit": qos_limit,
        "pending_submit_headroom": headroom,
        "pending_submit_slots_verified": 2,
        "limiting_factor": "association/QoS submit headroom",
        "evidence": {
            "commands": [
                "sacctmgr show assoc MaxSubmitJobs",
                "sacctmgr show qos MaxSubmitJobsPerUser",
                "squeue pending/running counts by runtime account or QoS",
            ],
            "raw_output_retained": False,
        },
    }


def _write_json_create_once(path: Path, record: Mapping[str, Any], label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"{label} already exists; choose a fresh pair ID") from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _validate_method(config: Any, expected: tuple[str, str, int, float, float]) -> None:
    profile, _suffix, k, lr, lr_critic = expected
    if config.backend.name != "nm5" or config.stage.name != "train":
        raise ValueError("paired preflight requires backend=nm5 and stage=train")
    method = method_identity(config.method)
    if (
        method["profile_id"] != profile
        or method["mode"] != "layer"
        or method["layer_start"] != 22
        or method["layer_end"] != 29
        or method["k_min"] != k
        or method["k_max"] != k
        or float(config.method.lr) != lr
        or float(config.method.lr_critic) != lr_critic
    ):
        raise ValueError(f"selected Hydra method does not match paired profile {profile}")
    require_training_seed(int(config.train.seed))
    if int(config.backend.train_gpus) != 4:
        raise ValueError("paired NM5 training requires exactly four GPUs")


def _normalized_pair_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    for key in ("scale", "run_id", "resolved_config_sha256"):
        normalized.pop(key, None)
    train = normalized.get("train")
    if not isinstance(train, dict):
        raise ValueError("paired training manifest must contain a train mapping")
    for key in ("max_steps", "log_iters", "timeout_seconds"):
        train.pop(key, None)
    return normalized


def _validated_smoke_manifest(path: Path, profile: str) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"smoke manifest is missing or unreadable for {profile}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("stage") != "train"
        or not isinstance(manifest.get("config"), dict)
    ):
        raise RuntimeError(f"smoke manifest is invalid for {profile}")
    canonical = json.dumps(
        {"stage": manifest["stage"], "config": manifest["config"]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("inputs_sha256"):
        raise RuntimeError(f"smoke manifest integrity check failed for {profile}")
    return manifest["config"]


def require_full_config_matches_smoke(
    smoke_manifest_config: Mapping[str, Any],
    full_manifest_config: Mapping[str, Any],
    *,
    profile: str,
) -> None:
    """Allow only scale and run-size/runtime fields to differ after smoke."""
    if _normalized_pair_config(smoke_manifest_config) != _normalized_pair_config(full_manifest_config):
        raise RuntimeError(
            f"full {profile} config differs from smoke in method, loss, LoRA, assets, prompts, tracking, or seed"
        )


def run_pair_preflight(
    pair_id: str,
    overrides: Sequence[str] = (),
    *,
    environment: Mapping[str, str] | None = None,
    check_stage_fn: Callable[..., Any] | None = None,
    run_command: Callable[..., Any] | None = None,
    smoke_pair_id: str | None = None,
) -> dict[str, Any]:
    """Check runtime roots, storage, dependencies, tracking, and both method configs."""
    env = os.environ if environment is None else environment
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,52}", pair_id):
        raise ValueError("pair ID must be 1–53 lowercase letters, digits, _ or -")
    if smoke_pair_id is not None and (
        not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,52}", smoke_pair_id)
        or smoke_pair_id == pair_id
    ):
        raise ValueError("full preflight requires a distinct valid smoke pair ID")
    deepresearch_root = Path(env.get("NM5_DEEPRESEARCH_ROOT", "")).expanduser().resolve()
    workspace = Path(env.get("SUE_WORKSPACE_ROOT", "")).expanduser().resolve()
    if not deepresearch_root.is_dir() or not workspace.is_dir():
        raise RuntimeError("NM5 DeepResearch root and SUE_WORKSPACE_ROOT must exist")
    if workspace != deepresearch_root / "workspace" / "looped-flow-matching":
        raise RuntimeError("SUE_WORKSPACE_ROOT does not match the selected NM5 workspace contract")
    exp_dir = Path(env.get("SUE_EXP_DIR", "")).expanduser().resolve()
    if not exp_dir.is_dir() or not (exp_dir / "config/config.yaml").is_file():
        raise RuntimeError("selected SUE_EXP_DIR is unavailable or lacks its Hydra config")
    selected_python = Path(env.get("SUE_PYTHON", "")).expanduser().resolve()
    if not selected_python.is_file() or selected_python != Path(sys.executable).resolve():
        raise RuntimeError("preflight dispatcher did not use the selected SUE_PYTHON interpreter")
    for dependency in ("hydra", "omegaconf", "torch", "wandb"):
        importlib.import_module(dependency)

    runtime = load_runtime_config(exp_dir)
    if runtime.get("backend", {}).get("primary") != "nm5":
        raise RuntimeError("paired comparison runtime backend must be NM5")
    output_roots = resolve_runtime_paths(exp_dir)
    resources = runtime["sandbox_resources"]["nm5"]
    if (
        int(resources.get("train_gpus", 0)) != 4
        or int(resources.get("gpus_per_node", 0)) != 4
        or int(resources.get("max_nodes_per_job", 0)) != 1
        or str(resources.get("gpu_type", "")).casefold() != "h100"
    ):
        raise RuntimeError("NM5 runtime capacity contract must be one four-GPU H100 node per training job")
    max_parallel = verify_max_parallel_artifact(
        exp_dir,
        pair_id,
        partition=str(resources["partition"]),
    )
    pending_submit = query_pending_submit_headroom(
        runtime,
        env,
        run_command=run_command,
    )
    pending_submit_path = output_roots["artifacts_root"] / "pairs" / pair_id / "pending_submit_preflight.json"
    pending_submit_sha256 = _write_json_create_once(
        pending_submit_path,
        {"schema_version": 1, "pair_id": pair_id, **pending_submit},
        "pending-submit preflight report",
    )
    asset_root = Path(env.get("SUE_ASSET_ROOT", "")).expanduser().resolve(strict=True)
    if not (asset_root / "checkpoints").is_dir() or not (asset_root / "wan_models").is_dir():
        raise RuntimeError("configured NM5 asset root lacks expected checkpoint/model directories")
    overlay_value = env.get("SUE_PYTHONPATH", "").strip()
    configured_overlay = runtime["environment"].get("python_overlay")
    if configured_overlay:
        overlay = (exp_dir / configured_overlay).resolve(strict=True)
        if not overlay.is_dir() or not _inside(overlay, exp_dir):
            raise RuntimeError("configured Hydra overlay is unavailable inside SUE_EXP_DIR")
        if Path(overlay_value).resolve() != overlay:
            raise RuntimeError("SUE_PYTHONPATH does not match runtime.environment.python_overlay")
    _check_private_cache_roots(runtime, env)
    free_bytes = _check_storage(output_roots)

    stage_check = check_stage_fn or check_stage
    prepared_profiles = []
    for profile in PAIR_METHODS:
        method_name, suffix, _k, _lr, _lr_critic = profile
        run_id = f"{pair_id}-{suffix}"
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("pair ID produces a run ID longer than 64 characters")
        config = compose_config(
            [
                "backend=nm5",
                "stage=train",
                f"method={method_name}",
                f"run_id={run_id}",
                *overrides,
            ],
            exp_dir=exp_dir,
        )
        _validate_method(config, profile)
        prepared = stage_check(config, "train", exp_dir=exp_dir)
        current_manifest_config = prepared.get("manifest") if isinstance(prepared, dict) else None
        if smoke_pair_id is not None:
            if not isinstance(current_manifest_config, dict):
                raise RuntimeError(f"full preflight did not materialize a training manifest for {method_name}")
            smoke_run_id = f"{smoke_pair_id}-{suffix}"
            smoke_manifest_path = (
                output_roots["artifacts_root"] / smoke_run_id / "config" / "train" / "manifest.json"
            )
            smoke_config = _validated_smoke_manifest(smoke_manifest_path, method_name)
            require_full_config_matches_smoke(
                smoke_config,
                current_manifest_config,
                profile=method_name,
            )
        prepared_profiles.append(
            {
                "profile": method_name,
                "run_id": run_id,
                "k": profile[2],
                "lr": profile[3],
                "lr_critic": profile[4],
                "max_steps": int(config.train.max_steps),
                "seed": int(config.train.seed),
                "manifest_inputs": current_manifest_config.get("resolved_config_sha256", "")
                if isinstance(current_manifest_config, dict)
                else "",
            }
        )
    return {
        "pair_id": pair_id,
        "sandbox": "nm5",
        "partition": str(resources["partition"]),
        "gpus_per_node": int(resources["gpus_per_node"]),
        "gpu_type_required": str(resources["gpu_type"]),
        "gpu_usable_memory_gib": float(resources["gpu_usable_memory_gib"]),
        "gpu_non_torch_reserve_gib": float(resources["gpu_non_torch_reserve_gib"]),
        "max_nodes_per_job": int(resources["max_nodes_per_job"]),
        "effective_gpu_parallel": 1,
        "max_parallel": max_parallel,
        "pending_submit": {
            "artifact_path": pending_submit_path.relative_to(exp_dir).as_posix(),
            "sha256": pending_submit_sha256,
            "headroom": pending_submit["pending_submit_headroom"],
            "slots_verified": pending_submit["pending_submit_slots_verified"],
        },
        "cache_roots_verified": True,
        "free_bytes": free_bytes,
        "profiles": prepared_profiles,
    }


def write_pair_preflight_report(
    exp_dir: str | Path,
    result: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> Path:
    """Persist a redacted summary of the static gates, atomically and once."""
    roots = resolve_runtime_paths(exp_dir)
    pair_id = str(result["pair_id"])
    report = {
        "schema_version": 1,
        "pair_id": pair_id,
        "created_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "sandbox": result["sandbox"],
        "partition": result["partition"],
        "resources": {
            "gpu_type_required": result["gpu_type_required"],
            "gpu_usable_memory_gib": result["gpu_usable_memory_gib"],
            "gpu_non_torch_reserve_gib": result["gpu_non_torch_reserve_gib"],
            "gpus_per_node": result["gpus_per_node"],
            "max_nodes_per_job": result["max_nodes_per_job"],
            "effective_gpu_parallel": result["effective_gpu_parallel"],
        },
        "max_parallel_artifact": result["max_parallel"],
        "pending_submit_artifact": result["pending_submit"],
        "storage_free_bytes": result["free_bytes"],
        "cache_roots_verified": bool(result["cache_roots_verified"]),
        "dependencies_verified": ["hydra", "omegaconf", "torch", "wandb"],
        "gpu_model_check": "required inside allocated NM5 worker before pipeline entry",
        "profiles": result["profiles"],
    }
    target = roots["artifacts_root"] / "pairs" / pair_id / "preflight.json"
    _write_json_create_once(target, report, "pair preflight report")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--smoke-pair-id")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    try:
        result = run_pair_preflight(
            args.pair_id,
            args.overrides,
            smoke_pair_id=args.smoke_pair_id,
        )
        report_path = write_pair_preflight_report(os.environ["SUE_EXP_DIR"], result)
    except Exception as error:  # keep private roots and asset paths out of Slurm logs
        message = str(error)
        for key in (
            "NM5_DEEPRESEARCH_ROOT",
            "NM5_WORKSPACE_ROOT",
            "SUE_WORKSPACE_ROOT",
            "SUE_EXP_DIR",
            "SUE_ASSET_ROOT",
            "SUE_PYTHON",
            "SUE_PYTHONPATH",
        ):
            value = os.environ.get(key, "")
            if value:
                message = message.replace(value, "<private-root>")
        for key in (
            "NM5_HF_HOME",
            "NM5_HF_HUB_CACHE",
            "NM5_MODELSCOPE_CACHE",
            "NM5_TORCH_HOME",
            "NM5_MPLCONFIGDIR",
            "NM5_WANDB_CACHE_DIR",
            "HF_HOME",
            "HF_HUB_CACHE",
            "MODELSCOPE_CACHE",
            "TORCH_HOME",
            "MPLCONFIGDIR",
            "WANDB_CACHE_DIR",
            "WANDB_API_KEY",
            "WANDB_ENTITY",
        ):
            value = os.environ.get(key, "")
            if value:
                message = message.replace(value, "<private-value>")
        print(f"paired NM5 preflight failed: {type(error).__name__}: {message}", file=sys.stderr)
        return 2
    print(
        "paired NM5 static preflight passed: dependencies, caches, storage, assets, "
        "configs, four-GPU H100 allocation contract, and Slurm submit headroom; "
        f"report={report_path.relative_to(Path(os.environ['SUE_EXP_DIR']).resolve()).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
