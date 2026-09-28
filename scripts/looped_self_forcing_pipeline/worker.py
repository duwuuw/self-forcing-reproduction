"""Prepare, launch, and verify one training or inference worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from looped_self_forcing_pipeline.checkpoint import (
    resolve_base_checkpoint_with_hash,
    sha256_file,
)
from looped_self_forcing_pipeline.config import (
    PACKAGE_ROOT,
    load_runtime_config,
    resolve_exp_dir,
    resolve_runtime_paths,
)
from looped_self_forcing_pipeline.ledger import write_result
from looped_self_forcing_pipeline.manifest import materialize_manifest
from looped_self_forcing_pipeline.verification import (
    prompt_count,
    require_ffprobe,
    require_fresh_inference_outputs,
    require_fresh_training_outputs,
    verify_inference_outputs,
    verify_training_checkpoint,
)


SOURCE_ROOT = PACKAGE_ROOT / "source"
SCRIPT_ROOT = PACKAGE_ROOT.parent
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


def _json_mapping(value: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    converted = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if not isinstance(converted, dict):
        raise TypeError("composed configuration section must be a mapping")
    return converted


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay inside {root}") from error
    return resolved


def bundle_path(exp_dir: str | Path, value: str, label: str) -> Path:
    root = Path(exp_dir).resolve()
    return _inside(root / value, root, label)


def dataset_path(exp_dir: str | Path, value: str, label: str) -> Path:
    """Resolve a prompt file beneath the SUE runtime's datasets root."""
    relative = Path(value).expanduser()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a relative path beneath paths.datasets_root")
    root = Path(exp_dir).resolve()
    datasets_root = resolve_runtime_paths(root)["datasets_root"]
    return _inside(datasets_root / relative, root, label)


def resolve_asset_root(value: str | Path | None = None) -> Path:
    """Resolve the explicit backend-local root containing model assets."""
    configured = value if value is not None else os.environ.get("SUE_ASSET_ROOT")
    if configured is None or not str(configured).strip() or str(configured).lower() == "none":
        raise ValueError("SUE_ASSET_ROOT must name the configured asset directory")
    try:
        root = Path(str(configured)).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise FileNotFoundError("configured SUE_ASSET_ROOT is unavailable or inaccessible") from error
    if not root.is_dir():
        raise NotADirectoryError("configured SUE_ASSET_ROOT must name a directory")
    return root


def asset_path(value: str, label: str, asset_root: str | Path | None = None) -> Path:
    root = resolve_asset_root(asset_root)
    relative = Path(value).expanduser()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a relative path beneath SUE_ASSET_ROOT")
    return (root / relative).resolve()


def method_identity(method_config: Any) -> dict[str, Any]:
    method = _json_mapping(method_config)
    loop = method.get("temporal_loop")
    lora = method.get("lora")
    if not isinstance(loop, dict) or not isinstance(lora, dict):
        raise ValueError("method config must contain temporal_loop and lora mappings")
    if not loop.get("enabled") or not loop.get("training_enabled"):
        raise ValueError("temporal_loop.enabled and temporal_loop.training_enabled must be true")
    if loop.get("mode") not in {"block", "layer"}:
        raise ValueError("temporal_loop.mode must be block or layer")
    start, end = int(loop["layer_start"]), int(loop["layer_end"])
    k_min, k_max = int(loop["k_min"]), int(loop["k_max"])
    if not (0 <= start <= end < 30 and 1 <= k_min <= k_max):
        raise ValueError("invalid temporal loop layer range or K bounds")
    if not lora.get("enabled") or tuple(lora.get("target_modules", ())) != (
        "self_attn.q",
        "self_attn.v",
    ):
        raise ValueError("method requires LoRA on self_attn.q and self_attn.v")
    return {
        "profile_id": method.get("profile_id", ""),
        "mode": str(loop["mode"]),
        "layer_start": start,
        "layer_end": end,
        "k_min": k_min,
        "k_max": k_max,
        "strength": float(loop["strength"]),
        "stop_grad_early": bool(loop["stop_grad_early"]),
        "schedule": str(loop["schedule"]),
        "training_enabled": bool(loop["training_enabled"]),
        "lora": lora,
    }


def build_experiment_name(method: dict[str, Any], run_id: str, stage: str) -> str:
    """Return the shared W&B/ledger/Slurm experiment-name body."""
    if stage not in {"train", "infer"}:
        raise ValueError(f"unsupported experiment stage: {stage}")
    if not _RUN_ID.fullmatch(str(run_id)):
        raise ValueError("run_id must be 1–64 lowercase letters, digits, _ or -")
    method_name = str(method.get("profile_id") or method.get("mode") or "").strip()
    if not method_name:
        raise ValueError("method identity must provide profile_id or mode")
    return f"{method_name}-{run_id}-{stage}"


def require_training_seed(value: Any) -> int:
    """Reject the upstream random-seed sentinel and unsupported seed values."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("training seed must be a non-zero positive integer")
    return value


def require_inference_latent_frames(value: Any) -> int:
    """Keep the copied T2V pipeline at its validated 123 latent-frame input."""
    if isinstance(value, bool) or not isinstance(value, int) or value != 123:
        raise ValueError("infer.num_output_frames must remain 123 latent frames for this inference profile")
    return value


def _configured_gpu_count(config: Any, stage: str, exp_dir: str | Path) -> int:
    runtime = load_runtime_config(exp_dir)
    policy = runtime.get("policy")
    resources = policy.get("sandbox_resources") if isinstance(policy, dict) else None
    backend_name = str(config.backend.name)
    backend_resources = resources.get(backend_name) if isinstance(resources, dict) else None
    if not isinstance(backend_resources, dict):
        raise ValueError(f"runtime policy lacks sandbox_resources.{backend_name}")
    stage_key = "train_gpus" if stage == "train" else "infer_gpus"
    configured = backend_resources.get(stage_key)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
        raise ValueError(f"runtime policy sandbox_resources.{backend_name}.{stage_key} must be positive")
    group_value = int(config.backend.train_gpus if stage == "train" else config.backend.infer_gpus)
    if group_value != configured:
        raise ValueError(
            f"Hydra backend.{stage_key}={group_value} disagrees with runtime policy "
            f"sandbox_resources.{backend_name}.{stage_key}={configured}"
        )
    scheduler = backend_resources.get("scheduler")
    group_scheduler = str(getattr(config.backend, "scheduler", ""))
    if scheduler and group_scheduler and str(scheduler) != group_scheduler:
        raise ValueError(
            f"Hydra backend.scheduler={group_scheduler} disagrees with runtime policy "
            f"sandbox_resources.{backend_name}.scheduler={scheduler}"
        )
    policy_tracking = policy.get("wandb_policy", {})
    tracking = _json_mapping(config.tracking)
    if "required" in policy_tracking and bool(tracking.get("required")) != bool(
        policy_tracking["required"]
    ):
        raise ValueError("Hydra tracking.required disagrees with runtime policy wandb_policy.required")
    return configured


def _required_weight_files(root: Path) -> tuple[Path, ...]:
    single = root / "diffusion_pytorch_model.safetensors"
    index_file = root / "diffusion_pytorch_model.safetensors.index.json"
    if single.is_file() and single.stat().st_size > 0:
        return (single,)
    if index_file.is_file():
        try:
            mapping = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
            files = tuple(root / name for name in sorted(set(mapping.values())))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid safetensors index under the configured asset root") from error
        if not files:
            raise ValueError("safetensors index contains no shards under the configured asset root")
        return files
    shards = tuple(sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors")))
    totals: set[int] = set()
    indices: set[int] = set()
    for shard in shards:
        match = re.fullmatch(r"diffusion_pytorch_model-(\d+)-of-(\d+)\.safetensors", shard.name)
        if not match:
            raise ValueError(f"unrecognized model shard name: {shard.name}")
        indices.add(int(match.group(1)))
        totals.add(int(match.group(2)))
    if not shards or len(totals) != 1 or indices != set(range(1, next(iter(totals)) + 1)):
        raise FileNotFoundError("model weight files are missing or incomplete under the configured asset root")
    return shards


def check_assets(
    config: Any, stage: str, *, exp_dir: str | Path | None = None
) -> dict[str, Path]:
    """Check configured model assets and the stage's prompt/checkpoint inputs."""
    exp_dir = resolve_exp_dir(exp_dir)
    assets = _json_mapping(config.assets)
    asset_root = resolve_asset_root(assets.get("root"))
    student = asset_path(str(assets["student_model_dir"]), "student model", asset_root)
    teacher = asset_path(str(assets["teacher_model_dir"]), "teacher model", asset_root)
    generator = asset_path(str(assets["generator_checkpoint"]), "generator checkpoint", asset_root)
    required = [
        student / "Wan2.1_VAE.pth",
        student / "models_t5_umt5-xxl-enc-bf16.pth",
        student / "google/umt5-xxl/spiece.model",
        student / "google/umt5-xxl/tokenizer.json",
        student / "google/umt5-xxl/tokenizer_config.json",
        student / "config.json",
        generator,
    ]
    if stage == "train":
        required.extend((teacher / "config.json", *_required_weight_files(teacher)))
        prompt_value = str(config.train.prompt_file)
    else:
        prompt_value = str(config.infer.prompt_file)
    required.extend(_required_weight_files(student))
    prompt_file = dataset_path(exp_dir, prompt_value, f"{stage} prompt file")
    required.append(prompt_file)
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        labels: list[str] = []
        for path in missing:
            try:
                relative = path.relative_to(asset_root)
                labels.append(f"SUE_ASSET_ROOT/{relative.as_posix()}")
            except ValueError:
                try:
                    labels.append(path.relative_to(exp_dir).as_posix())
                except ValueError:
                    labels.append(path.name)
        raise FileNotFoundError(
            "required local assets are missing or empty:\n" + "\n".join(labels)
        )
    return {
        "asset_root": asset_root,
        "student": student,
        "teacher": teacher,
        "generator": generator,
        "prompt": prompt_file,
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return payload


def _materialize_runtime_adapter(
    source_path: Path,
    source_provenance_path: Path,
    payload: dict[str, Any],
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    destination: Path,
) -> dict[str, Any]:
    """Create or verify a per-run adapter whose base path resolves on this backend."""
    import torch

    sidecar_path = destination.with_suffix(".provenance.json")
    source_digest = sha256_file(source_path)
    source_provenance_digest = sha256_file(source_provenance_path)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", base_checkpoint_sha256):
        raise ValueError("verified base checkpoint digest must be a SHA-256 hex string")
    base_digest = base_checkpoint_sha256.lower()
    identity = {
        "schema_version": 1,
        "source_checkpoint_sha256": source_digest,
        "source_provenance_sha256": source_provenance_digest,
        "base_checkpoint_sha256": base_digest,
    }

    def verify_existing() -> dict[str, Any]:
        try:
            recorded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing runtime adapter provenance is unreadable") from error
        if any(recorded.get(key) != value for key, value in identity.items()):
            raise ValueError("existing runtime adapter does not match the selected inputs; use a new run_id")
        actual_runtime_digest = sha256_file(destination)
        if recorded.get("runtime_checkpoint_sha256") != actual_runtime_digest:
            raise ValueError("existing runtime adapter bytes do not match its provenance")
        runtime_payload = _load_checkpoint(destination)
        runtime_metadata = runtime_payload.get("metadata")
        if (
            not isinstance(runtime_metadata, dict)
            or runtime_metadata.get("base_checkpoint") != str(base_checkpoint)
            or runtime_metadata.get("base_checkpoint_sha256") != base_digest
        ):
            raise ValueError("existing runtime adapter points to a different active base checkpoint")
        return {
            "path": destination,
            "sha256": actual_runtime_digest,
            "provenance_path": sidecar_path,
            "provenance_sha256": sha256_file(sidecar_path),
        }

    if destination.is_symlink() or sidecar_path.is_symlink():
        raise FileExistsError("runtime adapter paths must not be symlinks")
    if destination.exists() or sidecar_path.exists():
        if not (destination.is_file() and sidecar_path.is_file()):
            raise FileExistsError("runtime adapter is incomplete; use a new run_id")
        return verify_existing()

    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(payload.get("metadata") or {})
    metadata["base_checkpoint"] = str(base_checkpoint)
    metadata["base_checkpoint_sha256"] = base_digest
    runtime_payload = {**payload, "metadata": metadata}
    payload_fd, payload_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    provenance_fd, provenance_temporary = tempfile.mkstemp(
        prefix=f".{sidecar_path.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(payload_fd)
    try:
        torch.save(runtime_payload, payload_temporary)
        runtime_digest = sha256_file(payload_temporary)
        runtime_provenance = {
            **identity,
            "runtime_checkpoint_sha256": runtime_digest,
        }
        with os.fdopen(provenance_fd, "w", encoding="utf-8") as stream:
            json.dump(runtime_provenance, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        installed: list[Path] = []
        try:
            os.link(payload_temporary, destination)
            installed.append(destination)
            os.link(provenance_temporary, sidecar_path)
            installed.append(sidecar_path)
        except FileExistsError as error:
            for path in installed:
                path.unlink(missing_ok=True)
            raise FileExistsError("runtime adapter appeared concurrently; use one worker per run_id") from error
    finally:
        try:
            os.close(provenance_fd)
        except OSError:
            pass
        for temporary in (payload_temporary, provenance_temporary):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return verify_existing()


def _check_adapter_scope(state: dict[str, Any], method: dict[str, Any], label: str) -> None:
    targets = tuple(method["lora"]["target_modules"])
    expected = {
        (layer, target, matrix)
        for layer in range(method["layer_start"], method["layer_end"] + 1)
        for target in targets
        for matrix in ("A", "B")
    }
    actual: set[tuple[int, str, str]] = set()
    for name in state:
        match = re.search(
            r"(?:^|\.)blocks\.(\d+)\..*?(self_attn\.[qv])\.lora_([AB])(?:\.default)?\.weight$",
            str(name),
        )
        if not match:
            raise ValueError(f"{label} contains an unexpected adapter key: {name}")
        key = (int(match.group(1)), match.group(2), match.group(3))
        if key in actual:
            raise ValueError(f"{label} contains duplicate adapter scope: {key}")
        actual.add(key)
    if actual != expected:
        raise ValueError(
            f"{label} adapter scope differs from method config: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _validate_adapter_provenance(
    provenance: dict[str, Any], payload: dict[str, Any], method: dict[str, Any]
) -> str:
    """Cross-check exported sidecar claims against checkpoint-embedded identity."""
    if provenance.get("method") != method:
        raise ValueError("inference adapter provenance method differs from the selected Hydra method")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("inference adapter checkpoint lacks metadata")
    embedded_loop = metadata.get("temporal_loop")
    if not isinstance(embedded_loop, dict):
        raise ValueError("inference adapter lacks embedded temporal_loop metadata")
    loop_fields = (
        "mode",
        "layer_start",
        "layer_end",
        "k_min",
        "k_max",
        "strength",
        "stop_grad_early",
        "schedule",
        "training_enabled",
    )
    for key in loop_fields:
        if embedded_loop.get(key) != method[key]:
            raise ValueError(f"embedded temporal_loop.{key} disagrees with adapter provenance method")
    if metadata.get("lora") != method["lora"]:
        raise ValueError("embedded LoRA metadata disagrees with adapter provenance method")

    source_step = provenance.get("source_training_step")
    checkpoint_step = metadata.get("step")
    if (
        isinstance(source_step, bool)
        or not isinstance(source_step, int)
        or isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or source_step != checkpoint_step
    ):
        raise ValueError("adapter provenance training step does not match embedded checkpoint step")

    source_seed = provenance.get("seed")
    checkpoint_seed = metadata.get("seed")
    if (
        isinstance(source_seed, bool)
        or not isinstance(source_seed, int)
        or source_seed <= 0
        or isinstance(checkpoint_seed, bool)
        or not isinstance(checkpoint_seed, int)
        or source_seed != checkpoint_seed
    ):
        raise ValueError("adapter provenance seed does not match embedded checkpoint seed")

    sidecar_base_hash = provenance.get("base_checkpoint_sha256")
    embedded_base_hash = metadata.get("base_checkpoint_sha256")
    if (
        not isinstance(sidecar_base_hash, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", sidecar_base_hash)
        or embedded_base_hash != sidecar_base_hash
    ):
        raise ValueError("adapter provenance base hash does not match embedded checkpoint metadata")
    return sidecar_base_hash.lower()


def check_inference_checkpoint(
    config: Any,
    method: dict[str, Any],
    assets: dict[str, Path],
    *,
    exp_dir: str | Path | None = None,
    runtime_adapter_path: Path,
) -> dict[str, Any]:
    """Require an explicit, method-matched adapter with portable base identity."""
    checkpoint_value = config.infer.checkpoint
    if not checkpoint_value:
        raise ValueError("infer.checkpoint must be explicitly set to a verified adapter")
    checkpoint = bundle_path(resolve_exp_dir(exp_dir), str(checkpoint_value), "inference checkpoint")
    payload = _load_checkpoint(checkpoint)
    if payload.get("generator_format") != "lora_adapter":
        raise ValueError(f"checkpoint is not an adapter-only LoRA checkpoint: {checkpoint}")
    if payload.get("generator") is None or payload.get("generator_ema") is None:
        if bool(config.infer.use_ema):
            raise ValueError("EMA inference requested but checkpoint has no generator_ema")
        if payload.get("generator") is None:
            raise ValueError("inference checkpoint is missing generator adapter weights")
    _check_adapter_scope(payload["generator"], method, "generator")
    if payload.get("generator_ema") is not None:
        _check_adapter_scope(payload["generator_ema"], method, "generator_ema")

    sidecar = checkpoint.with_name("provenance.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"inference adapter provenance is required: {sidecar}")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if provenance.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("inference adapter bytes do not match provenance sidecar")
    metadata = payload.get("metadata")
    base_hash = _validate_adapter_provenance(provenance, payload, method)
    portable_metadata = dict(metadata)
    resolved_base, resolved_base_sha256 = resolve_base_checkpoint_with_hash(
        portable_metadata, assets["generator"]
    )
    runtime_adapter = _materialize_runtime_adapter(
        checkpoint,
        sidecar,
        payload,
        resolved_base,
        resolved_base_sha256,
        runtime_adapter_path,
    )
    return {
        "path": checkpoint,
        "provenance_path": sidecar,
        "runtime_path": runtime_adapter["path"],
        "runtime_sha256": runtime_adapter["sha256"],
        "runtime_provenance_path": runtime_adapter["provenance_path"],
        "runtime_provenance_sha256": runtime_adapter["provenance_sha256"],
        "payload": payload,
        "provenance": provenance,
        "base_checkpoint": resolved_base,
        "base_checkpoint_sha256": resolved_base_sha256,
    }


def _training_manifest_config(config: Any, assets: dict[str, Path]) -> dict[str, Any]:
    method = _json_mapping(config.method)
    method.pop("profile_id", None)
    return {
        "backend": str(config.backend.name),
        "method": method,
        "scale": _json_mapping(config.scale),
        "tracking": _json_mapping(config.tracking),
        "train": _json_mapping(config.train),
        "assets": {
            "generator_checkpoint": str(config.assets.generator_checkpoint),
            "generator_size": assets["generator"].stat().st_size,
            "generator_checkpoint_sha256": sha256_file(assets["generator"]),
            "teacher_model_dir": str(config.assets.teacher_model_dir),
            "student_model_dir": str(config.assets.student_model_dir),
        },
        "train_prompt": {
            "path": str(config.train.prompt_file),
            "size": assets["prompt"].stat().st_size,
            "sha256": sha256_file(assets["prompt"]),
        },
    }


def _inference_manifest_config(config: Any, assets: dict[str, Path], checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": str(config.backend.name),
        "method": method_identity(config.method),
        "tracking": _json_mapping(config.tracking),
        "infer": _json_mapping(config.infer),
        "prompt_file": {
            "path": str(config.infer.prompt_file),
            "size": assets["prompt"].stat().st_size,
            "sha256": sha256_file(assets["prompt"]),
        },
        "checkpoint": {
            "path": str(config.infer.checkpoint),
            "sha256": checkpoint["provenance"]["checkpoint_sha256"],
            "provenance_sha256": sha256_file(checkpoint["provenance_path"]),
            "runtime_checkpoint_sha256": checkpoint["runtime_sha256"],
            "runtime_provenance_sha256": checkpoint["runtime_provenance_sha256"],
            "base_checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
        },
    }


def prepare_stage(
    config: Any,
    stage: str,
    *,
    check_assets_enabled: bool = True,
    exp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate inputs, persist the exact stage config, and lock its manifest."""
    from omegaconf import OmegaConf

    if stage not in {"train", "infer"}:
        raise ValueError(f"unsupported worker stage: {stage}")
    exp_dir = resolve_exp_dir(exp_dir)
    output_roots = resolve_runtime_paths(exp_dir)
    run_id = str(config.run_id or "").strip()
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be provided as 1–64 lowercase letters, digits, _ or -")
    run_root = _inside(output_roots["artifacts_root"] / run_id, exp_dir, "run root")
    checkpoint_root = _inside(output_roots["ckpt_root"] / run_id, exp_dir, "checkpoint root")
    results_root = _inside(output_roots["final_result_root"] / run_id, exp_dir, "result root")
    method = method_identity(config.method)
    if stage == "train":
        require_training_seed(_json_mapping(config.train).get("seed"))
    required_gpus = _configured_gpu_count(config, stage, exp_dir)
    assets = check_assets(config, stage, exp_dir=exp_dir) if check_assets_enabled else {}
    if stage == "train":
        train = _json_mapping(config.train)
        max_steps, log_iters = int(train["max_steps"]), int(train["log_iters"])
        if max_steps <= 0 or log_iters <= 0 or max_steps % log_iters:
            raise ValueError("train.max_steps and train.log_iters must be positive and divisible")
        if required_gpus != 4:
            raise ValueError("this Self-Forcing DMD training profile requires four GPUs")
        resolved = _json_mapping(config.method)
        resolved["generator_ckpt"] = str(
            assets.get("generator")
            or asset_path(
                str(config.assets.generator_checkpoint),
                "generator checkpoint",
                str(config.assets.root),
            )
        )
        resolved["teacher_checkpoint"] = str(
            assets.get("teacher")
            or asset_path(
                str(config.assets.teacher_model_dir), "teacher model", str(config.assets.root)
            )
        )
        prompt = assets.get("prompt") or dataset_path(
            exp_dir, str(config.train.prompt_file), "train prompt file"
        )
        resolved["data_path"] = str(prompt)
        resolved["seed"] = int(config.train.seed)
        resolved["log_iters"] = log_iters
        resume_value = config.train.resume_checkpoint
        if resume_value:
            resume_path = bundle_path(exp_dir, str(resume_value), "resume checkpoint")
            resolved["resume_checkpoint"] = str(resume_path)
        output_name = "train.yaml"
        stage_config = _training_manifest_config(config, assets)
    else:
        if required_gpus != 1:
            raise ValueError("inference worker requires one GPU")
        require_inference_latent_frames(config.infer.num_output_frames)
        if int(config.infer.num_samples) <= 0:
            raise ValueError("infer.num_samples must be positive")
        checkpoint = (
            check_inference_checkpoint(
                config,
                method,
                assets,
                exp_dir=exp_dir,
                runtime_adapter_path=run_root / "runtime_inference_adapter.pt",
            )
            if check_assets_enabled
            else None
        )
        resolved = _json_mapping(config.method)
        resolved["generator_ckpt"] = str(
            assets.get("generator")
            or asset_path(
                str(config.assets.generator_checkpoint),
                "generator checkpoint",
                str(config.assets.root),
            )
        )
        output_name = "inference.yaml"
        stage_config = _inference_manifest_config(config, assets, checkpoint) if checkpoint else {
            "backend": str(config.backend.name),
            "method": method,
            "infer": _json_mapping(config.infer),
        }

    rendered = OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True)
    config_root = run_root / "config"
    stage_manifest_root = config_root / stage
    stage_manifest_root.mkdir(parents=True, exist_ok=True)
    stage_config["run_id"] = run_id
    stage_config["resolved_config_sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    materialize_manifest(stage_manifest_root, stage, stage_config)
    config_path = config_root / output_name
    if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"resolved {stage} configuration changed for this run_id: {config_path}")
    if not config_path.exists():
        temporary = config_path.with_name(config_path.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, config_path)
    return {
        "exp_dir": exp_dir,
        "run_root": run_root,
        "output_roots": output_roots,
        "checkpoint_root": checkpoint_root,
        "results_root": results_root,
        "final_model": checkpoint_root / "model.pt",
        "config_path": config_path,
        "method": method,
        "assets": assets,
        "checkpoint": checkpoint if stage == "infer" else None,
        "manifest": stage_config,
    }


def check_stage(
    config: Any, stage: str, *, exp_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run cheap asset/config preflight without allocating a GPU worker."""
    _validate_tracking_environment(config)
    if stage == "infer":
        require_ffprobe()
    prepared = prepare_stage(config, stage, exp_dir=exp_dir)
    if stage == "train":
        if shutil.which("timeout") is None:
            raise FileNotFoundError("GNU timeout is required by the training worker")
        entry = SOURCE_ROOT / "train_blockwise_lora_dmd.py"
        command = [
            sys.executable,
            str(entry),
            "--config_path",
            str(prepared["config_path"]),
            "--preflight",
        ]
        subprocess.run(command, cwd=SOURCE_ROOT, check=True)
    return prepared


def _resolved_run_config(config: Any) -> dict[str, Any]:
    return _json_mapping(config)


def _command_for(config: Any, stage: str, prepared: dict[str, Any]) -> list[str]:
    source = SOURCE_ROOT
    if stage == "train":
        timeout_seconds = int(config.train.get("timeout_seconds", 19800))
        return [
            "timeout",
            "-s",
            "TERM",
            "--kill-after=300",
            str(timeout_seconds),
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(PACKAGE_ROOT / "training_entry.py"),
            "--config_path",
            str(prepared["config_path"]),
            "--logdir",
            str(prepared["checkpoint_root"]),
            "--wandb-save-dir",
            str(prepared["run_root"] / "wandb"),
        ]
    checkpoint_path = prepared["checkpoint"]["runtime_path"]
    prompt = dataset_path(
        prepared["exp_dir"], str(config.infer.prompt_file), "infer prompt file"
    )
    output = prepared["results_root"]
    return [
        sys.executable,
        str(source / "inference.py"),
        "--config_path",
        str(prepared["config_path"]),
        "--checkpoint_path",
        str(checkpoint_path),
        "--data_path",
        str(prompt),
        "--output_folder",
        str(output),
        "--num_output_frames",
        str(int(config.infer.num_output_frames)),
        "--num_samples",
        str(int(config.infer.num_samples)),
        "--seed",
        str(int(config.infer.seed)),
        "--save_with_index",
        *(["--use_ema"] if bool(config.infer.use_ema) else []),
    ]


def _lightweight_model(full_checkpoint: Path, final_model: Path) -> None:
    import torch

    payload = _load_checkpoint(full_checkpoint)
    lightweight = {
        key: payload[key]
        for key in ("checkpoint_version", "generator_format", "generator", "generator_ema", "metadata")
        if key in payload
    }
    metadata = dict(lightweight.get("metadata") or {})
    metadata["final"] = True
    lightweight["metadata"] = metadata
    temporary = final_model.with_name(final_model.name + ".tmp")
    torch.save(lightweight, temporary)
    os.replace(temporary, final_model)


def _require_gpu_capacity(
    config: Any, stage: str, exp_dir: str | Path | None = None
) -> None:
    import torch

    requested = _configured_gpu_count(config, stage, resolve_exp_dir(exp_dir))
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible < requested:
        raise RuntimeError(
            f"{config.backend.name} {stage} worker requires {requested} visible CUDA GPUs; found {visible}"
        )


def _validate_tracking_environment(config: Any) -> None:
    """Fail before GPU execution when online W&B credentials are unavailable."""
    tracking = _json_mapping(config.tracking)
    if not str(tracking.get("project", "")).strip():
        raise ValueError("tracking.project must be set in the selected Hydra tracking group")
    if str(tracking.get("mode", "offline")) != "online":
        return
    required_names = (
        str(tracking.get("api_key_env", "WANDB_API_KEY")),
        str(tracking.get("entity_env", "WANDB_ENTITY")),
    )
    missing = [name for name in required_names if not os.environ.get(name, "").strip()]
    if missing:
        names = ", ".join(dict.fromkeys(missing))
        raise RuntimeError(f"online W&B tracking requires environment variable(s): {names}")


def _tracking_project(config: Any) -> str:
    tracking = _json_mapping(config.tracking)
    project = str(tracking.get("project", "")).strip()
    if not project:
        raise ValueError("tracking.project must be set in the selected Hydra tracking group")
    return project


def _build_worker_environment(config: Any, prepared: dict[str, Any], stage: str) -> dict[str, str]:
    """Build the child environment from the selected backend and stage config."""
    environment = os.environ.copy()
    backend = str(config.backend.name)
    tracking = _json_mapping(config.tracking)
    train_config = _json_mapping(config.train)
    environment["PYTHONPATH"] = os.pathsep.join((str(SOURCE_ROOT), str(SCRIPT_ROOT)))
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["WANDB_MODE"] = str(tracking.get("mode", "offline"))
    environment["SUE_WANDB_MODE"] = str(tracking.get("mode", "offline"))
    environment["SUE_BACKEND"] = backend
    environment["WANDB_DIR"] = str(prepared["run_root"] / "wandb")
    environment["SUE_WANDB_RUN_ID_PATH"] = str(prepared["run_root"] / "wandb_run_id.txt")
    environment["WANDB_PROJECT"] = _tracking_project(config)
    environment["WANDB_EXPERIMENT_NAME"] = (
        build_experiment_name(prepared["method"], str(config.run_id), stage)
    )
    environment["WANDB_GROUP"] = str(config.run_id)
    if stage == "train":
        environment["SUE_BASE_CHECKPOINT_RELATIVE_PATH"] = str(
            _json_mapping(config.assets)["generator_checkpoint"]
        )
        environment["SUE_MAX_STEPS"] = str(int(train_config["max_steps"]))
        environment["SUE_BASE_CHECKPOINT_SHA256"] = str(
            prepared["manifest"]["assets"]["generator_checkpoint_sha256"]
        )
        environment["SUE_CHECKPOINT_INTERVAL_SECONDS"] = str(
            int(train_config.get("checkpoint_interval_seconds", 3600))
        )
        environment["SUE_WANDB_LOG_INTERVAL"] = str(
            int(tracking.get("scalar_log_interval", 100))
        )
        environment["SUE_PROGRESS_LOG_INTERVAL"] = str(
            int(tracking.get("progress_log_interval", 10))
        )
    return environment


def _read_wandb_run_id(path: str | Path) -> str:
    """Read the exact run ID written by this worker attempt, if any."""
    marker = Path(path)
    if not marker.is_file():
        return ""
    return marker.read_text(encoding="utf-8").strip()


def _write_wandb_run_id(path: str | Path, run_id: str) -> None:
    from looped_self_forcing_pipeline.training_policy import persist_wandb_run_id

    persist_wandb_run_id(path, run_id)


def _start_inference_tracking(
    config: Any,
    prepared: dict[str, Any],
    *,
    run_id_path: str | Path,
    wandb_module: Any | None = None,
):
    """Create the inference run before execution so required tracking fails fast."""
    tracking = _json_mapping(config.tracking)
    if not bool(tracking.get("required", False)):
        return None
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as error:
            raise RuntimeError("required W&B tracking package is unavailable") from error

    provenance = prepared["checkpoint"]["provenance"]
    wandb_run = wandb_module.init(
        project=_tracking_project(config),
        entity=os.environ.get(str(tracking.get("entity_env", "WANDB_ENTITY"))) or None,
        name=build_experiment_name(prepared["method"], str(config.run_id), "infer"),
        group=str(config.run_id),
        job_type="inference",
        mode=str(tracking.get("mode", "offline")),
        dir=str(prepared["run_root"] / "wandb"),
        config={
            "method": prepared["method"],
            "backend": str(config.backend.name),
            "run_id": str(config.run_id),
            "seed": int(config.infer.seed),
            "num_output_frames": int(config.infer.num_output_frames),
            "num_samples": int(config.infer.num_samples),
            "use_ema": bool(config.infer.use_ema),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "base_checkpoint_sha256": prepared["checkpoint"]["base_checkpoint_sha256"],
            "source_training_step": provenance.get("source_training_step"),
        },
    )
    run_id = str(getattr(wandb_run, "id", "") or "").strip()
    if not wandb_run or not run_id:
        if bool(tracking.get("required", False)):
            raise RuntimeError("required inference W&B run did not provide a run ID")
        return None
    _write_wandb_run_id(run_id_path, run_id)
    return wandb_run


def _finish_tracking_run(wandb_run: Any, *, status: str, summary: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    target = getattr(wandb_run, "summary", None)
    if target is not None:
        target.update({**summary, "status": status})
    wandb_run.finish(exit_code=0 if status == "completed" else 1)


def execute_worker(
    config: Any, stage: str, *, exp_dir: str | Path | None = None
) -> dict[str, Any]:
    """Record worker entry before runtime gates, and mark every later failure."""
    if stage not in {"train", "infer"}:
        raise ValueError(f"unsupported worker stage: {stage}")
    selected_exp_dir = resolve_exp_dir(exp_dir)
    output_roots = resolve_runtime_paths(selected_exp_dir)
    run_id = str(config.run_id or "").strip()
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be provided as 1–64 lowercase letters, digits, _ or -")
    method_config = _json_mapping(config.method)
    loop_config = method_config.get("temporal_loop", {})
    method_mode = str(loop_config.get("mode", "unknown"))
    method_identity_for_row = {
        "profile_id": method_config.get("profile_id", ""),
        "mode": method_mode,
    }
    try:
        experiment_label = build_experiment_name(method_identity_for_row, run_id, stage)
    except ValueError:
        experiment_label = f"unknown-{run_id}-{stage}"
    run_root = output_roots["artifacts_root"] / run_id
    checkpoint_root = output_roots["ckpt_root"] / run_id
    results_root = output_roots["final_result_root"] / run_id
    ledger_path = output_roots["artifacts_root"] / "experiment_results.csv"
    try:
        seed = int(config.train.seed if stage == "train" else config.infer.seed)
    except (AttributeError, TypeError, ValueError):
        seed = ""
    ledger_row = {
        "method": method_mode,
        "experiment_name": experiment_label,
        "backend": str(config.backend.name),
        "status": "running",
        "seed": seed,
        "checkpoint_or_output_path": str(checkpoint_root if stage == "train" else results_root),
    }
    write_result(ledger_path, ledger_row)
    try:
        return _execute_worker_body(
            config,
            stage,
            exp_dir=selected_exp_dir,
            ledger_path=ledger_path,
            ledger_row=ledger_row,
        )
    except Exception as error:
        failed_row = {
            **ledger_row,
            "status": "failed",
            "wandb_run_id": _read_wandb_run_id(run_root / "wandb_run_id.txt"),
            "execution_id": os.environ.get(
                "SLURM_JOB_ID", os.environ.get("SUE_EXECUTION_ID", "")
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "error": f"{type(error).__name__}: {error}",
        }
        try:
            write_result(ledger_path, failed_row)
        except Exception:
            # Preserve the worker error; the initial running-row write is mandatory.
            pass
        raise


def _execute_worker_body(
    config: Any,
    stage: str,
    *,
    exp_dir: str | Path,
    ledger_path: Path,
    ledger_row: dict[str, Any],
) -> dict[str, Any]:
    """Run one prepared worker and record completion only after evidence checks."""
    _validate_tracking_environment(config)
    if stage == "infer":
        require_ffprobe()
    prepared = prepare_stage(config, stage, exp_dir=exp_dir)
    backend = str(config.backend.name)
    _require_gpu_capacity(config, stage, prepared["exp_dir"])
    run_root = prepared["run_root"]
    checkpoint_root = prepared["checkpoint_root"]
    results_root = prepared["results_root"]
    wandb_root = run_root / "wandb"
    wandb_run_id_path = run_root / "wandb_run_id.txt"
    if stage == "train":
        require_fresh_training_outputs(
            checkpoint_root,
            wandb_root,
            prepared["final_model"],
            wandb_run_id_path=wandb_run_id_path,
        )
        expected_steps = int(config.train.max_steps)
        expected_seed = require_training_seed(int(config.train.seed))
    else:
        require_fresh_inference_outputs(results_root, wandb_root, wandb_run_id_path)
        expected_steps = 0

    environment = _build_worker_environment(config, prepared, stage)
    command = _command_for(config, stage, prepared)
    tracking_run = None
    tracking_run_id = ""
    started_at = time.monotonic()
    try:
        if stage == "infer":
            tracking_run = _start_inference_tracking(
                config,
                prepared,
                run_id_path=wandb_run_id_path,
            )
            tracking_run_id = str(getattr(tracking_run, "id", "") or "")

        result = subprocess.run(command, cwd=SOURCE_ROOT, env=environment, check=False)
        elapsed = max(time.monotonic() - started_at, 1e-9)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)

        execution_id = os.environ.get(
            "SLURM_JOB_ID", os.environ.get("SUE_EXECUTION_ID", "")
        )
        if stage == "train":
            tracking_run_id = _read_wandb_run_id(wandb_run_id_path)
            if bool(config.tracking.required) and not tracking_run_id:
                raise RuntimeError("training exited without a required W&B run identity")
            latest = checkpoint_root / "latest.pt"
            if not latest.is_file():
                candidates = sorted(checkpoint_root.glob("checkpoint_model_*/model.pt"))
                if not candidates:
                    raise RuntimeError("training exited without a complete checkpoint")
                latest = candidates[-1]
            evidence = verify_training_checkpoint(
                latest,
                expected_step=expected_steps,
                expected_seed=expected_seed,
            )
            if evidence["metadata"].get("final") is not True:
                raise RuntimeError("final training checkpoint is not marked final=true")
            _lightweight_model(latest, prepared["final_model"])
            steps_completed = max(0, int(evidence["step"]))
            speed = steps_completed / elapsed
            write_result(
                ledger_path,
                {
                    **ledger_row,
                    "status": "completed",
                    "wandb_run_id": tracking_run_id,
                    "execution_id": execution_id,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                    "step": steps_completed,
                    "checkpoint_step": evidence["step"],
                    "training_speed": speed,
                    "trainable_parameter_count": evidence["trainable_parameter_count"],
                    "checkpoint_or_output_path": str(latest),
                },
            )
            return {"checkpoint": latest, "model": prepared["final_model"], "speed": speed}

        prompt = dataset_path(
            prepared["exp_dir"], str(config.infer.prompt_file), "infer prompt file"
        )
        expected_count = prompt_count(prompt) * int(config.infer.num_samples)
        expected_frames = 4 * int(config.infer.num_output_frames) - 3
        videos = verify_inference_outputs(
            results_root,
            expected_count=expected_count,
            expected_frames=expected_frames,
            expected_fps=16,
        )
        provenance = prepared["checkpoint"]["provenance"]
        source_step = provenance.get("source_training_step", "")
        tracking_summary = {
            "output_count": len(videos),
            "checkpoint_step": source_step,
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "seed": int(config.infer.seed),
        }
        write_result(
            ledger_path,
            {
                **ledger_row,
                "status": "completed",
                "wandb_run_id": tracking_run_id,
                "execution_id": execution_id,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "checkpoint_step": source_step,
                "checkpoint_or_output_path": str(results_root),
            },
        )
        _finish_tracking_run(tracking_run, status="completed", summary=tracking_summary)
        tracking_run = None
        return {"videos": videos, "output_dir": results_root}
    except Exception as error:
        if tracking_run is not None:
            try:
                _finish_tracking_run(
                    tracking_run,
                    status="failed",
                    summary={"error": f"{type(error).__name__}: {error}"},
                )
            except Exception:
                pass
        raise
