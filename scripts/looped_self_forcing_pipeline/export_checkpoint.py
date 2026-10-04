#!/usr/bin/env python3
"""Export a verified training checkpoint as a portable inference adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from looped_self_forcing_pipeline.checkpoint import (  # noqa: E402
    portable_base_checkpoint_metadata,
    sha256_file,
)
from looped_self_forcing_pipeline.config import (  # noqa: E402
    compose_config,
    resolve_exp_dir,
    resolve_runtime_paths,
)
from looped_self_forcing_pipeline.manifest import MANIFEST_SCHEMA_VERSION  # noqa: E402
from looped_self_forcing_pipeline.worker import (  # noqa: E402
    _check_adapter_scope,
    asset_path,
    bundle_path,
    method_identity,
    resolve_asset_root,
)


def _same_method(left: dict, right: dict) -> bool:
    return {key: value for key, value in left.items() if key != "profile_id"} == {
        key: value for key, value in right.items() if key != "profile_id"
    }


def _require_final_training_checkpoint(payload: dict, expected_steps: int) -> int:
    """Reject hourly/intermediate checkpoints from production adapter export."""
    if expected_steps <= 0:
        raise ValueError("same-run train manifest must specify positive train.max_steps")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("full checkpoint is missing training metadata")
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step != expected_steps:
        raise ValueError(
            f"selected checkpoint step {step!r} is partial; expected final step {expected_steps}"
        )
    if metadata.get("final") is not True:
        raise ValueError("selected checkpoint is not marked final=true")
    return step


def _require_checkpoint_seed(payload: dict, expected_seed: int) -> int:
    """Require exported checkpoint randomness to match immutable run config."""
    if isinstance(expected_seed, bool) or not isinstance(expected_seed, int) or expected_seed <= 0:
        raise ValueError("same-run train manifest must specify a positive non-zero train.seed")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("full checkpoint is missing training metadata")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != expected_seed:
        raise ValueError(
            f"checkpoint seed {seed!r} disagrees with train manifest seed {expected_seed}"
        )
    return seed


def _validate_training_checkpoint_objective(train_config, payload: dict) -> None:
    """Validate critic presence against the resolved trainer and objective."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("full checkpoint is missing training metadata")
    for key in ("critic", "critic_optimizer"):
        if key not in payload:
            raise ValueError(f"full checkpoint is missing {key}")

    trainer = train_config.get("trainer")
    objective = metadata.get("training_objective")
    critic = payload.get("critic")
    critic_optimizer = payload.get("critic_optimizer")

    if trainer == "diffusion":
        if objective != "supervised_flow_matching":
            raise ValueError("resolved trainer and checkpoint objective do not match")
        if critic is not None or critic_optimizer is not None:
            raise ValueError("supervised flow-matching checkpoint must not contain critic state")
        return

    if trainer not in (None, "score_distillation"):
        raise ValueError(f"unsupported resolved trainer for adapter export: {trainer}")
    if objective not in (None, "dmd"):
        raise ValueError("resolved trainer and checkpoint objective do not match")
    if critic is None or critic_optimizer is None:
        raise ValueError("DMD/legacy checkpoint requires critic weights and optimizer state")
    if not isinstance(critic, dict) or not isinstance(critic_optimizer, dict):
        raise ValueError("DMD/legacy critic weights and optimizer state must be mappings")


def build_inference_payload(
    full_payload: dict,
    *,
    base_checkpoint: str | Path,
    source_root: str | Path,
    use_ema: bool,
) -> dict:
    """Strip resume-only state and replace host-specific base paths."""
    if full_payload.get("generator_format") != "lora_adapter":
        raise ValueError("full checkpoint is not a LoRA adapter checkpoint")
    if not isinstance(full_payload.get("generator"), dict):
        raise ValueError("full checkpoint is missing generator adapter weights")
    if use_ema and full_payload.get("generator_ema") is None:
        raise ValueError("EMA inference requested but the full checkpoint has no EMA adapter")
    metadata = portable_base_checkpoint_metadata(
        full_payload.get("metadata") or {}, base_checkpoint, source_root
    )
    metadata["final"] = True
    return {
        "checkpoint_version": full_payload.get("checkpoint_version", 1),
        "generator_format": "lora_adapter",
        "generator": full_payload["generator"],
        "generator_ema": full_payload.get("generator_ema"),
        "metadata": metadata,
    }


def _atomic_export(destination: Path, payload: dict, provenance: dict) -> None:
    import torch

    sidecar = destination.with_name("provenance.json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite an existing adapter or provenance: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_fd, payload_tmp = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    sidecar_fd, sidecar_tmp = tempfile.mkstemp(
        prefix=".provenance.", suffix=".tmp", dir=destination.parent
    )
    os.close(payload_fd)
    try:
        torch.save(payload, payload_tmp)
        provenance["checkpoint_sha256"] = sha256_file(payload_tmp)
        with os.fdopen(sidecar_fd, "w", encoding="utf-8") as stream:
            json.dump(provenance, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(payload_tmp, destination)
        try:
            os.replace(sidecar_tmp, sidecar)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    finally:
        for temporary in (payload_tmp, sidecar_tmp):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def export_checkpoint(
    *,
    exp_dir: str | Path | None,
    run_id: str,
    method_group: str,
    backend: str,
    full_checkpoint: str,
    destination: str,
    allow_raw_smoke: bool = False,
    use_ema: bool = True,
) -> tuple[Path, dict]:
    """Validate same-run training evidence, then export an inference adapter."""
    import torch
    from omegaconf import OmegaConf

    bundle = resolve_exp_dir(exp_dir)
    source = bundle_path(bundle, full_checkpoint, "full checkpoint")
    destination_path = bundle_path(bundle, destination, "inference adapter destination")
    runtime_paths = resolve_runtime_paths(bundle)
    source_run_root = runtime_paths["artifacts_root"] / run_id
    source_checkpoint_root = runtime_paths["ckpt_root"] / run_id
    bundle_path(bundle, str(source_run_root), "source run root")
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"full checkpoint is missing or empty: {source}")
    if not source.is_relative_to(source_checkpoint_root.resolve()):
        raise ValueError("full checkpoint must be inside this run's checkpoints directory")

    train_config_path = source_run_root / "config" / "train.yaml"
    manifest_path = source_run_root / "config" / "train" / "manifest.json"
    if not train_config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("same-run resolved train config and manifest are required for export")
    train_config = OmegaConf.load(train_config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get("stage") != "train":
        raise ValueError("source training manifest has an unsupported schema or stage")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise ValueError("source training manifest lacks its resolved config")
    if manifest_config.get("resolved_config_sha256") != sha256_file(train_config_path):
        raise ValueError("source training config bytes do not match the immutable run manifest")
    if str(manifest_config.get("run_id")) != run_id:
        raise ValueError("source training manifest run_id differs from requested run_id")
    manifest_train = manifest_config.get("train")
    if not isinstance(manifest_train, dict):
        raise ValueError("source training manifest lacks its resolved train configuration")
    expected_steps = manifest_train.get("max_steps")
    if isinstance(expected_steps, bool) or not isinstance(expected_steps, int):
        raise ValueError("source training manifest lacks an integer train.max_steps")
    expected_seed = manifest_train.get("seed")
    if isinstance(expected_seed, bool) or not isinstance(expected_seed, int) or expected_seed <= 0:
        raise ValueError("source training manifest lacks a positive non-zero train.seed")
    if int(train_config.get("seed", 0)) != expected_seed:
        raise ValueError("resolved training config seed disagrees with the immutable train manifest")

    selected = compose_config(
        [
            f"method={method_group}",
            f"backend={backend}",
            "stage=train",
            f"run_id={run_id}",
        ],
        exp_dir=bundle,
    )
    method = method_identity(selected.method)
    source_method = method_identity(train_config)
    if not _same_method(source_method, method):
        raise ValueError("source training config method does not match selected Hydra method")

    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("generator_format") != "lora_adapter":
        raise ValueError("full checkpoint is not a LoRA adapter checkpoint")
    for key in ("generator", "generator_optimizer", "metadata"):
        if payload.get(key) is None:
            raise ValueError(f"full checkpoint is missing {key}")
    _validate_training_checkpoint_objective(train_config, payload)
    step = _require_final_training_checkpoint(payload, expected_steps)
    seed = _require_checkpoint_seed(payload, expected_seed)
    expected_directory_step = source.parent.name.removeprefix("checkpoint_model_")
    if source.name != "latest.pt" and source.name == "model.pt" and expected_directory_step.isdigit():
        if step != int(expected_directory_step):
            raise ValueError("checkpoint payload step disagrees with its checkpoint directory")
    if payload.get("generator_ema") is None and (not allow_raw_smoke or use_ema):
        raise ValueError("EMA-less export requires --allow-raw-smoke and use_ema=false")
    _check_adapter_scope(payload["generator"], method, "generator")
    if payload.get("generator_ema") is not None:
        _check_adapter_scope(payload["generator_ema"], method, "generator_ema")

    embedded = payload["metadata"].get("temporal_loop") or {}
    for key in (
        "mode",
        "layer_start",
        "layer_end",
        "k_min",
        "k_max",
        "strength",
        "stop_grad_early",
        "schedule",
        "training_enabled",
    ):
        if embedded.get(key) != method[key]:
            raise ValueError(f"full checkpoint temporal_loop.{key} does not match selected method")
    if payload["metadata"].get("lora") != method["lora"]:
        raise ValueError("full checkpoint LoRA config does not match selected method")

    asset_root = resolve_asset_root(selected.assets.root)
    base = asset_path(
        str(selected.assets.generator_checkpoint), "generator checkpoint", asset_root
    )
    if Path(str(train_config.generator_ckpt)).name != base.name:
        raise ValueError("source training config base checkpoint does not match the selected bundle asset")
    recorded_base_hash = (manifest_config.get("assets") or {}).get("generator_checkpoint_sha256")
    actual_base_hash = sha256_file(base)
    if recorded_base_hash != actual_base_hash:
        raise ValueError("selected base checkpoint bytes do not match the source run manifest")
    portable_metadata = portable_base_checkpoint_metadata(
        payload["metadata"],
        base,
        asset_root,
        relative_reference=str(selected.assets.generator_checkpoint),
        verified_sha256=actual_base_hash,
    )
    inference_payload = {
        "checkpoint_version": payload.get("checkpoint_version", 1),
        "generator_format": "lora_adapter",
        "generator": payload["generator"],
        "generator_ema": payload.get("generator_ema"),
        "metadata": portable_metadata,
    }
    provenance = {
        "schema_version": 1,
        "method": method,
        "backend": backend,
        "source_backend": manifest_config.get("backend"),
        "source_run_id": run_id,
        "source_training_step": step,
        "seed": seed,
        "source_full_checkpoint_sha256": sha256_file(source),
        "source_train_config_sha256": sha256_file(train_config_path),
        "source_run_manifest_sha256": sha256_file(manifest_path),
        "base_checkpoint_name": base.name,
        "base_checkpoint_sha256": portable_metadata["base_checkpoint_sha256"],
        "smoke_only_raw_generator": payload.get("generator_ema") is None,
        "mode_evidence": "same-run resolved training YAML and immutable train manifest",
    }
    _atomic_export(destination_path, inference_payload, provenance)
    return destination_path, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", help="SUE bundle root; defaults to SUE_EXP_DIR")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True, help="Hydra method group name")
    parser.add_argument("--backend", choices=("nm5", "autodl"), required=True)
    parser.add_argument("--full-checkpoint", required=True, help="path relative to SUE_EXP_DIR")
    parser.add_argument("--destination", required=True, help="adapter path relative to SUE_EXP_DIR")
    parser.add_argument("--allow-raw-smoke", action="store_true")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    destination, provenance = export_checkpoint(
        exp_dir=args.exp_dir,
        run_id=args.run_id,
        method_group=args.method,
        backend=args.backend,
        full_checkpoint=args.full_checkpoint,
        destination=args.destination,
        allow_raw_smoke=args.allow_raw_smoke,
        use_ema=args.use_ema,
    )
    print(
        f"exported {Path(args.destination).as_posix()}; "
        f"step={provenance['source_training_step']}; method={provenance['method']['mode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
