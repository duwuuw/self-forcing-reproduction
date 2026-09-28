from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "scripts" / "looped_self_forcing_pipeline"


def _module(name: str):
    module_path = PIPELINE_DIR / f"{name}.py"
    if not module_path.is_file():
        pytest.fail(f"pipeline helper is missing: {module_path}")
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(f"looped_self_forcing_pipeline.{name}")


def test_manifest_is_materialized_once_and_rejects_changed_configuration(tmp_path: Path):
    manifest_module = _module("manifest")
    run_root = tmp_path / "run-1"
    config = {
        "method": {"mode": "layer", "layer_start": 16, "layer_end": 23},
        "seed": 7,
        "checkpoint": "/models/adapter.pt",
    }

    saved = manifest_module.materialize_manifest(run_root, "infer", config)
    manifest_path = run_root / "manifest.json"
    original_bytes = manifest_path.read_bytes()

    assert saved["stage"] == "infer"
    assert saved["config"] == config
    assert json.loads(original_bytes) == saved
    assert manifest_module.materialize_manifest(run_root, "infer", dict(config)) == saved
    assert manifest_path.read_bytes() == original_bytes

    with pytest.raises(ValueError, match="run_id|manifest|configuration"):
        manifest_module.materialize_manifest(run_root, "infer", {**config, "seed": 8})

    assert manifest_path.read_bytes() == original_bytes


def test_manifest_rejects_reusing_run_root_for_a_different_stage(tmp_path: Path):
    manifest_module = _module("manifest")
    run_root = tmp_path / "run-1"
    manifest_module.materialize_manifest(run_root, "train", {"seed": 7})

    with pytest.raises(ValueError, match="run_id|manifest|configuration"):
        manifest_module.materialize_manifest(run_root, "infer", {"seed": 7})


def test_manifest_normalizes_paths_to_portable_json_values(tmp_path: Path):
    manifest_module = _module("manifest")
    input_path = tmp_path / "prompts.txt"

    saved = manifest_module.materialize_manifest(
        tmp_path / "run-1", "infer", {"prompt_file": input_path}
    )

    assert saved["config"]["prompt_file"] == str(input_path)


def test_sha256_file_matches_the_standard_digest(tmp_path: Path):
    checkpoint_module = _module("checkpoint")
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base checkpoint bytes")

    assert checkpoint_module.sha256_file(checkpoint) == (
        "4fdcf211ef0ae464c94ec806eda7ea4f49fb58e09eb3adcd883732cabac080d7"
    )


def test_base_checkpoint_resolution_uses_active_path_and_checks_identity(
    tmp_path: Path,
):
    checkpoint_module = _module("checkpoint")
    configured_base = tmp_path / "base.pt"
    configured_base.write_bytes(b"base checkpoint bytes")
    metadata = {
        "base_checkpoint": "/old/backend/mount/base.pt",
        "base_checkpoint_name": "base.pt",
        "base_checkpoint_sha256": (
            "4fdcf211ef0ae464c94ec806eda7ea4f49fb58e09eb3adcd883732cabac080d7"
        ),
    }

    assert checkpoint_module.resolve_base_checkpoint(metadata, configured_base) == (
        configured_base.resolve()
    )

    configured_base.write_bytes(b"different model bytes")
    with pytest.raises(ValueError, match="hash|identity"):
        checkpoint_module.resolve_base_checkpoint(metadata, configured_base)


def test_base_checkpoint_resolution_requires_a_recorded_hash(tmp_path: Path):
    checkpoint_module = _module("checkpoint")
    configured_base = tmp_path / "base.pt"
    configured_base.write_bytes(b"base checkpoint bytes")

    with pytest.raises(ValueError, match="hash|identity"):
        checkpoint_module.resolve_base_checkpoint(
            {"base_checkpoint": "/old/backend/base.pt"}, configured_base
        )


def test_base_identity_helper_returns_verified_hash_for_downstream_manifests(
    tmp_path: Path,
):
    checkpoint_module = _module("checkpoint")
    configured_base = tmp_path / "base.pt"
    configured_base.write_bytes(b"base checkpoint bytes")
    metadata = {
        "base_checkpoint_sha256": checkpoint_module.sha256_file(configured_base)
    }

    path, digest = checkpoint_module.resolve_base_checkpoint_with_hash(
        metadata, configured_base
    )

    assert path == configured_base.resolve()
    assert digest == metadata["base_checkpoint_sha256"]
