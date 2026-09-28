from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_module():
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("looped_self_forcing_pipeline.checkpoint")


def _exporter_module():
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("looped_self_forcing_pipeline.export_checkpoint")


def test_export_metadata_rewrites_machine_path_to_verified_relative_path(tmp_path: Path):
    checkpoint = _checkpoint_module()
    source_root = tmp_path / "source"
    base = source_root / "checkpoints" / "base.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"same base model")
    digest = hashlib.sha256(base.read_bytes()).hexdigest()

    portable = checkpoint.portable_base_checkpoint_metadata(
        {"base_checkpoint": "/old/nm5/mount/base.pt"}, base, source_root
    )

    assert portable["base_checkpoint"] == "checkpoints/base.pt"
    assert portable["base_checkpoint_name"] == "base.pt"
    assert portable["base_checkpoint_sha256"] == digest


def test_export_metadata_rejects_a_source_hash_that_disagrees(tmp_path: Path):
    checkpoint = _checkpoint_module()
    source_root = tmp_path / "source"
    base = source_root / "checkpoints" / "base.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base")

    with pytest.raises(ValueError, match="hash|identity"):
        checkpoint.portable_base_checkpoint_metadata(
            {"base_checkpoint_sha256": "0" * 64}, base, source_root
        )


def test_export_metadata_keeps_lexical_path_for_external_symlink(tmp_path: Path):
    checkpoint = _checkpoint_module()
    asset_root = tmp_path / "asset-root"
    outside = tmp_path / "shared/checkpoint.pt"
    lexical = asset_root / "checkpoints/base.pt"
    outside.parent.mkdir(parents=True)
    lexical.parent.mkdir(parents=True)
    outside.write_bytes(b"base model")
    lexical.symlink_to(outside)

    metadata = checkpoint.portable_base_checkpoint_metadata(
        {"base_checkpoint_sha256": hashlib.sha256(outside.read_bytes()).hexdigest()},
        outside,
        asset_root,
        relative_reference="checkpoints/base.pt",
    )

    assert metadata["base_checkpoint"] == "checkpoints/base.pt"
    assert metadata["base_checkpoint_sha256"] == hashlib.sha256(outside.read_bytes()).hexdigest()


def test_export_metadata_reuses_preverified_base_digest(tmp_path: Path, monkeypatch):
    checkpoint = _checkpoint_module()
    asset_root = tmp_path / "asset-root"
    base = asset_root / "checkpoints/base.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base weights")
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    monkeypatch.setattr(
        checkpoint,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("base was rehashed")),
    )

    metadata = checkpoint.portable_base_checkpoint_metadata(
        {}, base, asset_root, relative_reference="checkpoints/base.pt", verified_sha256=digest
    )

    assert metadata["base_checkpoint_sha256"] == digest


def test_inference_payload_uses_portable_base_reference_and_keeps_adapter_state(tmp_path: Path):
    torch = pytest.importorskip("torch")
    exporter = _exporter_module()
    source_root = tmp_path / "source"
    base = source_root / "checkpoints" / "base.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base model")
    generator = {"adapter": torch.tensor([1.0, 2.0])}
    full = {
        "checkpoint_version": 1,
        "generator_format": "lora_adapter",
        "generator": generator,
        "generator_ema": {"adapter": torch.tensor([3.0, 4.0])},
        "metadata": {"base_checkpoint": "/old/nm5/base.pt", "step": 600},
    }

    payload = exporter.build_inference_payload(
        full, base_checkpoint=base, source_root=source_root, use_ema=True
    )

    assert payload["generator"] is generator
    assert payload["metadata"]["base_checkpoint"] == "checkpoints/base.pt"
    assert payload["metadata"]["base_checkpoint_sha256"] == hashlib.sha256(
        base.read_bytes()
    ).hexdigest()


def test_inference_payload_rejects_ema_less_checkpoint_for_production(tmp_path: Path):
    torch = pytest.importorskip("torch")
    exporter = _exporter_module()
    source_root = tmp_path / "source"
    base = source_root / "checkpoints" / "base.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base model")
    full = {
        "generator_format": "lora_adapter",
        "generator": {"adapter": torch.tensor([1.0])},
        "generator_ema": None,
        "metadata": {},
    }

    with pytest.raises(ValueError, match="EMA"):
        exporter.build_inference_payload(
            full, base_checkpoint=base, source_root=source_root, use_ema=True
        )


def test_exporter_rejects_partial_checkpoint_even_when_raw_smoke_is_allowed():
    exporter = _exporter_module()

    with pytest.raises(ValueError, match="partial|final step"):
        exporter._require_final_training_checkpoint(
            {"metadata": {"step": 100, "final": False}}, expected_steps=600
        )


def test_exporter_requires_final_metadata_at_the_manifest_target_step():
    exporter = _exporter_module()

    with pytest.raises(ValueError, match="final=true"):
        exporter._require_final_training_checkpoint(
            {"metadata": {"step": 600, "final": False}}, expected_steps=600
        )

    assert exporter._require_final_training_checkpoint(
        {"metadata": {"step": 600, "final": True}}, expected_steps=600
    ) == 600


def test_exporter_rejects_checkpoint_seed_that_disagrees_with_train_manifest():
    exporter = _exporter_module()

    with pytest.raises(ValueError, match="seed.*manifest|manifest.*seed"):
        exporter._require_checkpoint_seed(
            {"metadata": {"seed": 7}}, expected_seed=8
        )

    assert exporter._require_checkpoint_seed(
        {"metadata": {"seed": 8}}, expected_seed=8
    ) == 8
