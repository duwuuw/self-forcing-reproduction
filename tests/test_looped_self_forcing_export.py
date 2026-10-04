from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _export_case(
    tmp_path: Path,
    monkeypatch,
    *,
    trainer: str | None,
    objective: str | None,
    critic,
    critic_optimizer,
    mutate_payload=None,
):
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf

    exporter = _exporter_module()
    bundle = tmp_path / "bundle"
    run_id = "run-1"
    artifacts_root = bundle / "artifacts"
    checkpoint_root = bundle / "checkpoints"
    source_run = artifacts_root / run_id
    checkpoint_dir = checkpoint_root / run_id / "checkpoint_model_000002"
    checkpoint_dir.mkdir(parents=True)

    asset_root = tmp_path / "assets"
    base = asset_root / "checkpoints" / "base.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base checkpoint bytes")
    base_hash = hashlib.sha256(base.read_bytes()).hexdigest()

    temporal_loop = {
        "enabled": True,
        "training_enabled": True,
        "mode": "layer",
        "layer_start": 0,
        "layer_end": 0,
        "k_min": 2,
        "k_max": 2,
        "strength": 1.0,
        "stop_grad_early": False,
        "schedule": "fixed",
    }
    lora = {
        "enabled": True,
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "target_modules": ["self_attn.q", "self_attn.v"],
    }
    method = {"temporal_loop": temporal_loop, "lora": lora}
    selected = SimpleNamespace(
        method=method,
        assets=SimpleNamespace(
            root=str(asset_root),
            generator_checkpoint="checkpoints/base.pt",
        ),
    )
    monkeypatch.setattr(
        exporter,
        "resolve_runtime_paths",
        lambda _bundle: {
            "artifacts_root": artifacts_root,
            "ckpt_root": checkpoint_root,
        },
    )
    monkeypatch.setattr(
        exporter,
        "compose_config",
        lambda *_args, **_kwargs: selected,
    )

    train_config = {
        "seed": 17,
        "generator_ckpt": "/old/backend/checkpoints/base.pt",
        "temporal_loop": temporal_loop,
        "lora": lora,
    }
    if trainer is not None:
        train_config["trainer"] = trainer
    train_config_path = source_run / "config" / "train.yaml"
    train_config_path.parent.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(train_config), train_config_path)
    manifest_config = {
        "run_id": run_id,
        "resolved_config_sha256": exporter.sha256_file(train_config_path),
        "backend": "nm5",
        "train": {"max_steps": 2, "seed": 17},
        "assets": {"generator_checkpoint_sha256": base_hash},
    }
    manifest_path = source_run / "config" / "train" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "stage": "train", "config": manifest_config}),
        encoding="utf-8",
    )

    generator_keys = (
        "blocks.0.self_attn.q.lora_A.weight",
        "blocks.0.self_attn.q.lora_B.weight",
        "blocks.0.self_attn.v.lora_A.weight",
        "blocks.0.self_attn.v.lora_B.weight",
    )
    metadata = {
        "step": 2,
        "seed": 17,
        "final": True,
        "training_objective": objective,
        "temporal_loop": {
            "mode": "layer",
            "layer_start": 0,
            "layer_end": 0,
            "k_min": 2,
            "k_max": 2,
            "strength": 1.0,
            "stop_grad_early": False,
            "schedule": "fixed",
            "training_enabled": True,
        },
        "lora": lora,
        "base_checkpoint": "checkpoints/base.pt",
        "base_checkpoint_sha256": base_hash,
    }
    if objective is None:
        metadata.pop("training_objective")
    payload = {
        "checkpoint_version": 1,
        "generator_format": "lora_adapter",
        "generator": {key: torch.ones(1) for key in generator_keys},
        "generator_ema": {key: torch.ones(1) for key in generator_keys},
        "generator_optimizer": {},
        "critic": critic,
        "critic_optimizer": critic_optimizer,
        "metadata": metadata,
    }
    if mutate_payload is not None:
        mutate_payload(payload)
    full_checkpoint = checkpoint_dir / "model.pt"
    torch.save(payload, full_checkpoint)

    return exporter, {
        "exp_dir": bundle,
        "run_id": run_id,
        "method_group": "test_method",
        "backend": "nm5",
        "full_checkpoint": full_checkpoint.relative_to(bundle).as_posix(),
        "destination": f"artifacts/{run_id}/inference/adapter.pt",
    }, bundle / f"artifacts/{run_id}/inference/adapter.pt"


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


def test_supervised_export_accepts_no_critic_and_strips_resume_state(
    tmp_path: Path, monkeypatch
):
    torch = pytest.importorskip("torch")
    exporter, export_args, destination = _export_case(
        tmp_path,
        monkeypatch,
        trainer="diffusion",
        objective="supervised_flow_matching",
        critic=None,
        critic_optimizer=None,
    )

    exported_path, _provenance = exporter.export_checkpoint(**export_args)
    payload = torch.load(exported_path, map_location="cpu", weights_only=True)

    assert exported_path == destination
    assert payload["metadata"]["training_objective"] == "supervised_flow_matching"
    assert set(payload) == {
        "checkpoint_version",
        "generator_format",
        "generator",
        "generator_ema",
        "metadata",
    }
    assert "critic" not in payload
    assert "critic_optimizer" not in payload
    assert "generator_optimizer" not in payload


@pytest.mark.parametrize(
    ("critic", "critic_optimizer"),
    [(None, {}), ({}, None)],
)
def test_supervised_export_rejects_one_sided_critic_state(
    tmp_path: Path, monkeypatch, critic, critic_optimizer
):
    exporter, export_args, _destination = _export_case(
        tmp_path,
        monkeypatch,
        trainer="diffusion",
        objective="supervised_flow_matching",
        critic=critic,
        critic_optimizer=critic_optimizer,
    )

    with pytest.raises(ValueError, match="critic"):
        exporter.export_checkpoint(**export_args)


@pytest.mark.parametrize(
    "missing_fields",
    [("critic",), ("critic_optimizer",), ("critic", "critic_optimizer")],
)
def test_supervised_export_requires_explicit_absent_critic_fields(
    tmp_path: Path, monkeypatch, missing_fields
):
    def remove_fields(payload):
        for key in missing_fields:
            payload.pop(key)

    exporter, export_args, _destination = _export_case(
        tmp_path,
        monkeypatch,
        trainer="diffusion",
        objective="supervised_flow_matching",
        critic=None,
        critic_optimizer=None,
        mutate_payload=remove_fields,
    )

    with pytest.raises(ValueError, match="critic"):
        exporter.export_checkpoint(**export_args)


@pytest.mark.parametrize("trainer", ["score_distillation", None])
def test_legacy_dmd_export_still_requires_and_strips_critic_state(
    tmp_path: Path, monkeypatch, trainer
):
    torch = pytest.importorskip("torch")
    exporter, export_args, _destination = _export_case(
        tmp_path,
        monkeypatch,
        trainer=trainer,
        objective=None,
        critic={},
        critic_optimizer={},
    )

    exported_path, _provenance = exporter.export_checkpoint(**export_args)
    payload = torch.load(exported_path, map_location="cpu", weights_only=True)

    assert "training_objective" not in payload["metadata"]
    assert "critic" not in payload
    assert "critic_optimizer" not in payload
    assert "generator_optimizer" not in payload


@pytest.mark.parametrize(
    ("critic", "critic_optimizer"),
    [(None, {}), ({}, None), (None, None)],
)
def test_legacy_dmd_export_rejects_missing_critic_state(
    tmp_path: Path, monkeypatch, critic, critic_optimizer
):
    exporter, export_args, _destination = _export_case(
        tmp_path,
        monkeypatch,
        trainer=None,
        objective=None,
        critic=critic,
        critic_optimizer=critic_optimizer,
    )

    with pytest.raises(ValueError, match="critic"):
        exporter.export_checkpoint(**export_args)


@pytest.mark.parametrize(
    ("trainer", "objective"),
    [
        ("diffusion", "dmd"),
        ("score_distillation", "supervised_flow_matching"),
    ],
)
def test_export_rejects_trainer_objective_mismatch(
    tmp_path: Path, monkeypatch, trainer, objective
):
    exporter, export_args, _destination = _export_case(
        tmp_path,
        monkeypatch,
        trainer=trainer,
        objective=objective,
        critic={},
        critic_optimizer={},
    )

    with pytest.raises(ValueError, match="trainer|objective"):
        exporter.export_checkpoint(**export_args)


@pytest.mark.parametrize(
    "corruption",
    [
        "final",
        "seed",
        "ema",
        "adapter_scope",
        "loop",
        "lora",
        "base_hash",
    ],
)
def test_objective_aware_export_keeps_existing_checkpoint_guards(
    tmp_path: Path, monkeypatch, corruption
):
    torch = pytest.importorskip("torch")

    def corrupt(payload):
        metadata = payload["metadata"]
        if corruption == "final":
            metadata["final"] = False
        elif corruption == "seed":
            metadata["seed"] = 18
        elif corruption == "ema":
            payload["generator_ema"] = None
        elif corruption == "adapter_scope":
            payload["generator"]["blocks.1.self_attn.q.lora_A.weight"] = torch.ones(1)
        elif corruption == "loop":
            metadata["temporal_loop"]["k_max"] = 3
        elif corruption == "lora":
            metadata["lora"]["rank"] = 4
        elif corruption == "base_hash":
            metadata["base_checkpoint_sha256"] = "0" * 64

    exporter, export_args, _destination = _export_case(
        tmp_path,
        monkeypatch,
        trainer="score_distillation",
        objective=None,
        critic={},
        critic_optimizer={},
        mutate_payload=corrupt,
    )

    with pytest.raises(ValueError):
        exporter.export_checkpoint(**export_args)


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
