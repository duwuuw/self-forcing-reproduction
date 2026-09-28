from __future__ import annotations

import importlib
import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module():
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("looped_self_forcing_pipeline.training_policy")


def test_checkpoint_policy_uses_wall_clock_and_exact_final_step():
    policy = _module()

    assert not policy.checkpoint_due(100, 100.0, 3599.0, max_steps=600, interval_seconds=3600)
    assert policy.checkpoint_due(100, 100.0, 3700.0, max_steps=600, interval_seconds=3600)
    assert policy.checkpoint_due(600, 3500.0, 3501.0, max_steps=600, interval_seconds=3600)


def test_log_policy_keeps_first_final_and_periodic_steps():
    policy = _module()

    assert policy.should_log_scalar(1, max_steps=600, interval=100)
    assert policy.should_log_scalar(100, max_steps=600, interval=100)
    assert policy.should_log_scalar(600, max_steps=600, interval=100)
    assert not policy.should_log_scalar(50, max_steps=600, interval=100)
    assert policy.should_log_progress(1, interval=10)
    assert policy.should_log_progress(10, interval=10)
    assert not policy.should_log_progress(9, interval=10)


def test_checkpoint_metadata_path_is_relative_to_external_asset_root(tmp_path: Path):
    policy = _module()
    asset_root = tmp_path / "asset-root"
    checkpoint = asset_root / "checkpoints" / "base.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"base")

    assert policy.portable_checkpoint_reference(
        checkpoint, asset_root, relative_path="checkpoints/base.pt"
    ) == (
        "checkpoints/base.pt"
    )


def test_checkpoint_metadata_preserves_lexical_name_for_external_symlink(tmp_path: Path):
    policy = _module()
    asset_root = tmp_path / "asset-root"
    outside = tmp_path / "shared-models/base-weights.pt"
    lexical = asset_root / "checkpoints/base.pt"
    outside.parent.mkdir(parents=True)
    lexical.parent.mkdir(parents=True)
    outside.write_bytes(b"external base weights")
    lexical.symlink_to(outside)

    assert policy.portable_checkpoint_reference(
        outside,
        asset_root,
        relative_path="checkpoints/base.pt",
    ) == "checkpoints/base.pt"


def test_training_checkpoint_metadata_keeps_asset_reference_seed_and_hash(
    tmp_path: Path, monkeypatch
):
    policy = _module()
    asset_root = tmp_path / "asset-root"
    outside = tmp_path / "shared-models/base.pt"
    lexical = asset_root / "checkpoints/base.pt"
    outside.parent.mkdir(parents=True)
    lexical.parent.mkdir(parents=True)
    outside.write_bytes(b"base weights")
    lexical.symlink_to(outside)
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_RELATIVE_PATH", "checkpoints/base.pt")
    base_hash = hashlib.sha256(outside.read_bytes()).hexdigest()
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_SHA256", base_hash)

    class Trainer:
        def save(self):
            pass

        def fwdbwd_one_step(self, batch, train_generator, *args, **kwargs):
            return None

        def _lora_checkpoint_metadata(self):
            return {"step": 12}

    trainer = Trainer()
    trainer.config = SimpleNamespace(
        generator_ckpt=str(outside),
        seed=17,
        temporal_loop=SimpleNamespace(
            mode="layer",
            stop_grad_early=True,
            training_enabled=True,
        ),
    )
    trainer.step = 12
    wandb = SimpleNamespace(
        init=lambda **_kwargs: None,
        log=lambda *_args, **_kwargs: None,
        run=None,
    )
    policy.install_training_policy(
        Trainer,
        wandb,
        max_steps=20,
        asset_root=asset_root,
    )

    metadata = trainer._lora_checkpoint_metadata()

    assert metadata["base_checkpoint"] == "checkpoints/base.pt"
    assert metadata["base_checkpoint_sha256"] == base_hash
    assert metadata["seed"] == 17
    assert metadata["temporal_loop"]["mode"] == "layer"


def test_wandb_init_persists_the_actual_run_id_for_the_parent_worker(
    tmp_path: Path, monkeypatch
):
    policy = _module()
    run_id_path = tmp_path / "run/wandb_run_id.txt"
    monkeypatch.setenv("SUE_WANDB_RUN_ID_PATH", str(run_id_path))

    class Trainer:
        def save(self):
            pass

        def fwdbwd_one_step(self, batch, train_generator, *args, **kwargs):
            return None

    wandb = SimpleNamespace(
        init=lambda **_kwargs: SimpleNamespace(id="actual-online-id"),
        log=lambda *_args, **_kwargs: None,
        run=None,
    )
    policy.install_training_policy(
        Trainer,
        wandb,
        max_steps=10,
        asset_root=tmp_path,
    )

    run = wandb.init(project="experiment")

    assert run.id == "actual-online-id"
    assert run_id_path.read_text(encoding="utf-8") == "actual-online-id\n"


def test_wandb_config_redacts_backend_paths_but_keeps_execution_values(
    tmp_path: Path, monkeypatch
):
    policy = _module()
    observed = {}
    monkeypatch.setenv("SUE_BACKEND", "nm5")
    monkeypatch.setenv("SUE_RUN_ID", "run-1")
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_EXPERIMENT_NAME", "layer-profile-run-1-train")
    monkeypatch.setenv("WANDB_PROJECT", "comparison-project")

    class Trainer:
        def save(self):
            pass

        def fwdbwd_one_step(self, batch, train_generator, *args, **kwargs):
            return None

    def init(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(id="run-id")

    wandb = SimpleNamespace(init=init, log=lambda *_args, **_kwargs: None, run=None)
    policy.install_training_policy(
        Trainer,
        wandb,
        max_steps=10,
        asset_root=tmp_path,
    )
    private_root = "/private/backend/mount"
    run_config = {
        "generator_ckpt": f"{private_root}/checkpoints/base.pt",
        "teacher_checkpoint": f"{private_root}/wan_models/teacher",
        "data_path": f"{private_root}/datasets/prompts.txt",
        "logdir": f"{private_root}/runs/run-1",
        "wandb_key": "private-api-key-value",
        "wandb_host": "https://private-wandb.example",
        "nested": {"wandb_save_dir": f"{private_root}/runs/wandb"},
        "max_steps": 10,
    }

    wandb.init(config=run_config)

    assert observed["config"] == {
        "generator_ckpt": "base.pt",
        "teacher_checkpoint": "teacher",
        "data_path": "prompts.txt",
        "logdir": "run-1",
        "wandb_key": "<redacted>",
        "wandb_host": "<redacted>",
        "nested": {"wandb_save_dir": "wandb"},
        "max_steps": 10,
        "train": {
            "max_steps": 10,
            "checkpoint_interval_seconds": 3600,
            "scalar_log_interval": 100,
            "progress_log_interval": 10,
        },
        "sue_runtime": {
            "backend": "nm5",
            "run_id": "run-1",
            "stage": "train",
            "experiment_name": "layer-profile-run-1-train",
            "tracking_mode": "offline",
            "project": "comparison-project",
        },
    }
    assert private_root not in repr(observed["config"])
    assert "private-api-key-value" not in repr(observed["config"])
    assert run_config["generator_ckpt"].startswith(private_root)


def test_wandb_config_includes_manifest_training_scale_and_runtime_identity(
    tmp_path: Path, monkeypatch
):
    policy = _module()
    observed = {}
    monkeypatch.setenv("SUE_BACKEND", "nm5")
    monkeypatch.setenv("SUE_RUN_ID", "pair-k3-lr5gen")
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    monkeypatch.setenv(
        "WANDB_EXPERIMENT_NAME", "layerwise_l23_30_k3_lr5gen-pair-k3-lr5gen-train"
    )
    monkeypatch.setenv("WANDB_PROJECT", "looped-self-forcing-lora-dmd")

    class Trainer:
        def save(self):
            pass

        def fwdbwd_one_step(self, batch, train_generator, *args, **kwargs):
            return None

    def init(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(id="run-id")

    wandb = SimpleNamespace(init=init, log=lambda *_args, **_kwargs: None, run=None)
    policy.install_training_policy(
        Trainer,
        wandb,
        max_steps=37,
        checkpoint_interval_seconds=180,
        scalar_log_interval=100,
        progress_log_interval=10,
        asset_root=tmp_path,
    )
    run_config = {
        "seed": 1,
        "log_iters": 5,
        "lr": 4e-7,
        "lr_critic": 4e-7,
        "distribution_loss": "dmd",
        "temporal_loop": {"layer_start": 22, "layer_end": 29, "k_min": 3, "k_max": 3},
        "lora": {"rank": 8},
    }

    wandb.init(config=run_config)
    saved = observed["config"]

    assert saved["train"]["max_steps"] == 37
    assert saved["train"]["seed"] == 1
    assert saved["train"]["log_iters"] == 5
    assert saved["train"]["checkpoint_interval_seconds"] == 180
    assert saved["lr"] == 4e-7 and saved["lr_critic"] == 4e-7
    assert saved["distribution_loss"] == "dmd"
    assert saved["sue_runtime"] == {
        "backend": "nm5",
        "run_id": "pair-k3-lr5gen",
        "stage": "train",
        "experiment_name": "layerwise_l23_30_k3_lr5gen-pair-k3-lr5gen-train",
        "tracking_mode": "offline",
        "project": "looped-self-forcing-lora-dmd",
    }
