from __future__ import annotations

import importlib
import hashlib
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest


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
            "git_commit": "unknown",
            "git_dirty": "unknown",
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
    monkeypatch.setenv("SUE_GIT_COMMIT", "abc123def456")
    monkeypatch.setenv("SUE_GIT_DIRTY", "clean")
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
        "git_commit": "abc123def456",
        "git_dirty": "clean",
    }


def test_diffusion_sue_wandb_runtime_overrides_native_online_and_identity(
    monkeypatch,
):
    policy = _module()
    observed = {}
    login_calls = []
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_PROJECT", "comparison-project")
    monkeypatch.setenv("WANDB_EXPERIMENT_NAME", "loop-run-train")
    monkeypatch.setenv("WANDB_GROUP", "loop-run")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    def init(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(id="offline-run-id")

    wandb = SimpleNamespace(init=init, login=lambda **kwargs: login_calls.append(kwargs))
    policy.install_sue_wandb_runtime(wandb)

    wandb.login(host="WANDB_HOST", key="WANDB_API_KEY")
    run = wandb.init(
        mode="online",
        name="WANDB_NAME",
        entity="WANDB_ENTITY",
        project="WANDB_PROJECT",
        config={
            "config_name": "WANDB_NAME",
            "wandb_entity": "WANDB_ENTITY",
            "wandb_project": "WANDB_PROJECT",
        },
        host="WANDB_HOST",
        key="WANDB_API_KEY",
    )

    assert run.id == "offline-run-id"
    assert login_calls == []
    assert observed["mode"] == "offline"
    assert observed["name"] == "loop-run-train"
    assert observed["project"] == "comparison-project"
    assert observed["group"] == "loop-run"
    assert "entity" not in observed
    assert "host" not in observed and "key" not in observed
    assert observed["config"] == {
        "config_name": "loop-run-train",
        "wandb_entity": "",
        "wandb_project": "comparison-project",
    }


@pytest.mark.parametrize(
    ("mode", "api_key", "entity", "error"),
    [
        ("offline", "", "", None),
        ("online", "", "team", "WANDB_API_KEY"),
        ("online", "key", "", "WANDB_ENTITY"),
        ("other", "key", "team", "SUE_WANDB_MODE"),
    ],
)
def test_sue_wandb_validation_requires_online_credentials_and_supported_mode(
    monkeypatch, mode, api_key, entity, error
):
    policy = _module()
    monkeypatch.setenv("SUE_WANDB_MODE", mode)
    if api_key:
        monkeypatch.setenv("WANDB_API_KEY", api_key)
    else:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
    if entity:
        monkeypatch.setenv("WANDB_ENTITY", entity)
    else:
        monkeypatch.delenv("WANDB_ENTITY", raising=False)

    if error:
        with pytest.raises(ValueError, match=error):
            policy.validate_sue_wandb_environment()
    else:
        assert policy.validate_sue_wandb_environment() == "offline"


def _install_diffusion_policy(policy, monkeypatch, tmp_path, *, max_steps=1000,
                              interval=3600, clock=None, events=None):
    shared_events = events if events is not None else []
    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    torch_module.cuda = SimpleNamespace(
        empty_cache=lambda: shared_events.append(("empty_cache",))
    )
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    asset_root = tmp_path / "assets"
    base_checkpoint = asset_root / "checkpoints" / "base.pt"
    base_checkpoint.parent.mkdir(parents=True)
    base_checkpoint.write_bytes(b"base weights")
    digest = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_RELATIVE_PATH", "checkpoints/base.pt")
    monkeypatch.setenv("SUE_BASE_CHECKPOINT_SHA256", digest)
    monkeypatch.setenv("SUE_WANDB_RUN_ID_PATH", str(tmp_path / "wandb_run_id.txt"))

    class DiffusionTrainer:
        def __init__(self):
            self.step = 0
            self.output_path = str(tmp_path / "checkpoints")
            self.is_main_process = True
            self.config = SimpleNamespace(
                generator_ckpt=str(base_checkpoint),
                seed=9,
                batch_size=2,
                temporal_loop=SimpleNamespace(
                    mode="layer", stop_grad_early=False, training_enabled=True
                ),
            )
            self.events = []
            self.save_error = None

        def train_one_step(self, batch):
            self.step += 1
            self.events.append(("step", self.step))
            return batch

        def save(self):
            self.events.append(("native_save", self.step))
            shared_events.append(("native_save", self.step))
            if self.save_error is not None:
                raise self.save_error
            checkpoint = (
                Path(self.output_path)
                / f"checkpoint_model_{self.step:06d}"
                / "model.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"adapter")

        def _lora_checkpoint_metadata(self):
            return {
                "step": self.step,
                "temporal_loop": {"layer_start": 3, "layer_end": 4},
            }

    wandb = SimpleNamespace(
        init=lambda **_kwargs: SimpleNamespace(id="diffusion-run-id"),
        login=lambda **_kwargs: None,
        log=lambda *_args, **_kwargs: None,
        run=SimpleNamespace(summary={}),
    )
    policy.install_diffusion_training_policy(
        DiffusionTrainer,
        wandb,
        max_steps=max_steps,
        checkpoint_interval_seconds=interval,
        scalar_log_interval=100,
        progress_log_interval=10,
        asset_root=asset_root,
        clock=clock or (lambda: 0.0),
    )
    return DiffusionTrainer, wandb, asset_root


def test_diffusion_policy_checks_each_step_and_saves_hourly_before_log_iters(
    monkeypatch, tmp_path
):
    policy = _module()
    times = iter((0.0, 3599.0, 3600.0))
    cache_events = []
    trainer_class, _wandb, _asset_root = _install_diffusion_policy(
        policy,
        monkeypatch,
        tmp_path,
        max_steps=1000,
        clock=lambda: next(times),
        events=cache_events,
    )
    trainer = trainer_class()
    decisions = []
    original_decision = policy._distributed_decision
    monkeypatch.setattr(
        policy,
        "_distributed_decision",
        lambda due: decisions.append(due) or original_decision(due),
    )

    trainer.step = 1
    assert trainer._sue_checkpoint_callback(trainer) is False
    trainer.step = 50
    assert trainer._sue_checkpoint_callback(trainer) is False
    trainer.step = 51
    assert trainer._sue_checkpoint_callback(trainer) is False

    assert decisions == [False, False, True]
    assert cache_events == [
        ("empty_cache",),
        ("native_save", 51),
        ("empty_cache",),
    ]
    assert trainer.events == [("native_save", 51)]
    latest = Path(trainer.output_path) / "latest.pt"
    assert latest.is_symlink()
    assert latest.resolve() == (
        Path(trainer.output_path) / "checkpoint_model_000051" / "model.pt"
    )


def test_diffusion_policy_final_step_saves_and_adds_exporter_metadata(
    monkeypatch, tmp_path
):
    policy = _module()
    times = iter((0.0, 1.0))
    trainer_class, wandb, _asset_root = _install_diffusion_policy(
        policy, monkeypatch, tmp_path, max_steps=7, interval=3600,
        clock=lambda: next(times)
    )
    trainer = trainer_class()
    trainer.step = 6
    assert trainer._sue_checkpoint_callback(trainer) is False
    trainer.step = 7

    assert trainer._sue_checkpoint_callback(trainer) is True
    assert trainer.events == [("native_save", 7)]
    metadata = trainer._lora_checkpoint_metadata()
    assert metadata["base_checkpoint"] == "checkpoints/base.pt"
    assert metadata["base_checkpoint_sha256"] == hashlib.sha256(
        (tmp_path / "assets" / "checkpoints" / "base.pt").read_bytes()
    ).hexdigest()
    assert metadata["seed"] == 9
    assert metadata["final"] is True
    assert metadata["checkpoint_interval_seconds"] == 3600
    assert wandb.run.summary["checkpoint_step"] == 7
    assert wandb.run.summary["final"] is True


def test_diffusion_policy_syncs_save_and_publication_failures_before_return(
    monkeypatch, tmp_path
):
    policy = _module()
    trainer_class, _wandb, _asset_root = _install_diffusion_policy(
        policy, monkeypatch, tmp_path, max_steps=1, interval=1, clock=lambda: 0.0
    )
    trainer = trainer_class()
    statuses = iter((False,))
    monkeypatch.setattr(policy, "_distributed_all_success", lambda _ok: next(statuses))
    trainer.save_error = OSError("disk full")
    trainer.step = 1

    with pytest.raises(RuntimeError, match="save failed"):
        trainer._sue_checkpoint_callback(trainer)

    trainer_class, _wandb, _asset_root = _install_diffusion_policy(
        policy, monkeypatch, tmp_path / "publish", max_steps=1,
        interval=1, clock=lambda: 0.0
    )
    trainer = trainer_class()
    statuses = iter((True, False))
    monkeypatch.setattr(policy, "_distributed_all_success", lambda _ok: next(statuses))
    monkeypatch.setattr(
        policy, "_publish_latest", lambda *_args: (_ for _ in ()).throw(OSError("rename failed"))
    )
    trainer.step = 1

    with pytest.raises(RuntimeError, match="publication failed"):
        trainer._sue_checkpoint_callback(trainer)


def test_diffusion_policy_orders_rank_wide_save_and_latest_publication(
    monkeypatch, tmp_path
):
    policy = _module()
    events = []
    trainer_class, _wandb, _asset_root = _install_diffusion_policy(
        policy,
        monkeypatch,
        tmp_path,
        max_steps=1,
        interval=3600,
        clock=lambda: 0.0,
        events=events,
    )
    trainer = trainer_class()
    monkeypatch.setattr(
        policy,
        "_distributed_decision",
        lambda due: events.append(("due_broadcast", due)) or due,
    )
    save_statuses = iter((True, True))
    monkeypatch.setattr(
        policy,
        "_distributed_all_success",
        lambda success: events.append(("success_reduce", success)) or next(save_statuses),
    )
    monkeypatch.setattr(
        policy,
        "_publish_latest",
        lambda *_args: events.append(("publish_latest",)),
    )
    trainer.step = 1

    assert trainer._sue_checkpoint_callback(trainer) is True

    assert events == [
        ("due_broadcast", True),
        ("empty_cache",),
        ("native_save", 1),
        ("empty_cache",),
        ("success_reduce", True),
        ("publish_latest",),
        ("success_reduce", True),
    ]


def test_diffusion_wandb_policy_wraps_sue_identity_and_persists_run_id(
    monkeypatch, tmp_path
):
    policy = _module()
    observed = {}
    run_id_path = tmp_path / "run" / "wandb_run_id.txt"
    base_checkpoint = tmp_path / "assets" / "base.pt"
    base_checkpoint.parent.mkdir(parents=True)
    base_checkpoint.write_bytes(b"base")
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_PROJECT", "comparison-project")
    monkeypatch.setenv("WANDB_EXPERIMENT_NAME", "supervised-run-train")
    monkeypatch.setenv("WANDB_GROUP", "supervised-run")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("SUE_WANDB_RUN_ID_PATH", str(run_id_path))

    class Trainer:
        def save(self):
            pass

        def train_one_step(self, _batch):
            self.step += 1

        def _lora_checkpoint_metadata(self):
            return {"temporal_loop": {}}

    def init(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(id="sue-offline-id")

    login_calls = []
    wandb = SimpleNamespace(
        init=init,
        login=lambda **kwargs: login_calls.append(kwargs),
        log=lambda *_args, **_kwargs: None,
        run=None,
    )
    policy.install_sue_wandb_runtime(wandb)
    policy.install_diffusion_training_policy(
        Trainer,
        wandb,
        max_steps=10,
        asset_root=tmp_path / "assets",
    )

    wandb.login(host="WANDB_HOST", key="WANDB_API_KEY")
    run = wandb.init(
        mode="online",
        name="WANDB_NAME",
        entity="WANDB_ENTITY",
        project="WANDB_PROJECT",
        config={
            "generator_ckpt": str(base_checkpoint),
            "wandb_entity": "WANDB_ENTITY",
        },
        host="WANDB_HOST",
        key="WANDB_API_KEY",
    )

    assert run.id == "sue-offline-id"
    assert run_id_path.read_text(encoding="utf-8") == "sue-offline-id\n"
    assert login_calls == []
    assert observed["mode"] == "offline"
    assert observed["project"] == "comparison-project"
    assert observed["name"] == "supervised-run-train"
    assert observed["group"] == "supervised-run"
    assert "entity" not in observed
    assert "host" not in observed and "key" not in observed
    assert observed["config"]["generator_ckpt"] == "base.pt"
    assert observed["config"]["wandb_entity"] == ""
    assert observed["config"]["sue_runtime"]["tracking_mode"] == "offline"


def test_distributed_save_status_uses_all_rank_minimum(monkeypatch):
    import torch

    policy = _module()
    events = []
    distributed = torch.distributed
    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def all_reduce(signal, *, op):
        events.append(("all_reduce", op))
        signal.fill_(0)

    monkeypatch.setattr(distributed, "all_reduce", all_reduce)

    assert policy._distributed_all_success(True) is False
    assert events == [("all_reduce", distributed.ReduceOp.MIN)]


def test_final_summary_failure_does_not_fail_the_saved_checkpoint(
    monkeypatch, tmp_path, capsys
):
    policy = _module()
    trainer_class, wandb, _asset_root = _install_diffusion_policy(
        policy, monkeypatch, tmp_path, max_steps=1, clock=lambda: 0.0
    )
    trainer = trainer_class()

    class FailingSummary(dict):
        def __setitem__(self, _key, _value):
            raise OSError("summary backend unavailable")

    wandb.run.summary = FailingSummary()
    trainer.step = 1

    assert trainer._sue_checkpoint_callback(trainer) is True
    assert (Path(trainer.output_path) / "latest.pt").is_symlink()
    assert "summary update failed" in capsys.readouterr().out
