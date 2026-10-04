from __future__ import annotations

import importlib
import sys
from importlib.machinery import PathFinder
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def test_training_entry_prioritizes_copied_pipeline_package_for_imports():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    search_path = [
        str(training_entry.PACKAGE_ROOT),
        str(training_entry.SOURCE_ROOT),
    ]

    prioritize_source_root = getattr(
        training_entry, "prioritize_source_root", None
    )
    assert callable(prioritize_source_root)
    prioritize_source_root(search_path)

    spec = PathFinder.find_spec("pipeline", search_path)

    assert spec is not None
    assert Path(spec.origin) == training_entry.SOURCE_ROOT / "pipeline" / "__init__.py"


def _fake_train(monkeypatch, trainer_name, *, preflight=False, no_save=False,
                disable_wandb=False, log_iters=10, result=17):
    calls = []

    class Parser:
        def parse_args(self, argv):
            calls.append(("parse_args", argv))
            return SimpleNamespace(
                config_path="resolved.yaml",
                temporal_loop_config=None,
                preflight=preflight,
                no_save=no_save,
                disable_wandb=disable_wandb,
            )

    train = ModuleType("train")
    train.build_parser = lambda: Parser()
    train.load_config = lambda *_args: SimpleNamespace(
        trainer=trainer_name, log_iters=log_iters
    )
    train.main = lambda argv=None: calls.append(("train.main", argv)) or result
    monkeypatch.setitem(sys.modules, "train", train)
    return calls


def _fake_trainer_package(monkeypatch, class_name):
    trainer = ModuleType("trainer")
    trainer.__path__ = []
    module = ModuleType(f"trainer.{class_name}")
    module.Trainer = type("Trainer", (), {"save": lambda self: None})
    monkeypatch.setitem(sys.modules, "trainer", trainer)
    monkeypatch.setitem(sys.modules, f"trainer.{class_name}", module)
    return module.Trainer


def test_diffusion_entry_calls_train_main_with_original_argv_without_dmd_wrapper(
    monkeypatch, tmp_path
):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    argv = ["--config_path", "resolved.yaml", "--logdir", "out"]
    calls = _fake_train(monkeypatch, "diffusion")
    trainer_class = _fake_trainer_package(monkeypatch, "diffusion")
    distillation = ModuleType("trainer.distillation")
    distillation.Trainer = type("DMDTrainer", (), {"save": lambda self: None})
    monkeypatch.setitem(sys.modules, "trainer.distillation", distillation)
    monkeypatch.setenv("SUE_ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("SUE_MAX_STEPS", "20")
    monkeypatch.setenv("SUE_CHECKPOINT_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())

    installed = []
    monkeypatch.setattr(
        training_entry,
        "install_training_policy",
        lambda *_args, **_kwargs: installed.append("dmd-policy"),
    )
    monkeypatch.setattr(
        training_entry,
        "install_sue_wandb_runtime",
        lambda _wandb: installed.append("wandb"),
        raising=False,
    )
    monkeypatch.setattr(
        training_entry,
        "install_diffusion_training_policy",
        lambda trainer_class, *_args, **_kwargs: installed.append(trainer_class),
        raising=False,
    )
    cleanup = []
    monkeypatch.setattr(
        training_entry,
        "_destroy_distributed_process_group",
        lambda: cleanup.append("destroy"),
    )
    dmd_calls = []
    wrapper = ModuleType("train_blockwise_lora_dmd")
    wrapper.__file__ = "/copied/train_blockwise_lora_dmd.py"
    wrapper.main = lambda: dmd_calls.append("main") or 99
    monkeypatch.setitem(sys.modules, "train_blockwise_lora_dmd", wrapper)

    assert training_entry.main(argv) == 17

    assert ("train.main", argv) in calls
    assert installed == ["wandb", trainer_class]
    assert dmd_calls == []
    assert cleanup == ["destroy"]


def test_score_distillation_entry_keeps_the_dmd_wrapper_argv_route(
    monkeypatch, tmp_path
):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    argv = ["--config_path", "resolved.yaml", "--logdir", "out"]
    calls = _fake_train(monkeypatch, "score_distillation")
    trainer_class = _fake_trainer_package(monkeypatch, "distillation")
    distillation = ModuleType("trainer.distillation")
    distillation.Trainer = trainer_class
    monkeypatch.setitem(sys.modules, "trainer.distillation", distillation)
    monkeypatch.setenv("SUE_ASSET_ROOT", str(tmp_path))
    wandb = SimpleNamespace(init=lambda **_kwargs: None, log=lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    wrapper = ModuleType("train_blockwise_lora_dmd")
    wrapper.__file__ = "/copied/train_blockwise_lora_dmd.py"
    wrapper_calls = []
    wrapper.main = lambda: wrapper_calls.append(list(sys.argv)) or 23
    monkeypatch.setitem(sys.modules, "train_blockwise_lora_dmd", wrapper)
    policy_calls = []
    monkeypatch.setattr(
        training_entry,
        "install_training_policy",
        lambda cls, *_args, **_kwargs: policy_calls.append(cls),
    )
    original_argv = list(sys.argv)

    try:
        assert training_entry.main(argv) == 23
    finally:
        sys.argv[:] = original_argv

    assert policy_calls == [trainer_class]
    assert wrapper_calls == [[wrapper.__file__, *argv]]
    assert not any(call[0] == "train.main" for call in calls)


def test_training_entry_preflight_delegates_without_runtime_policies(monkeypatch):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    argv = ["--config_path", "resolved.yaml", "--preflight"]
    calls = _fake_train(monkeypatch, "diffusion", preflight=True)
    monkeypatch.setattr(
        training_entry,
        "install_sue_wandb_runtime",
        lambda *_args: pytest.fail("preflight must not install W&B policy"),
    )
    monkeypatch.setattr(
        training_entry,
        "install_diffusion_training_policy",
        lambda *_args, **_kwargs: pytest.fail("preflight must not install trainer policy"),
    )

    assert training_entry.main(argv) == 17
    assert ("train.main", argv) in calls


@pytest.mark.parametrize(
    ("no_save", "disable_wandb", "max_steps", "log_iters", "interval", "mode", "key", "entity", "message"),
    [
        (True, False, "10", 5, "3600", "offline", "", "", "checkpoint saving"),
        (False, True, "10", 5, "3600", "offline", "", "", "W&B tracking"),
        (False, False, "0", 5, "3600", "offline", "", "", "SUE_MAX_STEPS"),
        (False, False, "10", 0, "3600", "offline", "", "", "log_iters"),
        (False, False, "10", 5, "0", "offline", "", "", "CHECKPOINT_INTERVAL"),
        (False, False, "10", 5, "3600", "online", "", "team", "WANDB_API_KEY"),
        (False, False, "10", 5, "3600", "online", "key", "", "WANDB_ENTITY"),
    ],
)
def test_diffusion_entry_rejects_invalid_sue_run_before_model_construction(
    monkeypatch, no_save, disable_wandb, max_steps, log_iters, interval,
    mode, key, entity, message,
):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    monkeypatch.setenv("SUE_MAX_STEPS", max_steps)
    monkeypatch.setenv("SUE_CHECKPOINT_INTERVAL_SECONDS", interval)
    monkeypatch.setenv("SUE_WANDB_MODE", mode)
    if key:
        monkeypatch.setenv("WANDB_API_KEY", key)
    else:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
    if entity:
        monkeypatch.setenv("WANDB_ENTITY", entity)
    else:
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
    args = SimpleNamespace(no_save=no_save, disable_wandb=disable_wandb)
    config = SimpleNamespace(log_iters=log_iters)

    with pytest.raises(ValueError, match=message):
        training_entry._validate_diffusion_sue_config(args, config)


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("SUE_WANDB_LOG_INTERVAL", "0"),
        ("SUE_WANDB_LOG_INTERVAL", "-1"),
        ("SUE_PROGRESS_LOG_INTERVAL", "0"),
        ("SUE_PROGRESS_LOG_INTERVAL", "-1"),
    ],
)
def test_diffusion_entry_rejects_nonpositive_tracking_intervals(
    monkeypatch, env_name, value
):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    monkeypatch.setenv("SUE_MAX_STEPS", "10")
    monkeypatch.setenv("SUE_CHECKPOINT_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    monkeypatch.setenv(env_name, value)
    args = SimpleNamespace(no_save=False, disable_wandb=False)
    config = SimpleNamespace(log_iters=5)

    with pytest.raises(ValueError, match=env_name):
        training_entry._validate_diffusion_sue_config(args, config)


def test_diffusion_entry_defaults_tracking_intervals_when_unset(monkeypatch):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    monkeypatch.setenv("SUE_MAX_STEPS", "10")
    monkeypatch.setenv("SUE_CHECKPOINT_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    monkeypatch.delenv("SUE_WANDB_LOG_INTERVAL", raising=False)
    monkeypatch.delenv("SUE_PROGRESS_LOG_INTERVAL", raising=False)

    assert training_entry._validate_diffusion_sue_config(
        SimpleNamespace(no_save=False, disable_wandb=False),
        SimpleNamespace(log_iters=5),
    ) == (10, 5, 3600, 100, 10)


def test_diffusion_entry_cleans_up_active_wandb_and_group_on_error(
    monkeypatch, tmp_path
):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    argv = ["--config_path", "resolved.yaml"]
    _fake_train(monkeypatch, "diffusion")
    train = sys.modules["train"]
    _fake_trainer_package(monkeypatch, "diffusion")
    monkeypatch.setenv("SUE_ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("SUE_MAX_STEPS", "10")
    monkeypatch.setenv("SUE_CHECKPOINT_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("SUE_WANDB_MODE", "offline")
    cleanup = []
    wandb = SimpleNamespace(
        init=lambda **_kwargs: None,
        run=SimpleNamespace(),
        finish=lambda: cleanup.append("finish"),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setattr(training_entry, "install_sue_wandb_runtime", lambda *_args: None)
    monkeypatch.setattr(
        training_entry,
        "install_diffusion_training_policy",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_entry,
        "_destroy_distributed_process_group",
        lambda: cleanup.append("destroy"),
    )
    monkeypatch.setattr(
        train,
        "main",
        lambda _argv: (_ for _ in ()).throw(RuntimeError("training failed")),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        training_entry.main(argv)

    assert cleanup == ["finish", "destroy"]
