from __future__ import annotations

import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "scripts" / "looped_self_forcing_pipeline"


def _module(name: str):
    """Load the backend-neutral pipeline module once the planned tree exists."""
    assert (PIPELINE_DIR / f"{name}.py").is_file(), (
        f"backend-neutral pipeline module is missing: {name}.py"
    )
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(f"looped_self_forcing_pipeline.{name}")


def test_hydra_composes_method_backend_stage_tracking_and_scale_groups():
    config_module = _module("config")

    config = config_module.compose_config(
        ["method=layerwise_l16_23_k2", "backend=autodl", "stage=infer", "tracking=offline", "scale=smoke"]
    )

    assert config.method.temporal_loop.mode == "layer"
    assert config.backend.name == "autodl"
    assert config.backend.train_gpus == 4
    assert config.backend.infer_gpus == 1
    assert config.stage.name == "infer"
    assert config.tracking.mode == "offline"
    assert config.tracking.project == "looped-self-forcing-lora-dmd"
    assert config.scale.name == "smoke"
    assert config.train.max_steps == 10
    assert config.train.log_iters == 10
    assert config.seed == 1
    assert config.train.seed == 1

    default = config_module.compose_config()
    assert default.backend.train_gpus == 4
    assert default.train.max_steps == 600
    assert default.train.log_iters == 50
    assert default.seed == 1


def test_online_tracking_group_uses_same_comparison_project():
    config_module = _module("config")
    config = config_module.compose_config(["tracking=online"])

    assert config.tracking.project == "looped-self-forcing-lora-dmd"


def test_resolve_experiment_name_uses_selected_method_profile(capsys):
    pipeline = _module("pipeline")

    result = pipeline.main(
        [
            "resolve-experiment-name",
            "method=layerwise_l08_15_k2",
            "backend=nm5",
            "stage=train",
            "run_id=run-1",
        ]
    )

    assert result == 0
    assert capsys.readouterr().out.strip() == "layerwise_l08_15_k2-run-1-train"


def test_training_seed_zero_is_rejected():
    worker = _module("worker")

    with pytest.raises(ValueError, match="non-zero"):
        worker.require_training_seed(0)

    assert worker.require_training_seed(17) == 17


def test_inference_latent_frames_keep_the_validated_profile_size():
    worker = _module("worker")

    assert worker.require_inference_latent_frames(123) == 123
    with pytest.raises(ValueError, match="123 latent frames"):
        worker.require_inference_latent_frames(120)


def test_sue_exp_dir_resolves_neutral_bundle_config_and_runtime(tmp_path: Path, monkeypatch):
    config_module = _module("config")
    exp_dir = tmp_path / "scale_up_outputs" / "looped_self_forcing_pipeline"
    exp_dir.mkdir(parents=True)
    monkeypatch.setenv("SUE_EXP_DIR", str(exp_dir))

    assert config_module.resolve_exp_dir() == exp_dir.resolve()
    assert config_module.resolve_config_dir() == (exp_dir / "config").resolve()
    assert config_module.resolve_runtime_path() == (exp_dir / "config/runtime.yaml").resolve()


def test_neutral_runtime_keeps_reusable_policy_under_policy_namespace():
    config_module = _module("config")
    runtime = config_module.load_runtime_config(
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline"
    )

    assert "sandbox_resources" not in runtime
    assert "wandb_policy" not in runtime
    assert runtime["policy"]["sandbox_resources"]["nm5"]["train_gpus"] == 4
    assert runtime["policy"]["sandbox_resources"]["autodl"]["infer_gpus"] == 1
    assert runtime["policy"]["wandb_policy"]["required"] is True


def test_hydra_backend_gpu_count_must_match_runtime_policy():
    worker = _module("worker")
    exp_dir = REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline"
    config = SimpleNamespace(
        backend=SimpleNamespace(
            name="autodl",
            scheduler="tmux",
            train_gpus=4,
            infer_gpus=1,
        ),
        tracking={"required": True},
    )

    assert worker._configured_gpu_count(config, "train", exp_dir) == 4

    config.backend.train_gpus = 8
    with pytest.raises(ValueError, match="disagrees with runtime policy"):
        worker._configured_gpu_count(config, "train", exp_dir)


def test_runtime_paths_resolve_under_the_selected_bundle(tmp_path: Path):
    config_module = _module("config")
    exp_dir = tmp_path / "selected-bundle"
    config = exp_dir / "config/runtime.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "paths:\n"
        "  config_root: config\n"
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "policy:\n"
        "  sandbox_resources: {}\n"
        "  wandb_policy:\n"
        "    required: true\n",
        encoding="utf-8",
    )

    roots = config_module.resolve_runtime_paths(exp_dir)

    assert roots == {
        "datasets_root": exp_dir / "datasets",
        "slurm_scripts_root": exp_dir / "slurm_scripts",
        "artifacts_root": exp_dir / "artifacts",
        "ckpt_root": exp_dir / "ckpt",
        "final_result_root": exp_dir / "final_result",
        "logs_root": exp_dir / "logs",
    }


def test_runtime_output_roots_reject_escape_paths(tmp_path: Path):
    config_module = _module("config")
    exp_dir = tmp_path / "selected-bundle"
    config = exp_dir / "config/runtime.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "paths:\n"
        "  config_root: config\n"
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: ../outside\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "policy:\n"
        "  sandbox_resources: {}\n"
        "  wandb_policy:\n"
        "    required: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inside|relative"):
        config_module.resolve_runtime_paths(exp_dir)


def test_runtime_config_root_is_fixed_for_the_bundle_bootstrap(tmp_path: Path):
    config_module = _module("config")
    exp_dir = tmp_path / "selected-bundle"
    config = exp_dir / "config/runtime.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "paths:\n"
        "  config_root: settings\n"
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "policy:\n"
        "  sandbox_resources: {}\n"
        "  wandb_policy:\n"
        "    required: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config_root is fixed"):
        config_module.resolve_runtime_paths(exp_dir)


def test_pipeline_check_command_passes_exp_dir_to_worker(monkeypatch, tmp_path: Path):
    pipeline_module = _module("pipeline")
    worker_module = _module("worker")
    exp_dir = tmp_path / "selected-bundle"
    observed = {}

    monkeypatch.setattr(
        pipeline_module,
        "compose_config",
        lambda _overrides, *, exp_dir: SimpleNamespace(
            stage=SimpleNamespace(name="infer"),
            backend=SimpleNamespace(name="autodl"),
            run_id="run-1",
        ),
    )

    def check_stage(_config, stage, *, exp_dir=None):
        observed.update(stage=stage, exp_dir=exp_dir)
        return {"config_path": exp_dir / "run-1/config/inference.yaml"}

    monkeypatch.setattr(worker_module, "check_stage", check_stage)

    assert pipeline_module.main(
        ["check-infer", "--exp-dir", str(exp_dir), "backend=autodl"]
    ) == 0
    assert observed == {"stage": "infer", "exp_dir": exp_dir.resolve()}


def test_pipeline_cli_rejects_private_asset_path_overrides(capsys):
    pipeline_module = _module("pipeline")

    with pytest.raises(SystemExit):
        pipeline_module._parse_args(["check-train", "assets.root=/private/mount/models"])

    assert "/private/mount/models" not in capsys.readouterr().err


def test_check_assets_uses_explicit_exp_dir_instead_of_environment(tmp_path: Path, monkeypatch):
    worker = _module("worker")
    package_root = tmp_path / "package"
    monkeypatch.setattr(worker, "PACKAGE_ROOT", package_root)
    selected_bundle = tmp_path / "selected-bundle"
    other_bundle = tmp_path / "environment-bundle"
    monkeypatch.setenv("SUE_EXP_DIR", str(other_bundle))
    runtime = selected_bundle / "config/runtime.yaml"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(
        "paths:\n"
        "  config_root: config\n"
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "policy:\n"
        "  sandbox_resources: {}\n"
        "  wandb_policy:\n"
        "    required: true\n",
        encoding="utf-8",
    )

    asset_root = tmp_path / "assets"
    student = asset_root / "wan_models/Wan2.1-T2V-1.3B"
    teacher = asset_root / "wan_models/Wan2.1-T2V-14B"
    for relative in (
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ):
        path = student / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    for relative in ("config.json", "diffusion_pytorch_model.safetensors"):
        path = teacher / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    generator = asset_root / "checkpoints/generator.pt"
    generator.parent.mkdir(parents=True)
    generator.write_bytes(b"generator")
    prompt = selected_bundle / "datasets/train.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("a prompt\n", encoding="utf-8")

    config = SimpleNamespace(
        assets={
            "root": str(asset_root),
            "student_model_dir": "wan_models/Wan2.1-T2V-1.3B",
            "teacher_model_dir": "wan_models/Wan2.1-T2V-14B",
            "generator_checkpoint": "checkpoints/generator.pt",
        },
        train=SimpleNamespace(prompt_file="train.txt"),
    )

    checked = worker.check_assets(config, "train", exp_dir=selected_bundle)

    assert checked["prompt"] == prompt.resolve()
    assert checked["asset_root"] == asset_root.resolve()


def test_asset_root_is_required_in_preflight(tmp_path: Path, monkeypatch):
    worker = _module("worker")
    monkeypatch.delenv("SUE_ASSET_ROOT", raising=False)
    config = SimpleNamespace(
        assets={
            "root": None,
            "student_model_dir": "wan_models/student",
            "teacher_model_dir": "wan_models/teacher",
            "generator_checkpoint": "checkpoints/generator.pt",
        },
        train=SimpleNamespace(prompt_file="datasets/train.txt"),
    )

    with pytest.raises(ValueError, match="SUE_ASSET_ROOT|assets.root"):
        worker.check_assets(config, "train", exp_dir=tmp_path)


def test_pipeline_cli_prints_composed_config_without_running_workers():
    pipeline = PIPELINE_DIR / "pipeline.py"
    assert pipeline.is_file(), "backend-neutral pipeline CLI is missing"
    result = subprocess.run(
        [
            sys.executable,
            str(pipeline),
            "--show-config",
            "backend=autodl",
            "stage=infer",
            "scale=smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "SUE_EXP_DIR": str(REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline"),
            "SUE_ASSET_ROOT": "/private/backend/model-store",
        },
    )

    rendered = json.loads(result.stdout)
    assert rendered["backend"]["name"] == "autodl"
    assert rendered["stage"]["name"] == "infer"
    assert rendered["scale"]["name"] == "smoke"
    assert rendered["assets"]["root"] == "<configured privately through SUE_ASSET_ROOT>"
    assert "/private/backend/model-store" not in result.stdout


def test_run_manifest_rejects_changed_inference_inputs(tmp_path: Path):
    manifest_module = _module("manifest")
    run_root = tmp_path / "run-1"
    first = {
        "method": {"mode": "layer", "layer_start": 16, "layer_end": 23},
        "backend": "autodl",
        "stage": "infer",
        "checkpoint": "/models/adapter.pt",
        "prompt_file": "datasets/prompts.txt",
        "num_output_frames": 123,
        "num_samples": 2,
        "seed": 7,
        "use_ema": True,
    }

    manifest_module.materialize_manifest(run_root, "infer", first)
    changed = {**first, "seed": 8}

    with pytest.raises(ValueError, match="run_id|manifest|configuration"):
        manifest_module.materialize_manifest(run_root, "infer", changed)


def test_adapter_base_checkpoint_is_resolved_from_active_backend_and_hashed(tmp_path: Path):
    checkpoint_module = _module("checkpoint")
    configured_base = tmp_path / "base.pt"
    configured_base.write_bytes(b"base checkpoint bytes")
    metadata = {
        "base_checkpoint": "/old/nm5/mount/base.pt",
        "base_checkpoint_name": configured_base.name,
        "base_checkpoint_sha256": checkpoint_module.sha256_file(configured_base),
    }

    resolved = checkpoint_module.resolve_base_checkpoint(metadata, configured_base)

    assert resolved == configured_base.resolve()

    configured_base.write_bytes(b"different model bytes")
    with pytest.raises(ValueError, match="hash|identity"):
        checkpoint_module.resolve_base_checkpoint(metadata, configured_base)


def test_backend_command_plan_uses_validated_launchers_for_both_stages(tmp_path: Path):
    launch_module = _module("launch")
    common = {"launcher_root": tmp_path / "slurm_scripts", "run_id": "run-1"}

    nm5_train = launch_module.build_backend_command("train", "nm5", **common)
    nm5_infer = launch_module.build_backend_command("infer", "nm5", **common)
    autodl_train = launch_module.build_backend_command("train", "autodl", **common)
    autodl_infer = launch_module.build_backend_command("infer", "autodl", **common)

    assert nm5_train == ["bash", str(tmp_path / "slurm_scripts/nm5_submit.sh"), "train", "run-1"]
    assert nm5_infer == ["bash", str(tmp_path / "slurm_scripts/nm5_submit.sh"), "infer", "run-1"]
    assert autodl_train == ["bash", str(tmp_path / "slurm_scripts/autodl_run.sh"), "train", "run-1"]
    assert autodl_infer == ["bash", str(tmp_path / "slurm_scripts/autodl_run.sh"), "infer", "run-1"]


def test_training_evidence_rejects_nonfinite_saved_tensors(tmp_path: Path):
    torch = pytest.importorskip("torch")
    worker_module = _module("worker")
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"metadata": {"step": 20}, "generator": {"adapter": torch.tensor([float("nan")])}},
        checkpoint,
    )

    with pytest.raises(ValueError, match="finite|NaN|Inf"):
        worker_module.verify_training_checkpoint(checkpoint, expected_step=20)


def test_inference_evidence_requires_exact_decodable_output_count(tmp_path: Path):
    worker_module = _module("worker")
    outputs = tmp_path / "results"
    outputs.mkdir()
    (outputs / "0-0_ema.mp4").write_bytes(b"video")

    def probe(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "1.0"}}),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="expected 2|exactly 2|count"):
        worker_module.verify_inference_outputs(
            outputs,
            expected_count=2,
            expected_frames=489,
            expected_fps=16,
            run_command=probe,
        )


def test_result_ledger_records_backend_seed_speed_and_parameter_count(tmp_path: Path):
    ledger_module = _module("ledger")
    path = tmp_path / "experiment_results.csv"
    row = {
        "method": "layer",
        "experiment_name": "layer-l16-23-k2-run-1-train",
        "backend": "autodl",
        "status": "completed",
        "seed": 7,
        "training_speed": 0.42,
        "trainable_parameter_count": 123456,
        "checkpoint_step": 600,
        "checkpoint_or_output_path": "/runs/run-1/checkpoints/latest.pt",
    }

    ledger_module.write_result(path, row)

    with path.open(newline="", encoding="utf-8") as stream:
        saved = next(csv.DictReader(stream))
    assert saved["backend"] == "autodl"
    assert saved["seed"] == "7"
    assert saved["training_speed"] == "0.42"
    assert saved["parameter_count"] == ""
    assert saved["trainable_parameter_count"] == "123456"
