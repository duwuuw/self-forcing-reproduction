from __future__ import annotations

import csv
import importlib
import json
import os
import shutil
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
    assert config.train.timeout_seconds == 1800
    assert config.infer.timeout_seconds == 19800
    assert config.seed == 1
    assert config.train.seed == 1

    default = config_module.compose_config()
    assert default.backend.train_gpus == 4
    assert default.train.max_steps == 600
    assert default.train.log_iters == 50
    assert default.train.timeout_seconds == 19800
    assert default.infer.timeout_seconds == 19800
    assert default.seed == 1


def test_runtime_does_not_require_slurm_email():
    config_module = _module("config")
    runtime = config_module.load_runtime_config()

    nm5 = runtime["backend_env"]["nm5"]
    assert "SUE_SLURM_MAIL_USER" not in nm5["required_keys"]
    assert "mail" not in nm5["source_note"].casefold()


def test_online_tracking_group_uses_same_comparison_project():
    config_module = _module("config")
    config = config_module.compose_config(["tracking=online"])

    assert config.tracking.project == "looped-self-forcing-lora-dmd"


def test_paired_layerwise_methods_use_same_layers_lora_and_loss_with_requested_k_lr():
    config_module = _module("config")
    baseline = config_module.compose_config(
        ["method=layerwise_l23_30_k3_lr5gen", "backend=nm5", "stage=train"]
    ).method
    low_lr = config_module.compose_config(
        ["method=layerwise_l23_30_k2_lr5both", "backend=nm5", "stage=train"]
    ).method

    assert baseline.temporal_loop.layer_start == low_lr.temporal_loop.layer_start == 22
    assert baseline.temporal_loop.layer_end == low_lr.temporal_loop.layer_end == 29
    assert baseline.temporal_loop.k_min == baseline.temporal_loop.k_max == 3
    assert low_lr.temporal_loop.k_min == low_lr.temporal_loop.k_max == 2
    assert baseline.lr == 4e-7
    assert baseline.lr_critic == 4e-7
    assert low_lr.lr == 4e-7
    assert low_lr.lr_critic == 8e-8

    from omegaconf import OmegaConf

    baseline_values = OmegaConf.to_container(baseline, resolve=True)
    low_lr_values = OmegaConf.to_container(low_lr, resolve=True)
    for values in (baseline_values, low_lr_values):
        values.pop("profile_id")
        values.pop("lr")
        values.pop("lr_critic")
        loop = values["temporal_loop"]
        loop["layer_start"] = 22
        loop["layer_end"] = 29
        loop["k_min"] = 1
        loop["k_max"] = 1
    assert baseline_values == low_lr_values


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

    assert "sandbox_resources" not in runtime["policy"]
    assert "wandb_policy" not in runtime
    assert runtime["sandbox_resources"]["nm5"]["partition"] == "acc"
    assert runtime["sandbox_resources"]["nm5"]["gpu_type"] == "H100"
    assert runtime["sandbox_resources"]["nm5"]["gpu_usable_memory_gib"] == 63.29
    assert runtime["sandbox_resources"]["nm5"]["gpu_non_torch_reserve_gib"] == 7.0
    assert runtime["sandbox_resources"]["nm5"]["gpus_per_node"] == 4
    assert runtime["sandbox_resources"]["nm5"]["max_nodes_per_job"] == 1
    assert runtime["sandbox_resources"]["nm5"]["cpus_per_gpu"] == 20
    assert runtime["sandbox_resources"]["nm5"]["timeout_kill_after_seconds"] == 300
    assert runtime["sandbox_resources"]["nm5"]["finalization_grace_seconds"] == 600
    assert runtime["sandbox_resources"]["nm5"]["train_gpus"] == 4
    assert runtime["sandbox_resources"]["autodl"]["infer_gpus"] == 1
    assert runtime["sandbox_script_folders"]["nm5"] == "slurm_scripts"
    assert runtime["backend_env"]["nm5"]["source_note"] == (
        "deepresearch-sandbox/config_nm5.txt"
    )
    assert "NM5_WORKSPACE_ROOT" in runtime["backend_env"]["nm5"]["required_keys"]
    assert "SUE_ASSET_ROOT" not in runtime["backend_env"]["nm5"]["required_keys"]
    assert "NM5_HF_HUB_CACHE" in runtime["backend_env"]["nm5"]["required_keys"]
    assert runtime["policy"]["wandb_policy"]["required"] is True


def test_neutral_runtime_names_nm5_python_asset_and_overlay_paths_without_absolute_roots():
    config_module = _module("config")
    runtime = config_module.load_runtime_config(
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline"
    )

    environment = runtime["environment"]
    assert environment["env_manager"] == "conda"
    assert environment["env_root"] == "scale_up_outputs/envs"
    assert environment["python_binary"].endswith("/bin/python")
    assert environment["asset_root"] == "Self-Forcing-blockwise-layerwise"
    assert environment["python_overlay"] == "envs/hydra_overlay"
    assert environment["allowed_external_roots"] == [
        "scale_up_outputs/envs",
        "Self-Forcing-blockwise-layerwise",
    ]
    assert environment["cache_env_sources"] == {
        "HF_HOME": "NM5_HF_HOME",
        "HF_HUB_CACHE": "NM5_HF_HUB_CACHE",
        "MODELSCOPE_CACHE": "NM5_MODELSCOPE_CACHE",
        "TORCH_HOME": "NM5_TORCH_HOME",
        "MPLCONFIGDIR": "NM5_MPLCONFIGDIR",
        "WANDB_CACHE_DIR": "NM5_WANDB_CACHE_DIR",
    }
    assert runtime["paths"]["autotune_hyperparam_root"] == "autotune_hyperparam"
    assert all(not Path(value).is_absolute() for value in environment["allowed_external_roots"])


def test_runtime_records_fixed_parameter_autotune_waiver():
    config_module = _module("config")
    runtime = config_module.load_runtime_config(
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline"
    )

    assert runtime["autotune_hyperparam"]["skip"] is True
    assert "fixed" in runtime["autotune_hyperparam"]["waiver_reason"].lower()
    runtime_text = (REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config/runtime.yaml").read_text(encoding="utf-8")
    assert runtime_text.count("autotune_hyperparam:") == 1


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
    with pytest.raises(ValueError, match="disagrees with runtime"):
        worker._configured_gpu_count(config, "train", exp_dir)


def test_runtime_paths_resolve_under_the_selected_bundle(tmp_path: Path):
    config_module = _module("config")
    exp_dir = tmp_path / "selected-bundle"
    config = exp_dir / "config/runtime.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "paths:\n"
        "  config_root: config\n"
        "  autotune_hyperparam_root: autotune_hyperparam\n"
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "  state_root: state\n"
        "  readiness_root: readiness\n"
        "  ledger_csv: artifacts/experiment_results.csv\n"
        "backend: {primary: nm5}\n"
        "sandbox_script_folders: {nm5: slurm_scripts}\n"
        "sandbox_resources: {nm5: {partition: acc, qos: acc_ehpc, gpus_per_node: 4, gpu_type: H100, gpu_usable_memory_gib: 63.29, gpu_non_torch_reserve_gib: 7.0, max_nodes_per_job: 1, cpus_per_gpu: 20, timeout_kill_after_seconds: 300, finalization_grace_seconds: 600, train_gpus: 4, infer_gpus: 1}}\n"
        "backend_env: {nm5: {source_note: deepresearch-sandbox/config_nm5.txt, required_keys: [NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT, NM5_LOGIN_SSH, NM5_HF_HOME, NM5_HF_HUB_CACHE, NM5_MODELSCOPE_CACHE, NM5_TORCH_HOME, NM5_MPLCONFIGDIR, NM5_WANDB_CACHE_DIR, WANDB_API_KEY, WANDB_ENTITY]}}\n"
        "environment: {env_manager: conda, env_root: scale_up_outputs/envs, python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python, asset_root: Self-Forcing-blockwise-layerwise, python_overlay: envs/hydra_overlay, allowed_external_roots: [scale_up_outputs/envs, Self-Forcing-blockwise-layerwise], cache_env_sources: {HF_HOME: NM5_HF_HOME, HF_HUB_CACHE: NM5_HF_HUB_CACHE, MODELSCOPE_CACHE: NM5_MODELSCOPE_CACHE, TORCH_HOME: NM5_TORCH_HOME, MPLCONFIGDIR: NM5_MPLCONFIGDIR, WANDB_CACHE_DIR: NM5_WANDB_CACHE_DIR}}\n"
        "policy:\n"
        "  wandb_policy:\n"
        "    required: true\n",
        encoding="utf-8",
    )

    roots = config_module.resolve_runtime_paths(exp_dir)

    assert roots == {
        "autotune_hyperparam_root": exp_dir / "autotune_hyperparam",
        "datasets_root": exp_dir / "datasets",
        "slurm_scripts_root": exp_dir / "slurm_scripts",
        "artifacts_root": exp_dir / "artifacts",
        "ckpt_root": exp_dir / "ckpt",
        "final_result_root": exp_dir / "final_result",
        "logs_root": exp_dir / "logs",
        "state_root": exp_dir / "state",
        "readiness_root": exp_dir / "readiness",
        "ledger_csv": exp_dir / "artifacts/experiment_results.csv",
    }


def test_runtime_rejects_external_environment_paths_outside_the_declared_allowlist(tmp_path: Path):
    config_module = _module("config")
    exp_dir = tmp_path / "selected-bundle"
    config_dir = exp_dir / "config"
    config_dir.mkdir(parents=True)
    runtime_text = (
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config/runtime.yaml"
    ).read_text(encoding="utf-8")
    runtime_text = runtime_text.replace(
        "python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python",
        "python_binary: ../outside/bin/python",
    )
    (config_dir / "runtime.yaml").write_text(runtime_text, encoding="utf-8")

    with pytest.raises(ValueError, match="environment.python_binary|allowed_external_roots"):
        config_module.resolve_runtime_paths(exp_dir)


def test_preflight_requires_fresh_standard_max_parallel_artifact(tmp_path: Path):
    preflight = _module("preflight")
    exp_dir = tmp_path / "bundle"
    artifact = exp_dir / "pair-1/max_parallel.json"
    artifact.parent.mkdir(parents=True)
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    record = {
        "timestamp": now.isoformat(),
        "sandbox": "nm5",
        "partition": "acc",
        "max_parallel": 4,
        "limiting_factor": "one four-GPU job per node",
        "evidence": {"squeue_summary": "redacted", "qos_summary": "redacted"},
    }
    artifact.write_text(json.dumps(record), encoding="utf-8")

    result = preflight.verify_max_parallel_artifact(
        exp_dir, "pair-1", partition="acc", now=now
    )

    assert result["max_parallel"] == 4
    assert result["artifact_path"] == "pair-1/max_parallel.json"


def test_pending_submit_query_requires_two_slots_without_recording_raw_output():
    preflight = _module("preflight")
    runtime = {"sandbox_resources": {"nm5": {"partition": "acc", "qos": "acc_ehpc"}}}
    environment = {"NM5_ACCOUNT": "private-account"}

    def query(_command, **_kwargs):
        command = _command
        if command[0] != "squeue" and "assoc" in command:
            output = "fixture-user|private-account|acc_ehpc|-1\n"
        elif command[0] != "squeue":
            output = "acc_ehpc|4\n"
        else:
            output = "RUNNING|private-account|acc_ehpc|acc\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    result = preflight.query_pending_submit_headroom(
        runtime, environment, run_command=query, username="fixture-user"
    )

    assert result["pending_submit_headroom"] == 3
    assert result["pending_submit_slots_verified"] == 2
    assert "private-account" not in json.dumps(result)
    assert "fixture-user" not in json.dumps(result)

    def insufficient(command, **_kwargs):
        if command[0] != "squeue" and "assoc" in command:
            output = "fixture-user|private-account|acc_ehpc|-1\n"
        elif command[0] != "squeue":
            output = "acc_ehpc|2\n"
        else:
            output = "RUNNING|private-account|acc_ehpc|acc\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    with pytest.raises(RuntimeError, match="two-job dependency pair"):
        preflight.query_pending_submit_headroom(
            runtime, environment, run_command=insufficient, username="fixture-user"
        )


def test_pending_submit_query_accepts_verified_association_with_unset_submit_limit():
    preflight = _module("preflight")
    runtime = {"sandbox_resources": {"nm5": {"partition": "acc", "qos": "acc_ehpc"}}}
    environment = {"NM5_ACCOUNT": "private-account"}

    def query(command, **_kwargs):
        if "assoc" in command:
            output = "fixture-user|private-account|acc_ehpc|\n"
        elif command[0] == "sacctmgr":
            output = "acc_ehpc|366\n"
        else:
            output = "RUNNING|private-account|acc_ehpc|acc\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    result = preflight.query_pending_submit_headroom(
        runtime, environment, run_command=query, username="fixture-user"
    )

    assert result["association_submit_limit"] is None
    assert result["qos_submit_limit"] == 366
    assert result["pending_submit_headroom"] == 365
    assert result["pending_submit_slots_verified"] == 2
    assert "private-account" not in json.dumps(result)
    assert "fixture-user" not in json.dumps(result)


def test_pending_submit_query_rejects_association_that_disallows_runtime_qos():
    preflight = _module("preflight")
    runtime = {"sandbox_resources": {"nm5": {"partition": "acc", "qos": "acc_ehpc"}}}
    environment = {"NM5_ACCOUNT": "private-account"}

    def query(command, **_kwargs):
        if "assoc" in command:
            output = "fixture-user|private-account|acc_debug|-1\n"
        elif command[0] == "sacctmgr":
            output = "acc_ehpc|366\n"
        else:
            output = ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    with pytest.raises(RuntimeError, match="runtime QoS"):
        preflight.query_pending_submit_headroom(
            runtime, environment, run_command=query, username="fixture-user"
        )


def test_pending_submit_query_requires_association_qos_evidence():
    preflight = _module("preflight")
    runtime = {"sandbox_resources": {"nm5": {"partition": "acc", "qos": "acc_ehpc"}}}
    environment = {"NM5_ACCOUNT": "private-account"}

    def query(command, **_kwargs):
        if "assoc" in command:
            output = "fixture-user|private-account||-1\n"
        elif command[0] == "sacctmgr":
            output = "acc_ehpc|366\n"
        else:
            output = ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    with pytest.raises(RuntimeError, match="no allowed-QoS evidence"):
        preflight.query_pending_submit_headroom(
            runtime, environment, run_command=query, username="fixture-user"
        )


def test_pending_submit_query_does_not_treat_minus_one_as_qos_wildcard():
    preflight = _module("preflight")
    runtime = {"sandbox_resources": {"nm5": {"partition": "acc", "qos": "acc_ehpc"}}}
    environment = {"NM5_ACCOUNT": "private-account"}

    def query(command, **_kwargs):
        if "assoc" in command:
            output = "fixture-user|private-account|-1|-1\n"
        elif command[0] == "sacctmgr":
            output = "acc_ehpc|366\n"
        else:
            output = ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    with pytest.raises(RuntimeError, match="runtime QoS"):
        preflight.query_pending_submit_headroom(
            runtime, environment, run_command=query, username="fixture-user"
        )


def test_preflight_cache_errors_name_only_the_private_environment_key():
    preflight = _module("preflight")
    config_module = _module("config")
    runtime = config_module.load_runtime_config(
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline"
    )
    private_path = "/private-fixture/cache/sentinel"
    env = {
        source: private_path if source == "NM5_HF_HOME" else "/private-fixture/cache/other"
        for source in runtime["environment"]["cache_env_sources"].values()
    }

    with pytest.raises(RuntimeError, match="NM5_HF_HOME") as caught:
        preflight._check_private_cache_roots(runtime, env)
    assert private_path not in str(caught.value)


def test_full_pair_manifest_comparison_allows_only_scale_run_size_and_identity_changes():
    preflight = _module("preflight")
    smoke = {
        "backend": "nm5",
        "method": {
            "lr": 4e-7,
            "lr_critic": 4e-7,
            "loss": {"dmd": 1.0, "cycle": 0.0},
            "lora": {"enabled": True, "target_modules": ["self_attn.q", "self_attn.v"]},
        },
        "tracking": {"mode": "offline", "project": "shared-project"},
        "scale": {"name": "smoke"},
        "train": {
            "max_steps": 10,
            "log_iters": 10,
            "timeout_seconds": 19800,
            "seed": 1,
            "checkpoint_interval_seconds": 3600,
            "resume_checkpoint": None,
        },
        "assets": {"generator_checkpoint_sha256": "a" * 64},
        "train_prompt": {"sha256": "b" * 64},
        "run_id": "smoke-k3",
        "resolved_config_sha256": "c" * 64,
    }
    full = json.loads(json.dumps(smoke))
    full["scale"] = {"name": "full"}
    full["train"].update(max_steps=600, log_iters=50, timeout_seconds=7200)
    full["run_id"] = "full-k3"
    full["resolved_config_sha256"] = "d" * 64

    preflight.require_full_config_matches_smoke(smoke, full, profile="K3")
    full["method"]["loss"]["cycle"] = 0.5
    with pytest.raises(RuntimeError, match="differs from smoke"):
        preflight.require_full_config_matches_smoke(smoke, full, profile="K3")


def test_pair_preflight_runs_static_gates_for_both_profiles(monkeypatch, tmp_path: Path):
    preflight = _module("preflight")
    exp_dir = tmp_path / "bundle"
    config_dir = exp_dir / "config"
    config_dir.mkdir(parents=True)
    source_config = REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config"
    shutil.copytree(source_config, config_dir, dirs_exist_ok=True)
    (exp_dir / "envs/hydra_overlay").mkdir(parents=True)
    deepresearch_root = tmp_path / "deepresearch"
    workspace = deepresearch_root / "workspace/looped-flow-matching"
    workspace.mkdir(parents=True)
    asset_root = tmp_path / "assets"
    (asset_root / "checkpoints").mkdir(parents=True)
    (asset_root / "wan_models").mkdir()
    caches = {}
    for source in (
        "NM5_HF_HOME",
        "NM5_HF_HUB_CACHE",
        "NM5_MODELSCOPE_CACHE",
        "NM5_TORCH_HOME",
        "NM5_MPLCONFIGDIR",
        "NM5_WANDB_CACHE_DIR",
    ):
        cache = tmp_path / "private-cache" / source.lower()
        cache.mkdir(parents=True)
        caches[source] = str(cache)
    capacity = exp_dir / "preflight-pair/max_parallel.json"
    capacity.parent.mkdir(parents=True)
    capacity.write_text(
        json.dumps(
            {
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "sandbox": "nm5",
                "partition": "acc",
                "max_parallel": 4,
                "limiting_factor": "one four-GPU job per node",
                "evidence": {"queue": "redacted summary"},
            }
        ),
        encoding="utf-8",
    )
    environment = {
        "NM5_DEEPRESEARCH_ROOT": str(deepresearch_root),
        "SUE_WORKSPACE_ROOT": str(workspace),
        "SUE_EXP_DIR": str(exp_dir),
        "SUE_PYTHON": sys.executable,
        "SUE_ASSET_ROOT": str(asset_root),
        "SUE_PYTHONPATH": str(exp_dir / "envs/hydra_overlay"),
        "NM5_ACCOUNT": "fixture-account",
        "HOME": str(tmp_path / "empty-home"),
        **caches,
    }
    Path(environment["HOME"]).mkdir()
    checked = []
    def fake_slurm_query(command, **_kwargs):
        if command[0] == "sacctmgr" and "assoc" in command:
            user = next(value.partition("=")[2] for value in command if value.startswith("user="))
            account = next(value.partition("=")[2] for value in command if value.startswith("account="))
            output = f"{user}|{account}|acc_ehpc|-1\n"
        elif command[0] == "sacctmgr":
            output = "acc_ehpc|4\n"
        else:
            output = "RUNNING|fixture-account|acc_ehpc|acc\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    import_module = preflight.importlib.import_module
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: object() if name == "wandb" else import_module(name),
    )
    result = preflight.run_pair_preflight(
        "preflight-pair",
        ["scale=smoke"],
        environment=environment,
        run_command=fake_slurm_query,
        check_stage_fn=lambda config, stage, *, exp_dir: checked.append(
            (config.method.profile_id, stage, config.run_id)
        ),
    )

    assert checked == [
        ("layerwise_l23_30_k3_lr5gen", "train", "preflight-pair-k3-lr5gen"),
        ("layerwise_l23_30_k2_lr5both", "train", "preflight-pair-k2-lr5both"),
    ]
    assert result["effective_gpu_parallel"] == 1
    assert result["pending_submit"]["headroom"] == 3
    pending_report_path = exp_dir / result["pending_submit"]["artifact_path"]
    pending_report = pending_report_path.read_text(encoding="utf-8")
    assert "fixture-account" not in pending_report
    assert "RUNNING|" not in pending_report
    report_path = preflight.write_pair_preflight_report(exp_dir, result)
    report_text = report_path.read_text(encoding="utf-8")
    assert report_path.relative_to(exp_dir).as_posix() == "artifacts/pairs/preflight-pair/preflight.json"
    assert str(exp_dir) not in report_text
    assert "preflight-pair/max_parallel.json" in report_text
    assert "pending_submit_preflight.json" in report_text


def test_runtime_output_roots_reject_escape_paths(tmp_path: Path):
    config_module = _module("config")
    exp_dir = tmp_path / "selected-bundle"
    config = exp_dir / "config/runtime.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "paths:\n"
        "  config_root: config\n"
        "  autotune_hyperparam_root: autotune_hyperparam\n"
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: ../outside\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "  state_root: state\n"
        "  readiness_root: readiness\n"
        "  ledger_csv: artifacts/experiment_results.csv\n"
        "backend: {primary: nm5}\n"
        "sandbox_script_folders: {nm5: slurm_scripts}\n"
        "sandbox_resources: {nm5: {partition: acc, qos: acc_ehpc, gpus_per_node: 4, gpu_type: H100, gpu_usable_memory_gib: 63.29, gpu_non_torch_reserve_gib: 7.0, max_nodes_per_job: 1, cpus_per_gpu: 20, timeout_kill_after_seconds: 300, finalization_grace_seconds: 600, train_gpus: 4, infer_gpus: 1}}\n"
        "backend_env: {nm5: {source_note: deepresearch-sandbox/config_nm5.txt, required_keys: [NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT, NM5_LOGIN_SSH, NM5_HF_HOME, NM5_HF_HUB_CACHE, NM5_MODELSCOPE_CACHE, NM5_TORCH_HOME, NM5_MPLCONFIGDIR, NM5_WANDB_CACHE_DIR, WANDB_API_KEY, WANDB_ENTITY]}}\n"
        "environment: {env_manager: conda, env_root: scale_up_outputs/envs, python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python, asset_root: Self-Forcing-blockwise-layerwise, python_overlay: envs/hydra_overlay, allowed_external_roots: [scale_up_outputs/envs, Self-Forcing-blockwise-layerwise], cache_env_sources: {HF_HOME: NM5_HF_HOME, HF_HUB_CACHE: NM5_HF_HUB_CACHE, MODELSCOPE_CACHE: NM5_MODELSCOPE_CACHE, TORCH_HOME: NM5_TORCH_HOME, MPLCONFIGDIR: NM5_MPLCONFIGDIR, WANDB_CACHE_DIR: NM5_WANDB_CACHE_DIR}}\n"
        "policy:\n"
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
        "  state_root: state\n"
        "  readiness_root: readiness\n"
        "  ledger_csv: artifacts/experiment_results.csv\n"
        "backend: {primary: nm5}\n"
        "sandbox_script_folders: {nm5: slurm_scripts}\n"
        "sandbox_resources: {nm5: {partition: acc, qos: acc_ehpc, gpus_per_node: 4, gpu_type: H100, gpu_usable_memory_gib: 63.29, gpu_non_torch_reserve_gib: 7.0, max_nodes_per_job: 1, cpus_per_gpu: 20, timeout_kill_after_seconds: 300, finalization_grace_seconds: 600, train_gpus: 4, infer_gpus: 1}}\n"
        "backend_env: {nm5: {source_note: deepresearch-sandbox/config_nm5.txt, required_keys: [NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT, NM5_LOGIN_SSH, NM5_HF_HOME, NM5_HF_HUB_CACHE, NM5_MODELSCOPE_CACHE, NM5_TORCH_HOME, NM5_MPLCONFIGDIR, NM5_WANDB_CACHE_DIR, WANDB_API_KEY, WANDB_ENTITY]}}\n"
        "environment: {env_manager: conda, env_root: scale_up_outputs/envs, python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python, asset_root: Self-Forcing-blockwise-layerwise, python_overlay: envs/hydra_overlay, allowed_external_roots: [scale_up_outputs/envs, Self-Forcing-blockwise-layerwise], cache_env_sources: {HF_HOME: NM5_HF_HOME, HF_HUB_CACHE: NM5_HF_HUB_CACHE, MODELSCOPE_CACHE: NM5_MODELSCOPE_CACHE, TORCH_HOME: NM5_TORCH_HOME, MPLCONFIGDIR: NM5_MPLCONFIGDIR, WANDB_CACHE_DIR: NM5_WANDB_CACHE_DIR}}\n"
        "policy:\n"
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


@pytest.mark.parametrize("method", [{"trainer": "diffusion"}, {"trainer": None}, {}])
def test_prepare_stage_rejects_unsupported_or_missing_trainer_before_resolving_outputs(
    tmp_path: Path, monkeypatch, method
):
    worker = _module("worker")
    exp_dir = tmp_path / "uncreated-bundle"

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("unsupported trainer reached bundle, asset, or output handling")

    monkeypatch.setattr(worker, "resolve_exp_dir", unexpected_call)
    monkeypatch.setattr(worker, "check_assets", unexpected_call)
    monkeypatch.setattr(worker, "materialize_manifest", unexpected_call)

    with pytest.raises(ValueError, match="supervised diffusion.*paired-video"):
        worker.prepare_stage(
            SimpleNamespace(method=method),
            "train",
            exp_dir=exp_dir,
        )

    assert not exp_dir.exists()


def test_prepare_stage_allows_score_distillation_and_infer_without_trainer_guard(
    tmp_path: Path, monkeypatch
):
    worker = _module("worker")

    def resolver_sentinel(*_args, **_kwargs):
        raise RuntimeError("next resolver reached")

    monkeypatch.setattr(worker, "resolve_exp_dir", resolver_sentinel)

    with pytest.raises(RuntimeError, match="next resolver reached"):
        worker.prepare_stage(
            SimpleNamespace(method={"trainer": "score_distillation"}),
            "train",
            exp_dir=tmp_path / "dmd-bundle",
        )

    with pytest.raises(RuntimeError, match="next resolver reached"):
        worker.prepare_stage(
            SimpleNamespace(method={}),
            "infer",
            exp_dir=tmp_path / "infer-bundle",
        )


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
        "  autotune_hyperparam_root: autotune_hyperparam\n"
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "  state_root: state\n"
        "  readiness_root: readiness\n"
        "  ledger_csv: artifacts/experiment_results.csv\n"
        "backend: {primary: nm5}\n"
        "sandbox_script_folders: {nm5: slurm_scripts}\n"
        "sandbox_resources: {nm5: {partition: acc, qos: acc_ehpc, gpus_per_node: 4, gpu_type: H100, gpu_usable_memory_gib: 63.29, gpu_non_torch_reserve_gib: 7.0, max_nodes_per_job: 1, cpus_per_gpu: 20, timeout_kill_after_seconds: 300, finalization_grace_seconds: 600, train_gpus: 4, infer_gpus: 1}}\n"
        "backend_env: {nm5: {source_note: deepresearch-sandbox/config_nm5.txt, required_keys: [NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT, NM5_LOGIN_SSH, NM5_HF_HOME, NM5_HF_HUB_CACHE, NM5_MODELSCOPE_CACHE, NM5_TORCH_HOME, NM5_MPLCONFIGDIR, NM5_WANDB_CACHE_DIR, WANDB_API_KEY, WANDB_ENTITY]}}\n"
        "environment: {env_manager: conda, env_root: scale_up_outputs/envs, python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python, asset_root: Self-Forcing-blockwise-layerwise, python_overlay: envs/hydra_overlay, allowed_external_roots: [scale_up_outputs/envs, Self-Forcing-blockwise-layerwise], cache_env_sources: {HF_HOME: NM5_HF_HOME, HF_HUB_CACHE: NM5_HF_HUB_CACHE, MODELSCOPE_CACHE: NM5_MODELSCOPE_CACHE, TORCH_HOME: NM5_TORCH_HOME, MPLCONFIGDIR: NM5_MPLCONFIGDIR, WANDB_CACHE_DIR: NM5_WANDB_CACHE_DIR}}\n"
        "policy:\n"
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
