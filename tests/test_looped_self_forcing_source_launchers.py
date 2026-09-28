from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "scripts" / "looped_self_forcing_pipeline"
BUNDLE_SCRIPTS = REPO_ROOT / "scale_up_outputs" / "looped_self_forcing_pipeline" / "slurm_scripts"
CUSTOM_LOGS_ROOT = "telemetry/pipeline-logs"
CUSTOM_SLURM_SCRIPTS_ROOT = "operations/slurm-launchers"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_source_tree_is_self_contained_code_without_runtime_assets():
    source = PIPELINE_DIR / "source"
    files = [path for path in source.rglob("*") if path.is_file() and "__pycache__" not in path.parts]

    assert (source / "train.py").is_file()
    assert (source / "inference.py").is_file()
    assert (source / "utils/temporal_loop.py").is_file()
    assert (source / "model/causvid.py").is_file()
    assert (source / "LICENSE").is_file()
    assert files
    assert all(path.suffix in {".py", ".yaml"} or path.name == "LICENSE" for path in files)
    assert not any(part in {"wan_models", "checkpoints", "datasets"} for path in files for part in path.parts)


def test_source_license_is_committed_and_matches_its_manifest_hash():
    copied_license = PIPELINE_DIR / "source" / "LICENSE"
    manifest = json.loads((PIPELINE_DIR / "COPY_MANIFEST.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["copy"] == "source/LICENSE")
    text = copied_license.read_text(encoding="utf-8")

    assert "Apache License" in text[:256]
    assert "Version 2.0" in text[:256]
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text
    assert _sha256(copied_license) == entry["sha256"]


def test_copy_manifest_covers_every_migrated_source_file_and_exporter():
    manifest_path = PIPELINE_DIR / "COPY_MANIFEST.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    copies = {entry["copy"] for entry in entries}
    source_files = {
        path.relative_to(PIPELINE_DIR).as_posix()
        for path in (PIPELINE_DIR / "source").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert copies == source_files | {"export_checkpoint.py"}
    for entry in entries:
        copied = PIPELINE_DIR / entry["copy"]
        assert copied.is_file(), entry["copy"]
        assert _sha256(copied) == entry["sha256"], entry["copy"]


def test_nm5_slurm_launcher_uses_private_runtime_inputs_and_durable_logs():
    submit = BUNDLE_SCRIPTS / "nm5_submit.sh"
    worker = BUNDLE_SCRIPTS / "nm5_worker.sbatch"
    common = BUNDLE_SCRIPTS / "launcher_common.sh"
    assert submit.is_file() and worker.is_file() and common.is_file()
    source = (
        submit.read_text(encoding="utf-8")
        + worker.read_text(encoding="utf-8")
        + common.read_text(encoding="utf-8")
    )

    for name in (
        "NM5_DEEPRESEARCH_ROOT",
        "NM5_ACCOUNT",
        "NM5_WORKSPACE_ROOT",
        "python_binary",
        "asset_root",
        "runtime_partition",
        "runtime_qos",
        "gpu_type",
        "timeout_kill_after_seconds",
        "finalization_grace_seconds",
        "load_runtime_config",
        "load_nm5_cache_environment",
        "cache_env_sources",
        "SUE_PYTHON",
        "SUE_ASSET_ROOT",
        "SUE_SCRIPTS_DIR",
    ):
        assert name in source
    assert "user.yaml" in submit.read_text(encoding="utf-8")
    assert "silly" in submit.read_text(encoding="utf-8")
    assert re.search(r'--output="\$logs_root', source)
    assert re.search(r'--error="\$logs_root', source)
    assert "resolve_logs_root" in source
    assert "gpus=4" in submit.read_text(encoding="utf-8")
    assert "gpus=1" in submit.read_text(encoding="utf-8")
    assert "--export=ALL" not in source
    assert "WANDB_API_KEY" in source
    assert "WANDB_ENTITY" in source
    assert "SUE_GPU_MODEL_PREFLIGHT" in worker.read_text(encoding="utf-8")


def test_autodl_direct_tmux_launcher_preflights_resources_and_logs_in_bundle():
    launcher = BUNDLE_SCRIPTS / "autodl_run.sh"
    assert launcher.is_file()
    source = launcher.read_text(encoding="utf-8")

    assert "AUTODL_DEEPRESEARCH_ROOT" in source
    assert "SUE_PYTHON" in source
    assert "tmux" in source
    assert "torch.cuda.device_count()" in source
    assert "expected_gpus" in source
    assert "SUE_EXP_DIR" in source and "logs_root" in source
    assert "resolve_logs_root" in source
    assert "SUE_ASSET_ROOT" in source
    assert "WANDB_API_KEY" in source
    assert "WANDB_ENTITY" in source
    assert "--export=ALL" not in source


def _runner_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    root = tmp_path / "deepresearch"
    workspace = root / "workspace" / "looped-flow-matching"
    pipeline = workspace / "scripts" / "looped_self_forcing_pipeline" / "pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.touch()
    (pipeline.parent / "__init__.py").touch()
    shutil.copyfile(REPO_ROOT / "scripts/looped_self_forcing_pipeline/config.py", pipeline.parent / "config.py")
    (workspace / "user.yaml").write_text("username: tester\n", encoding="utf-8")

    exp_dir = tmp_path / "bundle"
    source_config = REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config"
    shutil.copytree(source_config, exp_dir / "config", dirs_exist_ok=True)
    (exp_dir / "config/runtime.yaml").write_text(
        "paths:\n"
        "  config_root: config\n"
        "  autotune_hyperparam_root: autotune_hyperparam\n"
        "  datasets_root: prompt-data\n"
        f"  slurm_scripts_root: {CUSTOM_SLURM_SCRIPTS_ROOT}\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: checkpoints\n"
        "  final_result_root: final\n"
        f"  logs_root: {CUSTOM_LOGS_ROOT}\n"
        "  state_root: state\n"
        "  readiness_root: readiness\n"
        "  ledger_csv: artifacts/experiment_results.csv\n"
        "backend: {primary: nm5}\n"
        f"sandbox_script_folders:\n  nm5: {CUSTOM_SLURM_SCRIPTS_ROOT}\n"
        "sandbox_resources:\n"
        "  nm5:\n"
        "    partition: acc\n"
        "    qos: acc_ehpc\n"
        "    gpus_per_node: 4\n"
        "    gpu_type: H100\n"
        "    gpu_usable_memory_gib: 63.29\n"
        "    gpu_non_torch_reserve_gib: 7.0\n"
        "    timeout_kill_after_seconds: 300\n"
        "    finalization_grace_seconds: 600\n"
        "    max_nodes_per_job: 1\n"
        "    cpus_per_gpu: 20\n"
        "    train_gpus: 4\n"
        "    infer_gpus: 1\n"
        "backend_env:\n"
        "  nm5:\n"
        "    source_note: deepresearch-sandbox/config_nm5.txt\n"
        "    required_keys: [NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT, NM5_LOGIN_SSH, NM5_HF_HOME, NM5_HF_HUB_CACHE, NM5_MODELSCOPE_CACHE, NM5_TORCH_HOME, NM5_MPLCONFIGDIR, NM5_WANDB_CACHE_DIR, WANDB_API_KEY, WANDB_ENTITY]\n"
        "environment:\n"
        "  env_manager: conda\n"
        "  env_root: scale_up_outputs/envs\n"
        "  python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python\n"
        "  asset_root: Self-Forcing-blockwise-layerwise\n"
        "  python_overlay: envs/hydra_overlay\n"
        "  allowed_external_roots:\n"
        "    - scale_up_outputs/envs\n"
        "    - Self-Forcing-blockwise-layerwise\n"
        "  cache_env_sources: {HF_HOME: NM5_HF_HOME, HF_HUB_CACHE: NM5_HF_HUB_CACHE, MODELSCOPE_CACHE: NM5_MODELSCOPE_CACHE, TORCH_HOME: NM5_TORCH_HOME, MPLCONFIGDIR: NM5_MPLCONFIGDIR, WANDB_CACHE_DIR: NM5_WANDB_CACHE_DIR}\n"
        "policy:\n"
        "  wandb_policy: {}\n",
        encoding="utf-8",
    )
    (exp_dir / "slurm_scripts").mkdir()
    (exp_dir / "slurm_scripts/nm5_worker.sbatch").touch()
    (exp_dir / "envs/hydra_overlay").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_calls = tmp_path / "python_calls.txt"
    fake_python = tmp_path / "sue-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'ASSET=%s\\n' \"${SUE_ASSET_ROOT-}\"\n"
        "  printf 'SCRIPTS=%s\\n' \"${SUE_SCRIPTS_DIR-}\"\n"
        "  printf 'SUE_PYTHONPATH=%s\\n' \"${SUE_PYTHONPATH-}\"\n"
        "  printf 'CUSTOM_COMMON=%s\\n' \"${CUSTOM_COMMON_SELECTED-}\"\n"
        "  printf 'WANDB_API_KEY=%s\\n' \"${WANDB_API_KEY-}\"\n"
        "  printf 'WANDB_ENTITY=%s\\n' \"${WANDB_ENTITY-}\"\n"
        "  printf 'ARGS'\n"
        "  printf '\\t%s' \"$@\"\n"
        "  printf '\\n'\n"
        f"}} >> {shlex.quote(str(python_calls))}\n"
        "if [[ -n \"${ORDER_TRACE:-}\" ]]; then printf 'PY' >> \"$ORDER_TRACE\"; printf '\\t%s' \"$@\" >> \"$ORDER_TRACE\"; printf '\\n' >> \"$ORDER_TRACE\"; fi\n"
        "if [[ \"${2:-}\" == resolve-experiment-name ]]; then\n"
        "  if [[ -n \"${FAKE_EXPERIMENT_NAME:-}\" ]]; then printf '%s\\n' \"$FAKE_EXPERIMENT_NAME\"; exit 0; fi\n"
        "  method='unknown'; run_id='unknown'; stage='unknown'\n"
        "  for argument in \"$@\"; do\n"
        "    case \"$argument\" in method=*) method=\"${argument#method=}\";; run_id=*) run_id=\"${argument#run_id=}\";; stage=*) stage=\"${argument#stage=}\";; esac\n"
        "  done\n"
        "  printf '%s\\n' \"$method-$run_id-$stage\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == - ]]; then\n"
        "  source_code=\"$(cat)\"\n"
        "  if [[ \"$source_code\" == *SUE_GPU_MODEL_PREFLIGHT* ]]; then\n"
        "    if [[ -n \"$FAKE_GPU_MODEL_PREFLIGHT_FAIL\" ]]; then printf 'simulated allocated GPU model failure\\n' >&2; exit 42; fi\n"
        "    exit 0\n"
        "  fi\n"
        "  if [[ \"$source_code\" == *resolve_runtime_paths* || \"$source_code\" == *load_runtime_config* || \"$source_code\" == *compose_config* ]]; then\n"
        f"    printf '%s\\n' \"$source_code\" | {shlex.quote(sys.executable)} - \"${{@:2}}\"\n"
        "    exit $?\n"
        "  fi\n"
        "  if [[ \"${2:-}\" == */user.yaml ]]; then printf 'tester-\\n'; fi\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    sbatch_calls = tmp_path / "sbatch_args.bin"
    sbatch_count = tmp_path / "sbatch_count.txt"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"count=0; [[ -f {shlex.quote(str(sbatch_count))} ]] && count=\"$(cat {shlex.quote(str(sbatch_count))})\"\n"
        "count=$((count + 1))\n"
        f"printf '%s' \"$count\" > {shlex.quote(str(sbatch_count))}\n"
        f"printf '%s\\0' \"$@\" >> {shlex.quote(str(sbatch_calls))}\nprintf '\\0' >> {shlex.quote(str(sbatch_calls))}\n"
        "if [[ -n \"${ORDER_TRACE:-}\" ]]; then printf 'SBATCH probe=%s step=%s\\n' \"${SUE_MEM_PROBE-}\" \"${SUE_MEM_PROBE_STEP-}\" >> \"$ORDER_TRACE\"; fi\n"
        "printf '%s\\n' \"$((50122 + count))\"\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)

    scontrol = bin_dir / "scontrol"
    scontrol.write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"$1\" == show && \"$2\" == partition ]] || exit 2\n"
        "printf 'PartitionName=acc MaxTime=2-00:00:00\\n'\n",
        encoding="utf-8",
    )
    scontrol.chmod(0o755)

    tmux_calls = tmp_path / "tmux_calls.txt"
    tmux = bin_dir / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \" $* \" == *' has-session '* ]]; then exit 1; fi\n"
        "{\n"
        "  printf 'ARGS'\n"
        "  printf '\\t%s' \"$@\"\n"
        "  printf '\\nASSET=%s\\n' \"${SUE_ASSET_ROOT-}\"\n"
        "  printf 'EXECUTION_ID=%s\\n' \"${SUE_EXECUTION_ID-}\"\n"
        "  printf 'WANDB_API_KEY=%s\\n' \"${WANDB_API_KEY-}\"\n"
        "  printf 'WANDB_ENTITY=%s\\n' \"${WANDB_ENTITY-}\"\n"
        f"}} >> {shlex.quote(str(tmux_calls))}\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)

    assets = tmp_path / "private-fixture-assets"
    (assets / "checkpoints").mkdir(parents=True)
    (assets / "wan_models").mkdir()
    private_workspace = tmp_path / "private-workspace"
    derived_assets = private_workspace / "Self-Forcing-blockwise-layerwise"
    (derived_assets / "checkpoints").mkdir(parents=True)
    (derived_assets / "wan_models").mkdir()
    configured_python = (
        private_workspace
        / "scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python"
    )
    configured_python.parent.mkdir(parents=True)
    shutil.copyfile(fake_python, configured_python)
    configured_python.chmod(0o755)
    cache_directories = {}
    for target in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "MODELSCOPE_CACHE",
        "TORCH_HOME",
        "MPLCONFIGDIR",
        "WANDB_CACHE_DIR",
    ):
        directory = private_workspace / "cache" / target.lower()
        directory.mkdir(parents=True)
        cache_directories[target] = str(directory)
    environment = {
        **os.environ,
        "NM5_DEEPRESEARCH_ROOT": str(root),
        "AUTODL_DEEPRESEARCH_ROOT": str(root),
        "NM5_ACCOUNT": "test-account",
        "NM5_WORKSPACE_ROOT": str(private_workspace),
        "SUE_PYTHON": str(fake_python),
        "SUE_EXP_DIR": str(exp_dir),
        "SUE_ASSET_ROOT": str(assets),
        "SUE_SCRIPTS_DIR": str(BUNDLE_SCRIPTS),
        "WANDB_API_KEY": "fixture-only-key",
        "WANDB_ENTITY": "fixture-only-entity",
        "SBATCH_CAPTURE": str(sbatch_calls),
        "ORDER_TRACE": str(tmp_path / "order_trace.txt"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    for target, source_name in {
        "HF_HOME": "NM5_HF_HOME",
        "HF_HUB_CACHE": "NM5_HF_HUB_CACHE",
        "MODELSCOPE_CACHE": "NM5_MODELSCOPE_CACHE",
        "TORCH_HOME": "NM5_TORCH_HOME",
        "MPLCONFIGDIR": "NM5_MPLCONFIGDIR",
        "WANDB_CACHE_DIR": "NM5_WANDB_CACHE_DIR",
    }.items():
        environment[source_name] = cache_directories[target]
    dispatcher = root / "scripts" / "preflight.sh"
    dispatcher.parent.mkdir(parents=True, exist_ok=True)
    dispatcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ -n \"${ORDER_TRACE:-}\" ]]; then printf 'DISPATCH' >> \"$ORDER_TRACE\"; printf '\\t%s' \"$@\" >> \"$ORDER_TRACE\"; printf '\\n' >> \"$ORDER_TRACE\"; fi\n",
        encoding="utf-8",
    )
    dispatcher.chmod(0o755)
    return root, workspace, exp_dir, bin_dir, environment


def test_nm5_launcher_forwards_allowed_overrides_and_asset_env_to_slurm(tmp_path: Path):
    _, _, exp_dir, _, environment = _runner_fixture(tmp_path)
    script = BUNDLE_SCRIPTS / "nm5_submit.sh"
    overrides = [
        "method=layerwise_l08_15_k2",
        "tracking=offline",
        "scale=smoke",
        "seed=17",
        "train.max_steps=20",
        "train.log_iters=5",
        "train.timeout_seconds=300",
        "train.checkpoint_interval_seconds=60",
        "train.resume_checkpoint=ckpt/prior/latest.pt",
    ]
    environment["FAKE_EXPERIMENT_NAME"] = "layerwise_l08_15_k2-run-one-train"

    result = subprocess.run(
        ["bash", str(script), "train", "run-one", *overrides],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "50123"
    sbatch_args = (tmp_path / "sbatch_args.bin").read_bytes().decode().rstrip("\0").split("\0")
    assert sbatch_args[-(len(overrides) + 2) :] == ["train", "run-one", *overrides]
    export_values = next(arg.partition("=")[2] for arg in sbatch_args if arg.startswith("--export="))
    for name in (
        "SUE_ASSET_ROOT",
        "SUE_SCRIPTS_DIR",
        "SUE_PYTHONPATH",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
    ):
        assert name in export_values
    assert "SUE_MEM_PROBE" in export_values and "SUE_MEM_PROBE_STEP" in export_values
    for name in ("HF_HOME", "HF_HUB_CACHE", "MODELSCOPE_CACHE", "TORCH_HOME", "MPLCONFIGDIR", "WANDB_CACHE_DIR"):
        assert name in export_values
    assert environment["SUE_ASSET_ROOT"] not in " ".join(sbatch_args)
    assert environment["WANDB_API_KEY"] not in " ".join(sbatch_args)
    assert environment["WANDB_ENTITY"] not in " ".join(sbatch_args)
    assert all(value not in " ".join(sbatch_args) for value in environment.values() if value.startswith(str(Path(environment["NM5_WORKSPACE_ROOT"]))))
    calls = (tmp_path / "python_calls.txt").read_text(encoding="utf-8")
    assert "ASSET=" + environment["SUE_ASSET_ROOT"] in calls
    assert "WANDB_API_KEY=fixture-only-key" in calls
    assert "WANDB_ENTITY=fixture-only-entity" in calls
    assert "SCRIPTS=" + str(BUNDLE_SCRIPTS) in calls
    assert all(override in calls for override in overrides)
    assert "fixture-only-key" not in result.stdout + result.stderr
    assert environment["SUE_ASSET_ROOT"] not in result.stdout + result.stderr
    assert (exp_dir / CUSTOM_LOGS_ROOT).is_dir()
    assert any(arg.startswith("--output=") and f"/{CUSTOM_LOGS_ROOT}/" in arg for arg in sbatch_args)
    assert any(arg.startswith("--error=") and f"/{CUSTOM_LOGS_ROOT}/" in arg for arg in sbatch_args)
    assert "--time=00:20:00" in sbatch_args
    assert "--job-name=tester-layerwise_l08_15_k2-run-one-train" in sbatch_args
    assert str(BUNDLE_SCRIPTS / "nm5_worker.sbatch") in sbatch_args
    assert CUSTOM_LOGS_ROOT not in result.stdout + result.stderr


def test_nm5_launcher_resolves_python_and_assets_from_runtime_when_private_overrides_are_unset(
    tmp_path: Path,
):
    _, _, _, _, environment = _runner_fixture(tmp_path)
    environment.pop("SUE_PYTHON")
    environment.pop("SUE_ASSET_ROOT")

    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "nm5_submit.sh"), "train", "run-runtime-env", "scale=smoke"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    expected_python = Path(environment["NM5_WORKSPACE_ROOT"]) / (
        "scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python"
    )
    expected_assets = Path(environment["NM5_WORKSPACE_ROOT"]) / "Self-Forcing-blockwise-layerwise"
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "python_calls.txt").read_text(encoding="utf-8")
    assert "ASSET=" + str(expected_assets) in calls
    assert "SCRIPTS=" + str(BUNDLE_SCRIPTS) in calls
    assert str(expected_python) not in result.stdout + result.stderr
    assert str(expected_assets) not in result.stdout + result.stderr


def test_nm5_launcher_rejects_zero_training_seed_before_submission(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)

    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "nm5_submit.sh"), "train", "run-zero", "seed=0"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "non-zero" in result.stderr
    assert not (tmp_path / "sbatch_args.bin").exists()


def test_nm5_launcher_rejects_timeout_above_partition_maxtime(tmp_path: Path):
    _, _, _, bin_dir, environment = _runner_fixture(tmp_path)
    scontrol = bin_dir / "scontrol"
    scontrol.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'PartitionName=acc MaxTime=00:04:00\\n'\n",
        encoding="utf-8",
    )
    scontrol.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(BUNDLE_SCRIPTS / "nm5_submit.sh"),
            "train",
            "run-too-long",
            "train.timeout_seconds=300",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "MaxTime" in result.stderr
    assert not (tmp_path / "sbatch_args.bin").exists()


def test_nm5_pair_submit_preflights_both_then_submits_serial_four_gpu_jobs(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)
    script = BUNDLE_SCRIPTS / "nm5_submit_pair_23_30.sh"

    result = subprocess.run(
        ["bash", str(script), "pair-23-30", "scale=smoke"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    captured = (tmp_path / "sbatch_args.bin").read_bytes()
    calls = [call.decode().split("\0") for call in captured.split(b"\0\0") if call]
    assert len(calls) == 2
    baseline, low_lr = calls
    assert "--gres=gpu:4" in baseline and "--gres=gpu:4" in low_lr
    assert "pair-23-30-k3-lr5gen" in baseline[-3:]
    assert "pair-23-30-k2-lr5both" in low_lr[-3:]
    assert "--job-name=tester-layerwise_l23_30_k3_lr5gen-pair-23-30-k3-lr5gen-train" in baseline
    assert "--job-name=tester-layerwise_l23_30_k2_lr5both-pair-23-30-k2-lr5both-train" in low_lr
    assert "--dependency=afterany:50123" in low_lr
    assert not any("--dependency=" in argument for argument in baseline)
    assert baseline.index("method=layerwise_l23_30_k3_lr5gen") >= 0
    assert low_lr.index("method=layerwise_l23_30_k2_lr5both") >= 0

    trace = (tmp_path / "order_trace.txt").read_text(encoding="utf-8").splitlines()
    first_sbatch = next(index for index, line in enumerate(trace) if line.startswith("SBATCH"))
    dispatch_events = [index for index, line in enumerate(trace) if line.startswith("DISPATCH")]
    assert dispatch_events and dispatch_events[0] < first_sbatch
    assert "--adapter" in trace[dispatch_events[0]]
    assert "looped_self_forcing_pipeline/preflight.py" in trace[dispatch_events[0]]
    preflight_events = [
        index for index, line in enumerate(trace) if "check-train" in line
    ]
    assert len(preflight_events) >= 2
    assert preflight_events[0] < first_sbatch and preflight_events[1] < first_sbatch
    assert [line for line in trace if line.startswith("SBATCH")] == [
        "SBATCH probe=1 step=3",
        "SBATCH probe=1 step=3",
    ]


def test_nm5_pair_full_requires_after_smoke_before_preflight_or_submission(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(BUNDLE_SCRIPTS / "nm5_submit_pair_23_30.sh"),
            "full-23-30",
            "scale=full",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "after-smoke" in result.stderr
    assert not (tmp_path / "sbatch_args.bin").exists()
    assert not (tmp_path / "order_trace.txt").exists()


def test_nm5_pair_full_requires_measured_timeout_before_preflight_or_submission(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(BUNDLE_SCRIPTS / "nm5_submit_pair_23_30.sh"),
            "full-23-30",
            "scale=full",
            "--after-smoke",
            "smoke-23-30",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "train.timeout_seconds" in result.stderr
    assert not (tmp_path / "sbatch_args.bin").exists()
    assert not (tmp_path / "order_trace.txt").exists()


def test_nm5_pair_full_verifies_smoke_before_static_preflight_and_disables_probe(
    tmp_path: Path,
):
    _, _, _, _, environment = _runner_fixture(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(BUNDLE_SCRIPTS / "nm5_submit_pair_23_30.sh"),
            "full-23-30",
            "scale=full",
            "--after-smoke",
            "smoke-23-30",
            "train.timeout_seconds=3600",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    trace = (tmp_path / "order_trace.txt").read_text(encoding="utf-8").splitlines()
    gate_index = next(index for index, line in enumerate(trace) if "check-smoke-readiness" in line)
    dispatch_index = next(index for index, line in enumerate(trace) if line.startswith("DISPATCH"))
    first_submit = next(index for index, line in enumerate(trace) if line.startswith("SBATCH"))
    assert gate_index < dispatch_index < first_submit
    dispatch_line = trace[dispatch_index]
    assert "--smoke-pair-id" in dispatch_line and "smoke-23-30" in dispatch_line
    assert [line for line in trace if line.startswith("SBATCH")] == [
        "SBATCH probe=0 step=3",
        "SBATCH probe=0 step=3",
    ]
    captured = (tmp_path / "sbatch_args.bin").read_bytes().decode().rstrip("\0").split("\0")
    assert "--time=01:15:00" in captured


@pytest.mark.parametrize(
    ("stage", "override", "message"),
    [
        ("train", "train.seed=0", "non-zero"),
        ("infer", "infer.num_output_frames=120", "fixed at 123"),
    ],
)
def test_autodl_launcher_rejects_random_training_seed_and_nonstandard_latent_frames(
    tmp_path: Path, stage: str, override: str, message: str
):
    _, _, _, _, environment = _runner_fixture(tmp_path)

    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "autodl_run.sh"), stage, "run-invalid", override],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "tmux_calls.txt").exists()


def test_nm5_worker_reuses_overrides_for_preflight_and_worker(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)
    overrides = [
        "method=layerwise_l08_15_k2",
        "scale=smoke",
        "seed=13",
        "train.max_steps=15",
        "train.log_iters=5",
    ]

    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "nm5_worker.sbatch"), "train", "run-worker", *overrides],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "python_calls.txt").read_text(encoding="utf-8")
    argument_lines = [line for line in calls.splitlines() if line.startswith("ARGS")]
    assert len(argument_lines) == 3
    assert "-\t" in argument_lines[0]
    assert "check-train" in argument_lines[1]
    assert "train-worker" in argument_lines[2]
    assert all(all(override in line for override in overrides) for line in argument_lines[1:])
    assert calls.count("ASSET=" + environment["SUE_ASSET_ROOT"]) == 3


def test_nm5_worker_stops_before_pipeline_when_allocated_gpu_model_check_fails(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)
    environment["FAKE_GPU_MODEL_PREFLIGHT_FAIL"] = "1"

    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "nm5_worker.sbatch"), "train", "run-h100-check"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 42
    assert "simulated allocated GPU model failure" in result.stderr
    calls = (tmp_path / "python_calls.txt").read_text(encoding="utf-8")
    assert "check-train" not in calls
    assert "train-worker" not in calls


def test_nm5_worker_sources_submitter_supplied_durable_scripts_dir(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)
    durable_scripts = tmp_path / "durable-script-root"
    durable_scripts.mkdir()
    shutil.copyfile(
        BUNDLE_SCRIPTS / "launcher_common.sh",
        durable_scripts / "launcher_common.sh",
    )
    with (durable_scripts / "launcher_common.sh").open("a", encoding="utf-8") as stream:
        stream.write("export CUSTOM_COMMON_SELECTED=durable-root\n")
    environment["SUE_SCRIPTS_DIR"] = str(durable_scripts)

    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "nm5_worker.sbatch"), "train", "run-worker"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "python_calls.txt").read_text(encoding="utf-8")
    assert calls.count("SCRIPTS=" + str(durable_scripts)) == 3
    assert calls.count("CUSTOM_COMMON=durable-root") == 3


def test_nm5_worker_fails_closed_without_durable_scripts_dir(tmp_path: Path):
    _, _, _, _, environment = _runner_fixture(tmp_path)
    environment.pop("SUE_SCRIPTS_DIR")

    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "nm5_worker.sbatch"), "train", "run-worker"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "SUE_SCRIPTS_DIR" in result.stderr
    assert not (tmp_path / "python_calls.txt").exists()


def test_autodl_launcher_forwards_infer_overrides_and_selected_env_to_tmux(tmp_path: Path):
    _, _, exp_dir, _, environment = _runner_fixture(tmp_path)
    script = BUNDLE_SCRIPTS / "autodl_run.sh"
    overrides = [
        "method=layerwise_l16_23_k2",
        "tracking=offline",
        "scale=smoke",
        "seed=23",
        "infer.checkpoint=ckpt/run-one/inference.pt",
        "infer.num_samples=2",
        "infer.num_output_frames=123",
        "infer.seed=5",
        "infer.use_ema=true",
    ]

    result = subprocess.run(
        ["bash", str(script), "infer", "run-two", *overrides],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    tmux_capture = (tmp_path / "tmux_calls.txt").read_text(encoding="utf-8")
    python_capture = (tmp_path / "python_calls.txt").read_text(encoding="utf-8")
    assert "ASSET=" + environment["SUE_ASSET_ROOT"] in python_capture
    assert "WANDB_API_KEY=fixture-only-key" in python_capture
    assert "WANDB_ENTITY=fixture-only-entity" in python_capture
    assert "ASSET=" + environment["SUE_ASSET_ROOT"] in tmux_capture
    assert "EXECUTION_ID=loop-run-two-infer" in tmux_capture
    assert "WANDB_API_KEY=fixture-only-key" in tmux_capture
    assert "WANDB_ENTITY=fixture-only-entity" in tmux_capture
    assert all(override in tmux_capture for override in overrides)
    assert environment["SUE_ASSET_ROOT"] not in tmux_capture.partition("ARGS")[2].partition("\nASSET=")[0]
    assert "fixture-only-key" not in tmux_capture.partition("ARGS")[2].partition("\nASSET=")[0]
    assert "tmux -L" in result.stdout
    assert "fixture-only-key" not in result.stdout + result.stderr
    assert environment["SUE_ASSET_ROOT"] not in result.stdout + result.stderr
    assert list((exp_dir / CUSTOM_LOGS_ROOT).glob("run-two_infer_*.log"))
    assert CUSTOM_LOGS_ROOT not in result.stdout + result.stderr


def test_autodl_defaults_online_only_when_both_credentials_exist(tmp_path: Path):
    _, _, _, _, with_credentials = _runner_fixture(tmp_path / "with-creds")
    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "autodl_run.sh"), "infer", "run-online"],
        check=False,
        capture_output=True,
        text=True,
        env=with_credentials,
    )
    assert result.returncode == 0, result.stderr
    assert "tracking=online" in (tmp_path / "with-creds/python_calls.txt").read_text()
    assert "tracking=online" in (tmp_path / "with-creds/tmux_calls.txt").read_text()

    _, _, _, _, without_credentials = _runner_fixture(tmp_path / "without-creds")
    without_credentials.pop("WANDB_API_KEY")
    without_credentials.pop("WANDB_ENTITY")
    result = subprocess.run(
        ["bash", str(BUNDLE_SCRIPTS / "autodl_run.sh"), "infer", "run-offline"],
        check=False,
        capture_output=True,
        text=True,
        env=without_credentials,
    )
    assert result.returncode == 0, result.stderr
    assert "tracking=offline" in (tmp_path / "without-creds/python_calls.txt").read_text()
    assert "tracking=offline" in (tmp_path / "without-creds/tmux_calls.txt").read_text()


def test_launchers_reject_protected_and_unapproved_overrides_before_backend_access(tmp_path: Path):
    cases = [
        ("stage=infer", "stage"),
        ("run_id=other", "run_id"),
        ("backend=autodl", "backend"),
        ("assets.root=/private/assets", "assets"),
        ("infer.prompt_file=/private/prompts.txt", "infer.prompt_file"),
        ("infer.checkpoint=/private/checkpoint.pt", "relative"),
        ("train.max_steps=3\ntracking=online", "newline"),
    ]
    for case, label in cases:
        result = subprocess.run(
            ["bash", str(BUNDLE_SCRIPTS / "autodl_run.sh"), "infer", "run-one", case],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": os.environ.get("PATH", "")},
        )
        assert result.returncode == 2, case
        assert label in result.stderr.lower(), result.stderr


def test_pipeline_docs_resolve_legacy_launcher_and_asset_prompt_contracts():
    readme = (PIPELINE_DIR / "README.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    runtime = (REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config/runtime.yaml").read_text(encoding="utf-8")
    method_configs = [
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config/method/layerwise_l16_23_k2.yaml",
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config/method/layerwise_l08_15_k2.yaml",
        REPO_ROOT / "scale_up_outputs/looped_self_forcing_pipeline/config/method/blockwise_l16_23_k2.yaml",
    ]

    assert "SUE_ASSET_ROOT" in readme
    assert "checkpoints/" in readme and "wan_models/" in readme
    assert "paths.datasets_root" in readme
    assert "train.prompt_file" in readme and "infer.prompt_file" in readme
    assert "vidprom_filtered_extended.txt" in readme
    assert "inference_prompts.txt" in readme
    assert "nm5_submit.sh" in readme and "autodl_run.sh" in readme
    assert "paths.logs_root" in readme
    assert "paths.config_root` is fixed to `config`" in readme
    assert "non-zero seed `1`" in readme
    assert "The AutoDL launcher selects online tracking by default" in readme
    assert "otherwise it keeps offline mode" in readme
    assert "not nested output path components" in runtime
    assert "W&B experiment" in readme
    assert "123 latent frames" in readme and "489 video frames at 16 fps" in readme
    assert "trainable_parameter_count" in readme
    assert "nm5_submit_pair_23_30.sh" in readme
    assert "layerwise_l23_30_k3_lr5gen" in readme
    assert "layerwise_l23_30_k2_lr5both" in readme
    assert "afterany" in readme and "scale=smoke" in readme and "scale=full" in readme
    assert 'SMOKE_PAIR_ID="<fresh-smoke-pair-id>"' in readme
    assert '--verify-smoke "$SMOKE_PAIR_ID"' in readme
    assert 'FULL_PAIR_ID="<fresh-full-pair-id>"' in readme
    assert '--after-smoke "$SMOKE_PAIR_ID"' in readme
    assert "train.timeout_seconds=<smoke-derived-seconds>" in readme
    assert "SUE_EXP_DIR/<pair_id>/max_parallel.json" in readme
    assert "SUE_MEM_PROBE=1" in readme and "step 3" in readme
    assert "56.29 GiB" in readme and "full EMA checkpoint" in readme
    assert "LoRA, model/checkpoint hashes" in readme and "must match the smoke manifest" in readme
    assert "native save/checkpoint opportunity cadence" in readme
    assert "scalar logging every 100 steps" in readme
    assert "console progress every 10 steps" in readme
    assert "WANDB_API_KEY" in readme and "WANDB_ENTITY" in readme
    assert "export_checkpoint.py" in readme and "--full-checkpoint" in readme
    assert "infer.checkpoint" in readme and "ckpt/$TRAIN_RUN_ID/inference.pt" in readme
    assert "assets.root=" not in readme
    assert "older VBench search" in agents
    assert "Self-Forcing-blockwise-layerwise/" in agents
    assert "SUE_ASSET_ROOT` is an optional private override" in agents
    assert "scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/" in agents
    assert "this workspace's `SCALE_UP.md`" in agents
    assert "`memory/sue/SCALE_UP.md`" in agents
    for path in method_configs:
        text = path.read_text(encoding="utf-8")
        assert "scripts/looped_self_forcing_pipeline/source/model/base.py" in text
        assert "scripts/looped_self_forcing_pipeline/source/train_blockwise_lora_dmd.py" in text
        assert "scripts/looped_self_forcing_pipeline/source/utils/temporal_loop.py" in text
        assert "scripts/looped_self_forcing_pipeline/source/pipeline/self_forcing_training.py" in text
        assert "(model/base.py:" not in text
