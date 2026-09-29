from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module(name: str):
    import importlib
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    scripts = str(repo_root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(f"looped_self_forcing_pipeline.{name}")


def _write_valid_smoke_pair(exp_dir: Path, pair_id: str = "smoke-pair"):
    torch = pytest.importorskip("torch")
    config_module = _module("config")
    manifest_module = _module("manifest")
    ledger_module = _module("ledger")
    roots = config_module.resolve_runtime_paths(exp_dir)
    capacity_path = exp_dir / pair_id / "max_parallel.json"
    capacity_path.parent.mkdir(parents=True)
    capacity_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sandbox": "nm5",
                "partition": "acc",
                "max_parallel": 4,
                "limiting_factor": "one four-GPU job per node",
                "evidence": {"queue_summary": "redacted"},
            }
        ),
        encoding="utf-8",
    )
    definitions = (
        ("layerwise_l23_30_k3_lr5gen", "k3-lr5gen", 3, 4e-7, 4e-7),
        ("layerwise_l23_30_k2_lr5both", "k2-lr5both", 2, 4e-7, 8e-8),
    )
    job_ids = []
    for profile, suffix, k, lr, lr_critic in definitions:
        run_id = f"{pair_id}-{suffix}"
        job_id = str(50100 + len(job_ids) + 1)
        job_ids.append(job_id)
        method = {
            "temporal_loop": {
                "mode": "layer",
                "layer_start": 22,
                "layer_end": 29,
                "k_min": k,
                "k_max": k,
                "training_enabled": True,
                "stop_grad_early": True,
            },
            "lora": {"enabled": True, "target_modules": ["self_attn.q", "self_attn.v"]},
            "lr": lr,
            "lr_critic": lr_critic,
        }
        train = {"max_steps": 10, "seed": 1}
        base_hash = "a" * 64
        manifest_config = {
            "backend": "nm5",
            "method": method,
            "train": train,
            "assets": {"generator_checkpoint_sha256": base_hash},
        }
        manifest_root = roots["artifacts_root"] / run_id / "config" / "train"
        manifest_module.materialize_manifest(manifest_root, "train", manifest_config)

        checkpoint_root = roots["ckpt_root"] / run_id
        checkpoint_root.mkdir(parents=True)
        torch.save(
            {
                "metadata": {
                    "step": 10,
                    "seed": 1,
                    "final": True,
                    "base_checkpoint_sha256": base_hash,
                    "temporal_loop": {
                        "mode": "layer",
                        "layer_start": 22,
                        "layer_end": 29,
                        "k_min": k,
                        "k_max": k,
                        "training_enabled": True,
                        "stop_grad_early": True,
                    },
                },
                "generator": {"lora.weight": torch.ones(2, dtype=torch.float32)},
            },
            checkpoint_root / "latest.pt",
        )
        (checkpoint_root / "model.pt").write_bytes(b"lightweight-final-model")

        run_root = roots["artifacts_root"] / run_id
        (run_root / "wandb_run_id.txt").write_text("wandb-smoke-id\n", encoding="utf-8")
        wandb_dir = run_root / "wandb"
        wandb_dir.mkdir()
        (wandb_dir / "offline-run-smoke.wandb").write_bytes(b"run")
        output_log = roots["logs_root"] / f"{run_id}_train_{job_id}.out"
        error_log = roots["logs_root"] / f"{run_id}_train_{job_id}.err"
        output_log.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for rank in range(4):
            lines.append(f"===== MEMPROBE rank={rank} deep trace at step 3 (train_generator=False) =====")
            lines.append(
                f"MEMPROBE step=3 rank={rank} step_exit: allocated=1.00GiB reserved=2.00GiB "
                "frag=1.00GiB peak=3.00GiB"
            )
        lines.append("SUE_GPU_MODEL_PREFLIGHT passed expected=H100 count=4")
        output_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        error_log.write_text("", encoding="utf-8")
        ledger_module.write_result(
            roots["ledger_csv"],
            {
                "method": "layer",
                "experiment_name": f"{profile}-{run_id}-train",
                "wandb_run_id": "wandb-smoke-id",
                "backend": "nm5",
                "status": "completed",
                "execution_id": job_id,
                "slurm_job_id": job_id,
                "seed": 1,
                "step": 10,
                "checkpoint_step": 10,
                "training_speed": 1.0,
                "checkpoint_or_output_path": str(checkpoint_root / "latest.pt"),
            },
        )
    with roots["ledger_csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    receipt_jobs = []
    for profile, suffix, _k, _lr, _lr_critic in definitions:
        run_id = f"{pair_id}-{suffix}"
        row = next(row for row in rows if row["experiment_name"] == f"{profile}-{run_id}-train")
        receipt_jobs.append(
            {
                "profile": profile,
                "run_id": run_id,
                "job_id": row["slurm_job_id"],
                "submitted_at": datetime.fromisoformat(row["timestamp"]).isoformat(),
            }
        )
    receipt_path = roots["artifacts_root"] / "pairs" / pair_id / "submission.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pair_id": pair_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "jobs": receipt_jobs,
            }
        ),
        encoding="utf-8",
    )
    return roots, job_ids


def test_nm5_gpu_contract_rejects_non_h100_or_mixed_visible_devices():
    worker = _module("worker")
    h100_names = ["NVIDIA H100 80GB HBM3"] * 4

    assert worker.require_gpu_model_names(
        h100_names, expected_model="H100", expected_count=4
    ) == tuple(h100_names)
    with pytest.raises(RuntimeError, match="device model mismatch"):
        worker.require_gpu_model_names(
            [*h100_names[:3], "NVIDIA A100-SXM4-80GB"],
            expected_model="H100",
            expected_count=4,
        )
    with pytest.raises(RuntimeError, match="exactly 4"):
        worker.require_gpu_model_names(
            h100_names[:3], expected_model="H100", expected_count=4
        )


def test_nm5_worker_timeout_uses_configured_kill_after_grace():
    worker = _module("worker")
    config = SimpleNamespace(
        backend=SimpleNamespace(name="nm5"),
        train={"timeout_seconds": 123},
    )
    prepared = {
        "timeout_kill_after_seconds": 300,
        "config_path": Path("train.yaml"),
        "checkpoint_root": Path("checkpoints/run"),
        "run_root": Path("artifacts/run"),
    }

    command = worker._command_for(config, "train", prepared)

    assert command[:5] == ["timeout", "-s", "TERM", "--kill-after=300", "123"]


def test_training_evidence_requires_exact_step_and_finite_state(tmp_path: Path):
    torch = pytest.importorskip("torch")
    worker = _module("worker")
    path = tmp_path / "checkpoint.pt"

    torch.save({"metadata": {"step": 9}, "generator": {"adapter": torch.tensor([1.0])}}, path)
    with pytest.raises(ValueError, match="step|expected"):
        worker.verify_training_checkpoint(path, expected_step=10)

    torch.save(
        {"metadata": {"step": 10}, "generator": {"adapter": torch.tensor([float("inf")])}},
        path,
    )
    with pytest.raises(ValueError, match="finite|NaN|Inf"):
        worker.verify_training_checkpoint(path, expected_step=10)


def test_training_evidence_seed_matches_the_ledger_seed(tmp_path: Path):
    torch = pytest.importorskip("torch")
    worker = _module("worker")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": {"step": 10, "seed": 17},
            "generator": {"adapter": torch.tensor([1.0])},
        },
        path,
    )

    evidence = worker.verify_training_checkpoint(
        path, expected_step=10, expected_seed=17
    )

    assert evidence["metadata"]["seed"] == 17
    with pytest.raises(ValueError, match="seed.*expected|expected.*seed"):
        worker.verify_training_checkpoint(path, expected_step=10, expected_seed=18)


def test_training_adapter_scope_rejects_valid_lora_key_outside_selected_layers():
    worker = _module("worker")
    method = {
        "layer_start": 22,
        "layer_end": 29,
        "lora": {"target_modules": ["self_attn.q", "self_attn.v"]},
    }
    state = {
        f"blocks.{layer}.self_attn.{target}.lora_{matrix}.weight": 1
        for layer in range(22, 30)
        for target in ("q", "v")
        for matrix in ("A", "B")
    }
    state["blocks.21.self_attn.q.lora_A.weight"] = 1

    with pytest.raises(ValueError, match="adapter scope differs.*extra"):
        worker._check_adapter_scope(state, method, "generator")


def test_inference_evidence_checks_each_expected_video_is_decodable(tmp_path: Path):
    worker = _module("worker")
    outputs = tmp_path / "output"
    outputs.mkdir()
    (outputs / "one.mp4").write_bytes(b"container")

    def invalid_probe(_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": []}), stderr="")

    with pytest.raises(RuntimeError, match="decod|video stream"):
        worker.verify_inference_outputs(
            outputs,
            expected_count=1,
            expected_frames=489,
            expected_fps=16,
            run_command=invalid_probe,
        )


def test_inference_evidence_checks_configured_frame_count_and_fps(tmp_path: Path):
    worker = _module("worker")
    outputs = tmp_path / "output"
    outputs.mkdir()
    (outputs / "one.mp4").write_bytes(b"video")

    def truncated_probe(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "nb_read_frames": "488",
                            "avg_frame_rate": "16/1",
                        }
                    ],
                    "format": {"duration": "30.5"},
                }
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="frame count.*488|expected.*489"):
        worker.verify_inference_outputs(
            outputs,
            expected_count=1,
            expected_frames=489,
            expected_fps=16,
            run_command=truncated_probe,
        )


def test_inference_evidence_accepts_full_489_frame_16_fps_output(tmp_path: Path):
    worker = _module("worker")
    outputs = tmp_path / "output"
    outputs.mkdir()
    video = outputs / "one.mp4"
    video.write_bytes(b"video")

    def valid_probe(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "nb_read_frames": "489",
                            "avg_frame_rate": "16/1",
                        }
                    ],
                    "format": {"duration": "30.5625"},
                }
            ),
            stderr="",
        )

    assert worker.verify_inference_outputs(
        outputs,
        expected_count=1,
        expected_frames=489,
        expected_fps=16,
        run_command=valid_probe,
    ) == [video]


def test_smoke_pair_requires_four_gpu_step_three_probe_and_matching_run_evidence(tmp_path: Path):
    exp_dir = tmp_path / "bundle"
    config_dir = exp_dir / "config"
    config_dir.mkdir(parents=True)
    source_runtime = Path(__file__).resolve().parents[1] / (
        "scale_up_outputs/looped_self_forcing_pipeline/config/runtime.yaml"
    )
    (config_dir / "runtime.yaml").write_bytes(source_runtime.read_bytes())
    _, job_ids = _write_valid_smoke_pair(exp_dir)
    verification = _module("verification")

    def completed_sacct(command, **_kwargs):
        job_id = command[command.index("-j") + 1]
        assert job_id in job_ids
        return SimpleNamespace(
            returncode=0,
            stdout=f"{job_id}|COMPLETED\n{job_id}.batch|COMPLETED\n",
            stderr="",
        )

    evidence = verification.verify_smoke_pair_evidence(
        exp_dir, "smoke-pair", run_command=completed_sacct
    )

    assert [item["job_id"] for item in evidence] == job_ids
    assert [item["k"] for item in evidence] == [3, 2]
    assert [item["probe_peak_gib"] for item in evidence] == [[3.0] * 4, [3.0] * 4]
    assert [item["probe_reserved_gib"] for item in evidence] == [[2.0] * 4, [2.0] * 4]
    assert [item["reserved_memory_headroom_gib"] for item in evidence] == [61.29, 61.29]
    assert [item["reserved_memory_headroom_by_rank_gib"] for item in evidence] == [
        [61.29] * 4,
        [61.29] * 4,
    ]


def test_smoke_readiness_stamp_is_relative_immutable_and_rechecks_hashes(tmp_path: Path):
    exp_dir = tmp_path / "bundle"
    config_dir = exp_dir / "config"
    config_dir.mkdir(parents=True)
    source_runtime = Path(__file__).resolve().parents[1] / (
        "scale_up_outputs/looped_self_forcing_pipeline/config/runtime.yaml"
    )
    (config_dir / "runtime.yaml").write_bytes(source_runtime.read_bytes())
    roots, job_ids = _write_valid_smoke_pair(exp_dir)
    verification = _module("verification")

    def completed_sacct(command, **_kwargs):
        job_id = command[command.index("-j") + 1]
        assert job_id in job_ids
        return SimpleNamespace(returncode=0, stdout=f"{job_id}|COMPLETED\n", stderr="")

    evidence = verification.verify_smoke_pair_evidence(
        exp_dir, "smoke-pair", run_command=completed_sacct
    )
    stamp = verification.write_smoke_readiness_stamp(exp_dir, "smoke-pair", evidence)
    stamp_text = stamp.read_text(encoding="utf-8")
    assert stamp.relative_to(exp_dir).as_posix() == "readiness/layerwise_23_30_smoke-pair.json"
    assert "smoke-pair/max_parallel.json" in stamp_text
    assert str(exp_dir) not in stamp_text

    checked_stamp, checked_evidence = verification.verify_smoke_readiness_stamp(
        exp_dir, "smoke-pair", run_command=completed_sacct
    )
    assert checked_stamp == stamp
    assert [item["run_id"] for item in checked_evidence] == [
        "smoke-pair-k3-lr5gen",
        "smoke-pair-k2-lr5both",
    ]

    log = roots["logs_root"] / f"smoke-pair-k3-lr5gen_train_{job_ids[0]}.out"
    with log.open("a", encoding="utf-8") as stream:
        stream.write("tampered after readiness stamp\n")
    with pytest.raises(RuntimeError, match="hashes or evidence"):
        verification.verify_smoke_readiness_stamp(
            exp_dir, "smoke-pair", run_command=completed_sacct
        )


@pytest.mark.parametrize("failure", ["pending", "missing-probe", "missing-gpu-model", "oom", "high-reserved"])
def test_smoke_pair_rejects_incomplete_scheduler_or_memory_probe_evidence(
    tmp_path: Path, failure: str
):
    exp_dir = tmp_path / "bundle"
    config_dir = exp_dir / "config"
    config_dir.mkdir(parents=True)
    source_runtime = Path(__file__).resolve().parents[1] / (
        "scale_up_outputs/looped_self_forcing_pipeline/config/runtime.yaml"
    )
    (config_dir / "runtime.yaml").write_bytes(source_runtime.read_bytes())
    roots, job_ids = _write_valid_smoke_pair(exp_dir)
    verification = _module("verification")
    if failure == "missing-probe":
        first_run = "smoke-pair-k3-lr5gen"
        (roots["logs_root"] / f"{first_run}_train_{job_ids[0]}.out").write_text(
            "SUE_GPU_MODEL_PREFLIGHT passed expected=H100 count=4\n"
            "===== MEMPROBE rank=0 deep trace at step 3 =====\n",
            encoding="utf-8",
        )
    elif failure == "missing-gpu-model":
        first_run = "smoke-pair-k3-lr5gen"
        output_log = roots["logs_root"] / f"{first_run}_train_{job_ids[0]}.out"
        output_log.write_text(
            output_log.read_text(encoding="utf-8").replace(
                "SUE_GPU_MODEL_PREFLIGHT passed expected=H100 count=4\n", ""
            ),
            encoding="utf-8",
        )
    elif failure == "oom":
        first_run = "smoke-pair-k3-lr5gen"
        with (roots["logs_root"] / f"{first_run}_train_{job_ids[0]}.err").open("w") as stream:
            stream.write("RuntimeError: CUDA out of memory\n")
    elif failure == "high-reserved":
        first_run = "smoke-pair-k3-lr5gen"
        output_log = roots["logs_root"] / f"{first_run}_train_{job_ids[0]}.out"
        output_log.write_text(
            output_log.read_text(encoding="utf-8").replace(
                "rank=2 step_exit: allocated=1.00GiB reserved=2.00GiB",
                "rank=2 step_exit: allocated=58.00GiB reserved=60.00GiB",
            ),
            encoding="utf-8",
        )

    def sacct(command, **_kwargs):
        job_id = command[command.index("-j") + 1]
        state = "RUNNING" if failure == "pending" else "COMPLETED"
        return SimpleNamespace(returncode=0, stdout=f"{job_id}|{state}\n", stderr="")

    with pytest.raises(RuntimeError, match="COMPLETED|MEMPROBE|out of memory|OOM evidence|H100|reserved-memory|rank-labelled|model check"):
        verification.verify_smoke_pair_evidence(
            exp_dir, "smoke-pair", run_command=sacct
        )


def test_inference_evidence_rejects_wrong_frame_rate(tmp_path: Path):
    worker = _module("worker")
    outputs = tmp_path / "output"
    outputs.mkdir()
    (outputs / "one.mp4").write_bytes(b"video")

    def wrong_rate_probe(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "nb_read_frames": "489",
                            "avg_frame_rate": "15/1",
                        }
                    ],
                    "format": {"duration": "32.6"},
                }
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="fps.*expected 16"):
        worker.verify_inference_outputs(
            outputs,
            expected_count=1,
            expected_frames=489,
            expected_fps=16,
            run_command=wrong_rate_probe,
        )


@pytest.mark.parametrize("stale_name", ["notes.txt", ".partial", "nested/old.mp4"])
def test_inference_stale_output_guard_rejects_any_existing_entry(
    tmp_path: Path, stale_name: str
):
    verification = _module("verification")
    outputs = tmp_path / "output"
    stale = outputs / stale_name
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="already contains|not empty"):
        verification.require_empty_results_dir(outputs)


def test_training_stale_artifact_guard_rejects_prior_checkpoints_wandb_or_model(
    tmp_path: Path,
):
    verification = _module("verification")
    checkpoints = tmp_path / "run/checkpoints"
    wandb = tmp_path / "run/wandb"
    model = tmp_path / "run/model.pt"
    checkpoints.mkdir(parents=True)
    wandb.mkdir()
    (checkpoints / "latest.pt").write_bytes(b"old checkpoint")

    with pytest.raises(FileExistsError, match="training.*artifacts|new run_id"):
        verification.require_fresh_training_outputs(checkpoints, wandb, model)


def test_inference_stale_wandb_artifact_guard_rejects_partial_prior_run(tmp_path: Path):
    verification = _module("verification")
    results = tmp_path / "run/final_result"
    wandb = tmp_path / "run/wandb"
    marker = tmp_path / "run/wandb_run_id.txt"
    wandb.mkdir(parents=True)
    (wandb / "offline-run-stale").mkdir()

    with pytest.raises(FileExistsError, match="W&B artifacts|new run_id"):
        verification.require_fresh_inference_outputs(results, wandb, marker)


def test_worker_reads_the_run_id_persisted_by_its_training_child(tmp_path: Path):
    worker = _module("worker")
    path = tmp_path / "run/wandb_run_id.txt"
    path.parent.mkdir()
    path.write_text("offline-id-123\n", encoding="utf-8")

    assert worker._read_wandb_run_id(path) == "offline-id-123"
    assert worker._read_wandb_run_id(path.with_name("missing.txt")) == ""


def test_training_worker_environment_preserves_online_mode_for_source_wrapper(
    tmp_path: Path, monkeypatch
):
    worker = _module("worker")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    overlay = str(tmp_path / "hydra-overlay")
    monkeypatch.setenv("SUE_PYTHONPATH", overlay)
    config = SimpleNamespace(
        backend=SimpleNamespace(name="autodl"),
        tracking={"mode": "online", "required": True, "project": "shared-project"},
        train={"max_steps": 10, "checkpoint_interval_seconds": 3600},
        assets={"generator_checkpoint": "checkpoints/base.pt"},
        method={"profile_id": "layer-profile"},
        run_id="run-1",
    )
    run_root = tmp_path / "run-1"
    prepared = {
        "run_root": run_root,
        "method": {"profile_id": "layer-profile"},
        "manifest": {"assets": {"generator_checkpoint_sha256": "a" * 64}},
    }

    environment = worker._build_worker_environment(config, prepared, "train")

    assert environment["WANDB_MODE"] == "online"
    assert environment["SUE_WANDB_MODE"] == "online"
    assert environment["WANDB_PROJECT"] == "shared-project"
    assert environment["SUE_WANDB_RUN_ID_PATH"] == str(run_root / "wandb_run_id.txt")
    assert environment["SUE_BASE_CHECKPOINT_RELATIVE_PATH"] == "checkpoints/base.pt"
    assert environment["PYTHONPATH"].split(":")[0] == overlay


def test_inference_tracking_starts_required_run_and_records_actual_run_id(
    tmp_path: Path, monkeypatch
):
    worker = _module("worker")
    monkeypatch.setenv("SUE_GIT_COMMIT", "abc123def456")
    monkeypatch.setenv("SUE_GIT_DIRTY", "clean")
    config = SimpleNamespace(
        tracking={"mode": "offline", "required": True, "project": "shared-project"},
        backend=SimpleNamespace(name="nm5"),
        method={"profile_id": "layer-profile"},
        run_id="infer-1",
        seed=7,
        infer=SimpleNamespace(
            seed=7,
            num_output_frames=123,
            num_samples=1,
            use_ema=True,
        ),
    )
    run_root = tmp_path / "infer-1"
    run_root.mkdir()
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base")
    prepared = {
        "run_root": run_root,
        "method": {"profile_id": "layer-profile"},
        "checkpoint": {
            "base_checkpoint": base_checkpoint,
            "base_checkpoint_sha256": worker.sha256_file(base_checkpoint),
            "provenance": {"checkpoint_sha256": "c" * 64},
        },
    }
    arguments = {}

    class Run:
        id = "actual-inference-run"
        summary = {}

        def finish(self, **_kwargs):
            pass

    def init(**kwargs):
        arguments.update(kwargs)
        return Run()

    tracking_run = worker._start_inference_tracking(
        config,
        prepared,
        run_id_path=run_root / "wandb_run_id.txt",
        wandb_module=SimpleNamespace(init=init),
    )

    assert tracking_run.id == "actual-inference-run"
    assert arguments["mode"] == "offline"
    assert arguments["project"] == "shared-project"
    assert arguments["group"] == "infer-1"
    assert arguments["job_type"] == "inference"
    assert arguments["config"]["checkpoint_sha256"] == "c" * 64
    assert arguments["config"]["git_commit"] == "abc123def456"
    assert arguments["config"]["git_dirty"] == "clean"
    assert (run_root / "wandb_run_id.txt").read_text(encoding="utf-8") == (
        "actual-inference-run\n"
    )


def test_online_tracking_preflight_checks_names_without_disclosing_values(
    monkeypatch,
):
    worker = _module("worker")
    monkeypatch.setenv("WANDB_API_KEY", "private-api-key-value")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    config = SimpleNamespace(
        tracking={
            "mode": "online",
            "required": True,
            "project": "comparison-project",
            "api_key_env": "WANDB_API_KEY",
            "entity_env": "WANDB_ENTITY",
        }
    )

    with pytest.raises(RuntimeError, match="WANDB_ENTITY") as error:
        worker._validate_tracking_environment(config)

    assert "private-api-key-value" not in str(error.value)


def test_worker_marks_early_preflight_failure_in_ledger(tmp_path: Path, monkeypatch):
    worker = _module("worker")
    exp_dir = tmp_path / "bundle"
    runtime = exp_dir / "config/runtime.yaml"
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
        "backend_env: {nm5: {source_note: deepresearch-sandbox/config_nm5.txt plus operator Git config for SUE_SLURM_MAIL_USER, required_keys: [NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT, NM5_LOGIN_SSH, NM5_HF_HOME, NM5_HF_HUB_CACHE, NM5_MODELSCOPE_CACHE, NM5_TORCH_HOME, NM5_MPLCONFIGDIR, NM5_WANDB_CACHE_DIR, SUE_SLURM_MAIL_USER, WANDB_API_KEY, WANDB_ENTITY]}}\n"
        "environment: {env_manager: conda, env_root: scale_up_outputs/envs, python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python, asset_root: Self-Forcing-blockwise-layerwise, python_overlay: envs/hydra_overlay, allowed_external_roots: [scale_up_outputs/envs, Self-Forcing-blockwise-layerwise], cache_env_sources: {HF_HOME: NM5_HF_HOME, HF_HUB_CACHE: NM5_HF_HUB_CACHE, MODELSCOPE_CACHE: NM5_MODELSCOPE_CACHE, TORCH_HOME: NM5_TORCH_HOME, MPLCONFIGDIR: NM5_MPLCONFIGDIR, WANDB_CACHE_DIR: NM5_WANDB_CACHE_DIR}}\n"
        "policy:\n"
        "  wandb_policy:\n"
        "    required: true\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    config = SimpleNamespace(
        run_id="failed-run",
        backend=SimpleNamespace(name="autodl"),
        method={"profile_id": "layer-profile", "temporal_loop": {"mode": "layer"}},
        tracking={
            "mode": "online",
            "required": True,
            "project": "comparison-project",
            "api_key_env": "WANDB_API_KEY",
            "entity_env": "WANDB_ENTITY",
        },
        train=SimpleNamespace(seed=1),
        infer=SimpleNamespace(seed=1),
    )

    with pytest.raises(RuntimeError, match="online W&B tracking requires"):
        worker.execute_worker(config, "train", exp_dir=exp_dir)

    with (exp_dir / "artifacts/experiment_results.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["experiment_name"] == "layer-profile-failed-run-train"
    assert row["status"] == "failed"
    assert "WANDB_API_KEY" in row["error"]


def test_completed_training_ledger_seed_matches_checkpoint_evidence(
    tmp_path: Path, monkeypatch
):
    worker = _module("worker")
    exp_dir = tmp_path / "bundle"
    runtime = exp_dir / "config/runtime.yaml"
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
        "backend_env: {nm5: {source_note: deepresearch-sandbox/config_nm5.txt plus operator Git config for SUE_SLURM_MAIL_USER, required_keys: [NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT, NM5_LOGIN_SSH, NM5_HF_HOME, NM5_HF_HUB_CACHE, NM5_MODELSCOPE_CACHE, NM5_TORCH_HOME, NM5_MPLCONFIGDIR, NM5_WANDB_CACHE_DIR, SUE_SLURM_MAIL_USER, WANDB_API_KEY, WANDB_ENTITY]}}\n"
        "environment: {env_manager: conda, env_root: scale_up_outputs/envs, python_binary: scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python, asset_root: Self-Forcing-blockwise-layerwise, python_overlay: envs/hydra_overlay, allowed_external_roots: [scale_up_outputs/envs, Self-Forcing-blockwise-layerwise], cache_env_sources: {HF_HOME: NM5_HF_HOME, HF_HUB_CACHE: NM5_HF_HUB_CACHE, MODELSCOPE_CACHE: NM5_MODELSCOPE_CACHE, TORCH_HOME: NM5_TORCH_HOME, MPLCONFIGDIR: NM5_MPLCONFIGDIR, WANDB_CACHE_DIR: NM5_WANDB_CACHE_DIR}}\n"
        "policy:\n"
        "  wandb_policy:\n"
        "    required: false\n",
        encoding="utf-8",
    )
    run_id = "train-run"
    run_root = exp_dir / "artifacts" / run_id
    checkpoint_root = exp_dir / "ckpt" / run_id
    results_root = exp_dir / "final_result" / run_id
    checkpoint_root.mkdir(parents=True)
    asset_root = tmp_path / "configured-assets"
    asset_root.mkdir()
    prepared = {
        "exp_dir": exp_dir.resolve(),
        "run_root": run_root,
        "checkpoint_root": checkpoint_root,
        "results_root": results_root,
        "final_model": checkpoint_root / "model.pt",
        "output_roots": {
            "artifacts_root": exp_dir / "artifacts",
            "ckpt_root": exp_dir / "ckpt",
            "final_result_root": exp_dir / "final_result",
            "logs_root": exp_dir / "logs",
            "datasets_root": exp_dir / "datasets",
            "slurm_scripts_root": exp_dir / "slurm_scripts",
        },
        "method": {
            "profile_id": "layer-profile",
            "mode": "layer",
            "layer_start": 22,
            "layer_end": 29,
            "lora": {"target_modules": ["self_attn.q", "self_attn.v"]},
        },
        "assets": {"asset_root": asset_root.resolve()},
        "manifest": {"assets": {"generator_checkpoint_sha256": "a" * 64}},
    }
    observed = {}

    def run_command(_command, **_kwargs):
        (checkpoint_root / "latest.pt").write_bytes(b"checkpoint")
        return SimpleNamespace(returncode=0)

    def verify(_path, *, expected_step, expected_seed):
        observed.update(expected_step=expected_step, expected_seed=expected_seed)
        generator = {
            f"blocks.{layer}.self_attn.{target}.lora_{matrix}.weight": 1
            for layer in range(22, 30)
            for target in ("q", "v")
            for matrix in ("A", "B")
        }
        return {
            "step": expected_step,
            "metadata": {"step": expected_step, "seed": expected_seed, "final": True},
            "trainable_parameter_count": 8,
            "payload": {"generator": generator},
        }

    original_scope_check = worker._check_adapter_scope

    def check_scope(state, method, label):
        observed["scope"] = (len(state), method["layer_start"], method["layer_end"], label)
        return original_scope_check(state, method, label)

    monkeypatch.setattr(worker, "_validate_tracking_environment", lambda _config: None)
    monkeypatch.setattr(worker, "prepare_stage", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(worker, "_require_gpu_capacity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "require_fresh_training_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "_build_worker_environment", lambda *_args: {})
    monkeypatch.setattr(worker, "_command_for", lambda *_args: ["worker"])
    monkeypatch.setattr(worker.subprocess, "run", run_command)
    monkeypatch.setattr(worker, "verify_training_checkpoint", verify)
    monkeypatch.setattr(worker, "_check_adapter_scope", check_scope)
    monkeypatch.setattr(worker, "_lightweight_model", lambda *_args: None)
    config = SimpleNamespace(
        run_id=run_id,
        backend=SimpleNamespace(name="nm5"),
        method={"profile_id": "layer-profile", "temporal_loop": {"mode": "layer"}},
        tracking=SimpleNamespace(mode="offline", required=False, project="comparison"),
        train=SimpleNamespace(max_steps=20, seed=17),
        infer=SimpleNamespace(seed=0),
    )

    worker.execute_worker(config, "train", exp_dir=exp_dir)

    with (exp_dir / "artifacts/experiment_results.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        row = next(csv.DictReader(stream))
    assert observed == {
        "expected_step": 20,
        "expected_seed": 17,
        "scope": (32, 22, 29, "generator"),
    }
    assert row["seed"] == str(observed["expected_seed"])
    assert row["trainable_parameter_count"] == "8"
    assert row["parameter_count"] == ""


def test_runtime_adapter_is_derived_once_from_verified_inputs(tmp_path: Path):
    torch = pytest.importorskip("torch")
    worker = _module("worker")
    source = tmp_path / "portable-adapter.pt"
    source_sidecar = tmp_path / "provenance.json"
    base = tmp_path / "asset-root/checkpoints/base.pt"
    destination = tmp_path / "run/runtime_inference_adapter.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base weights")
    base_hash = worker.sha256_file(base)
    original_payload = {
        "generator_format": "lora_adapter",
        "generator": {"adapter": torch.tensor([1.0])},
        "generator_ema": {"adapter": torch.tensor([2.0])},
        "metadata": {
            "base_checkpoint": "checkpoints/base.pt",
            "base_checkpoint_sha256": base_hash,
        },
    }
    torch.save(original_payload, source)
    original_bytes = source.read_bytes()
    source_sidecar.write_text('{"source":"verified"}\n', encoding="utf-8")

    first = worker._materialize_runtime_adapter(
        source,
        source_sidecar,
        original_payload,
        base.resolve(),
        base_hash,
        destination,
    )
    second = worker._materialize_runtime_adapter(
        source,
        source_sidecar,
        original_payload,
        base.resolve(),
        base_hash,
        destination,
    )
    runtime_payload = torch.load(destination, map_location="cpu", weights_only=True)

    assert source.read_bytes() == original_bytes
    assert runtime_payload["metadata"]["base_checkpoint"] == str(base.resolve())
    assert first["sha256"] == second["sha256"]
    assert first["provenance_sha256"] == second["provenance_sha256"]


def test_runtime_adapter_refuses_stale_source_identity(tmp_path: Path):
    torch = pytest.importorskip("torch")
    worker = _module("worker")
    source = tmp_path / "adapter.pt"
    source_sidecar = tmp_path / "provenance.json"
    base = tmp_path / "asset-root/checkpoints/base.pt"
    destination = tmp_path / "run/runtime_inference_adapter.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base weights")
    payload = {
        "generator_format": "lora_adapter",
        "generator": {"adapter": torch.tensor([1.0])},
        "metadata": {"base_checkpoint_sha256": worker.sha256_file(base)},
    }
    torch.save(payload, source)
    source_sidecar.write_text('{"source":"first"}\n', encoding="utf-8")
    worker._materialize_runtime_adapter(
        source,
        source_sidecar,
        payload,
        base.resolve(),
        worker.sha256_file(base),
        destination,
    )

    source_sidecar.write_text('{"source":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the selected inputs|source identity"):
        worker._materialize_runtime_adapter(
            source,
            source_sidecar,
            payload,
            base.resolve(),
            worker.sha256_file(base),
            destination,
        )


def _adapter_provenance_fixture():
    method = {
        "profile_id": "layerwise_l16_23_k2",
        "mode": "layer",
        "layer_start": 16,
        "layer_end": 23,
        "k_min": 2,
        "k_max": 2,
        "strength": 1.0,
        "stop_grad_early": True,
        "schedule": "temporal_uniform",
        "training_enabled": True,
        "lora": {"enabled": True, "rank": 8},
    }
    loop = {key: value for key, value in method.items() if key != "profile_id" and key != "lora"}
    digest = "a" * 64
    payload = {
        "metadata": {
            "step": 600,
            "seed": 7,
            "base_checkpoint_sha256": digest,
            "temporal_loop": loop,
            "lora": method["lora"],
        }
    }
    provenance = {
        "method": dict(method),
        "source_training_step": 600,
        "seed": 7,
        "base_checkpoint_sha256": digest,
    }
    return method, payload, provenance


def test_adapter_provenance_matches_embedded_method_base_step_and_seed():
    worker = _module("worker")
    method, payload, provenance = _adapter_provenance_fixture()

    assert worker._validate_adapter_provenance(provenance, payload, method) == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", {"mode": "block"}),
        ("base_checkpoint_sha256", "b" * 64),
        ("source_training_step", 599),
        ("seed", 8),
    ],
)
def test_adapter_provenance_rejects_sidecar_identity_mismatch(field: str, value):
    worker = _module("worker")
    method, payload, provenance = _adapter_provenance_fixture()
    provenance[field] = value

    with pytest.raises(ValueError, match="provenance|differs|does not match|step|seed"):
        worker._validate_adapter_provenance(provenance, payload, method)


def test_adapter_provenance_requires_embedded_temporal_loop_mode():
    worker = _module("worker")
    method, payload, provenance = _adapter_provenance_fixture()
    payload["metadata"]["temporal_loop"].pop("mode")

    with pytest.raises(ValueError, match="temporal_loop.mode|embedded"):
        worker._validate_adapter_provenance(provenance, payload, method)


def test_ledger_upserts_a_row_with_reproducibility_fields(tmp_path: Path):
    ledger = _module("ledger")
    path = tmp_path / "results.csv"
    row = {
        "method": "layer",
        "experiment_name": "layer-run-1-train",
        "backend": "nm5",
        "status": "completed",
        "seed": 7,
        "training_speed": 0.5,
        "trainable_parameter_count": 123,
    }

    ledger.write_result(path, row)
    ledger.write_result(path, {**row, "backend": "autodl"})

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["backend"] == "autodl"
    assert rows[0]["seed"] == "7"
    assert rows[0]["training_speed"] == "0.5"
    assert rows[0]["parameter_count"] == ""
    assert rows[0]["trainable_parameter_count"] == "123"


@pytest.mark.parametrize("stage", ["train", "infer"])
def test_worker_subprocess_runs_from_configured_asset_root(
    tmp_path: Path, monkeypatch, stage: str
):
    worker = _module("worker")
    exp_dir = tmp_path / "bundle"
    asset_root = tmp_path / "configured-assets"
    asset_root.mkdir()
    output_roots = {
        "artifacts_root": exp_dir / "artifacts",
        "ckpt_root": exp_dir / "ckpt",
        "final_result_root": exp_dir / "final_result",
        "ledger_csv": exp_dir / "artifacts/experiment_results.csv",
    }
    run_id = f"cwd-{stage}"
    checkpoint_root = output_roots["ckpt_root"] / run_id
    results_root = output_roots["final_result_root"] / run_id
    prepared = {
        "exp_dir": exp_dir.resolve(),
        "run_root": output_roots["artifacts_root"] / run_id,
        "checkpoint_root": checkpoint_root,
        "results_root": results_root,
        "final_model": checkpoint_root / "model.pt",
        "output_roots": output_roots,
        "method": {"profile_id": "layer-profile", "mode": "layer"},
        "assets": {"asset_root": asset_root.resolve()},
        "manifest": {"assets": {"generator_checkpoint_sha256": "a" * 64}},
    }
    observed = {}

    def run_command(_command, **kwargs):
        observed["cwd"] = Path(kwargs["cwd"])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(worker, "_validate_tracking_environment", lambda _config: None)
    monkeypatch.setattr(worker, "require_ffprobe", lambda: None)
    monkeypatch.setattr(worker, "prepare_stage", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(worker, "_require_gpu_capacity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "require_fresh_training_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "require_fresh_inference_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "_build_worker_environment", lambda *_args: {})
    monkeypatch.setattr(worker, "_command_for", lambda *_args: ["worker"])
    monkeypatch.setattr(worker.subprocess, "run", run_command)
    config = SimpleNamespace(
        run_id=run_id,
        backend=SimpleNamespace(name="nm5"),
        method={"profile_id": "layer-profile", "temporal_loop": {"mode": "layer"}},
        tracking={"mode": "offline", "required": False, "project": "comparison"},
        train=SimpleNamespace(max_steps=1, seed=7),
        infer=SimpleNamespace(seed=7),
    )

    with pytest.raises(subprocess.CalledProcessError):
        worker._execute_worker_body(
            config,
            stage,
            exp_dir=exp_dir,
            ledger_path=output_roots["ledger_csv"],
            ledger_row={},
        )

    assert observed["cwd"] == asset_root.resolve()
