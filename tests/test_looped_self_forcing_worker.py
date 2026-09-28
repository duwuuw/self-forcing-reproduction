from __future__ import annotations

import csv
import json
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


def test_inference_tracking_starts_required_run_and_records_actual_run_id(
    tmp_path: Path,
):
    worker = _module("worker")
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
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "policy:\n"
        "  sandbox_resources:\n"
        "    autodl:\n"
        "      train_gpus: 4\n"
        "      infer_gpus: 1\n"
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
        "  datasets_root: datasets\n"
        "  slurm_scripts_root: slurm_scripts\n"
        "  artifacts_root: artifacts\n"
        "  ckpt_root: ckpt\n"
        "  final_result_root: final_result\n"
        "  logs_root: logs\n"
        "policy:\n"
        "  sandbox_resources:\n"
        "    nm5:\n"
        "      train_gpus: 4\n"
        "      infer_gpus: 1\n"
        "  wandb_policy:\n"
        "    required: false\n",
        encoding="utf-8",
    )
    run_id = "train-run"
    run_root = exp_dir / "artifacts" / run_id
    checkpoint_root = exp_dir / "ckpt" / run_id
    results_root = exp_dir / "final_result" / run_id
    checkpoint_root.mkdir(parents=True)
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
        "method": {"profile_id": "layer-profile", "mode": "layer"},
        "manifest": {"assets": {"generator_checkpoint_sha256": "a" * 64}},
    }
    observed = {}

    def run_command(_command, **_kwargs):
        (checkpoint_root / "latest.pt").write_bytes(b"checkpoint")
        return SimpleNamespace(returncode=0)

    def verify(_path, *, expected_step, expected_seed):
        observed.update(expected_step=expected_step, expected_seed=expected_seed)
        return {
            "step": expected_step,
            "metadata": {"step": expected_step, "seed": expected_seed, "final": True},
            "trainable_parameter_count": 8,
        }

    monkeypatch.setattr(worker, "_validate_tracking_environment", lambda _config: None)
    monkeypatch.setattr(worker, "prepare_stage", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(worker, "_require_gpu_capacity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "require_fresh_training_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "_build_worker_environment", lambda *_args: {})
    monkeypatch.setattr(worker, "_command_for", lambda *_args: ["worker"])
    monkeypatch.setattr(worker.subprocess, "run", run_command)
    monkeypatch.setattr(worker, "verify_training_checkpoint", verify)
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
    assert observed == {"expected_step": 20, "expected_seed": 17}
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
