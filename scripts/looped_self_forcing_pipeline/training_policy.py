"""Training cadence adapters kept outside the copied model/trainer sources."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


def portable_checkpoint_reference(
    checkpoint_path: str | Path,
    asset_root: str | Path,
    *,
    relative_path: str | Path,
) -> str:
    """Store the configured asset name, validating it resolves to the loaded file."""
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    root = Path(asset_root).expanduser().resolve(strict=True)
    relative = Path(relative_path).expanduser()
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("base checkpoint reference must be relative to SUE_ASSET_ROOT")
    configured_file = (root / relative).resolve(strict=True)
    if not configured_file.is_file() or configured_file != checkpoint:
        raise ValueError(
            "configured base checkpoint reference does not resolve to the loaded generator checkpoint"
        )
    return relative.as_posix()


def checkpoint_due(
    step: int,
    last_checkpoint_time: float,
    now: float,
    *,
    max_steps: int,
    interval_seconds: int,
) -> bool:
    """Save at the wall-clock interval or at the exact final optimizer step."""
    if interval_seconds <= 0:
        raise ValueError("checkpoint interval must be positive")
    return (max_steps > 0 and step >= max_steps) or now - last_checkpoint_time >= interval_seconds


def should_log_scalar(step: int, *, max_steps: int, interval: int = 100) -> bool:
    """Keep W&B scalar volume low while retaining the first and final steps."""
    if interval <= 0:
        raise ValueError("scalar logging interval must be positive")
    return step <= 1 or step == max_steps or step % interval == 0


def should_log_progress(step: int, *, interval: int = 10) -> bool:
    if interval <= 0:
        raise ValueError("progress logging interval must be positive")
    return step <= 1 or step % interval == 0


def _distributed_decision(due: bool) -> bool:
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return due
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    signal = torch.tensor([int(due if rank == 0 else 0)], device=device, dtype=torch.int32)
    dist.broadcast(signal, src=0)
    return bool(signal.item())


def _publish_latest(checkpoint_root: Path, checkpoint_file: Path) -> None:
    latest = checkpoint_root / "latest.pt"
    temporary = checkpoint_root / f".latest.pt.{os.getpid()}.tmp"
    temporary.unlink(missing_ok=True)
    relative_target = os.path.relpath(checkpoint_file, checkpoint_root)
    os.symlink(relative_target, temporary)
    os.replace(temporary, latest)
    current_directory = checkpoint_file.parent
    for candidate in checkpoint_root.glob("checkpoint_model_*"):
        if candidate != current_directory and candidate.is_dir():
            shutil.rmtree(candidate)


def persist_wandb_run_id(path: str | Path, run_id: str) -> None:
    """Atomically persist the exact W&B ID for the parent worker and ledger."""
    value = str(run_id).strip()
    if not value:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_text(encoding="utf-8").strip()
        if existing != value:
            raise RuntimeError(f"multiple W&B run IDs were initialized for one worker: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, target)
        except FileExistsError:
            existing = target.read_text(encoding="utf-8").strip()
            if existing != value:
                raise RuntimeError(
                    f"multiple W&B run IDs were initialized for one worker: {target}"
                )
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


_PRIVATE_PATH_CONFIG_KEYS = {
    "generator_ckpt",
    "teacher_checkpoint",
    "data_path",
    "resume_checkpoint",
    "logdir",
    "wandb_save_dir",
}
_PRIVATE_VALUE_CONFIG_KEYS = {"wandb_key", "wandb_api_key", "wandb_host", "api_key", "token"}


def _sanitize_wandb_config(value: Any) -> Any:
    """Copy run config while replacing backend-local paths with stable names."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in _PRIVATE_VALUE_CONFIG_KEYS and item:
                sanitized[key] = "<redacted>"
            elif normalized_key in _PRIVATE_PATH_CONFIG_KEYS and item:
                sanitized[key] = Path(str(item)).name
            else:
                sanitized[key] = _sanitize_wandb_config(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_wandb_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_wandb_config(item) for item in value)
    return value


def validate_sue_wandb_environment() -> str:
    """Validate the tracking mode before a SUE trainer constructs its model."""
    mode = os.environ.get("SUE_WANDB_MODE", "offline").strip().casefold()
    if mode not in {"offline", "online"}:
        raise ValueError("SUE_WANDB_MODE must be either offline or online")
    if mode == "online":
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise ValueError("online SUE tracking requires WANDB_API_KEY")
        if not os.environ.get("WANDB_ENTITY", "").strip():
            raise ValueError("online SUE tracking requires WANDB_ENTITY")
    return mode


def install_sue_wandb_runtime(wandb_module: Any) -> None:
    """Override copied-trainer W&B arguments with the resolved SUE identity."""
    if getattr(wandb_module, "_looped_sue_wandb_runtime", False):
        return

    mode = validate_sue_wandb_environment()
    original_init = wandb_module.init
    placeholders = {"WANDB_ENTITY", "WANDB_PROJECT", "WANDB_NAME", "WANDB_GROUP"}

    def init_with_sue_identity(*args, **kwargs):
        kwargs = dict(kwargs)
        values = {
            "entity": os.environ.get("WANDB_ENTITY", "").strip(),
            "project": os.environ.get("WANDB_PROJECT", "").strip(),
            "name": (
                os.environ.get("WANDB_EXPERIMENT_NAME", "").strip()
                or os.environ.get("WANDB_NAME", "").strip()
            ),
            "group": os.environ.get("WANDB_GROUP", "").strip(),
        }
        for key, value in values.items():
            if value and value not in placeholders:
                kwargs[key] = value
            elif kwargs.get(key) in placeholders:
                kwargs.pop(key, None)

        kwargs.pop("host", None)
        kwargs.pop("key", None)
        kwargs["mode"] = mode

        config = kwargs.get("config")
        if isinstance(config, dict):
            config = dict(config)
            config_fields = {
                "entity": "wandb_entity",
                "project": "wandb_project",
                "name": "config_name",
            }
            for identity_key, config_key in config_fields.items():
                value = values[identity_key]
                if value and value not in placeholders:
                    config[config_key] = value
                elif config.get(config_key) in placeholders:
                    config[config_key] = ""
            kwargs["config"] = config
        return original_init(*args, **kwargs)

    def login_from_environment(*_args, **_kwargs):
        # Online credentials are validated above and consumed from WANDB_API_KEY.
        return True

    wandb_module.init = init_with_sue_identity
    wandb_module.login = login_from_environment
    wandb_module._looped_sue_wandb_runtime = True


def _install_wandb_observation_policy(
    wandb_module: Any,
    *,
    max_steps: int,
    checkpoint_interval_seconds: int,
    scalar_log_interval: int,
    progress_log_interval: int,
) -> None:
    original_wandb_log = wandb_module.log
    original_wandb_init = wandb_module.init

    def init_with_run_id(*args, **kwargs):
        kwargs = dict(kwargs)
        if isinstance(kwargs.get("config"), dict):
            run_config = _sanitize_wandb_config(kwargs["config"])
            train_config = dict(run_config.get("train") or {})
            train_config.update(
                {
                    "max_steps": max_steps,
                    "checkpoint_interval_seconds": checkpoint_interval_seconds,
                    "scalar_log_interval": scalar_log_interval,
                    "progress_log_interval": progress_log_interval,
                }
            )
            if run_config.get("seed") is not None:
                train_config["seed"] = run_config["seed"]
            if run_config.get("log_iters") is not None:
                train_config["log_iters"] = run_config["log_iters"]
            run_config["train"] = train_config
            run_config["sue_runtime"] = {
                "backend": os.environ.get("SUE_BACKEND", ""),
                "run_id": os.environ.get("SUE_RUN_ID", ""),
                "stage": os.environ.get("SUE_STAGE", "train"),
                "experiment_name": os.environ.get("WANDB_EXPERIMENT_NAME", ""),
                "tracking_mode": os.environ.get("SUE_WANDB_MODE", "offline"),
                "project": os.environ.get("WANDB_PROJECT", ""),
                **git_provenance(),
            }
            kwargs["config"] = run_config
        run = original_wandb_init(*args, **kwargs)
        target = os.environ.get("SUE_WANDB_RUN_ID_PATH", "").strip()
        run_id = getattr(run, "id", "")
        if target and run_id:
            persist_wandb_run_id(target, str(run_id))
        return run

    def log_with_policy(data=None, *args, step=None, **kwargs):
        scalar_batch = isinstance(data, dict) and all(
            isinstance(value, (int, float, bool)) for value in data.values()
        )
        if scalar_batch and step is not None and not should_log_scalar(
            int(step), max_steps=max_steps, interval=scalar_log_interval
        ):
            return None
        return original_wandb_log(data, *args, step=step, **kwargs)

    wandb_module.init = init_with_run_id
    wandb_module.log = log_with_policy


def _distributed_all_success(success: bool) -> bool:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return success
    import torch

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    signal = torch.tensor([int(success)], device=device, dtype=torch.int32)
    dist.all_reduce(signal, op=dist.ReduceOp.MIN)
    return bool(signal.item())


def git_provenance() -> dict[str, str]:
    commit = os.environ.get("SUE_GIT_COMMIT", "").strip().casefold()
    dirty = os.environ.get("SUE_GIT_DIRTY", "unknown").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        commit = "unknown"
    if dirty not in {"clean", "dirty", "unknown"}:
        dirty = "unknown"
    return {"git_commit": commit, "git_dirty": dirty}


def install_training_policy(
    trainer_class: type,
    wandb_module: Any,
    *,
    max_steps: int,
    checkpoint_interval_seconds: int = 3600,
    scalar_log_interval: int = 100,
    progress_log_interval: int = 10,
    asset_root: str | Path,
    clock=time.monotonic,
) -> None:
    """Install timed checkpointing and low-volume scalar/progress logging.

    The copied Trainer remains the owner of optimizer/model math and the actual
    distributed state-dict gather. This adapter only chooses when its existing
    ``save`` callback runs and publishes the resulting checkpoint path.
    """
    if getattr(trainer_class.save, "_looped_training_policy", False):
        return

    native_save = trainer_class.save
    native_metadata = getattr(trainer_class, "_lora_checkpoint_metadata", None)
    native_step = trainer_class.fwdbwd_one_step
    original_wandb_log = wandb_module.log
    original_wandb_init = wandb_module.init

    def init_with_run_id(*args, **kwargs):
        kwargs = dict(kwargs)
        if isinstance(kwargs.get("config"), dict):
            run_config = _sanitize_wandb_config(kwargs["config"])
            train_config = dict(run_config.get("train") or {})
            train_config.update({
                "max_steps": max_steps,
                "checkpoint_interval_seconds": checkpoint_interval_seconds,
                "scalar_log_interval": scalar_log_interval,
                "progress_log_interval": progress_log_interval,
            })
            if run_config.get("seed") is not None:
                train_config["seed"] = run_config["seed"]
            if run_config.get("log_iters") is not None:
                train_config["log_iters"] = run_config["log_iters"]
            run_config["train"] = train_config
            run_config["sue_runtime"] = {
                "backend": os.environ.get("SUE_BACKEND", ""),
                "run_id": os.environ.get("SUE_RUN_ID", ""),
                "stage": os.environ.get("SUE_STAGE", "train"),
                "experiment_name": os.environ.get("WANDB_EXPERIMENT_NAME", ""),
                "tracking_mode": os.environ.get("SUE_WANDB_MODE", "offline"),
                "project": os.environ.get("WANDB_PROJECT", ""),
                **git_provenance(),
            }
            kwargs["config"] = run_config
        run = original_wandb_init(*args, **kwargs)
        target = os.environ.get("SUE_WANDB_RUN_ID_PATH", "").strip()
        run_id = getattr(run, "id", "")
        if target and run_id:
            persist_wandb_run_id(target, str(run_id))
        return run

    wandb_module.init = init_with_run_id

    def save_with_policy(self) -> None:
        now = clock()
        if not hasattr(self, "_looped_last_checkpoint_time"):
            self._looped_last_checkpoint_time = now
        last = float(self._looped_last_checkpoint_time)
        due = checkpoint_due(
            int(self.step),
            last,
            float(now),
            max_steps=max_steps,
            interval_seconds=checkpoint_interval_seconds,
        )
        if not _distributed_decision(due):
            return
        native_save(self)
        if getattr(self, "is_main_process", False):
            checkpoint_root = Path(self.output_path)
            checkpoint_file = checkpoint_root / f"checkpoint_model_{self.step:06d}" / "model.pt"
            if not checkpoint_file.is_file():
                raise FileNotFoundError(f"native trainer did not write a checkpoint: {checkpoint_file}")
            _publish_latest(checkpoint_root, checkpoint_file)
        self._looped_last_checkpoint_time = now
        if int(self.step) >= max_steps and getattr(self, "is_main_process", False):
            run = getattr(wandb_module, "run", None)
            if run is not None:
                run.summary["checkpoint_step"] = int(self.step)
                run.summary["final"] = True
                run.summary["examples_seen"] = int(self.step) * int(
                    getattr(self.config, "total_batch_size", 1)
                )

    save_with_policy._looped_training_policy = True
    trainer_class.save = save_with_policy

    if native_metadata is not None:
        def metadata_with_final(self):
            metadata = native_metadata(self)
            temporal_loop = dict(metadata.get("temporal_loop") or {})
            temporal_loop.update(
                {
                    "mode": str(self.config.temporal_loop.mode),
                    "stop_grad_early": bool(self.config.temporal_loop.stop_grad_early),
                    "training_enabled": bool(self.config.temporal_loop.training_enabled),
                }
            )
            metadata["temporal_loop"] = temporal_loop
            relative_base = os.environ.get("SUE_BASE_CHECKPOINT_RELATIVE_PATH", "").strip()
            if not relative_base:
                raise ValueError("SUE_BASE_CHECKPOINT_RELATIVE_PATH must name the configured asset")
            metadata["base_checkpoint"] = portable_checkpoint_reference(
                self.config.generator_ckpt,
                asset_root,
                relative_path=relative_base,
            )
            base_hash = os.environ.get("SUE_BASE_CHECKPOINT_SHA256", "")
            if len(base_hash) != 64:
                raise ValueError("SUE_BASE_CHECKPOINT_SHA256 must be set to the resolved base asset hash")
            metadata["base_checkpoint_sha256"] = base_hash
            seed = int(self.config.seed)
            if seed <= 0:
                raise ValueError("training seed must be non-zero and positive")
            metadata["seed"] = seed
            metadata["final"] = bool(max_steps > 0 and int(self.step) >= max_steps)
            metadata["checkpoint_interval_seconds"] = checkpoint_interval_seconds
            return metadata

        trainer_class._lora_checkpoint_metadata = metadata_with_final

    def step_with_progress(self, batch, train_generator, *args, **kwargs):
        output = native_step(self, batch, train_generator, *args, **kwargs)
        step = int(self.step) + (0 if train_generator else 1)
        if (
            not train_generator
            and getattr(self, "is_main_process", False)
            and should_log_progress(step, interval=progress_log_interval)
            and getattr(self, "_looped_last_progress_step", None) != step
        ):
            self._looped_last_progress_step = step
            print(f"TRAIN_PROGRESS step={step}/{max_steps}", flush=True)
        return output

    trainer_class.fwdbwd_one_step = step_with_progress

    def log_with_policy(data=None, *args, step=None, **kwargs):
        scalar_batch = isinstance(data, dict) and all(
            isinstance(value, (int, float, bool)) for value in data.values()
        )
        if scalar_batch and step is not None and not should_log_scalar(
            int(step), max_steps=max_steps, interval=scalar_log_interval
        ):
            return None
        return original_wandb_log(data, *args, step=step, **kwargs)

    wandb_module.log = log_with_policy


def install_diffusion_training_policy(
    trainer_class: type,
    wandb_module: Any,
    *,
    max_steps: int,
    checkpoint_interval_seconds: int = 3600,
    scalar_log_interval: int = 100,
    progress_log_interval: int = 10,
    asset_root: str | Path,
    clock=time.monotonic,
) -> None:
    """Install SUE callbacks for the supervised trainer's one-step API."""
    if getattr(trainer_class, "_sue_diffusion_training_policy", False):
        return
    if not hasattr(trainer_class, "train_one_step"):
        raise TypeError("DiffusionTrainer must expose train_one_step")

    native_save = trainer_class.save
    native_metadata = getattr(trainer_class, "_lora_checkpoint_metadata", None)
    native_step = trainer_class.train_one_step
    _install_wandb_observation_policy(
        wandb_module,
        max_steps=max_steps,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
        scalar_log_interval=scalar_log_interval,
        progress_log_interval=progress_log_interval,
    )

    if native_metadata is not None:
        def metadata_with_sue_identity(self):
            metadata = native_metadata(self)
            temporal_loop = dict(metadata.get("temporal_loop") or {})
            temporal_loop.update(
                {
                    "mode": str(self.config.temporal_loop.mode),
                    "stop_grad_early": bool(self.config.temporal_loop.stop_grad_early),
                    "training_enabled": bool(self.config.temporal_loop.training_enabled),
                }
            )
            metadata["temporal_loop"] = temporal_loop
            relative_base = os.environ.get(
                "SUE_BASE_CHECKPOINT_RELATIVE_PATH", ""
            ).strip()
            if not relative_base:
                raise ValueError("SUE_BASE_CHECKPOINT_RELATIVE_PATH must name the configured asset")
            metadata["base_checkpoint"] = portable_checkpoint_reference(
                self.config.generator_ckpt,
                asset_root,
                relative_path=relative_base,
            )
            base_hash = os.environ.get("SUE_BASE_CHECKPOINT_SHA256", "")
            if len(base_hash) != 64:
                raise ValueError("SUE_BASE_CHECKPOINT_SHA256 must be set to the resolved base asset hash")
            metadata["base_checkpoint_sha256"] = base_hash
            seed = int(self.config.seed)
            if seed <= 0:
                raise ValueError("training seed must be non-zero and positive")
            metadata["seed"] = seed
            metadata["final"] = bool(max_steps > 0 and int(self.step) >= max_steps)
            metadata["checkpoint_interval_seconds"] = checkpoint_interval_seconds
            return metadata

        trainer_class._lora_checkpoint_metadata = metadata_with_sue_identity

    def checkpoint_step(self) -> bool:
        now = float(clock())
        if not hasattr(self, "_looped_last_checkpoint_time"):
            self._looped_last_checkpoint_time = now
        final_step = max_steps > 0 and int(self.step) >= max_steps
        due = checkpoint_due(
            int(self.step),
            float(self._looped_last_checkpoint_time),
            now,
            max_steps=max_steps,
            interval_seconds=checkpoint_interval_seconds,
        )
        if not _distributed_decision(due):
            return final_step

        import torch

        torch.cuda.empty_cache()
        save_error = None
        try:
            native_save(self)
        except Exception as error:  # noqa: BLE001 - synchronize recoverable save failures
            save_error = error
        finally:
            torch.cuda.empty_cache()
        if not _distributed_all_success(save_error is None):
            raise RuntimeError(
                "SUE diffusion checkpoint save failed on at least one rank"
            ) from save_error

        publish_error = None
        if getattr(self, "is_main_process", False):
            try:
                checkpoint_root = Path(self.output_path)
                checkpoint_file = (
                    checkpoint_root
                    / f"checkpoint_model_{self.step:06d}"
                    / "model.pt"
                )
                if not checkpoint_file.is_file():
                    raise FileNotFoundError(
                        f"native trainer did not write a checkpoint: {checkpoint_file}"
                    )
                _publish_latest(checkpoint_root, checkpoint_file)
            except Exception as error:  # noqa: BLE001 - publish status is synchronized below
                publish_error = error
        if not _distributed_all_success(
            publish_error is None if getattr(self, "is_main_process", False) else True
        ):
            raise RuntimeError(
                "SUE diffusion checkpoint publication failed on at least one rank"
            ) from publish_error

        self._looped_last_checkpoint_time = now
        if final_step and getattr(self, "is_main_process", False):
            run = getattr(wandb_module, "run", None)
            if run is not None:
                try:
                    run.summary["checkpoint_step"] = int(self.step)
                    run.summary["final"] = True
                    total_batch_size = getattr(self.config, "total_batch_size", None)
                    if total_batch_size is None:
                        total_batch_size = int(getattr(self.config, "batch_size", 1)) * int(
                            getattr(self, "world_size", 1)
                        )
                    run.summary["examples_seen"] = int(self.step) * int(total_batch_size)
                except Exception as error:  # noqa: BLE001 - checkpoint metadata is authoritative
                    print(
                        "TRAINING_POLICY WARNING: final W&B summary update failed: "
                        f"{type(error).__name__}",
                        flush=True,
                    )
        return final_step

    trainer_class._sue_checkpoint_callback = staticmethod(checkpoint_step)
    trainer_class._sue_max_steps = max_steps

    def train_one_step_with_progress(self, batch, *args, **kwargs):
        output = native_step(self, batch, *args, **kwargs)
        step = int(self.step)
        if (
            getattr(self, "is_main_process", False)
            and should_log_progress(step, interval=progress_log_interval)
            and getattr(self, "_looped_last_progress_step", None) != step
        ):
            self._looped_last_progress_step = step
            print(f"TRAIN_PROGRESS step={step}/{max_steps}", flush=True)
        return output

    trainer_class.train_one_step = train_one_step_with_progress
    trainer_class._sue_diffusion_training_policy = True
