"""Apply pipeline-level run policies, then delegate to the copied trainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from looped_self_forcing_pipeline.training_policy import (
    install_diffusion_training_policy,
    install_sue_wandb_runtime,
    install_training_policy,
    validate_sue_wandb_environment,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT / "source"


def prioritize_source_root(search_path: list[str]) -> None:
    source_root = str(SOURCE_ROOT)
    search_path[:] = [entry for entry in search_path if entry != source_root]
    search_path.insert(0, source_root)


def _validate_diffusion_sue_config(args, config) -> tuple[int, int, int, int, int]:
    if args.no_save:
        raise ValueError("SUE diffusion training requires checkpoint saving")
    if args.disable_wandb:
        raise ValueError("SUE diffusion training requires W&B tracking")

    max_steps = int(os.environ.get("SUE_MAX_STEPS", "0"))
    if max_steps <= 0:
        raise ValueError("SUE_MAX_STEPS must be positive for diffusion training")
    log_iters = int(getattr(config, "log_iters", 0))
    if log_iters <= 0:
        raise ValueError("training config log_iters must be positive")
    checkpoint_interval = int(
        os.environ.get("SUE_CHECKPOINT_INTERVAL_SECONDS", "3600")
    )
    if checkpoint_interval <= 0:
        raise ValueError("SUE_CHECKPOINT_INTERVAL_SECONDS must be positive")
    scalar_log_interval = int(os.environ.get("SUE_WANDB_LOG_INTERVAL", "100"))
    if scalar_log_interval <= 0:
        raise ValueError("SUE_WANDB_LOG_INTERVAL must be positive")
    progress_log_interval = int(os.environ.get("SUE_PROGRESS_LOG_INTERVAL", "10"))
    if progress_log_interval <= 0:
        raise ValueError("SUE_PROGRESS_LOG_INTERVAL must be positive")
    validate_sue_wandb_environment()
    return (
        max_steps,
        log_iters,
        checkpoint_interval,
        scalar_log_interval,
        progress_log_interval,
    )


def _destroy_distributed_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _finish_active_wandb(wandb_module) -> None:
    if getattr(wandb_module, "run", None) is not None:
        try:
            wandb_module.finish()
        except Exception as error:  # noqa: BLE001 - preserve the training exception
            print(
                f"TRAINING_ENTRY WARNING: W&B cleanup failed: {type(error).__name__}: {error}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    prioritize_source_root(sys.path)
    import train

    args = train.build_parser().parse_args(argv)
    config = train.load_config(args.config_path, args.temporal_loop_config)
    if args.preflight:
        return int(train.main(argv) or 0)

    if config.trainer == "score_distillation":
        import wandb
        import train_blockwise_lora_dmd as copied_wrapper
        import trainer.distillation as distillation

        asset_root = os.environ.get("SUE_ASSET_ROOT", "").strip()
        if not asset_root:
            raise ValueError("SUE_ASSET_ROOT must be provided to the training worker")
        install_training_policy(
            distillation.Trainer,
            wandb,
            max_steps=int(os.environ.get("SUE_MAX_STEPS", "0")),
            checkpoint_interval_seconds=int(
                os.environ.get("SUE_CHECKPOINT_INTERVAL_SECONDS", "3600")
            ),
            scalar_log_interval=int(os.environ.get("SUE_WANDB_LOG_INTERVAL", "100")),
            progress_log_interval=int(os.environ.get("SUE_PROGRESS_LOG_INTERVAL", "10")),
            asset_root=asset_root,
        )
        if argv is not None:
            sys.argv = [str(Path(copied_wrapper.__file__)), *argv]
        return int(copied_wrapper.main() or 0)

    if config.trainer != "diffusion":
        raise ValueError(f"SUE training entry does not support trainer={config.trainer!r}")

    (
        max_steps,
        _log_iters,
        checkpoint_interval,
        scalar_log_interval,
        progress_log_interval,
    ) = _validate_diffusion_sue_config(args, config)
    asset_root = os.environ.get("SUE_ASSET_ROOT", "").strip()
    if not asset_root:
        raise ValueError("SUE_ASSET_ROOT must be provided to the training worker")

    import wandb
    from trainer.diffusion import Trainer as DiffusionTrainer

    install_sue_wandb_runtime(wandb)
    install_diffusion_training_policy(
        DiffusionTrainer,
        wandb,
        max_steps=max_steps,
        checkpoint_interval_seconds=checkpoint_interval,
        scalar_log_interval=scalar_log_interval,
        progress_log_interval=progress_log_interval,
        asset_root=asset_root,
    )
    try:
        result = train.main(argv)
    except BaseException:
        _finish_active_wandb(wandb)
        try:
            _destroy_distributed_process_group()
        except Exception as error:  # noqa: BLE001 - preserve the training exception
            print(
                "TRAINING_ENTRY WARNING: distributed cleanup failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
        raise
    _destroy_distributed_process_group()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
