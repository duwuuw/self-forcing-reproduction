"""Apply pipeline-level run policies, then delegate to the copied trainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from looped_self_forcing_pipeline.training_policy import install_training_policy


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT / "source"


def main(argv: list[str] | None = None) -> int:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
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


if __name__ == "__main__":
    raise SystemExit(main())
