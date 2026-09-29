import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils.training_config import validate_training_preflight


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--temporal_loop_config", type=str, default=None,
                        help="Optional YAML file containing a temporal_loop namespace")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate config and local checkpoints without constructing models or training",
    )
    return parser


def load_config(config_path, temporal_loop_config_path=None):
    default_config_path = (
        Path(__file__).resolve().parent / "configs" / "default_config.yaml"
    )
    default_config = OmegaConf.load(default_config_path)
    config = OmegaConf.load(config_path)
    if temporal_loop_config_path:
        temporal_loop_config = OmegaConf.load(temporal_loop_config_path)
        return OmegaConf.merge(default_config, config, temporal_loop_config)
    return OmegaConf.merge(default_config, config)


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = load_config(args.config_path, args.temporal_loop_config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.preflight = args.preflight

    if args.preflight:
        validate_training_preflight(config)
        print("training preflight passed")
        return 0

    from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer
    import wandb

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
