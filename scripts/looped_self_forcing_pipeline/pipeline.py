#!/usr/bin/env python3
"""Hydra-backed command line entry point for looped Self-Forcing workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from looped_self_forcing_pipeline.config import (
    compose_config,
    resolve_exp_dir,
    resolve_runtime_paths,
)


_COMMANDS = (
    "show-config",
    "check-train",
    "check-infer",
    "train-worker",
    "infer-worker",
    "record-submit",
    "submit-plan",
    "resolve-experiment-name",
)


def _parse_args(argv: Sequence[str] | None = None):
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    if "--show-config" in tokens:
        tokens.remove("--show-config")
        if tokens and tokens[0] in _COMMANDS:
            tokens[0] = "show-config"
        else:
            tokens.insert(0, "show-config")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=_COMMANDS, default="show-config")
    parser.add_argument("--exp-dir", help="resolved SUE experiment directory; defaults to SUE_EXP_DIR")
    parser.add_argument("--job-id", default="", help="scheduler or detached-session identifier")
    args, overrides = parser.parse_known_args(tokens)
    if any(value.startswith("--") for value in overrides):
        parser.error("Hydra overrides must use key=value syntax")
    if any(value.startswith("assets.") for value in overrides):
        parser.error(
            "asset paths must come from the private SUE_ASSET_ROOT environment variable"
        )
    return parser, args, overrides


def _stage_overrides(overrides: list[str], stage: str) -> list[str]:
    found = [value.partition("=")[2] for value in overrides if value.startswith("stage=")]
    if found and any(value != stage for value in found):
        raise ValueError(f"command requires stage={stage}; received {found}")
    return list(overrides) if found else [*overrides, f"stage={stage}"]


def main(argv: Sequence[str] | None = None) -> int:
    parser, args, overrides = _parse_args(argv)
    show_config = args.command == "show-config"
    if args.command in {"check-train", "train-worker"}:
        stage = "train"
        overrides = _stage_overrides(overrides, stage)
    elif args.command in {"check-infer", "infer-worker"}:
        stage = "infer"
        overrides = _stage_overrides(overrides, stage)
    elif args.command == "record-submit" and "stage=" not in " ".join(overrides):
        parser.error("record-submit requires a stage=train|infer Hydra override")
    else:
        stage = None

    exp_dir = resolve_exp_dir(args.exp_dir)
    config = compose_config(overrides, exp_dir=exp_dir)
    if show_config:
        from omegaconf import OmegaConf

        rendered_config = OmegaConf.to_container(config, resolve=True)
        if isinstance(rendered_config, dict) and isinstance(rendered_config.get("assets"), dict):
            rendered_config["assets"]["root"] = (
                "<configured privately through SUE_ASSET_ROOT>"
                if rendered_config["assets"].get("root")
                else "<not configured>"
            )
        print(
            json.dumps(
                rendered_config,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if stage is not None and str(config.stage.name) != stage:
        raise ValueError(f"composed stage {config.stage.name!r} differs from command {stage!r}")

    if args.command.startswith("check-"):
        from looped_self_forcing_pipeline.worker import check_stage

        prepared = check_stage(config, stage, exp_dir=exp_dir)
        print(
            f"{stage} preflight passed: run_id={config.run_id}; "
            f"backend={config.backend.name}; config={prepared['config_path']}"
        )
        return 0

    if args.command in {"train-worker", "infer-worker"}:
        from looped_self_forcing_pipeline.worker import execute_worker

        result = execute_worker(config, stage, exp_dir=exp_dir)
        print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False))
        return 0

    if args.command == "record-submit":
        from looped_self_forcing_pipeline.ledger import write_result
        from looped_self_forcing_pipeline.worker import build_experiment_name, method_identity

        run_id = str(config.run_id or "").strip()
        method = method_identity(config.method)
        runtime_paths = resolve_runtime_paths(exp_dir)
        output_path = (
            runtime_paths["ckpt_root"] / run_id
            if str(config.stage.name) == "train"
            else runtime_paths["final_result_root"] / run_id
        )
        stage_seed = int(
            config.train.seed if str(config.stage.name) == "train" else config.infer.seed
        )
        ledger_path = runtime_paths["artifacts_root"] / "experiment_results.csv"
        write_result(
            ledger_path,
            {
                "method": method["mode"],
                "experiment_name": build_experiment_name(
                    method, run_id, str(config.stage.name)
                ),
                "backend": config.backend.name,
                "status": "submitted",
                "execution_id": args.job_id,
                "slurm_job_id": args.job_id if config.backend.name == "nm5" else "",
                "seed": stage_seed,
                "checkpoint_or_output_path": str(output_path),
            },
        )
        return 0

    if args.command == "resolve-experiment-name":
        from looped_self_forcing_pipeline.worker import build_experiment_name, method_identity

        method = method_identity(config.method)
        name = build_experiment_name(
            method,
            str(config.run_id or "").strip(),
            str(config.stage.name),
        )
        print(name)
        return 0

    if args.command == "submit-plan":
        from looped_self_forcing_pipeline.launch import build_backend_command

        run_id = str(config.run_id or "").strip()
        stage = str(config.stage.name)
        output_roots = resolve_runtime_paths(exp_dir)
        launcher_overrides = [
            override
            for override in overrides
            if override.partition("=")[0] not in {"backend", "stage", "run_id"}
        ]
        command = build_backend_command(
            stage,
            str(config.backend.name),
            launcher_root=output_roots["slurm_scripts_root"],
            run_id=run_id,
            overrides=launcher_overrides,
        )
        print(json.dumps(command, ensure_ascii=False))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
