#!/usr/bin/env python3
"""Resolve a paper preset plus overrides without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffusion_loop.cli import add_runtime_config_arguments, resolve_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the exact runtime configuration after composing Dense/Sparse "
            "Token Loop, optional Loop Guidance, and hyperparameter overrides."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; existing files are never overwritten.",
    )
    add_runtime_config_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_from_args(payload, args)
    rendered = json.dumps(resolved.to_record(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
