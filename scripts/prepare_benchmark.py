#!/usr/bin/env python3
"""Resolve a paper configuration into a deterministic generation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffusion_loop.cli import add_runtime_config_arguments, resolve_from_args


def read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("prompt snapshot must be nonempty and contain no blank lines")
    return [line.strip() for line in lines]


def build_plan(
    prompts: list[str],
    benchmark: str,
    resolved,
    output_root: Path,
) -> list[dict]:
    seed = int(resolved.generation["seed"])
    return [
        {
            "schema_version": 1,
            "benchmark": benchmark,
            "prompt_id": f"prompt{index:05d}",
            "prompt": prompt,
            "seed": seed,
            "method_id": resolved.method_id,
            "output_image": str(
                output_root
                / resolved.method_id
                / benchmark
                / f"prompt{index:05d}.png"
            ),
            "resolved_config": resolved.to_record(),
        }
        for index, prompt in enumerate(prompts)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        choices=("geneval", "dpgbench", "geneval2"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int)
    add_runtime_config_arguments(parser)
    args = parser.parse_args()
    if args.max_prompts is not None and args.max_prompts < 1:
        parser.error("--max-prompts must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    if args.plan.exists():
        raise FileExistsError(f"refusing to overwrite: {args.plan}")
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_from_args(payload, args)
    prompts = read_prompts(args.prompt_file)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    rows = build_plan(prompts, args.benchmark, resolved, args.output_root)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "method_id": resolved.method_id,
                "paper_preset": resolved.paper_preset,
                "samples": len(rows),
                "plan": str(args.plan),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
