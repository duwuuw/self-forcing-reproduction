#!/usr/bin/env python3
"""Generate one image or a complete prompt snapshot with a Diffusers backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from diffusion_loop.cli import add_runtime_config_arguments, resolve_from_args
from diffusion_loop.flux2 import apply_flux2_loop_patch, set_flux2_step_metadata
from diffusion_loop.pixart import apply_pixart_loop_patch, set_pixart_step_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--records",
        type=Path,
        help="Write one JSONL generation record per image; required for prompt files.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("geneval", "dpgbench", "geneval2", "custom"),
        default="custom",
    )
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--allow-download", action="store_true")
    add_runtime_config_arguments(parser)
    args = parser.parse_args()
    if (args.prompt is None) != (args.output is None):
        parser.error("--prompt requires --output; --prompt-file requires --output-dir")
    if args.prompt_file is not None and args.records is None:
        parser.error("--prompt-file requires --records")
    if args.max_prompts is not None and args.max_prompts < 1:
        parser.error("--max-prompts must be >= 1")
    return args


def load_pipeline(backend: str, model_path: str, dtype, allow_download: bool):
    local_only = not allow_download
    if backend == "pixart":
        from diffusers import PixArtAlphaPipeline

        return PixArtAlphaPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
    if backend == "flux2":
        from diffusers import Flux2Pipeline

        pipeline = Flux2Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        pipeline.enable_model_cpu_offload()
        return pipeline
    raise ValueError("this entry point supports pixart and flux2 configs only")


def read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("--prompt must not be blank")
        return [args.prompt]
    lines = args.prompt_file.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("prompt snapshots must be nonempty and contain no blank lines")
    prompts = [line.strip() for line in lines]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    return prompts


def output_paths(args: argparse.Namespace, count: int) -> list[Path]:
    if args.output is not None:
        return [args.output]
    return [args.output_dir / f"prompt{index:05d}.png" for index in range(count)]


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    prompts = read_prompts(args)
    paths = output_paths(args, len(prompts))
    occupied = [path for path in paths if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite: {occupied[0]}")
    if args.records is not None and args.records.exists():
        raise FileExistsError(f"refusing to overwrite: {args.records}")
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_from_args(payload, args)
    generation = resolved.generation
    loop_config = resolved.loop_config
    print(json.dumps(resolved.to_record(), indent=2, sort_keys=True))

    dtype = torch.bfloat16 if generation["dtype"] == "bfloat16" else torch.float32
    pipeline = load_pipeline(
        payload["backend"],
        args.model_path,
        dtype,
        args.allow_download,
    )
    if payload["backend"] == "pixart":
        pipeline.to("cuda")

    total_steps = int(generation["num_inference_steps"])
    if loop_config.enabled:
        if payload["backend"] == "pixart":
            apply_pixart_loop_patch(pipeline, loop_config)
        else:
            apply_flux2_loop_patch(pipeline, loop_config)

    for index, (prompt, output_path) in enumerate(zip(prompts, paths, strict=True)):
        if loop_config.enabled:
            if payload["backend"] == "pixart":
                set_pixart_step_metadata(pipeline, total_steps=total_steps)
            else:
                set_flux2_step_metadata(pipeline, total_steps=total_steps)
        seed = int(generation["seed"])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                height=int(generation["height"]),
                width=int(generation["width"]),
                num_inference_steps=total_steps,
                guidance_scale=float(generation["model_guidance"]),
                generator=generator,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)
        if args.records is not None:
            append_record(
                args.records,
                {
                    "schema_version": 1,
                    "benchmark": args.benchmark,
                    "prompt_id": f"prompt{index:05d}",
                    "prompt": prompt,
                    "seed": seed,
                    "method_id": resolved.method_id,
                    "output_image": str(output_path.resolve()),
                    "resolved_config": resolved.to_record(),
                },
            )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
