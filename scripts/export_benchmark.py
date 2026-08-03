#!/usr/bin/env python3
"""Validate generation records and export official-evaluator directory layouts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("prompt snapshot must be nonempty and contain no blank lines")
    return [line.strip() for line in lines]


def resolve_image(records: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = records.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"generated image is missing: {path}")
    return path


def validated_rows(
    records: Path,
    prompts: list[str],
    benchmark: str,
    method_id: str,
    seed: int,
    allow_incomplete: bool,
) -> list[tuple[int, dict, Path]]:
    selected: dict[int, tuple[dict, Path]] = {}
    for row in read_jsonl(records):
        if row.get("benchmark") != benchmark:
            continue
        if row.get("method_id") != method_id or int(row.get("seed", -1)) != seed:
            continue
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id.startswith("prompt") or not prompt_id[6:].isdigit():
            raise ValueError(f"invalid prompt_id: {prompt_id!r}")
        index = int(prompt_id[6:])
        if index >= len(prompts):
            raise ValueError(f"prompt index is outside snapshot: {prompt_id}")
        if row.get("prompt") != prompts[index]:
            raise ValueError(f"prompt text mismatch at {prompt_id}")
        if index in selected:
            raise ValueError(f"duplicate generation record at {prompt_id}")
        selected[index] = (row, resolve_image(records, str(row["output_image"])))
    expected = set(range(len(prompts)))
    missing = sorted(expected - selected.keys())
    if missing and not allow_incomplete:
        raise ValueError(
            f"missing {len(missing)} generations; first missing prompt{missing[0]:05d}"
        )
    return [(index, *selected[index]) for index in sorted(selected)]


def transfer(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    if copy:
        shutil.copy2(source, destination)
    else:
        os.symlink(source, destination)


def export_geneval(
    rows: list[tuple[int, dict, Path]],
    annotations: Path,
    destination: Path,
    copy: bool,
) -> None:
    metadata = read_jsonl(annotations)
    for index, _, image in rows:
        if index >= len(metadata):
            raise ValueError(f"GenEval annotation missing at prompt{index:05d}")
        prompt_dir = destination / f"{index:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=False)
        (prompt_dir / "metadata.jsonl").write_text(
            json.dumps(metadata[index], ensure_ascii=False) + "\n", encoding="utf-8"
        )
        transfer(image, prompt_dir / "samples" / "0000.png", copy)


def export_dpgbench(
    rows: list[tuple[int, dict, Path]], destination: Path, copy: bool
) -> None:
    items = read_jsonl(ROOT / "prompts" / "dpgbench_1065_items.jsonl")
    for index, row, image in rows:
        if index >= len(items) or items[index].get("prompt") != row["prompt"]:
            raise ValueError(f"DPG-Bench item mapping mismatch at prompt{index:05d}")
        transfer(image, destination / f"{items[index]['item_id']}.png", copy)


def export_geneval2(
    rows: list[tuple[int, dict, Path]], destination: Path, copy: bool
) -> None:
    image_paths = {}
    for index, row, image in rows:
        target = destination / "images" / f"{index:05d}.png"
        transfer(image, target, copy)
        image_paths[row["prompt"]] = str(target.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "image_paths.json").write_text(
        json.dumps(image_paths, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--benchmark", choices=("geneval", "dpgbench", "geneval2"), required=True
    )
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.benchmark == "geneval" and args.annotations is None:
        parser.error("GenEval export requires --annotations from the official release")
    return args


def main() -> int:
    args = parse_args()
    destination = args.out_dir / args.method_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    prompts = read_prompts(args.prompt_file)
    rows = validated_rows(
        args.records,
        prompts,
        args.benchmark,
        args.method_id,
        args.seed,
        args.allow_incomplete,
    )
    if args.benchmark == "geneval":
        export_geneval(rows, args.annotations, destination, args.copy)
    elif args.benchmark == "dpgbench":
        export_dpgbench(rows, destination, args.copy)
    else:
        export_geneval2(rows, destination, args.copy)
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "method_id": args.method_id,
                "samples": len(rows),
                "out_dir": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
