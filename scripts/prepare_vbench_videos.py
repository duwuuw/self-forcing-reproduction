#!/usr/bin/env python3
"""Map index-named Self-Forcing videos to the official VBench layout."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VIDEO_RE = re.compile(r"^(\d+)-0_(?:ema|regular)\.mp4$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prompts = args.prompt_file.read_text(encoding="utf-8").splitlines()
    videos = {}
    for path in args.source_dir.glob("*.mp4"):
        match = VIDEO_RE.match(path.name)
        if match:
            videos[int(match.group(1))] = path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    missing = []
    for index, prompt in enumerate(prompts):
        source = videos.get(index)
        if source is None:
            missing.append(index)
            continue
        target = args.output_dir / f"{prompt}-0.mp4"
        if not target.exists():
            target.symlink_to(source.resolve())
        manifest.append({
            "index": index,
            "prompt": prompt,
            "source": str(source.resolve()),
            "target": str(target),
        })

    (args.output_dir / "layout_manifest.json").write_text(
        json.dumps({
            "prompt_count": len(prompts),
            "mapped_count": len(manifest),
            "missing_indices": missing,
            "duplicate_prompt_targets_reused": len(manifest) - len({row["target"] for row in manifest}),
            "entries": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if missing:
        raise SystemExit(f"missing {len(missing)} generated videos; first indices: {missing[:10]}")
    print(f"mapped {len(manifest)} prompts into {args.output_dir}")


if __name__ == "__main__":
    main()
