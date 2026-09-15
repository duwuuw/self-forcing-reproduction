#!/usr/bin/env python3
"""Copy N randomly chosen generated videos into a flat sample dir with a manifest.

Usage:
    python scripts/pick_sample_videos.py <workspace_root> <n> [out_dir_name] [source_dir]

`source_dir` is relative to <workspace_root> and defaults to the main run's
`scale_up_outputs/nm5_looped_self_forcing/videos/vbench_long_640`; pass e.g.
`scale_up_outputs/nm5_looped_self_forcing/variants/k2_81f/videos/rank_000` to
sample a variant. Videos are named `<index>-0_ema.mp4`, so the prompt for each is
looked up by index in `Self-Forcing/prompts/vbench/all_dimension_extended.txt`.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

EXP_REL = "scale_up_outputs/nm5_looped_self_forcing"


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    n = int(sys.argv[2])
    out_name = sys.argv[3] if len(sys.argv) > 3 else "sample10"
    source_rel = sys.argv[4] if len(sys.argv) > 4 else f"{EXP_REL}/videos/vbench_long_640"

    video_dir = root / source_rel
    prompt_file = root / "Self-Forcing/prompts/vbench/all_dimension_extended.txt"

    videos = sorted(video_dir.glob("*.mp4"), key=lambda p: int(p.name.split("-")[0]))
    if not videos:
        raise SystemExit(f"no videos under {video_dir}")
    prompts = prompt_file.read_text(encoding="utf-8").splitlines()

    picked = random.SystemRandom().sample(videos, min(n, len(videos)))
    out = root / "artifacts" / out_name
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for video in sorted(picked, key=lambda p: int(p.name.split("-")[0])):
        index = int(video.name.split("-")[0])
        shutil.copy2(video, out / video.name)
        manifest.append({
            "index": index,
            "file": video.name,
            "prompt": prompts[index] if index < len(prompts) else "",
        })

    (out / "manifest.json").write_text(
        json.dumps({
            "source_dir": str(video_dir),
            "total_generated": len(videos),
            "sample_size": len(manifest),
            "seed": "SystemRandom (non-reproducible)",
            "videos": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"sampled {len(manifest)} of {len(videos)} videos into {out}")
    for row in manifest:
        print(f"  {row['index']:>4}  {row['file']:<16} {row['prompt'][:80]}")


if __name__ == "__main__":
    main()
