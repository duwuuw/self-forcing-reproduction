#!/usr/bin/env python3
"""CPU-only validation for the anonymous Code and Data Supplement."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_loop import LoopConfig, resolve_run_config  # noqa: E402
from diffusion_loop.loop_config import VALID_SELECTORS  # noqa: E402


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
}
FORBIDDEN_PATH_FRAGMENTS = (
    "/" + "workspace/",
    "/" + "gpfs/",
    "/" + "scratch/",
    "/" + "home/",
    "/" + "users/",
    "file" + "://",
)
FORBIDDEN_METHOD_FRAGMENTS = (
    "ab" + "ab",
    "aba" + "cb",
    "aba" + "cacheb",
    "t" + "mix",
)
FORBIDDEN_IDENTITY_FRAGMENTS = (
    "n" + "m5",
    "a" + "login",
    "ehpc" + "821",
    "t" + "long",
    "y" + "yy",
)
EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+",
    re.IGNORECASE,
)
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".jpeg",
    ".jpg",
    ".log",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".webp",
}
PAPER_METHOD_IDS = {
    "without_loop",
    "dense_token_loop",
    "sparse_token_loop",
    "loop_guidance_dense_token_loop",
    "loop_guidance_sparse_token_loop",
}
REQUIRED_RELEASE_FILES = {
    "LICENSE",
    "THIRD_PARTY.md",
    "ENVIRONMENT.md",
    "REPRODUCE.md",
    "prompts/manifest.json",
    "requirements/scale_rae.txt",
    "scripts/aggregate_metrics.py",
    "scripts/export_benchmark.py",
    "scripts/prepare_benchmark.py",
    "scripts/run_diffusers.py",
}


def validate_configs() -> int:
    count = 0
    for path in sorted((ROOT / "configs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported schema in {path.name}")
        if payload.get("model", {}).get("weights_included") is not False:
            raise ValueError(f"weights_included must be false in {path.name}")
        revision = payload.get("model", {}).get("revision")
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"invalid model revision in {path.name}")
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, dict):
            raise ValueError(f"missing capabilities in {path.name}")
        token_loops = set(capabilities.get("token_loops", []))
        guided_loops = set(capabilities.get("loop_guidance_token_loops", []))
        sparse_selectors = set(capabilities.get("sparse_selectors", []))
        if not token_loops or not token_loops <= {"dense", "sparse"}:
            raise ValueError(f"invalid token-loop capabilities in {path.name}")
        if not guided_loops <= token_loops:
            raise ValueError(f"Loop Guidance capability lacks its base loop in {path.name}")
        if not sparse_selectors <= set(VALID_SELECTORS) - {"all"}:
            raise ValueError(f"invalid sparse selectors in {path.name}")
        if ("sparse" in token_loops) != bool(sparse_selectors):
            raise ValueError(f"sparse selector capabilities mismatch in {path.name}")
        for method_name, raw in payload["methods"].items():
            if method_name not in PAPER_METHOD_IDS:
                raise ValueError(f"non-paper method ID {method_name!r} in {path.name}")
            config = LoopConfig.from_dict(raw)
            if method_name == "without_loop" and config.enabled:
                raise ValueError(f"without_loop must be disabled in {path.name}")
            if method_name == "dense_token_loop" and (
                config.token_operator != "dense" or config.loopguidance_enabled
            ):
                raise ValueError(f"dense_token_loop semantics mismatch in {path.name}")
            if method_name == "sparse_token_loop" and (
                config.token_operator != "sparse" or config.loopguidance_enabled
            ):
                raise ValueError(f"sparse_token_loop semantics mismatch in {path.name}")
            if method_name == "loop_guidance_dense_token_loop" and (
                config.token_operator != "dense" or not config.loopguidance_enabled
            ):
                raise ValueError(f"dense Loop Guidance semantics mismatch in {path.name}")
            if method_name == "loop_guidance_sparse_token_loop" and (
                config.token_operator != "sparse" or not config.loopguidance_enabled
            ):
                raise ValueError(f"sparse Loop Guidance semantics mismatch in {path.name}")
            count += 1

        for token_loop in sorted(token_loops):
            resolve_run_config(payload, token_loop=token_loop)
            if token_loop in guided_loops:
                guided_id = f"loop_guidance_{token_loop}_token_loop"
                kwargs = {}
                if guided_id not in payload["methods"]:
                    kwargs["loop_guidance_weight"] = 2.0
                resolved = resolve_run_config(
                    payload,
                    token_loop=token_loop,
                    loop_guidance=True,
                    **kwargs,
                )
                if resolved.loop_config.token_operator != token_loop:
                    raise ValueError(f"guided composition changed token loop in {path.name}")

        if payload.get("backend") in {"scale_rae", "raev2"}:
            expected = {"dense", "sparse"}
            if token_loops != expected or guided_loops != expected:
                raise ValueError(
                    f"RAE backend must expose Dense/Sparse with/without Loop Guidance in {path.name}"
                )
    return count


def validate_csv() -> int:
    count = 0
    for path in sorted((ROOT / "data").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty CSV: {path.name}")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError(f"inconsistent CSV width: {path.name}")
        count += len(rows)
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prompts() -> int:
    manifest = json.loads((ROOT / "prompts" / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("seed") != 42:
        raise ValueError("unexpected prompt manifest protocol")
    total = 0
    for benchmark, spec in manifest.get("benchmarks", {}).items():
        path = ROOT / spec["file"]
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != int(spec["count"]) or any(not line.strip() for line in lines):
            raise ValueError(f"prompt count/content mismatch for {benchmark}")
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"prompt checksum mismatch for {benchmark}")
        if "items_file" in spec:
            items_path = ROOT / spec["items_file"]
            items = [
                json.loads(line)
                for line in items_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(items) != len(lines):
                raise ValueError(f"item mapping count mismatch for {benchmark}")
            if any(item.get("prompt") != prompt for item, prompt in zip(items, lines)):
                raise ValueError(f"item mapping prompt mismatch for {benchmark}")
            if sha256(items_path) != spec["items_sha256"]:
                raise ValueError(f"item mapping checksum mismatch for {benchmark}")
        total += len(lines)
    if set(manifest.get("benchmarks", {})) != {"geneval", "dpgbench", "geneval2"}:
        raise ValueError("prompt manifest must contain all three paper benchmarks")
    return total


def validate_tree() -> int:
    missing = sorted(path for path in REQUIRED_RELEASE_FILES if not (ROOT / path).is_file())
    if missing:
        raise ValueError(f"required release files are missing: {missing}")
    count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            raise ValueError(f"forbidden artifact type: {path.relative_to(ROOT)}")
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"unexpected large file: {path.relative_to(ROOT)}")
        if (
            path.suffix.lower() not in TEXT_SUFFIXES
            and path.name not in {".gitignore", "LICENSE"}
        ):
            continue
        text = path.read_text(encoding="utf-8").lower()
        for fragment in (
            FORBIDDEN_PATH_FRAGMENTS
            + FORBIDDEN_METHOD_FRAGMENTS
            + FORBIDDEN_IDENTITY_FRAGMENTS
        ):
            if fragment in text:
                raise ValueError(
                    f"forbidden release fragment in {path.relative_to(ROOT)}"
                )
        if EMAIL_PATTERN.search(text):
            raise ValueError(f"possible email address in {path.relative_to(ROOT)}")
        count += 1
    return count


def main() -> int:
    configs = validate_configs()
    rows = validate_csv()
    prompts = validate_prompts()
    files = validate_tree()
    print(
        "release validation passed: "
        f"{configs} configs, {rows} data rows, {prompts} prompts, {files} text files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
