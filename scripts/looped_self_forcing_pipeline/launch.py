"""Backend command plans that use the validated local SUE launchers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path


_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


def build_backend_command(
    stage: str,
    backend: str,
    *,
    launcher_root: str | Path,
    run_id: str,
    overrides: Sequence[str] = (),
) -> list[str]:
    """Build the command that invokes the backend's validated submission launcher."""
    if stage not in {"train", "infer"}:
        raise ValueError(f"unsupported stage: {stage}")
    if backend not in {"nm5", "autodl"}:
        raise ValueError(f"unsupported backend: {backend}")
    if not _RUN_ID.fullmatch(str(run_id)):
        raise ValueError("run_id must be 1–64 lowercase letters, digits, _ or -")

    normalized_overrides = []
    for override in overrides:
        key, separator, _value = str(override).partition("=")
        if not separator:
            raise ValueError("launch overrides must use key=value syntax")
        if key in {"stage", "run_id", "backend"} or key.startswith("assets."):
            raise ValueError(f"launcher-owned or private config override is not allowed: {key}")
        normalized_overrides.append(str(override))

    launcher_name = "nm5_submit.sh" if backend == "nm5" else "autodl_run.sh"
    return [
        "bash",
        str(Path(launcher_root) / launcher_name),
        stage,
        run_id,
        *normalized_overrides,
    ]
