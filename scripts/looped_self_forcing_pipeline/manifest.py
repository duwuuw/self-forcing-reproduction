"""Immutable, local run manifests for the looped self-forcing pipeline."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def _json_value(value: Any, *, location: str = "config") -> Any:
    """Convert common config values into stable JSON data or fail clearly."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return _json_value(value.value, location=location)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value), location=location)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} keys must be strings; got {type(key).__name__}")
            normalized[key] = _json_value(item, location=f"{location}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported value type {type(value).__name__}; "
        "normalize it to plain JSON data before materializing the run manifest"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity_digest(stage: str, config: dict[str, Any]) -> str:
    identity = {"stage": stage, "config": config}
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _read_existing_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"existing run manifest is unreadable: {path}") from error

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("stage"), str)
        or not isinstance(manifest.get("config"), dict)
        or not isinstance(manifest.get("inputs_sha256"), str)
    ):
        raise ValueError(f"existing run manifest has an unsupported or invalid format: {path}")

    expected_digest = _identity_digest(manifest["stage"], manifest["config"])
    if manifest["inputs_sha256"] != expected_digest:
        raise ValueError(f"existing run manifest failed its configuration integrity check: {path}")
    return manifest


def _create_manifest_once(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Install a complete manifest atomically, without replacing an existing one."""
    encoded = (_canonical_json(manifest) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MANIFEST_FILENAME}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            # A hard link is an atomic create-if-absent operation on the same filesystem.
            os.link(temporary_path, path)
        except FileExistsError:
            existing = _read_existing_manifest(path)
            if existing != manifest:
                raise ValueError(
                    f"run manifest already exists with different configuration: {path}"
                )
            return existing
        return manifest
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize_manifest(
    run_root: str | os.PathLike[str], stage: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a run manifest once and reject any later change to its inputs.

    The manifest is stored at ``<run_root>/manifest.json``. Repeating the call
    with equivalent JSON configuration returns the existing manifest without
    rewriting it; changing the stage or any configuration value raises
    ``ValueError``.
    """
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of JSON-compatible run inputs")

    normalized_config = _json_value(config)
    if not isinstance(normalized_config, dict):
        raise TypeError("config must normalize to a JSON object")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "config": normalized_config,
        "inputs_sha256": _identity_digest(stage, normalized_config),
    }
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_FILENAME

    if manifest_path.exists():
        existing = _read_existing_manifest(manifest_path)
        if existing != manifest:
            raise ValueError(
                f"run manifest already exists with different stage or configuration: {manifest_path}"
            )
        return existing

    return _create_manifest_once(manifest_path, manifest)
