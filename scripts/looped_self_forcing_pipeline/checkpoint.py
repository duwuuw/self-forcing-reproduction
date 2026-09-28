"""Checkpoint identity helpers that remain valid after backend relocation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from os import PathLike
from typing import Any


_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | PathLike[str]) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_base_checkpoint(
    metadata: Mapping[str, Any], configured_base: str | PathLike[str]
) -> Path:
    """Resolve the active base checkpoint after validating its byte identity."""
    resolved_path, _ = resolve_base_checkpoint_with_hash(metadata, configured_base)
    return resolved_path


def resolve_base_checkpoint_with_hash(
    metadata: Mapping[str, Any], configured_base: str | PathLike[str]
) -> tuple[Path, str]:
    """Resolve the active backend's base checkpoint and verify its recorded hash.

    ``metadata['base_checkpoint']`` is retained as provenance only: it can be
    an absolute path from another backend. The configured path is authoritative
    after relocation, and must contain exactly the checkpoint bytes identified
    by ``base_checkpoint_sha256`` in the adapter metadata.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")

    expected_hash = metadata.get("base_checkpoint_sha256")
    if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(expected_hash):
        raise ValueError("adapter metadata is missing a valid base checkpoint identity hash")

    try:
        resolved_path = Path(configured_base).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise FileNotFoundError("configured base checkpoint is unavailable under SUE_ASSET_ROOT") from error

    if not resolved_path.is_file():
        raise ValueError("configured base checkpoint is not a regular file under SUE_ASSET_ROOT")

    actual_hash = sha256_file(resolved_path)
    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            "configured base checkpoint hash does not match the adapter's recorded identity: "
            f"expected {expected_hash.lower()}, got {actual_hash}"
        )
    return resolved_path, actual_hash


def portable_base_checkpoint_metadata(
    metadata: Mapping[str, Any],
    configured_base: str | PathLike[str],
    asset_root: str | PathLike[str],
    *,
    relative_reference: str | PathLike[str] | None = None,
    verified_sha256: str | None = None,
) -> dict[str, Any]:
    """Replace a machine-local base path with its configured relative name and hash."""
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    base = Path(configured_base).expanduser().resolve(strict=True)
    root = Path(asset_root).expanduser().resolve(strict=True)
    if not base.is_file():
        raise ValueError("configured base checkpoint is not a regular file under SUE_ASSET_ROOT")
    if relative_reference is not None:
        configured_relative = Path(relative_reference).expanduser()
        if (
            configured_relative.is_absolute()
            or ".." in configured_relative.parts
            or not configured_relative.parts
        ):
            raise ValueError("base checkpoint reference must be relative to SUE_ASSET_ROOT")
        resolved_reference = (root / configured_relative).resolve(strict=True)
        if resolved_reference != base:
            raise ValueError(
                "configured base checkpoint reference does not resolve to the selected checkpoint"
            )
        relative = configured_relative.as_posix()
    else:
        try:
            relative = base.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("base checkpoint must be inside the configured asset root") from error
    if verified_sha256 is not None:
        if not isinstance(verified_sha256, str) or not _SHA256_PATTERN.fullmatch(verified_sha256):
            raise ValueError("verified base checkpoint digest must be a SHA-256 hex string")
        actual_hash = verified_sha256.lower()
    else:
        actual_hash = sha256_file(base)
    expected_hash = metadata.get("base_checkpoint_sha256")
    if expected_hash is not None and (
        not isinstance(expected_hash, str)
        or not _SHA256_PATTERN.fullmatch(expected_hash)
        or expected_hash.lower() != actual_hash.lower()
    ):
        raise ValueError("full checkpoint base checkpoint hash does not match configured asset")
    portable = dict(metadata)
    portable["base_checkpoint"] = relative
    portable["base_checkpoint_name"] = base.name
    portable["base_checkpoint_sha256"] = actual_hash
    return portable
