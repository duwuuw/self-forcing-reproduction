"""Pure evidence checks for worker-produced checkpoints and videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


def verify_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_step: int,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Check the final training checkpoint's step and floating-point tensors."""
    import torch

    path = Path(checkpoint_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"training checkpoint is missing or empty: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("training checkpoint must contain a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("training checkpoint is missing metadata")
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step != expected_step:
        raise ValueError(
            f"training checkpoint step {step!r} does not match expected step {expected_step}"
        )
    if expected_seed is not None:
        seed = metadata.get("seed")
        if (
            isinstance(expected_seed, bool)
            or not isinstance(expected_seed, int)
            or expected_seed <= 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed != expected_seed
        ):
            raise ValueError(
                f"training checkpoint seed {seed!r} does not match expected ledger seed {expected_seed}"
            )
    generator = payload.get("generator")
    if not isinstance(generator, dict) or not generator:
        raise ValueError("training checkpoint is missing generator weights")
    tensors = list(_iter_tensors(generator))
    if not tensors:
        raise ValueError("training checkpoint contains no generator tensors")
    if payload.get("generator_ema") is not None:
        ema_tensors = list(_iter_tensors(payload["generator_ema"]))
        for tensor in ema_tensors:
            _require_finite(tensor)
    else:
        ema_tensors = []
    for tensor in tensors:
        _require_finite(tensor)
    return {
        "step": step,
        "trainable_parameter_count": sum(int(tensor.numel()) for tensor in tensors),
        "checkpoint_path": str(path.resolve()),
        "metadata": metadata,
        "payload": payload,
    }


def _iter_tensors(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tensors(nested)


def _require_finite(tensor: Any) -> None:
    import torch

    if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
        raise ValueError("training checkpoint contains non-finite values")


def prompt_count(prompt_path: str | Path) -> int:
    """Count prompts with the same line semantics as the copied TextDataset."""
    path = Path(prompt_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"prompt file is missing or empty: {path}")
    return len(path.read_text(encoding="utf-8").splitlines())


def require_ffprobe(which: Callable[[str], str | None] = shutil.which) -> str:
    """Resolve ffprobe before inference allocates a GPU or writes outputs."""
    executable = which("ffprobe")
    if not executable:
        raise FileNotFoundError(
            "ffprobe is required to verify generated MP4 files; add it to PATH before inference"
        )
    return executable


def require_empty_results_dir(results_dir: str | Path) -> Path:
    """Refuse to mix any prior artifact into a new inference run."""
    path = Path(results_dir)
    if path.is_symlink():
        raise FileExistsError(f"inference output directory must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"inference output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"inference output directory is not empty: {path}; use a new run_id"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_fresh_training_outputs(
    checkpoint_dir: str | Path,
    wandb_dir: str | Path,
    final_model: str | Path,
    *,
    wandb_run_id_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Refuse stale training evidence before reusing an immutable run id."""
    directories = (Path(checkpoint_dir), Path(wandb_dir))
    for directory in directories:
        if directory.is_symlink():
            raise FileExistsError(f"training artifact directory must not be a symlink: {directory}")
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(f"training artifact path is not a directory: {directory}")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"training artifacts already exist under {directory}; use a new run_id"
            )

    files = [Path(final_model), Path(final_model).with_name(Path(final_model).name + ".tmp")]
    if wandb_run_id_path is not None:
        files.append(Path(wandb_run_id_path))
    for file_path in files:
        if file_path.exists() or file_path.is_symlink():
            raise FileExistsError(
                f"training artifacts already exist at {file_path}; use a new run_id"
            )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def require_fresh_inference_outputs(
    results_dir: str | Path,
    wandb_dir: str | Path,
    wandb_run_id_path: str | Path,
) -> tuple[Path, Path]:
    """Require new video and W&B output locations for one inference run."""
    results = require_empty_results_dir(results_dir)
    tracking = Path(wandb_dir)
    if tracking.is_symlink():
        raise FileExistsError(f"inference W&B directory must not be a symlink: {tracking}")
    if tracking.exists() and not tracking.is_dir():
        raise FileExistsError(f"inference W&B path is not a directory: {tracking}")
    if tracking.exists() and any(tracking.iterdir()):
        raise FileExistsError(
            f"inference W&B artifacts already exist under {tracking}; use a new run_id"
        )
    marker = Path(wandb_run_id_path)
    if marker.exists() or marker.is_symlink():
        raise FileExistsError(
            f"inference W&B run identity already exists at {marker}; use a new run_id"
        )
    tracking.mkdir(parents=True, exist_ok=True)
    return results, tracking


def verify_inference_outputs(
    results_dir: str | Path,
    *,
    expected_count: int,
    expected_frames: int,
    expected_fps: int,
    run_command: Callable[..., Any] | None = None,
) -> list[Path]:
    """Require exact video count, decoded frame count, and frame rate."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if expected_frames <= 0 or expected_fps <= 0:
        raise ValueError("expected video frames and fps must be positive")
    path = Path(results_dir)
    videos = sorted(path.glob("*.mp4"))
    if len(videos) != expected_count:
        raise RuntimeError(
            f"expected exactly {expected_count} inference videos, found {len(videos)} in {path}"
        )
    probe = run_command or subprocess.run
    if run_command is None:
        require_ffprobe()
    for video in videos:
        if video.stat().st_size == 0:
            raise RuntimeError(f"empty inference output: {video}")
        result = probe(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "format=duration:stream=codec_type,nb_read_frames,avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError(f"ffprobe could not decode inference output: {video}")
        try:
            info = json.loads(result.stdout)
            video_streams = [
                stream
                for stream in info.get("streams", [])
                if stream.get("codec_type") == "video"
            ]
            duration = float(info.get("format", {}).get("duration", 0))
            if not video_streams:
                raise RuntimeError(f"inference output has no decodable video stream: {video}")
            stream = video_streams[0]
            decoded_frames = int(stream.get("nb_read_frames", "0"))
            rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
            frame_rate = Fraction(rate_text)
        except (TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as error:
            raise RuntimeError(f"ffprobe returned invalid metadata for {video}") from error
        if duration <= 0:
            raise RuntimeError(f"inference output has no decodable video stream: {video}")
        if decoded_frames != expected_frames:
            raise RuntimeError(
                f"inference output frame count {decoded_frames} does not match expected "
                f"{expected_frames}: {video}"
            )
        if frame_rate != expected_fps:
            raise RuntimeError(
                f"inference output fps {float(frame_rate):g} does not match expected "
                f"{expected_fps}: {video}"
            )
    return videos
