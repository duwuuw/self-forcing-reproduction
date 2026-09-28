"""Resolve a SUE bundle and compose its Hydra configuration groups."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_EXP_DIR = WORKSPACE_ROOT / "scale_up_outputs" / "looped_self_forcing_pipeline"
RUNTIME_PATH_KEYS = (
    "datasets_root",
    "slurm_scripts_root",
    "artifacts_root",
    "ckpt_root",
    "final_result_root",
    "logs_root",
)


def resolve_exp_dir(
    exp_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the active bundle root from an argument, SUE_EXP_DIR, or default."""
    environment = os.environ if environ is None else environ
    configured = exp_dir if exp_dir is not None else environment.get("SUE_EXP_DIR")
    if configured is None or not str(configured).strip():
        return DEFAULT_EXP_DIR.resolve()
    return Path(configured).expanduser().resolve()


def resolve_config_dir(exp_dir: str | Path | None = None) -> Path:
    """Return the Hydra config directory beneath the selected bundle."""
    return (resolve_exp_dir(exp_dir) / "config").resolve()


def resolve_runtime_path(exp_dir: str | Path | None = None) -> Path:
    """Return the SUE runtime contract next to this bundle's Hydra configs."""
    return (resolve_config_dir(exp_dir) / "runtime.yaml").resolve()


def load_runtime_config(exp_dir: str | Path | None = None) -> dict[str, object]:
    """Read the active bundle's parsed runtime contract."""
    from omegaconf import OmegaConf

    runtime_path = resolve_runtime_path(exp_dir)
    if not runtime_path.is_file():
        raise FileNotFoundError(f"SUE runtime config does not exist: {runtime_path}")
    runtime = OmegaConf.to_container(OmegaConf.load(runtime_path), resolve=True)
    if not isinstance(runtime, dict):
        raise ValueError(f"SUE runtime config must contain a mapping: {runtime_path}")
    return runtime


def resolve_runtime_paths(exp_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve runtime dataset and output roots beneath the active SUE bundle."""
    bundle = resolve_exp_dir(exp_dir)
    runtime_path = resolve_runtime_path(bundle)
    runtime = load_runtime_config(bundle)
    if not isinstance(runtime.get("paths"), dict):
        raise ValueError(f"SUE runtime config must contain a paths mapping: {runtime_path}")
    if runtime["paths"].get("config_root") != "config":
        raise ValueError(
            "paths.config_root is fixed to 'config' because Hydra and runtime.yaml share that bootstrap directory"
        )
    if "sandbox_resources" in runtime or "wandb_policy" in runtime:
        raise ValueError("runtime sandbox_resources and wandb_policy must be nested under policy:")
    policy = runtime.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"SUE runtime config must contain a policy mapping: {runtime_path}")
    if not isinstance(policy.get("sandbox_resources"), dict) or not isinstance(
        policy.get("wandb_policy"), dict
    ):
        raise ValueError(
            f"SUE runtime config policy must contain sandbox_resources and wandb_policy: {runtime_path}"
        )

    roots: dict[str, Path] = {}
    for key in RUNTIME_PATH_KEYS:
        value = runtime["paths"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SUE runtime config paths.{key} must be a non-empty relative path")
        relative = Path(value).expanduser()
        if relative.is_absolute():
            raise ValueError(f"SUE runtime config paths.{key} must be relative to {bundle}")
        resolved = (bundle / relative).resolve()
        try:
            resolved.relative_to(bundle)
        except ValueError as error:
            raise ValueError(
                f"SUE runtime config paths.{key} must resolve inside {bundle}"
            ) from error
        roots[key] = resolved
    return roots


def compose_config(
    overrides: Sequence[str] = (),
    *,
    exp_dir: str | Path | None = None,
):
    """Compose method/backend/stage/tracking/scale groups from the active bundle."""
    from hydra import compose, initialize_config_dir

    config_dir = resolve_config_dir(exp_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory does not exist: {config_dir}")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=list(overrides))
