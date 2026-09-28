"""Resolve a SUE bundle and compose its Hydra configuration groups."""

from __future__ import annotations

import os
import math
from collections.abc import Mapping, Sequence
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_EXP_DIR = WORKSPACE_ROOT / "scale_up_outputs" / "looped_self_forcing_pipeline"
RUNTIME_PATH_KEYS = (
    "autotune_hyperparam_root",
    "datasets_root",
    "slurm_scripts_root",
    "artifacts_root",
    "ckpt_root",
    "final_result_root",
    "logs_root",
    "state_root",
    "readiness_root",
    "ledger_csv",
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


def _relative_runtime_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime {label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"runtime {label} must not be absolute or contain traversal components")
    return path


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
    if "wandb_policy" in runtime:
        raise ValueError("runtime wandb_policy must be nested under policy:")
    policy = runtime.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"SUE runtime config must contain a policy mapping: {runtime_path}")
    if any(key in policy for key in ("sandbox_resources", "sandbox_script_folders", "backend_env")):
        raise ValueError("runtime sandbox resources, script folders, and backend env sources are concrete top-level values")
    if not isinstance(policy.get("wandb_policy"), dict):
        raise ValueError(f"SUE runtime config policy must contain wandb_policy: {runtime_path}")
    resources = runtime.get("sandbox_resources")
    script_folders = runtime.get("sandbox_script_folders")
    backend_env = runtime.get("backend_env")
    if not isinstance(resources, dict) or not isinstance(script_folders, dict) or not isinstance(
        backend_env, dict
    ):
        raise ValueError(
            f"SUE runtime config must contain top-level sandbox_resources, sandbox_script_folders, and backend_env mappings: {runtime_path}"
        )
    backend_config = runtime.get("backend")
    primary = backend_config.get("primary") if isinstance(backend_config, dict) else None
    if not isinstance(primary, str) or not isinstance(resources.get(primary), dict):
        raise ValueError("selected backend must have a top-level sandbox_resources entry")
    if not isinstance(script_folders.get(primary), str) or not script_folders[primary].strip():
        raise ValueError("selected backend must have a top-level sandbox_script_folders entry")
    if script_folders[primary] != runtime["paths"].get("slurm_scripts_root"):
        raise ValueError(
            "sandbox_script_folders for the selected backend must match paths.slurm_scripts_root"
        )
    backend_env_entry = backend_env.get(primary)
    if (
        not isinstance(backend_env_entry, dict)
        or not isinstance(backend_env_entry.get("source_note"), str)
        or not isinstance(backend_env_entry.get("required_keys"), list)
        or not backend_env_entry["required_keys"]
        or any(not isinstance(key, str) or not key.strip() for key in backend_env_entry["required_keys"])
    ):
        raise ValueError(
            f"runtime backend_env.{primary} must contain source_note and required_keys metadata"
        )
    if primary == "nm5":
        resources_for_nm5 = resources[primary]
        for key in ("partition", "qos"):
            if not isinstance(resources_for_nm5.get(key), str) or not resources_for_nm5[key].strip():
                raise ValueError(f"runtime sandbox_resources.nm5.{key} must be set")
        if not isinstance(resources_for_nm5.get("gpu_type"), str) or not resources_for_nm5["gpu_type"].strip():
            raise ValueError("runtime sandbox_resources.nm5.gpu_type must be set")
        memory_usable = resources_for_nm5.get("gpu_usable_memory_gib")
        memory_reserve = resources_for_nm5.get("gpu_non_torch_reserve_gib")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in (memory_usable, memory_reserve)
        ) or memory_reserve >= memory_usable:
            raise ValueError("runtime NM5 GPU memory and non-PyTorch reserve must be finite and positive")
        for key in (
            "gpus_per_node",
            "max_nodes_per_job",
            "cpus_per_gpu",
            "train_gpus",
            "infer_gpus",
            "timeout_kill_after_seconds",
            "finalization_grace_seconds",
        ):
            value = resources_for_nm5.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"runtime sandbox_resources.nm5.{key} must be a positive integer")
        required_env_keys = set(backend_env_entry["required_keys"])
        missing_env_keys = {
            "NM5_DEEPRESEARCH_ROOT",
            "NM5_WORKSPACE_ROOT",
            "NM5_ACCOUNT",
            "NM5_LOGIN_SSH",
            "WANDB_API_KEY",
            "WANDB_ENTITY",
        } - required_env_keys
        if missing_env_keys:
            raise ValueError(
                "runtime backend_env.nm5.required_keys lacks: "
                + ", ".join(sorted(missing_env_keys))
            )

        environment = runtime.get("environment")
        if not isinstance(environment, dict):
            raise ValueError("runtime must contain an environment mapping for the selected NM5 backend")
        if environment.get("env_manager") != "conda":
            raise ValueError("runtime environment.env_manager must be conda for NM5")
        env_root = _relative_runtime_path(environment.get("env_root"), "environment.env_root")
        python_binary = _relative_runtime_path(
            environment.get("python_binary"), "environment.python_binary"
        )
        asset_root = _relative_runtime_path(environment.get("asset_root"), "environment.asset_root")
        allowed_roots_raw = environment.get("allowed_external_roots")
        if not isinstance(allowed_roots_raw, list) or not allowed_roots_raw:
            raise ValueError("runtime environment.allowed_external_roots must be a non-empty list")
        allowed_roots = [
            _relative_runtime_path(value, "environment.allowed_external_roots[]")
            for value in allowed_roots_raw
        ]
        for label, candidate in (
            ("environment.env_root", env_root),
            ("environment.python_binary", python_binary),
            ("environment.asset_root", asset_root),
        ):
            if not any(candidate == allowed or allowed in candidate.parents for allowed in allowed_roots):
                raise ValueError(f"runtime {label} must resolve beneath environment.allowed_external_roots")
        if "python_overlay" in environment:
            overlay = _relative_runtime_path(
                environment.get("python_overlay"), "environment.python_overlay"
            )
            overlay_root = (bundle / overlay).resolve()
            try:
                overlay_root.relative_to(bundle)
            except ValueError as error:
                raise ValueError("runtime environment.python_overlay must stay inside the selected bundle") from error
        expected_cache_sources = {
            "HF_HOME": "NM5_HF_HOME",
            "HF_HUB_CACHE": "NM5_HF_HUB_CACHE",
            "MODELSCOPE_CACHE": "NM5_MODELSCOPE_CACHE",
            "TORCH_HOME": "NM5_TORCH_HOME",
            "MPLCONFIGDIR": "NM5_MPLCONFIGDIR",
            "WANDB_CACHE_DIR": "NM5_WANDB_CACHE_DIR",
        }
        cache_sources = environment.get("cache_env_sources")
        if cache_sources != expected_cache_sources:
            raise ValueError("runtime environment.cache_env_sources does not match the NM5 cache contract")
        missing_cache_keys = set(expected_cache_sources.values()) - required_env_keys
        if missing_cache_keys:
            raise ValueError(
                "runtime backend_env.nm5.required_keys lacks cache key(s): "
                + ", ".join(sorted(missing_cache_keys))
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
