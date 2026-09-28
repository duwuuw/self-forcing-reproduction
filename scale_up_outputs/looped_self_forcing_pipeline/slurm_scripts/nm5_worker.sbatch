#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ -z "${SUE_SCRIPTS_DIR:-}" ]]; then
  printf 'SUE_SCRIPTS_DIR must be exported by the NM5 submitter\n' >&2
  exit 2
fi
if ! durable_scripts_dir="$(realpath -e -- "$SUE_SCRIPTS_DIR" 2>/dev/null)" || \
  [[ ! -r "$durable_scripts_dir/launcher_common.sh" ]]; then
  printf 'SUE_SCRIPTS_DIR must point to a readable durable launcher directory\n' >&2
  exit 2
fi
SUE_SCRIPTS_DIR="$durable_scripts_dir"
source "$SUE_SCRIPTS_DIR/launcher_common.sh"

[[ $# -ge 2 ]] || { printf 'NM5 worker requires stage and run_id arguments\n' >&2; exit 2; }
stage=$1
run_id=$2
[[ "$stage" == train || "$stage" == infer ]] || { printf 'invalid stage\n' >&2; exit 2; }
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || { printf 'invalid run_id\n' >&2; exit 2; }
shift 2
validate_hydra_overrides "$stage" "$@"
overrides=("$@")

: "${NM5_DEEPRESEARCH_ROOT:?NM5_DEEPRESEARCH_ROOT was not exported to the job}"
: "${SUE_PYTHON:?SUE_PYTHON was not exported to the job}"
: "${SUE_EXP_DIR:?SUE_EXP_DIR was not exported to the job}"
: "${SUE_ASSET_ROOT:?SUE_ASSET_ROOT was not exported to the job}"
[[ "$SUE_ASSET_ROOT" == /* && -d "$SUE_ASSET_ROOT" ]] || {
  printf 'SUE_ASSET_ROOT must name an existing absolute directory\n' >&2
  exit 2
}
NM5_DEEPRESEARCH_ROOT="$(realpath -e -- "$NM5_DEEPRESEARCH_ROOT")"
SUE_PYTHON="$(realpath -e -- "$SUE_PYTHON")"
SUE_EXP_DIR="$(realpath -e -- "$SUE_EXP_DIR")"
workspace="$NM5_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
pipeline="$workspace/scripts/looped_self_forcing_pipeline/pipeline.py"
[[ -d "$workspace" && -f "$pipeline" ]] || { printf 'workspace or pipeline is unavailable\n' >&2; exit 2; }
[[ -x "$SUE_PYTHON" ]] || { printf 'SUE_PYTHON is not executable\n' >&2; exit 2; }

if [[ -n "${SUE_PYTHONPATH:-}" ]]; then
  SUE_PYTHONPATH="$(realpath -e -- "$SUE_PYTHONPATH" 2>/dev/null)" || {
    printf 'configured NM5 Python overlay is unavailable\n' >&2
    exit 2
  }
  [[ -d "$SUE_PYTHONPATH" && "$SUE_PYTHONPATH" == "$SUE_EXP_DIR"/* ]] || {
    printf 'configured NM5 Python overlay must stay inside SUE_EXP_DIR\n' >&2
    exit 2
  }
  PYTHONPATH="$SUE_PYTHONPATH"
else
  PYTHONPATH=""
fi
SUE_MEM_PROBE="${SUE_MEM_PROBE:-0}"
SUE_MEM_PROBE_STEP="${SUE_MEM_PROBE_STEP:-3}"
[[ "$SUE_MEM_PROBE" == 0 || "$SUE_MEM_PROBE" == 1 ]] || {
  printf 'SUE_MEM_PROBE must be 0 or 1\n' >&2
  exit 2
}
[[ "$SUE_MEM_PROBE_STEP" =~ ^[1-9][0-9]*$ ]] || {
  printf 'SUE_MEM_PROBE_STEP must be a positive integer\n' >&2
  exit 2
}
export SUE_PYTHONPATH PYTHONPATH SUE_MEM_PROBE SUE_MEM_PROBE_STEP

cd "$workspace"
# SUE_GPU_MODEL_PREFLIGHT: validate allocated device names before either
# stage preflight or upstream model construction.
"$SUE_PYTHON" - "$SUE_EXP_DIR" "$workspace" "$stage" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
from looped_self_forcing_pipeline.config import load_runtime_config
from looped_self_forcing_pipeline.worker import require_gpu_model_names
import torch

runtime = load_runtime_config(sys.argv[1])
resources = runtime["sandbox_resources"]["nm5"]
stage = sys.argv[3]
expected_count = int(resources["train_gpus"] if stage == "train" else resources["infer_gpus"])
expected_model = str(resources["gpu_type"])
visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
names = [torch.cuda.get_device_name(index) for index in range(visible)]
require_gpu_model_names(names, expected_model=expected_model, expected_count=expected_count)
print(f"SUE_GPU_MODEL_PREFLIGHT passed expected={expected_model} count={expected_count}", flush=True)
PY
cd "$workspace"
"$SUE_PYTHON" "$pipeline" "check-$stage" --exp-dir "$SUE_EXP_DIR" \
  backend=nm5 "run_id=$run_id" "$@"
exec "$SUE_PYTHON" "$pipeline" "${stage}-worker" --exp-dir "$SUE_EXP_DIR" \
  backend=nm5 "run_id=$run_id" "$@"
