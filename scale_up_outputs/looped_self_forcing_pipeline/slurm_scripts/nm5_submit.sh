#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/launcher_common.sh"

usage() {
  printf 'usage: %s train|infer <run_id> [key=value ...]\n' "${0##*/}" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
stage=$1
run_id=$2
[[ "$stage" == train || "$stage" == infer ]] || usage
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || usage
shift 2
validate_hydra_overrides "$stage" "$@"
overrides=("$@")

: "${NM5_DEEPRESEARCH_ROOT:?load the private NM5 configuration first}"
: "${NM5_ACCOUNT:?load the private NM5 configuration first}"
: "${NM5_PARTITION:?load the private NM5 configuration first}"
: "${NM5_QOS:?load the private NM5 configuration first}"
: "${SUE_PYTHON:?set SUE_PYTHON to the prepared NM5 interpreter}"
: "${SUE_ASSET_ROOT:?load the private model-asset root configuration first}"
[[ -d "$NM5_DEEPRESEARCH_ROOT" ]] || { printf 'NM5 DeepResearch root is unavailable\n' >&2; exit 2; }
[[ -x "$SUE_PYTHON" ]] || { printf 'SUE_PYTHON must name an executable interpreter\n' >&2; exit 2; }
[[ "$SUE_ASSET_ROOT" == /* && -d "$SUE_ASSET_ROOT" ]] || {
  printf 'SUE_ASSET_ROOT must name an existing absolute directory\n' >&2
  exit 2
}
NM5_DEEPRESEARCH_ROOT="$(realpath -e -- "$NM5_DEEPRESEARCH_ROOT")"
SUE_PYTHON="$(realpath -e -- "$SUE_PYTHON")"

workspace="$NM5_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
pipeline="$workspace/scripts/looped_self_forcing_pipeline/pipeline.py"
[[ -d "$workspace" && -f "$pipeline" ]] || {
  printf 'looped-flow-matching workspace or pipeline is missing under the configured NM5 root\n' >&2
  exit 2
}

if [[ -z "${SUE_EXP_DIR:-}" ]]; then
  SUE_EXP_DIR="$workspace/scale_up_outputs/looped_self_forcing_pipeline"
elif [[ "$SUE_EXP_DIR" != /* ]]; then
  SUE_EXP_DIR="$workspace/$SUE_EXP_DIR"
fi
export SUE_EXP_DIR
[[ -f "$SUE_EXP_DIR/config/config.yaml" && -f "$SUE_EXP_DIR/config/runtime.yaml" ]] || {
  printf 'selected SUE_EXP_DIR lacks the pipeline config contract\n' >&2
  exit 2
}
SUE_EXP_DIR="$(realpath -e -- "$SUE_EXP_DIR")"
export NM5_DEEPRESEARCH_ROOT SUE_PYTHON SUE_EXP_DIR
export SUE_ASSET_ROOT
for name in LD_LIBRARY_PATH CUDA_HOME ROCM_PATH WANDB_API_KEY WANDB_ENTITY; do
  if [[ -v $name ]]; then
    export "$name"
  fi
done
logs_root="$(resolve_logs_root "$workspace/scripts" "$SUE_EXP_DIR")"
mkdir -p "$logs_root"

cd "$workspace"
"$SUE_PYTHON" "$pipeline" "check-$stage" --exp-dir "$SUE_EXP_DIR" \
  backend=nm5 "run_id=$run_id" "${overrides[@]}"

job_prefix="$("$SUE_PYTHON" - "$workspace/user.yaml" <<'PY'
import re
import sys
from pathlib import Path

prefix = "silly-"
try:
    import yaml

    config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    username = config.get("username") if isinstance(config, dict) else None
    if isinstance(username, str) and re.fullmatch(r"[a-z0-9]{2,12}", username):
        prefix = f"{username}-"
except Exception:
    pass
print(prefix)
PY
)"
experiment_name="$("$SUE_PYTHON" "$pipeline" resolve-experiment-name \
  --exp-dir "$SUE_EXP_DIR" "stage=$stage" backend=nm5 "run_id=$run_id" "${overrides[@]}")"
[[ -n "$experiment_name" && "$experiment_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  printf 'could not resolve a valid experiment name from the selected Hydra method\n' >&2
  exit 2
}
job_name="${job_prefix}${experiment_name}"

if [[ "$stage" == train ]]; then
  gpus=4
else
  gpus=1
fi

job_id="$(sbatch --parsable \
  --chdir="$workspace" \
  --account="$NM5_ACCOUNT" \
  --partition="$NM5_PARTITION" \
  --qos="$NM5_QOS" \
  --gres="gpu:$gpus" \
  --cpus-per-task="$((20 * gpus))" \
  --job-name="$job_name" \
  --output="$logs_root/${run_id}_${stage}_%j.out" \
  --error="$logs_root/${run_id}_${stage}_%j.err" \
  --export=NM5_DEEPRESEARCH_ROOT,SUE_PYTHON,SUE_EXP_DIR,SUE_ASSET_ROOT,PATH,LD_LIBRARY_PATH,CUDA_HOME,ROCM_PATH,WANDB_API_KEY,WANDB_ENTITY \
  "$script_dir/nm5_worker.sbatch" "$stage" "$run_id" "${overrides[@]}")"

if ! "$SUE_PYTHON" "$pipeline" record-submit --exp-dir "$SUE_EXP_DIR" \
  --job-id "$job_id" "stage=$stage" backend=nm5 "run_id=$run_id" "${overrides[@]}"; then
  printf 'submitted job id: %s (ledger update failed)\n' "$job_id" >&2
  exit 1
fi
printf '%s\n' "$job_id"
