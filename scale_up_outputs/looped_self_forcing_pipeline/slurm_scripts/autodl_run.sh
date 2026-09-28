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
has_tracking_override=0
for override in "${overrides[@]}"; do
  if [[ "${override%%=*}" == tracking ]]; then
    has_tracking_override=1
    break
  fi
done
if (( ! has_tracking_override )); then
  if [[ -n "${WANDB_API_KEY:-}" && -n "${WANDB_ENTITY:-}" ]]; then
    overrides+=(tracking=online)
  else
    overrides+=(tracking=offline)
  fi
fi

: "${AUTODL_DEEPRESEARCH_ROOT:?set the explicit AutoDL DeepResearch root}"
: "${SUE_PYTHON:?set SUE_PYTHON to the prepared AutoDL interpreter}"
: "${SUE_ASSET_ROOT:?load the private model-asset root configuration first}"
[[ -d "$AUTODL_DEEPRESEARCH_ROOT" ]] || { printf 'AutoDL DeepResearch root is unavailable\n' >&2; exit 2; }
[[ -x "$SUE_PYTHON" ]] || { printf 'SUE_PYTHON must name an executable interpreter\n' >&2; exit 2; }
command -v tmux >/dev/null 2>&1 || { printf 'tmux is required for detached AutoDL runs\n' >&2; exit 2; }
[[ "$SUE_ASSET_ROOT" == /* && -d "$SUE_ASSET_ROOT" ]] || {
  printf 'SUE_ASSET_ROOT must name an existing absolute directory\n' >&2
  exit 2
}
AUTODL_DEEPRESEARCH_ROOT="$(realpath -e -- "$AUTODL_DEEPRESEARCH_ROOT")"
SUE_PYTHON="$(realpath -e -- "$SUE_PYTHON")"

workspace="$AUTODL_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
pipeline="$workspace/scripts/looped_self_forcing_pipeline/pipeline.py"
[[ -d "$workspace" && -f "$pipeline" ]] || {
  printf 'looped-flow-matching workspace or pipeline is missing under the configured AutoDL root\n' >&2
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
export SUE_EXP_DIR SUE_ASSET_ROOT
for name in WANDB_API_KEY WANDB_ENTITY; do
  if [[ -v $name ]]; then
    export "$name"
  fi
done
logs_root="$(resolve_logs_root "$workspace/scripts" "$SUE_EXP_DIR")"

run_python() {
  "$SUE_PYTHON" "$@"
}

run_tmux_clean() (
  local name
  while IFS= read -r name; do
    case "$name" in
      BASHOPTS|SHELLOPTS) continue ;;
    esac
    export -n "$name"
  done < <(compgen -e)

  export PATH SUE_ASSET_ROOT SUE_EXP_DIR SUE_EXECUTION_ID
  for name in HOME LD_LIBRARY_PATH CUDA_HOME ROCM_PATH CUDA_VISIBLE_DEVICES WANDB_API_KEY WANDB_ENTITY; do
    if [[ -v $name ]]; then
      export "$name"
    fi
  done
  tmux "$@"
)

if [[ "$stage" == train ]]; then
  expected_gpus=4
else
  expected_gpus=1
fi
run_python - "$expected_gpus" <<'PY'
import sys

import torch

expected_gpus = int(sys.argv[1])
available_gpus = torch.cuda.device_count()
if not torch.cuda.is_available() or available_gpus < expected_gpus:
    raise SystemExit(
        f"AutoDL {expected_gpus}-GPU preflight failed; visible CUDA GPUs: {available_gpus}"
    )
PY

cd "$workspace"
run_python "$pipeline" "check-$stage" --exp-dir "$SUE_EXP_DIR" \
  backend=autodl "run_id=$run_id" "${overrides[@]}"

log_dir="$logs_root"
mkdir -p "$log_dir"
session_name="loop-${run_id}-${stage}"
socket_name="lsf-${run_id}-${stage}"
SUE_EXECUTION_ID="$session_name"
export SUE_EXECUTION_ID
if run_tmux_clean -L "$socket_name" has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session for this run and stage is already active\n' >&2
  exit 2
fi
log_file="$(mktemp "$log_dir/${run_id}_${stage}_XXXXXX.log")"

printf -v worker_command '%q ' "$SUE_PYTHON" "$pipeline" "${stage}-worker" \
  --exp-dir "$SUE_EXP_DIR" backend=autodl "run_id=$run_id" "${overrides[@]}"
printf -v quoted_log_file '%q' "$log_file"
command_string="exec ${worker_command}>>$quoted_log_file 2>&1"
run_tmux_clean -L "$socket_name" \
  new-session -d -s "$session_name" -c "$workspace" "$command_string"
printf 'started %s; attach with: tmux -L %s attach-session -t %s\n' \
  "$session_name" "$socket_name" "$session_name"
