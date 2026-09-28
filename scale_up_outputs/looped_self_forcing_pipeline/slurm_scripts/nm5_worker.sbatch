#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/launcher_common.sh"

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

cd "$workspace"
"$SUE_PYTHON" "$pipeline" "check-$stage" --exp-dir "$SUE_EXP_DIR" \
  backend=nm5 "run_id=$run_id" "$@"
exec "$SUE_PYTHON" "$pipeline" "${stage}-worker" --exp-dir "$SUE_EXP_DIR" \
  backend=nm5 "run_id=$run_id" "$@"
