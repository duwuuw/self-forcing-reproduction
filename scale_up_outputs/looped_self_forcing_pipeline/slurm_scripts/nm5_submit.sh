#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/launcher_common.sh"

usage() {
  printf 'usage: %s train|infer <run_id> [--preflight-only] [--dependency-afterany <job_id>] [key=value ...]\n' "${0##*/}" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
stage=$1
run_id=$2
[[ "$stage" == train || "$stage" == infer ]] || usage
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || usage
shift 2
preflight_only=0
dependency_job=""
overrides=()
while (($#)); do
  case "$1" in
    --preflight-only)
      (( preflight_only == 0 )) || usage
      preflight_only=1
      shift
      ;;
    --dependency-afterany)
      [[ -z "$dependency_job" && $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || usage
      dependency_job=$2
      shift 2
      ;;
    --*)
      usage
      ;;
    *)
      overrides+=("$1")
      shift
      ;;
  esac
done
validate_hydra_overrides "$stage" "${overrides[@]}"

: "${NM5_DEEPRESEARCH_ROOT:?load the private NM5 configuration first}"
: "${NM5_ACCOUNT:?load the private NM5 configuration first}"
: "${NM5_WORKSPACE_ROOT:?load the private NM5 configuration first}"
[[ -d "$NM5_DEEPRESEARCH_ROOT" ]] || { printf 'NM5 DeepResearch root is unavailable\n' >&2; exit 2; }
[[ -d "$NM5_WORKSPACE_ROOT" ]] || { printf 'NM5 workspace root is unavailable\n' >&2; exit 2; }
NM5_WORKSPACE_ROOT="$(realpath -e -- "$NM5_WORKSPACE_ROOT")"
NM5_DEEPRESEARCH_ROOT="$(realpath -e -- "$NM5_DEEPRESEARCH_ROOT")"

workspace="$NM5_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
pipeline="$workspace/scripts/looped_self_forcing_pipeline/pipeline.py"
[[ -d "$workspace" && -f "$pipeline" ]] || {
  printf 'looped-flow-matching workspace or pipeline is missing under the configured NM5 root\n' >&2
  exit 2
}
configure_code_provenance "$workspace"

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

configure_nm5_runtime_environment "$NM5_WORKSPACE_ROOT" "$SUE_EXP_DIR"
SUE_SCRIPTS_DIR="$(realpath -e -- "$script_dir")"
export NM5_DEEPRESEARCH_ROOT SUE_PYTHON SUE_EXP_DIR SUE_SCRIPTS_DIR SUE_PYTHONPATH PYTHONPATH
export SUE_ASSET_ROOT
load_nm5_cache_environment "$SUE_EXP_DIR" "$workspace"
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
export SUE_MEM_PROBE SUE_MEM_PROBE_STEP
for name in LD_LIBRARY_PATH CUDA_HOME ROCM_PATH WANDB_API_KEY WANDB_ENTITY SUE_PYTHONPATH PYTHONPATH; do
  if [[ -v $name ]]; then
    export "$name"
  fi
done
logs_root="$(resolve_logs_root "$workspace/scripts" "$SUE_EXP_DIR")"
mkdir -p "$logs_root"

runtime_resource_values="$("$SUE_PYTHON" - "$SUE_EXP_DIR" "$workspace" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
from looped_self_forcing_pipeline.config import load_runtime_config

resources = load_runtime_config(sys.argv[1])["sandbox_resources"]["nm5"]
fields = (
    "partition",
    "qos",
    "gpus_per_node",
    "max_nodes_per_job",
    "cpus_per_gpu",
    "gpu_type",
    "timeout_kill_after_seconds",
    "finalization_grace_seconds",
)
values = [resources.get(field) for field in fields]
if any(value is None for value in values):
    raise SystemExit("runtime sandbox_resources.nm5 is incomplete")
print("\t".join(str(value) for value in values))
PY
)"
IFS=$'\t' read -r runtime_partition runtime_qos runtime_gpus_per_node runtime_max_nodes runtime_cpus_per_gpu runtime_gpu_type runtime_kill_after runtime_finalization_grace <<<"$runtime_resource_values"
[[ "$runtime_partition" =~ ^[A-Za-z0-9_-]+$ && "$runtime_qos" =~ ^[A-Za-z0-9_-]+$ ]] || {
  printf 'runtime NM5 partition/QoS must be simple scheduler labels\n' >&2
  exit 2
}
[[ "$runtime_gpus_per_node" =~ ^[1-9][0-9]*$ && "$runtime_max_nodes" == 1 && "$runtime_cpus_per_gpu" =~ ^[1-9][0-9]*$ ]] || {
  printf 'runtime NM5 GPU/node/CPU resource settings are invalid for the single-node pipeline\n' >&2
  exit 2
}
[[ "$runtime_gpu_type" == H100 && "$runtime_kill_after" =~ ^[1-9][0-9]*$ && "$runtime_finalization_grace" =~ ^[1-9][0-9]*$ ]] || {
  printf 'runtime NM5 GPU model and timeout finalization settings are invalid\n' >&2
  exit 2
}

timeout_seconds="$("$SUE_PYTHON" - "$SUE_EXP_DIR" "$workspace" "$stage" "$run_id" "${overrides[@]}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
from looped_self_forcing_pipeline.config import compose_config

exp_dir, stage, run_id = sys.argv[1], sys.argv[3], sys.argv[4]
config = compose_config(
    [f"backend=nm5", f"stage={stage}", f"run_id={run_id}", *sys.argv[5:]],
    exp_dir=exp_dir,
)
section = config.train if stage == "train" else config.infer
seconds = section.get("timeout_seconds", None)
if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
    raise SystemExit(f"{stage}.timeout_seconds must be a positive integer")
print(seconds)
PY
)" || {
  printf 'selected Hydra timeout_seconds could not be resolved\n' >&2
  exit 2
}
slurm_time_seconds=$((timeout_seconds + runtime_kill_after + runtime_finalization_grace))
require_nm5_partition_time "$runtime_partition" "$slurm_time_seconds"

cd "$workspace"
"$SUE_PYTHON" "$pipeline" "check-$stage" --exp-dir "$SUE_EXP_DIR" \
  backend=nm5 "run_id=$run_id" "${overrides[@]}"
if (( preflight_only )); then
  printf 'preflight passed for %s/%s\n' "$stage" "$run_id"
  exit 0
fi

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
(( gpus <= runtime_gpus_per_node )) || {
  printf 'requested pipeline GPUs exceed runtime sandbox_resources.nm5.gpus_per_node\n' >&2
  exit 2
}

dependency_args=()
if [[ -n "$dependency_job" ]]; then
  dependency_args+=("--dependency=afterany:$dependency_job")
fi
unset SBATCH_MAIL_USER SBATCH_MAIL_TYPE
job_id="$(sbatch --parsable \
  --chdir="$workspace" \
  --account="$NM5_ACCOUNT" \
  --partition="$runtime_partition" \
  --qos="$runtime_qos" \
  --gres="gpu:$gpus" \
  --nodes="$runtime_max_nodes" \
  --cpus-per-task="$((runtime_cpus_per_gpu * gpus))" \
  --time="$SUE_SLURM_REQUEST_TIME" \
  --job-name="$job_name" \
  --mail-type=NONE \
  "${dependency_args[@]}" \
  --output="$logs_root/${run_id}_${stage}_%j.out" \
  --error="$logs_root/${run_id}_${stage}_%j.err" \
  --export=NM5_DEEPRESEARCH_ROOT,SUE_PYTHON,SUE_EXP_DIR,SUE_SCRIPTS_DIR,SUE_ASSET_ROOT,SUE_GIT_COMMIT,SUE_GIT_DIRTY,SUE_PYTHONPATH,PYTHONPATH,SUE_MEM_PROBE,SUE_MEM_PROBE_STEP,PATH,LD_LIBRARY_PATH,CUDA_HOME,ROCM_PATH,WANDB_API_KEY,WANDB_ENTITY,$SUE_NM5_CACHE_EXPORTS \
  "$script_dir/nm5_worker.sbatch" "$stage" "$run_id" "${overrides[@]}")"

if ! "$SUE_PYTHON" "$pipeline" record-submit --exp-dir "$SUE_EXP_DIR" \
  --job-id "$job_id" "stage=$stage" backend=nm5 "run_id=$run_id" "${overrides[@]}"; then
  printf 'submitted job id: %s (ledger update failed)\n' "$job_id" >&2
  exit 1
fi
printf '%s\n' "$job_id"
