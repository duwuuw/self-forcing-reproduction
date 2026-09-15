#!/usr/bin/env bash
set -euo pipefail

: "${SUE_BASE_ENV_SCRIPT:?SUE_BASE_ENV_SCRIPT is required}"
: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
: "${SUE_EXP:?SUE_EXP is required}"
: "${SUE_RUN_ID:?SUE_RUN_ID is required}"
source "$SUE_BASE_ENV_SCRIPT"

EXPECTED_WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
if [[ "${WS:-}" != "$EXPECTED_WS" ]]; then
  echo "PATH ERROR: base environment resolved an unexpected workspace root" >&2
  exit 2
fi
case "$SUE_EXP" in
  "$EXPECTED_WS/scale_up_outputs/nm5_looped_self_forcing_search_20260912") ;;
  *) echo "PATH ERROR: experiment root is outside the selected SUE bundle" >&2; exit 2 ;;
esac

BASE_OVERLAY="$OVERLAY"
BASE_TORCH_HOME="$TORCH_HOME"
BASE_VBENCH_CACHE_DIR="$VBENCH_CACHE_DIR"
BASE_HF_HOME="$HF_HOME"
BASE_LOCAL_BIN="$SUE_LOCAL_BIN"
export EXP="$SUE_EXP"
export SUE_EXP_DIR="$SUE_EXP"
export DEEPRESEARCH_SCRIPTS="$SF"
export PROJECT_SCRIPTS="$EXPECTED_WS/scripts"
export OVERLAY="$BASE_OVERLAY"
export TORCH_HOME="$BASE_TORCH_HOME"
export VBENCH_CACHE_DIR="$BASE_VBENCH_CACHE_DIR"
export HF_HOME="$BASE_HF_HOME"
export SUE_LOCAL_BIN="$BASE_LOCAL_BIN"
export PATH="$V/bin:$PATH"
export PYTHONPATH="$SF:$SF/VBench:$OVERLAY:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONHOME=''
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SUE_NUM_OUTPUT_FRAMES=123
export SCALEUP_WORKFLOW_TYPE=generation
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-looped-self-forcing-vbench-long}"
export WANDB_GROUP="$SUE_RUN_ID"
export WANDB_EXPERIMENT_NAME="$WANDB_PROJECT"

RUN_ROOT="$SUE_EXP/$SUE_RUN_ID"
case "$RUN_ROOT" in
  "$SUE_EXP"/*) ;;
  *) echo "PATH ERROR: run root escaped SUE_EXP_DIR" >&2; exit 2 ;;
esac
export RUN_ROOT
export TASK_CLI="$SUE_DEEPRESEARCH_ROOT/scripts/environment/sue_tasks_cli.py"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$EXP/env/wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$EXP/env/wandb_data}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$EXP/env/wandb_config}"
# $RUN_ROOT/run_state holds the per-stage completion markers
# (generation_complete.json / eval_complete.json) written by the packed
# launchers; $RUN_ROOT/state holds the task ledger from sue_tasks_cli. Both must
# exist before the stage tails write into them.
mkdir -p "$RUN_ROOT/timings_parts" "$RUN_ROOT/worker_logs" "$RUN_ROOT/wandb" "$RUN_ROOT/run_state" "$RUN_ROOT/variants" "$SUE_EXP/logs" "$SUE_EXP/run_state" "$SUE_EXP/artifacts" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_CONFIG_DIR"

task_update() {
  local task_id="$1"; shift
  "$V/bin/python" "$TASK_CLI" update --run-dir "$RUN_ROOT" --task-id "$task_id" "$@" >/dev/null
}

record_timing() {
  local state="$1"
  local end_epoch
  end_epoch="$(date +%s)"
  "$V/bin/python" - "$RUN_ROOT/timings_parts/job_${SLURM_JOB_ID}_${SUE_STAGE}_${SUE_VARIANT}.jsonl" "$state" "$START_EPOCH" "$end_epoch" "$SUE_STAGE" "$SUE_VARIANT" "$SUE_RUN_ID" "$SLURM_JOB_ID" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, state, start, end, stage, variant, run_id, job_id = sys.argv[1:]
payload = {
    "job_id": job_id,
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    "job_name": os.environ.get("SLURM_JOB_NAME", ""),
    "stage": stage,
    "variant": variant,
    "run_id": run_id,
    "start_time_epoch": int(start),
    "end_time_epoch": int(end),
    "start_time": datetime.fromtimestamp(int(start), timezone.utc).isoformat(),
    "end_time": datetime.fromtimestamp(int(end), timezone.utc).isoformat(),
    "state": state,
    "gpus_allocated": os.environ.get("SLURM_GPUS", os.environ.get("SLURM_GPUS_PER_NODE", "4")),
    "nodes_allocated": os.environ.get("SLURM_NNODES", "1"),
    "elapsed_seconds": int(end) - int(start),
    "source": "launcher",
}
Path(path).parent.mkdir(parents=True, exist_ok=True)
with Path(path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + chr(10))
PY
}

gpu_sample() {
  local label="$1"
  {
    printf '%s,' "$(date -Is)"
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null || true
  } >> "$RUN_ROOT/worker_logs/gpu_util_${SLURM_JOB_ID}_${SUE_STAGE}_${SUE_VARIANT}_${label}.csv"
}

verify_allocation() {
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "GPU PREFLIGHT ERROR: nvidia-smi is unavailable inside the allocation" >&2
    return 2
  }
  local visible_count
  visible_count="$(nvidia-smi -L 2>/dev/null | awk 'NF {count += 1} END {print count + 0}')"
  [[ "$visible_count" -eq 4 ]] || {
    echo "GPU PREFLIGHT ERROR: expected four visible GPUs, found $visible_count" >&2
    return 2
  }
  printf '%s\n' "ALLOCATION_GPU_CHECK=ok slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset} visible_gpu_count=$visible_count"
}

verify_worker_gpu() {
  local gpu="$1"
  [[ "$gpu" =~ ^[0-3]$ ]] || {
    echo "GPU WORKER ERROR: invalid packed GPU index $gpu" >&2
    return 2
  }
  # `nvidia-smi -L` enumerates the physical devices and ignores
  # CUDA_VISIBLE_DEVICES, so it always reports every GPU on the node and cannot
  # verify a packed binding. Only the CUDA runtime honours the variable, so the
  # assertion has to go through torch.
  CUDA_VISIBLE_DEVICES="$gpu" "$V/bin/python" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(f"GPU WORKER ERROR: torch sees {count} GPUs after binding")
print("WORKER_GPU_CHECK=ok torch_visible_gpu_count=1")
PY
}

task_heartbeat() {
  local task_id="$1"
  local expected="$2"
  while true; do
    sleep 60
    task_update "$task_id" --state running --expected "$expected" --slurm-job-id "$SLURM_JOB_ID" || true
  done
}
