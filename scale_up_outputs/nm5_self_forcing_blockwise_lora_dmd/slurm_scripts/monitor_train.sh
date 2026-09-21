#!/bin/bash -l
# Durable detached monitor for the block-wise LoRA DMD training job.
#
# Runs on the NM5 login node inside a detached tmux session (the job itself
# runs under Slurm and does not depend on this). Appends scheduler state, GPU
# sampling, log tails and offline-W&B growth to a durable file so a later
# session can reconstruct what happened without re-querying the scheduler.
#
# Usage: monitor_train.sh <run_id> <job_id> [poll_seconds]
set -uo pipefail

RUN_ID="${1:?run_id required}"
JOB_ID="${2:?job_id required}"
POLL="${3:-300}"

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
EXP="$WS/scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd"
RUN_ROOT="$EXP/$RUN_ID"
LOG="$EXP/logs/monitor_${RUN_ID}_${JOB_ID}.log"
mkdir -p "$EXP/logs"

echo "MONITOR_START run_id=$RUN_ID job_id=$JOB_ID poll=${POLL}s $(date -u +%FT%TZ)" >> "$LOG"

while true; do
  {
    echo "=== $(date -u +%FT%TZ) ==="
    sacct -j "$JOB_ID" -o JobID,JobName%50,State,Elapsed,Timelimit,NNodes,Partition -X 2>/dev/null || true
    squeue -j "$JOB_ID" -o "%.12i %.8T %.10M %.20R" 2>/dev/null || true
    echo "--- offline wandb dirs: $(find "$RUN_ROOT/wandb" -maxdepth 3 -name 'offline-run-*' -type d 2>/dev/null | wc -l) ---"
    echo "--- checkpoints: $(find "$RUN_ROOT/checkpoints" -maxdepth 2 -name 'model.pt' 2>/dev/null | wc -l) ---"
    echo "--- du run root: $(du -sh "$RUN_ROOT" 2>/dev/null | cut -f1) ---"
    tail -n 5 "$EXP/logs/${RUN_ID}_smoke_${JOB_ID}.out" 2>/dev/null || true
    tail -n 5 "$EXP/logs/${RUN_ID}_fullrun_${JOB_ID}.out" 2>/dev/null || true
    echo "--- err tail ---"
    tail -n 5 "$EXP/logs/${RUN_ID}_smoke_${JOB_ID}.err" 2>/dev/null || true
    tail -n 5 "$EXP/logs/${RUN_ID}_fullrun_${JOB_ID}.err" 2>/dev/null || true
  } >> "$LOG" 2>&1

  state="$(sacct -j "$JOB_ID" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')"
  case "$state" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      echo "MONITOR_END state=$state $(date -u +%FT%TZ)" >> "$LOG"
      exit 0 ;;
  esac
  sleep "$POLL"
done
