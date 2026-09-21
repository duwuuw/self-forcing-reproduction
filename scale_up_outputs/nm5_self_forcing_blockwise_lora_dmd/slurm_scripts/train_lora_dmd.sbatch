#!/bin/bash -l
# NM5 Slurm launcher for block-wise selected-block LoRA DMD training on
# Self-Forcing-blockwise (branch block-wise-temporal-loop @ 168d4de).
#
# Submitted by submit_train.sh, which passes SUE_SCRIPTS_DIR explicitly because
# Slurm spools this file to a per-job spool directory and $BASH_SOURCE would
# resolve to the spool directory, not the durable bundle.
set -euo pipefail

SCRIPT_DIR="${SUE_SCRIPTS_DIR:?SUE_SCRIPTS_DIR must be passed via --export}"
[[ -r "$SCRIPT_DIR/common_train_env.sh" ]] || {
  echo "TRAIN LAUNCHER ERROR: cannot read $SCRIPT_DIR/common_train_env.sh" >&2; exit 2; }

# shellcheck disable=SC1090
source "$SCRIPT_DIR/common_train_env.sh"

echo "TRAIN_START run_id=${SUE_RUN_ID} job=${SLURM_JOB_ID:-none} host=$(hostname) $(date -u +%FT%TZ)"
echo "TRAIN_SF=$SF"
echo "TRAIN_CONFIG=$SUE_TRAIN_CONFIG"
echo "TRAIN_RUN_ROOT=$RUN_ROOT"
echo "TRAIN_SECONDS=$SUE_TRAIN_SECONDS"
[[ -n "${SLURM_JOB_ID:-}" ]] && scontrol show job "$SLURM_JOB_ID" 2>/dev/null | \
  grep -E "JobName|Partition|QOS|Account|NumCPUs|Gres|TimeLimit" || true

# ------------------------------------------------------------------ allocation
command -v nvidia-smi >/dev/null || { echo "TRAIN LAUNCHER ERROR: nvidia-smi unavailable" >&2; exit 2; }
visible_gpus="$(nvidia-smi -L 2>/dev/null | awk 'NF {count += 1} END {print count + 0}')"
[[ "$visible_gpus" -eq 4 ]] || {
  echo "TRAIN LAUNCHER ERROR: expected 4 visible GPUs, found $visible_gpus" >&2; exit 2; }
echo "ALLOCATION_GPU_CHECK=ok visible_gpu_count=$visible_gpus"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ------------------------------------------------------------------- preflight
sue_train_preflight

# The block-wise training entry point is the branch's own train.py, invoked
# through an out-of-tree wrapper that forces offline W&B (see
# scripts/train_blockwise_lora_dmd.py). No branch file is modified.
WRAPPER="$PROJECT_SCRIPTS/train_blockwise_lora_dmd.py"
[[ -f "$WRAPPER" ]] || { echo "TRAIN LAUNCHER ERROR: missing wrapper $WRAPPER" >&2; exit 2; }

echo "TRAIN_PREFLIGHT_MODE start $(date -u +%FT%TZ)"
( cd "$SF" && "$V/bin/python" "$WRAPPER" --config_path "$SUE_TRAIN_CONFIG" --preflight ) || {
  echo "TRAIN LAUNCHER ERROR: branch preflight rejected the configuration" >&2; exit 2; }
echo "TRAIN_PREFLIGHT_MODE done $(date -u +%FT%TZ)"

# --------------------------------------------------------- checkpoint retention
# Project policy: keep only the latest resumable checkpoint. The branch writes
# checkpoint_model_<step>/model.pt every log_iters steps; each payload carries
# the full critic plus its optimizer state (tens of GB), which would otherwise
# accumulate across a long run on a 97%-full scratch filesystem.
prune_checkpoints() {
  local keep="${SUE_KEEP_CHECKPOINTS:-1}"
  [[ "$keep" -ge 0 ]] || return 0
  local -a dirs
  mapfile -t dirs < <(find "$SUE_CKPT_DIR" -maxdepth 1 -type d -name 'checkpoint_model_*' 2>/dev/null | sort)
  local n="${#dirs[@]}"
  (( n > keep )) || return 0
  local idx
  for (( idx = 0; idx < n - keep; idx++ )); do
    # Never delete a directory written in the last 5 minutes: save() may still
    # be streaming into it.
    if [[ -z "$(find "${dirs[idx]}" -newermt '-5 minutes' -print -quit 2>/dev/null)" ]]; then
      echo "CHECKPOINT_PRUNE removing ${dirs[idx]}"
      rm -rf -- "${dirs[idx]}"
    fi
  done
}
PRUNER_PID=""
if [[ "${SUE_KEEP_CHECKPOINTS:-1}" -ge 0 ]]; then
  ( while true; do sleep 300; prune_checkpoints || true; done ) &
  PRUNER_PID="$!"
fi
cleanup() {
  [[ -n "$PRUNER_PID" ]] && kill "$PRUNER_PID" 2>/dev/null || true
  gpu_sample end || true
}
trap cleanup EXIT

gpu_sample start || true

# --------------------------------------------------------------------- train
# The training loop in trainer/distillation.py is `while True:` -- there is no
# step bound anywhere in the branch or the preset. Upstream bounded its
# reference run by wall clock ("600 iterations ... in under 2 hours using 64
# H100 GPUs"). We bound it the same way, so `timeout` exit 124 is an EXPECTED
# outcome, not a failure; the durable evidence is the checkpoint, the offline
# W&B run and the loss lines in this log.
cd "$SF"
set +e
timeout -s TERM --kill-after=300 "$SUE_TRAIN_SECONDS" \
  torchrun --nnodes=1 --nproc_per_node=4 \
           --master_port="$SUE_TRAIN_MASTER_PORT" --standalone \
           "$WRAPPER" \
           --config_path "$SUE_TRAIN_CONFIG" \
           --logdir "$SUE_CKPT_DIR" \
           --wandb-save-dir "$WANDB_DIR"
rc=$?
set -e

echo "TRAIN_TORCHRUN_RC=$rc $(date -u +%FT%TZ)"
case "$rc" in
  124) echo "TRAIN_NOTE: wall-clock bound reached (expected for a step-unbounded loop)" ;;
  137) echo "TRAIN_NOTE: SIGKILL after TERM grace period" ;;
  143) echo "TRAIN_NOTE: terminated by SIGTERM" ;;
  0)   echo "TRAIN_NOTE: torchrun exited cleanly" ;;
  *)   echo "TRAIN FAILURE: torchrun exited $rc" >&2 ;;
esac

# ------------------------------------------------------------------ evidence
echo "=== TRAIN_EVIDENCE $(date -u +%FT%TZ) ==="
echo "--- offline wandb runs ---"
find "$WANDB_DIR" -maxdepth 3 -name 'offline-run-*' -type d 2>/dev/null | sort | tail -5
echo "--- checkpoint payloads ---"
find "$SUE_CKPT_DIR" -maxdepth 2 -name 'model.pt' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -5
echo "--- gpu util samples ---"
tail -2 "$RUN_ROOT/worker_logs"/gpu_util_*.csv 2>/dev/null || true
echo "=== TRAIN_END rc=$rc $(date -u +%FT%TZ) ==="

# A non-zero timeout code is the documented success shape for a bounded smoke;
# any other non-zero code is a real failure.
if [[ "$rc" -ne 0 && "$rc" -ne 124 && "$rc" -ne 137 && "$rc" -ne 143 ]]; then
  exit "$rc"
fi

# The step-unbounded loop is expected to be cut; require that it produced real
# evidence before reporting success.
ckpts="$(find "$SUE_CKPT_DIR" -maxdepth 2 -name 'model.pt' 2>/dev/null | wc -l)"
runs="$(find "$WANDB_DIR" -maxdepth 3 -name 'offline-run-*' -type d 2>/dev/null | wc -l)"
if [[ "$runs" -lt 1 ]]; then
  echo "TRAIN EVIDENCE ERROR: no offline W&B run directory was produced" >&2
  exit 3
fi
if [[ "$ckpts" -lt 1 ]]; then
  echo "TRAIN EVIDENCE ERROR: no checkpoint payload was written; the run was cut before reaching log_iters steps." >&2
  echo "  Raise SUE_TRAIN_SECONDS, or lower log_iters for the dryrun scale." >&2
  exit 4
fi
echo "TRAIN_EVIDENCE_OK checkpoints=$ckpts offline_wandb_runs=$runs"
exit 0
