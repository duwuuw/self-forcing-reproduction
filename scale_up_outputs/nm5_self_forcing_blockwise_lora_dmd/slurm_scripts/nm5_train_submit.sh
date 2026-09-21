#!/bin/bash -l
# NM5 execution-plane submitter for block-wise LoRA DMD training.
#
# Resolves the private backend config *on the execution plane* (never on the
# control plane, never materialised into a file), exports the SUE_* launcher
# contract, and delegates to submit_train.sh.
#
# Usage: nm5_train_submit.sh <run_id> <smoke|fullrun>
#   Run from the NM5 HPC login node (the `nm5` ssh alias), NOT from
#   `nm5-transfer`, which is the data-mover cluster and exposes no GPUs.
#
# Optional overrides:
#   SUE_QOS=acc_debug|acc_ehpc       (default: acc_debug for smoke, acc_ehpc for fullrun)
#   SUE_TRAIN_SECONDS=<int>          wall-clock bound handed to `timeout`
#   SUE_TIME_LIMIT=HH:MM:SS          Slurm --time
#   SUE_TRAIN_LOG_ITERS=<int>        checkpoint/W&B cadence (dryrun scale knob)
#   SUE_PLAN_ONLY=1                  print the sbatch command without submitting
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <run_id> <smoke|fullrun>" >&2
  exit 2
fi
RUN_ID="$1"
MODE="$2"
[[ "$MODE" == smoke || "$MODE" == fullrun ]] || {
  echo "MODE ERROR: expected smoke or fullrun, got $MODE" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[a-z0-9_]+$ ]] || { echo "RUN_ID ERROR: $RUN_ID" >&2; exit 2; }

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
CONFIG="$SUE_DEEPRESEARCH_ROOT/deepresearch-sandbox/config_nm5.txt"
[[ -r "$CONFIG" ]] || { echo "CONFIG ERROR: missing $CONFIG on the execution plane" >&2; exit 3; }

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
EXP="$WS/scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd"
: "${NM5_ACCOUNT:?NM5_ACCOUNT missing from $CONFIG}"
export SUE_DATASET_PATH="$EXP/datasets/vidprom_filtered_extended.txt"

# NM5_ACCOUNT is the compute account; storage roots may live under a different
# storage account, so never infer the account from a path. Always pass it
# explicitly rather than letting Slurm fall back to the user's default.
export SUE_DEEPRESEARCH_ROOT
export SUE_EXP="$EXP"
export SUE_RUN_ID="$RUN_ID"
export SUE_ACCOUNT="$NM5_ACCOUNT"
export SUE_PARTITION="${NM5_PARTITION:-acc}"
if [[ "$MODE" == smoke ]]; then
  # acc_debug: priority weight 100x, but MaxJobsPU=1 and wall <= 02:00:00.
  # Only ever used for the single short smoke.
  export SUE_QOS="${SUE_QOS:-acc_debug}"
else
  export SUE_QOS="${SUE_QOS:-acc_ehpc}"
fi

mkdir -p "$EXP/logs"
LOG="$EXP/logs/submit_${RUN_ID}_${MODE}.log"

cd "$WS"
echo "TRAIN_SUBMIT_START run_id=$RUN_ID mode=$MODE qos=$SUE_QOS"
# The outer launcher is a login shell so NM5's module setup is available.  Do
# not make the child launcher a login shell: the NM5 profile may cd to $HOME,
# which would trip submit_train.sh's workspace-root guard before it can build
# the Slurm plan.
bash "$EXP/slurm_scripts/submit_train.sh" "$MODE" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
echo "TRAIN_SUBMIT_RC=$rc log=$LOG"
exit "$rc"
