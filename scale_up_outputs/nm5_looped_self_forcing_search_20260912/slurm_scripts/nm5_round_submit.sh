#!/bin/bash -l
# NM5 round submitter. Resolves the private backend config *on the execution
# plane* (never on the control plane, never materialised into a file), exports
# the SUE_* launcher contract, and delegates to submit_search.sh.
#
# This script contains no private values: it reads them from
# $SUE_DEEPRESEARCH_ROOT/deepresearch-sandbox/config_nm5.txt, which is the same
# ignored source the SUE skills use. Keeping the read on the sandbox means
# credentials never leave NM5.
#
# Usage: nm5_round_submit.sh <run_id> <dryrun|fullrun> <variant> [variant...]
# Run it from the NM5 HPC login node (the `nm5` ssh alias), not from
# `nm5-transfer`, which is the data-mover cluster and exposes no GPUs.
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 <run_id> <dryrun|fullrun> <variant> [variant...]" >&2
  exit 2
fi
RUN_ID="$1"
MODE="$2"
shift 2
[[ "$MODE" == dryrun || "$MODE" == fullrun ]] || { echo "MODE ERROR: expected dryrun or fullrun, got $MODE" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[a-z0-9_]+$ ]] || { echo "RUN_ID ERROR: $RUN_ID" >&2; exit 2; }

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
CONFIG="$SUE_DEEPRESEARCH_ROOT/deepresearch-sandbox/config_nm5.txt"
[[ -r "$CONFIG" ]] || { echo "CONFIG ERROR: missing $CONFIG on the execution plane" >&2; exit 3; }

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
EXP="$WS/scale_up_outputs/nm5_looped_self_forcing_search_20260912"
: "${NM5_ACCOUNT:?NM5_ACCOUNT missing from $CONFIG}"

export SUE_DEEPRESEARCH_ROOT
export SUE_EXP="$EXP"
export SUE_RUN_ID="$RUN_ID"
export SUE_MODE="$MODE"
export SUE_ACCOUNT="$NM5_ACCOUNT"
export SUE_PARTITION="${NM5_PARTITION:-acc}"
export SUE_QOS="${NM5_QOS:-acc_ehpc}"
export SUE_MAX_PARALLEL="${SUE_MAX_PARALLEL:-1}"
export SUE_DRYRUN_PROMPT_COUNT="${SUE_DRYRUN_PROMPT_COUNT:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$EXP/logs"
LOG="$EXP/logs/submit_${RUN_ID}.log"

# submit_search.sh needs a login shell for `module`; invoking it with an
# explicit `bash -l` is equivalent to its shebang and does not require the
# file to be executable.
cd "$WS"
echo "ROUND_SUBMIT_START run_id=$RUN_ID mode=$MODE variants=$*"
bash -l "$EXP/slurm_scripts/submit_search.sh" "$@" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
echo "ROUND_SUBMIT_RC=$rc log=$LOG"
exit "$rc"
