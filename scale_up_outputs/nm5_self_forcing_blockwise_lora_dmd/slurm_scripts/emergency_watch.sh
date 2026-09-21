#!/bin/bash -l
# Durable server-side watcher for a training job.
#
# Lives on the NM5 login node inside a detached tmux session and is therefore
# independent of any control-plane (Mac mini / MacBook) session: it keeps running
# whether or not the coding agent or its IDE is connected.
#
# Every POLL seconds it appends a status line to a durable log, and when the job
# leaves the queue it writes RUN_STATUS.md next to that log containing the
# outcome plus a ready-to-paste resume recipe. It deliberately does **not**
# resubmit anything by itself -- spending cluster allocation is a human decision.
#
# Usage: emergency_watch.sh <run_id> <job_id> [poll_seconds] [target_steps]
set -uo pipefail

RUN_ID="${1:?run_id required}"
JOB_ID="${2:?job_id required}"
POLL="${3:-600}"
TARGET_STEPS="${4:-0}"

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
EXP="$WS/scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd"
RUN_ROOT="$EXP/$RUN_ID"
STAGE="$EXP/logs"
LOG="$STAGE/watch_${RUN_ID}_${JOB_ID}.log"
STATUS="$STAGE/RUN_STATUS.md"
mkdir -p "$STAGE"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "WATCH_START run_id=$RUN_ID job_id=$JOB_ID poll=${POLL}s target_steps=$TARGET_STEPS"

last_step() {
  # Highest checkpoint payload written so far; the payload metadata is the
  # authoritative record of how far training actually got.
  local newest
  newest="$(find "$RUN_ROOT/checkpoints" -maxdepth 1 -type d -name 'checkpoint_model_*' 2>/dev/null | sort | tail -1)"
  if [ -n "$newest" ]; then basename "$newest" | sed 's/checkpoint_model_0*//'; else echo 0; fi
}

while true; do
  state="$(sacct -j "$JOB_ID" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')"
  elapsed="$(sacct -j "$JOB_ID" -X -n -o Elapsed 2>/dev/null | head -1 | tr -d ' ')"
  step="$(last_step)"
  oom="$(grep -ac 'out of memory' "$STAGE/${RUN_ID}_fullrun_${JOB_ID}.err" 2>/dev/null || echo 0)"
  errs="$(grep -ac 'CheckpointError\|Traceback' "$STAGE/${RUN_ID}_fullrun_${JOB_ID}.err" 2>/dev/null || echo 0)"
  samples="$(find "$RUN_ROOT/wandb" -name '*.wandb' 2>/dev/null | xargs -r stat -c%s 2>/dev/null | head -1)"
  log "state=$state elapsed=$elapsed step=$step oom=$oom traces=$errs wandb_bytes=${samples:-0}"

  case "$state" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      case "$state" in
        COMPLETED) verdict="completed" ;;
        *) verdict="died early: $state" ;;
      esac
      {
        echo "# Run status: $RUN_ID"
        echo
        echo "- job: \`$JOB_ID\`  state: **$state**  elapsed: ${elapsed:-?}"
        echo "- highest checkpoint step: **$step** (target ${TARGET_STEPS:-?})"
        echo "- OOM lines: $oom   traceback/checkpoint lines: $errs"
        echo "- written: $(date -u +%FT%TZ)"
        echo
        echo "## Outcome"
        echo
        echo "$verdict"
        echo
        echo "## If you need to resume"
        echo
        echo 'Run this on the NM5 login node:'
        echo
        echo '```bash'
        echo "export SUE_DEEPRESEARCH_ROOT=$SUE_DEEPRESEARCH_ROOT"
        echo "cd \$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
        echo "# point the run at the newest surviving checkpoint, then resubmit"
        echo "ls scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd/$RUN_ID/checkpoints/"
        echo "export SUE_TRAIN_OVERRIDES=\"text_encoder_cpu_offload=true\""
        echo "export SUE_MAX_STEPS=600 SUE_SAMPLE_EVERY=0"
        echo "export SUE_TRAIN_SECONDS=27000 SUE_TIME_LIMIT=08:00:00"
        echo "SUE_QOS=acc_ehpc bash scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd/slurm_scripts/nm5_train_submit.sh <new_run_id> fullrun"
        echo '```'
        echo
        echo "The branch reads \`config.resume_checkpoint\` to continue from a saved payload;"
        echo "leave it unset for a fresh 600-step run, or set it to a surviving"
        echo "\`checkpoint_model_*/model.pt\` path to continue."
      } > "$STATUS"
      log "WATCH_END state=$state step=$step wrote=$STATUS"
      exit 0 ;;
  esac
  sleep "$POLL"
done
