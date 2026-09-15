#!/bin/bash -l
set -euo pipefail

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
: "${SUE_EXP:?SUE_EXP is required}"
: "${SUE_RUN_ID:?SUE_RUN_ID is required}"
: "${SUE_MODE:?SUE_MODE is required}"
: "${SUE_ACCOUNT:?SUE_ACCOUNT is required}"
: "${SUE_PARTITION:?SUE_PARTITION is required}"
: "${SUE_QOS:?SUE_QOS is required}"

SUE_BASE_ENV_SCRIPT="${SUE_BASE_ENV_SCRIPT:-$SUE_EXP/slurm_scripts/_env.sh}"
export SUE_BASE_ENV_SCRIPT

MODE="$SUE_MODE"
DRYRUN_PROMPT_COUNT="${SUE_DRYRUN_PROMPT_COUNT:-1}"
MAX_PARALLEL="${SUE_MAX_PARALLEL:-1}"
if [[ "$MODE" != dryrun && "$MODE" != fullrun ]]; then
  echo "MODE ERROR: expected dryrun or fullrun" >&2
  exit 2
fi
[[ "$MAX_PARALLEL" =~ ^[0-9]+$ && "$MAX_PARALLEL" -ge 1 ]] || {
  echo "MAX_PARALLEL ERROR: positive integer required" >&2
  exit 2
}
if [[ "$#" -ne 4 ]]; then
  echo "SUBMISSION ERROR: pass exactly four variant names" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
EXPECTED_WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
[[ "${WS:-}" == "$EXPECTED_WS" ]] || { echo "PATH ERROR: workspace identity mismatch" >&2; exit 2; }
[[ "$(pwd -P)" == "$EXPECTED_WS" ]] || { echo "PATH ERROR: submit from workspace root only" >&2; exit 2; }
export EXP="$SUE_EXP"
export SUE_EXP_DIR="$SUE_EXP"
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-looped-self-forcing-vbench-long}"
export WANDB_GROUP="$SUE_RUN_ID"
export WANDB_EXPERIMENT_NAME="$WANDB_PROJECT"
EXPECTED_PROMPT_COUNT=128
if [[ "$MODE" == dryrun ]]; then EXPECTED_PROMPT_COUNT="$DRYRUN_PROMPT_COUNT"; fi

# Canonical Slurm prefix resolver: user.yaml only; missing/invalid -> silly-.
SUE_SLURM_JOB_PREFIX="$("$V/bin/python" - "$EXPECTED_WS" <<'PY'
import re
import sys
from pathlib import Path
import yaml

workspace = Path(sys.argv[1])
username = ""
try:
    path = workspace / "user.yaml"
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        username = str(loaded.get("username", "")).strip().lower()
except Exception:
    username = ""
if not re.fullmatch(r"[a-z0-9]{2,12}", username):
    print("notice: user.yaml missing or username unset/invalid; using default Slurm prefix 'silly-'", file=sys.stderr)
    username = "silly"
print(username + "-")
PY
)"
export SUE_SLURM_JOB_PREFIX
if [[ -n "${SUE_EXPECTED_SLURM_JOB_PREFIX:-}" && "$SUE_EXPECTED_SLURM_JOB_PREFIX" != "$SUE_SLURM_JOB_PREFIX" ]]; then
  echo "JOB NAME ERROR: caller prefix disagrees with canonical user.yaml resolver" >&2
  exit 2
fi
case "$SUE_SLURM_JOB_PREFIX" in
  *-) ;;
  *) echo "JOB NAME ERROR: prefix must end with '-'" >&2; exit 2 ;;
esac

REL_EXP=scale_up_outputs/nm5_looped_self_forcing_search_20260912
RUN_ROOT="$SUE_EXP/$SUE_RUN_ID"
mkdir -p "$RUN_ROOT/state" "$SUE_EXP/run_state" "$SUE_EXP/logs" "$SUE_EXP/artifacts"
declare -a variants=( "$@" )
for variant in "${variants[@]}"; do
  [[ "$variant" =~ ^[a-z0-9_]+$ ]] || { echo "VARIANT ERROR: $variant" >&2; exit 2; }
  [[ -f "$SUE_EXP/config/${variant}.yaml" ]] || { echo "CONFIG ERROR: missing $SUE_EXP/config/${variant}.yaml" >&2; exit 2; }
done

TASK_INIT_JSON="$RUN_ROOT/state/tasks_init.json"
PYTHONPATH="$SUE_DEEPRESEARCH_ROOT/scripts/environment" "$V/bin/python" - "$RUN_ROOT" "$SUE_RUN_ID" "$TASK_INIT_JSON" "${variants[@]}" <<'PY'
import json
import sys
from pathlib import Path
from sue_tasks import init_tasks

run_dir, run_id, output, *variants = sys.argv[1:]
prompt_count = 1 if __import__("os").environ.get("SUE_MODE") == "dryrun" else 128
specs = []
for variant in variants:
    for stage in ("generation", "vbench_long_eval"):
        specs.append({
            "task_id": f"{variant}:{stage}",
            "variant": variant,
            "method": "looped-self-forcing",
            "dataset": "vbench-long-selected128",
            "seed": 0,
            "expected_total_units": prompt_count,
            "unit": "videos",
            "gpus_allocated": 1,
            "nodes_allocated": 1,
            "partition": "acc",
            "state": "queued",
        })
rows = init_tasks(run_dir, run_id, specs, exp_version_id="3", replace=False)
Path(output).write_text(json.dumps({"task_count": len(rows), "task_ids": [r["task_id"] for r in rows]}, indent=2) + chr(10), encoding="utf-8")
PY

parse_job_id() {
  printf '%s\n' "$1" | awk -F ';' '/^[0-9]+([.][0-9]+)?(;|$)/ {job=$1} END {if (job=="") exit 2; print job}'
}
job_name() {
  local stage="$1"
  local body="${WANDB_EXPERIMENT_NAME}-${SUE_RUN_ID}-${stage}"
  body="${body:0:70}"
  printf '%s%s' "$SUE_SLURM_JOB_PREFIX" "$body"
}
# SUE_SCRIPTS_DIR is required: Slurm spools the batch script to
# /scratch/slurm/job<N>/, so the workers cannot locate common_env.sh relative to
# BASH_SOURCE. Pass the durable bundle directory explicitly.
export_args="ALL,SUE_SCRIPTS_DIR=$SCRIPT_DIR,SUE_BASE_ENV_SCRIPT=$SUE_BASE_ENV_SCRIPT,SUE_DEEPRESEARCH_ROOT=$SUE_DEEPRESEARCH_ROOT,SUE_EXP=$SUE_EXP,SUE_EXP_DIR=$SUE_EXP,SUE_RUN_ID=$SUE_RUN_ID,SUE_MODE=$MODE,SUE_DRYRUN_PROMPT_COUNT=$DRYRUN_PROMPT_COUNT,SUE_MAX_PARALLEL=$MAX_PARALLEL,SUE_ACCOUNT=$SUE_ACCOUNT,SUE_PARTITION=$SUE_PARTITION,SUE_QOS=$SUE_QOS,SUE_SLURM_JOB_PREFIX=$SUE_SLURM_JOB_PREFIX,WANDB_MODE=offline,WANDB_PROJECT=$WANDB_PROJECT,WANDB_GROUP=$SUE_RUN_ID"
if [[ -n "${WANDB_ENTITY:-}" ]]; then export_args="$export_args,WANDB_ENTITY=$WANDB_ENTITY"; fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then export_args="$export_args,WANDB_API_KEY=$WANDB_API_KEY"; fi
# Alternative Self-Forcing source tree (e.g. the layer-wise selected-layer stack
# loop checkout). Passed explicitly rather than relying on --export=ALL so the
# worker's _env.sh resolves SF deterministically.
if [[ -n "${SUE_SF_DIR:-}" ]]; then export_args="$export_args,SUE_SF_DIR=$SUE_SF_DIR"; fi

GEN_TIME=06:00:00
EVAL_TIME=24:00:00
if [[ "$MODE" == dryrun ]]; then GEN_TIME=02:00:00; EVAL_TIME=02:00:00; fi
# Optional wall-time overrides. runtime.yaml records eval_walltime=24:00:00, but
# an over-long request is hard for Slurm to backfill: a 128-prompt VBench-Long
# eval finishes in ~2 h, and holding a 24 h reservation can leave the job
# PENDING for a long time on a busy partition. Lowering it (in place, with
# `scontrol update jobid=<id> TimeLimit=...`, so queue seniority is kept) starts
# the eval far sooner. Keep a comfortable multiple of the expected runtime:
# VBenchLong accumulates in memory and only writes on the last dimension, so a
# timeout loses every score.
GEN_TIME="${SUE_GEN_TIME:-$GEN_TIME}"
EVAL_TIME="${SUE_EVAL_TIME:-$EVAL_TIME}"
# Optional per-stage QoS override. acc_debug carries 100x the priority weight of
# acc_ehpc but caps a user at one job (MaxJobsPU=1, MaxSubmitPU=1) and 02:00:00
# wall time, so it suits a short dryrun and never the fullrun. Defaults keep
# both stages on the recorded runtime.yaml QoS.
GEN_QOS="${SUE_GEN_QOS:-$SUE_QOS}"
EVAL_QOS="${SUE_EVAL_QOS:-$SUE_QOS}"
GEN_NAME="$(job_name generation)"
EVAL_NAME="$(job_name evaluation)"
gen_output="$(sbatch --parsable --chdir="$EXPECTED_WS" --account="$SUE_ACCOUNT" --partition="$SUE_PARTITION" --qos="$GEN_QOS" --gres=gpu:4 --cpus-per-task=80 --time="$GEN_TIME" --job-name="$GEN_NAME" --output="$SUE_EXP/logs/${SUE_RUN_ID}_generation_%j.out" --error="$SUE_EXP/logs/${SUE_RUN_ID}_generation_%j.err" --export="$export_args" "$SCRIPT_DIR/gen_packed.sbatch" "${variants[@]}")"
gen_id="$(parse_job_id "$gen_output")"
eval_output="$(sbatch --parsable --chdir="$EXPECTED_WS" --account="$SUE_ACCOUNT" --partition="$SUE_PARTITION" --qos="$EVAL_QOS" --dependency="afterok:$gen_id" --gres=gpu:4 --cpus-per-task=80 --time="$EVAL_TIME" --job-name="$EVAL_NAME" --output="$SUE_EXP/logs/${SUE_RUN_ID}_evaluation_%j.out" --error="$SUE_EXP/logs/${SUE_RUN_ID}_evaluation_%j.err" --export="$export_args" "$SCRIPT_DIR/eval_packed.sbatch" "${variants[@]}")"
eval_id="$(parse_job_id "$eval_output")"

for variant in "${variants[@]}"; do
  # The rows start as queued before sbatch workers can race ahead. Only add
  # scheduler IDs here; never write queued after a worker has become running.
  task_update "${variant}:generation" --expected "${EXPECTED_PROMPT_COUNT}" --slurm-job-id "$gen_id"
  task_update "${variant}:vbench_long_eval" --expected "${EXPECTED_PROMPT_COUNT}" --slurm-job-id "$eval_id"
done

"$V/bin/python" - "$RUN_ROOT/state/submitted_jobs.json" "$SUE_RUN_ID" "$MODE" "$MAX_PARALLEL" "$SUE_SLURM_JOB_PREFIX" "$gen_id" "$eval_id" "${variants[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, run_id, mode, max_parallel, prefix, gen_id, eval_id, *variants = sys.argv[1:]
payload = {
    "run_id": run_id,
    "mode": mode,
    "max_parallel": int(max_parallel),
    "slurm_job_prefix": prefix,
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "jobs": [
        {"job_id": gen_id, "stage": "generation", "job_name": "generation", "variants": variants},
        {"job_id": eval_id, "stage": "vbench_long_eval", "job_name": "evaluation", "depends_on": [gen_id], "variants": variants},
    ],
    "protocol": {"num_output_frames": 123, "derived_internal_processing_counts": {"raw_decoded_frames": 489, "eval_frames": 480}, "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts", "fps": 16, "duration_seconds": 30, "seed": 0, "use_ema": True},
}
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY
"$V/bin/python" - "$RUN_ROOT/state/launch_manifest.json" "$SUE_RUN_ID" "$MODE" "$MAX_PARALLEL" "$SUE_SLURM_JOB_PREFIX" "$gen_id" "$eval_id" "${variants[@]}" <<'PY'
import json
import sys
from pathlib import Path

path, run_id, mode, max_parallel, prefix, gen_id, eval_id, *variants = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "mode": mode,
    "variants": variants,
    "max_parallel": int(max_parallel),
    "resources": {"nodes": 1, "gpus_per_node": 4, "cpus_per_gpu": 20, "gpu_type": "H100"},
    "jobs": {"generation": gen_id, "vbench_long_eval": eval_id, "dependency": f"afterok:{gen_id}"},
    "slurm_job_prefix": prefix,
    "protocol": {"num_output_frames": 123, "derived_internal_processing_counts": {"raw_decoded_frames": 489, "eval_frames": 480}, "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts", "fps": 16, "duration_seconds": 30, "seed": 0, "use_ema": True},
}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY

MONITOR_SESSION="sue-${SUE_RUN_ID}"
MONITOR_LOG="$SUE_EXP/logs/monitor_${SUE_RUN_ID}.log"
MONITOR_INTERVAL=600
if [[ "$MODE" == dryrun ]]; then MONITOR_INTERVAL=60; fi
# NM5 ships tmux as a module, not in the base PATH. The Slurm jobs above are
# already durable through --output/--error, so the monitor session is a
# convenience: warn and continue rather than aborting after submission.
if ! command -v tmux >/dev/null 2>&1; then
  # tmux/3.3a declares prereq_any("ncurses/6.4"), so loading it alone fails.
  module load ncurses/6.4 tmux/3.3a 2>/dev/null || true
fi
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$MONITOR_SESSION" 2>/dev/null || tmux new-session -d -s "$MONITOR_SESSION" "bash '$SCRIPT_DIR/monitor_run.sh' '$SUE_EXP' '$SUE_RUN_ID' '$gen_id' '$eval_id' '$MODE' '$MONITOR_INTERVAL' >> '$MONITOR_LOG' 2>&1"
  tmux has-session -t "$MONITOR_SESSION"
  printf '%s\n' "MONITOR_SESSION=$MONITOR_SESSION"
else
  printf 'MONITOR WARNING: tmux unavailable even after module load; Slurm output/error logs remain the durable monitor\n' >&2
fi
printf '%s\n' "SUBMISSION_DONE run_id=$SUE_RUN_ID mode=$MODE generation_job=$gen_id evaluation_job=$eval_id monitor_session=$MONITOR_SESSION"
printf '%s\n' "DURABLE_LOGS $SUE_EXP/logs/${SUE_RUN_ID}_generation_%j.out $SUE_EXP/logs/${SUE_RUN_ID}_evaluation_%j.out $MONITOR_LOG"
