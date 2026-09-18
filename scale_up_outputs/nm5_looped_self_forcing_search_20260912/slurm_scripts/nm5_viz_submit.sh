#!/bin/bash -l
# Submit the three-way qualitative rollout (baseline / blockwise / layerwise) as
# a single four-GPU allocation. Private backend values are read on the execution
# plane from the ignored config, exactly like nm5_round_submit.sh, so they never
# reach the control plane.
#
# Usage: nm5_viz_submit.sh [run_id]        (default viz_r5_20260917)
# Run from the NM5 HPC login node (the `nm5` ssh alias).
set -euo pipefail

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
CONFIG="$SUE_DEEPRESEARCH_ROOT/deepresearch-sandbox/config_nm5.txt"
[[ -r "$CONFIG" ]] || { echo "CONFIG ERROR: missing $CONFIG on the execution plane" >&2; exit 3; }

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
EXP="$WS/scale_up_outputs/nm5_looped_self_forcing_search_20260912"
RUN_ID="${1:-viz_r5_20260917}"
[[ "$RUN_ID" =~ ^[a-z0-9_]+$ ]] || { echo "RUN_ID ERROR: $RUN_ID" >&2; exit 2; }
: "${NM5_ACCOUNT:?NM5_ACCOUNT missing from $CONFIG}"

for f in "$EXP/config/viz_baseline.yaml" "$EXP/config/k2_l08_15.yaml" \
         "$EXP/config/k2_s08_15.yaml" "$EXP/slurm_scripts/viz_triplet.sbatch"; do
  [[ -f "$f" ]] || { echo "CONFIG ERROR: missing $f" >&2; exit 2; }
done

export SUE_DEEPRESEARCH_ROOT
export SUE_EXP="$EXP"
export SUE_RUN_ID="$RUN_ID"
export SUE_BASE_ENV_SCRIPT="$EXP/slurm_scripts/_env.sh"
export SUE_SCRIPTS_DIR="$EXP/slurm_scripts"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Canonical Slurm prefix resolver: user.yaml only; missing/invalid -> silly-.
V="$WS/scale_up_outputs/envs/miniconda3/envs/looped-self-forcing/bin/python"
[[ -x "$V" ]] || { echo "ENV ERROR: missing env interpreter $V" >&2; exit 2; }
JOB_PREFIX="$("$V" - "$WS" <<'PY'
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

export_args="ALL,SUE_SCRIPTS_DIR=$SUE_SCRIPTS_DIR,SUE_BASE_ENV_SCRIPT=$SUE_BASE_ENV_SCRIPT,SUE_DEEPRESEARCH_ROOT=$SUE_DEEPRESEARCH_ROOT,SUE_EXP=$SUE_EXP,SUE_RUN_ID=$SUE_RUN_ID,WANDB_MODE=offline"
if [[ -n "${WANDB_ENTITY:-}" ]]; then export_args="$export_args,WANDB_ENTITY=$WANDB_ENTITY"; fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then export_args="$export_args,WANDB_API_KEY=$WANDB_API_KEY"; fi

mkdir -p "$EXP/logs"
# acc_debug carries 100x the acc_ehpc priority weight and this job is far shorter
# than its 02:00:00 wall cap, so the three short rollouts schedule quickly.
QOS="${NM5_VIZ_QOS:-acc_debug}"
TIME="${NM5_VIZ_TIME:-01:30:00}"

echo "VIZ_SUBMIT_START run_id=$RUN_ID qos=$QOS time=$TIME prefix=$JOB_PREFIX"
job_id="$(sbatch --parsable \
  --chdir="$WS" \
  --account="$NM5_ACCOUNT" \
  --partition="${NM5_PARTITION:-acc}" \
  --qos="$QOS" \
  --gres=gpu:4 \
  --cpus-per-task=80 \
  --time="$TIME" \
  --job-name="${JOB_PREFIX}viz-triplet-${RUN_ID}" \
  --output="$EXP/logs/${RUN_ID}_viz_%j.out" \
  --error="$EXP/logs/${RUN_ID}_viz_%j.err" \
  --export="$export_args" \
  "$EXP/slurm_scripts/viz_triplet.sbatch")"
echo "VIZ_SUBMIT_DONE run_id=$RUN_ID job=$job_id"
echo "VIZ_LOG $EXP/logs/${RUN_ID}_viz_${job_id}.out"
echo "VIZ_ROOT $EXP/$RUN_ID"
