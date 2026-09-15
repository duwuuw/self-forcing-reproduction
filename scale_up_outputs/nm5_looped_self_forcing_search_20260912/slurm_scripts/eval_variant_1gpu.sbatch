#!/bin/bash -l
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#
# One-variant VBench-Long evaluation on a single GPU.
#
# This is an alternative decomposition of eval_packed.sbatch, not a
# replacement. Each variant evaluates independently and writes its own
# metrics.json, W&B run, and CSV ledger row (record_vbench_result.py locks the
# ledger with flock, so concurrent writers are safe). It exists because a
# 1-GPU request backfills onto a saturated cluster far more readily than the
# packed 4-GPU single-node request, which needs an entire node free.
#
# Pack-level roll-up (reduce_vbench_history.py, eval_complete.json,
# readiness_nm5.json) is NOT done here; run it once after every variant job for
# the same run has finished.
#
# Usage:
#   SUE_SCRIPTS_DIR=<bundle>/slurm_scripts \
#   sbatch --account=... --qos=... --export=ALL,SUE_SCRIPTS_DIR=...,SUE_EXP=...,\
#          SUE_RUN_ID=...,SUE_MODE=fullrun,SUE_DEEPRESEARCH_ROOT=... \
#          eval_variant_1gpu.sbatch <variant>
set -euo pipefail

: "${SUE_EXP:?SUE_EXP is required}"
: "${SUE_RUN_ID:?SUE_RUN_ID is required}"
: "${SUE_MODE:?SUE_MODE is required}"
: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"

if [[ "$#" -ne 1 ]]; then
  echo "PACK ERROR: expected exactly one variant, got $#" >&2
  exit 2
fi
VARIANT="$1"
[[ "$VARIANT" =~ ^[a-z0-9_]+$ ]] || { echo "VARIANT ERROR: $VARIANT" >&2; exit 2; }
[[ "$SUE_MODE" == dryrun || "$SUE_MODE" == fullrun ]] || { echo "MODE ERROR: $SUE_MODE" >&2; exit 2; }

# Slurm spools this script, so BASH_SOURCE does not point at the bundle.
SCRIPT_DIR="${SUE_SCRIPTS_DIR:-}"
if [[ -z "$SCRIPT_DIR" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[[ -r "$SCRIPT_DIR/common_env.sh" ]] || {
  echo "PATH ERROR: common_env.sh not found under SCRIPT_DIR=$SCRIPT_DIR" >&2
  exit 2
}
source "$SCRIPT_DIR/common_env.sh"
export SUE_STAGE=vbench_long_eval SUE_VARIANT="$VARIANT"

module load ffmpeg/7.0.1-gcc
export PATH="$SUE_LOCAL_BIN:$PATH"
sue_tools
sue_preflight

PROMPT_COUNT=128
if [[ "$SUE_MODE" == dryrun ]]; then PROMPT_COUNT="${SUE_DRYRUN_PROMPT_COUNT:-1}"; fi
EXPECTED_WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
REL_EXP=scale_up_outputs/nm5_looped_self_forcing_search_20260912
RUN_ROOT="$SUE_EXP/$SUE_RUN_ID"
LEDGER_PATH="$SUE_EXP/artifacts/experiment_results.csv"

require_eval_file() {
  [[ -f "$1" ]] || { echo "EVAL PREFLIGHT ERROR: missing $1" >&2; exit 2; }
}
require_eval_dir() {
  [[ -d "$1" ]] || { echo "EVAL PREFLIGHT ERROR: missing $1" >&2; exit 2; }
}

# Keep these in step with eval_packed.sbatch: they are what fail loudly before
# GPU time is spent on a partially staged asset tree.
require_eval_file "$VBENCH_CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
require_eval_file "$VBENCH_CACHE_DIR/amt_model/amt-s.pth"
require_eval_file "$VBENCH_CACHE_DIR/raft_model/models/raft-things.pth"
require_eval_file "$VBENCH_CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
require_eval_file "$SF/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml"
require_eval_file "$HOME/.cache/clip/ViT-B-32.pt"
require_eval_file "$HOME/.cache/clip/ViT-L-14.pt"

# torch.hub resolves under $TORCH_HOME/hub; this bundle stages one level up.
mkdir -p "$TORCH_HOME/hub/checkpoints"
for repo in facebookresearch_dino_main facebookresearch_dinov2_main; do
  if [[ -d "$TORCH_HOME/$repo" && ! -e "$TORCH_HOME/hub/$repo" ]]; then
    ln -sfn "$TORCH_HOME/$repo" "$TORCH_HOME/hub/$repo"
  fi
done
for weight in dino_vitbase16_pretrain.pth dinov2_vitb14_pretrain.pth; do
  if [[ -f "$TORCH_HOME/checkpoints/$weight" && ! -e "$TORCH_HOME/hub/checkpoints/$weight" ]]; then
    ln -sfn "$TORCH_HOME/checkpoints/$weight" "$TORCH_HOME/hub/checkpoints/$weight"
  fi
done
require_eval_dir "$TORCH_HOME/hub/facebookresearch_dino_main"
require_eval_file "$TORCH_HOME/hub/checkpoints/dino_vitbase16_pretrain.pth"
require_eval_dir "$TORCH_HOME/hub/facebookresearch_dinov2_main"
require_eval_file "$TORCH_HOME/hub/checkpoints/dinov2_vitb14_pretrain.pth"
require_eval_dir "$HOME/.cache/facebookresearch_dino_main"
require_eval_file "$HOME/.cache/checkpoints/dino_vitbase16_pretrain.pth"
require_eval_file "$HOME/.cache/dino_vitb16_pretrain.pth"
require_eval_file "$HOME/.cache/open_clip_vitb16_pretrain.pth.tar"
require_eval_file "$HOME/.cache/clip_vitb16_pretrain.pth.tar"
require_eval_dir "$HOME/.cache/ensemble_lora"
require_eval_file "$HOME/.cache/ensemble_lora/adapter_config.json"
dreamsim_adapter_weight="$(find "$HOME/.cache/ensemble_lora" -maxdepth 1 -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \) -print -quit)"
[[ -n "$dreamsim_adapter_weight" ]] || {
  echo "EVAL PREFLIGHT ERROR: DreamSim ensemble adapter weights are missing" >&2
  exit 2
}
[[ -n "${WANDB_ENTITY:-}" ]] || { echo "EVAL PREFLIGHT ERROR: WANDB_ENTITY is empty" >&2; exit 2; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "EVAL PREFLIGHT ERROR: WANDB_API_KEY is empty" >&2; exit 2; }

GPU=0
rel_run="$REL_EXP/$SUE_RUN_ID"
rel_var="$rel_run/variants/$VARIANT"
videos_dir="$EXPECTED_WS/$rel_var/videos/final"
eval_dir="$EXPECTED_WS/$rel_var/eval/results"
config_abs="$SUE_EXP/config/${VARIANT}.yaml"
task_id="${VARIANT}:vbench_long_eval"
marker="$RUN_ROOT/run_state/eval_${VARIANT}_complete.json"

[[ -f "$config_abs" ]] || { echo "PREFLIGHT ERROR: missing $config_abs" >&2; exit 2; }
[[ -d "$videos_dir" ]] || { echo "EVAL PREFLIGHT ERROR: missing $videos_dir" >&2; exit 2; }
video_count="$(find "$videos_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
[[ "$video_count" -eq "$PROMPT_COUNT" ]] || {
  echo "EVAL PREFLIGHT ERROR: $VARIANT has $video_count videos, expected $PROMPT_COUNT" >&2
  exit 2
}
verify_worker_gpu "$GPU"
mkdir -p "$eval_dir" "$RUN_ROOT/run_state"

task_update "$task_id" --state running --processed 0 --expected "$PROMPT_COUNT" --slurm-job-id "$SLURM_JOB_ID"
task_heartbeat "$task_id" "$PROMPT_COUNT" &
heartbeat_pid="$!"
START_EPOCH="$(date +%s)"
export START_EPOCH
export WANDB_DIR="$RUN_ROOT/wandb/$VARIANT"
mkdir -p "$WANDB_DIR"
gpu_sample start
( while true; do gpu_sample tick; sleep 60; done ) &
gpu_sampler_pid="$!"

finish_one() {
  local status="$?"
  trap - EXIT
  kill "$gpu_sampler_pid" 2>/dev/null || true
  wait "$gpu_sampler_pid" 2>/dev/null || true
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  if [[ "$status" -eq 0 ]]; then
    task_update "$task_id" --state done --processed "$PROMPT_COUNT" --expected "$PROMPT_COUNT" \
      --slurm-job-id "$SLURM_JOB_ID" --result-path "$RUN_ROOT/variants/$VARIANT/eval/metrics.json" || true
    record_timing COMPLETED || true
    "$V/bin/python" - "$marker" "$VARIANT" "$PROMPT_COUNT" "$SLURM_JOB_ID" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, variant, prompts, job_id = sys.argv[1:]
Path(path).write_text(json.dumps({
    "stage": "vbench_long_eval",
    "variant": variant,
    "prompt_count": int(prompts),
    "job_id": job_id,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + chr(10), encoding="utf-8")
PY
  else
    task_update "$task_id" --state failed --last-error "VBench worker failed (rc=$status)" --slurm-job-id "$SLURM_JOB_ID" || true
    record_timing FAILED || true
  fi
  exit "$status"
}
trap finish_one EXIT

echo "EVAL_START $(date -Is) variant=$VARIANT gpu=$GPU mode=$SUE_MODE videos=$video_count job=$SLURM_JOB_ID"
CUDA_VISIBLE_DEVICES="$GPU" "$V/bin/python" -u "$SF/VBench/vbench2_beta_long/eval_long.py" \
  --videos_path "$videos_dir" \
  --dimension subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
  --mode long_custom_input --dev_flag --output_path "$eval_dir" \
  2>&1 | tee "$RUN_ROOT/worker_logs/eval_${SLURM_JOB_ID}_${VARIANT}.log"

result_path="$(find "$eval_dir" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)"
full_info_path="$(find "$eval_dir" -maxdepth 1 -type f -name '*_full_info.json' -print -quit)"
[[ -n "$result_path" && -f "$result_path" ]] || { echo "EVAL ERROR: missing aggregate result for $VARIANT" >&2; exit 2; }
[[ -n "$full_info_path" && -f "$full_info_path" ]] || { echo "EVAL ERROR: missing full-info result for $VARIANT" >&2; exit 2; }
"$V/bin/python" - "$result_path" <<'PY'
import json
import sys

required = {
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
}
data = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit("EVAL ERROR: aggregate VBench result is not a JSON object")
missing = sorted(required.difference(data))
if missing:
    raise SystemExit("EVAL ERROR: aggregate VBench result missing dimensions: " + ",".join(missing))
# With --dev_flag each dimension is [aggregate, [per-clip details...]]; the
# aggregate is the leading element.
for key in required:
    value = data[key][0] if isinstance(data[key], list) else data[key]
    if not isinstance(value, (int, float)):
        raise SystemExit("EVAL ERROR: non-numeric aggregate value for " + key)
print("VBench aggregate result covers all six dimensions")
PY

"$V/bin/python" "$PROJECT_SCRIPTS/record_vbench_result.py" \
  --result-path "$result_path" \
  --metrics-path "$RUN_ROOT/variants/$VARIANT/eval/metrics.json" \
  --ledger-path "$LEDGER_PATH" --output-path "$videos_dir" --config-path "$config_abs" \
  --run-id "$SUE_RUN_ID" --variant "$VARIANT" --mode "$SUE_MODE" --job-id "$SLURM_JOB_ID" \
  --experiment-version-id 3 --prompt-count "$PROMPT_COUNT" --expected-prompt-count "$PROMPT_COUNT" \
  --wandb-project "$WANDB_PROJECT"

"$V/bin/python" - "$videos_dir" "$RUN_ROOT/variants/$VARIANT/eval/temp_cleanup.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

videos_dir = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2])
removed = []
for name in ("split_clip", "split_scene"):
    target = videos_dir / name
    if not target.is_dir() or target.parent != videos_dir:
        continue
    file_count = sum(1 for item in target.rglob("*") if item.is_file())
    shutil.rmtree(target)
    removed.append({"name": name, "files_removed": file_count})
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps({
    "stage": "vbench_long_eval",
    "videos_dir": str(videos_dir),
    "removed": removed,
    "policy": "retain final videos and durable metrics/history; remove VBench split trees",
}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY

echo "EVAL_DONE $(date -Is) variant=$VARIANT gpu=$GPU result=$result_path"
