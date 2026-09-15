#!/bin/bash -l
#SBATCH --partition=acc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
set -euo pipefail

: "${SUE_EXP:?SUE_EXP is required}"
: "${SUE_RUN_ID:?SUE_RUN_ID is required}"
: "${SUE_MODE:?SUE_MODE is required}"

# Slurm spools this batch script to /scratch/slurm/job<N>/ before running it,
# so BASH_SOURCE resolves to the spool directory, not to the durable bundle.
# submit_search.sh passes the real directory via --export=SUE_SCRIPTS_DIR.
SCRIPT_DIR="${SUE_SCRIPTS_DIR:-}"
if [[ -z "$SCRIPT_DIR" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[[ -r "$SCRIPT_DIR/common_env.sh" ]] || {
  echo "PATH ERROR: common_env.sh not found under SCRIPT_DIR=$SCRIPT_DIR" >&2
  echo "PATH ERROR: submit through submit_search.sh (it passes SUE_SCRIPTS_DIR) or export SUE_SCRIPTS_DIR" >&2
  exit 2
}
source "$SCRIPT_DIR/common_env.sh"
export SUE_STAGE=vbench_long_eval

if [[ "$SUE_MODE" != dryrun && "$SUE_MODE" != fullrun ]]; then
  echo "MODE ERROR: expected dryrun or fullrun" >&2
  exit 2
fi
if [[ "$#" -ne 4 ]]; then
  echo "PACK ERROR: expected exactly four variants, got $#" >&2
  exit 2
fi

module load ffmpeg/7.0.1-gcc
export PATH="$SUE_LOCAL_BIN:$PATH"
command -v ffprobe >/dev/null 2>&1 || { echo "MEDIA PREFLIGHT ERROR: ffprobe is missing" >&2; exit 2; }
sue_tools
sue_preflight
verify_allocation

PROMPT_COUNT=128
if [[ "$SUE_MODE" == dryrun ]]; then
  PROMPT_COUNT="${SUE_DRYRUN_PROMPT_COUNT:-1}"
fi
EXPECTED_WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
REL_EXP=scale_up_outputs/nm5_looped_self_forcing_search_20260912
RUN_ROOT="$SUE_EXP/$SUE_RUN_ID"
LEDGER_PATH="$SUE_EXP/artifacts/experiment_results.csv"
HISTORY_PATH="$SUE_EXP/artifacts/eval_results_history.md"

require_eval_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "EVAL PREFLIGHT ERROR: missing evaluator file $path" >&2; exit 2; }
}

require_eval_dir() {
  local path="$1"
  [[ -d "$path" ]] || { echo "EVAL PREFLIGHT ERROR: missing evaluator directory $path" >&2; exit 2; }
}

# These four assets are resolved from VBENCH_CACHE_DIR by VBench's base metrics.
require_eval_file "$VBENCH_CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
require_eval_file "$VBENCH_CACHE_DIR/amt_model/amt-s.pth"
require_eval_file "$VBENCH_CACHE_DIR/raft_model/models/raft-things.pth"
require_eval_file "$VBENCH_CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
require_eval_file "$SF/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml"

# Base subject/background/aesthetic metrics use the OpenAI CLIP/DINO loaders.
require_eval_file "$HOME/.cache/clip/ViT-B-32.pt"
require_eval_file "$HOME/.cache/clip/ViT-L-14.pt"

# torch.hub resolves repo dirs and weights under $TORCH_HOME/hub, but this
# bundle stages them one level up at $TORCH_HOME/. Bridge the two layouts
# here (self-healing, like _env.sh rebuilding the ffmpeg link on every run)
# rather than weakening the checks below, which are what actually fail loudly
# before GPU time is spent.
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

# --dev_flag enables the long clip2clip branches.  The DINOv2 branch uses the
# normal TORCH_HOME hub cache; DreamSim resets torch.hub to ~/.cache and uses
# different DINO-v1/DreamSim filenames there.
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
if [[ -z "$dreamsim_adapter_weight" ]]; then
  echo "EVAL PREFLIGHT ERROR: DreamSim ensemble adapter weights are missing" >&2
  exit 2
fi
[[ -n "${WANDB_ENTITY:-}" ]] || { echo "EVAL PREFLIGHT ERROR: WANDB_ENTITY is empty" >&2; exit 2; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "EVAL PREFLIGHT ERROR: WANDB_API_KEY is empty" >&2; exit 2; }

run_one() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"
  local rel_run="$REL_EXP/$SUE_RUN_ID"
  local rel_var="$rel_run/variants/$variant"
  local videos_dir="$EXPECTED_WS/$rel_var/videos/final"
  local eval_dir="$EXPECTED_WS/$rel_var/eval/results"
  local result_path=""
  local full_info_path=""
  local task_id="${variant}:vbench_long_eval"
  local rc=0

  export SUE_VARIANT="$variant"
  local config_abs="$SUE_EXP/config/${variant}.yaml"
  [[ "$variant" =~ ^[a-z0-9_]+$ ]] || { echo "VARIANT ERROR: $variant" >&2; return 2; }
  [[ -f "$config_abs" ]] || { echo "PREFLIGHT ERROR: missing $config_abs" >&2; return 2; }
  [[ -d "$videos_dir" ]] || { echo "EVAL PREFLIGHT ERROR: missing $videos_dir" >&2; return 2; }
  local video_count
  video_count="$(find "$videos_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  [[ "$video_count" -eq "$PROMPT_COUNT" ]] || {
    echo "EVAL PREFLIGHT ERROR: $variant has $video_count videos, expected $PROMPT_COUNT" >&2
    return 2
  }
  verify_worker_gpu "$gpu"
  mkdir -p "$eval_dir"

  task_update "${variant}:vbench_long_eval" --state running --processed 0 --expected "$PROMPT_COUNT" --slurm-job-id "$SLURM_JOB_ID"
  task_heartbeat "${variant}:vbench_long_eval" "$PROMPT_COUNT" &
  local heartbeat_pid="$!"
  START_EPOCH="$(date +%s)"
  export START_EPOCH
  export WANDB_DIR="$RUN_ROOT/wandb/$variant"
  mkdir -p "$WANDB_DIR"
  gpu_sample start
  (
    while true; do
      gpu_sample tick
      sleep 60
    done
  ) &
  local gpu_sampler_pid="$!"
  finish_one() {
    local status="$?"
    trap - EXIT
    kill "$gpu_sampler_pid" 2>/dev/null || true
    wait "$gpu_sampler_pid" 2>/dev/null || true
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    if [[ "$status" -eq 0 ]]; then
      task_update "$task_id" --state done --processed "$PROMPT_COUNT" --expected "$PROMPT_COUNT" --slurm-job-id "$SLURM_JOB_ID" --result-path "$RUN_ROOT/variants/$variant/eval/metrics.json" || true
      record_timing COMPLETED || true
    else
      task_update "$task_id" --state failed --last-error "VBench worker failed (rc=$status)" --slurm-job-id "$SLURM_JOB_ID" || true
      record_timing FAILED || true
    fi
    exit "$status"
  }
  trap finish_one EXIT
  echo "EVAL_START $(date -Is) variant=$variant gpu=$gpu mode=$SUE_MODE videos=$video_count job=$SLURM_JOB_ID"
  CUDA_VISIBLE_DEVICES="$gpu" "$V/bin/python" -u "$SF/VBench/vbench2_beta_long/eval_long.py" --videos_path "$videos_dir" --dimension subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality --mode long_custom_input --dev_flag --output_path "$eval_dir" 2>&1 | tee "$RUN_ROOT/worker_logs/eval_${SLURM_JOB_ID}_${variant}.log"
  result_path="$(find "$eval_dir" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)"
  full_info_path="$(find "$eval_dir" -maxdepth 1 -type f -name '*_full_info.json' -print -quit)"
  [[ -n "$result_path" && -f "$result_path" ]] || { echo "EVAL ERROR: missing aggregate result for $variant" >&2; return 2; }
  [[ -n "$full_info_path" && -f "$full_info_path" ]] || { echo "EVAL ERROR: missing full-info result for $variant" >&2; return 2; }
  "$V/bin/python" - "$result_path" <<'PY'
import json
import sys

required = {
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
}
data = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit("EVAL ERROR: aggregate VBench result is not a JSON object")
missing = sorted(required.difference(data))
if missing:
    raise SystemExit("EVAL ERROR: aggregate VBench result missing dimensions: " + ",".join(missing))
for key in required:
    value = data[key][0] if isinstance(data[key], list) else data[key]
    if not isinstance(value, (int, float)):
        raise SystemExit("EVAL ERROR: non-numeric aggregate value for " + key)
print("VBench aggregate result covers all six dimensions")
PY

  "$V/bin/python" "$PROJECT_SCRIPTS/record_vbench_result.py" --result-path "$result_path" --metrics-path "$RUN_ROOT/variants/$variant/eval/metrics.json" --ledger-path "$LEDGER_PATH" --output-path "$videos_dir" --config-path "$config_abs" --run-id "$SUE_RUN_ID" --variant "$variant" --mode "$SUE_MODE" --job-id "$SLURM_JOB_ID" --experiment-version-id 3 --prompt-count "$PROMPT_COUNT" --expected-prompt-count "$PROMPT_COUNT" --wandb-project "$WANDB_PROJECT"
  "$V/bin/python" - "$videos_dir" "$RUN_ROOT/variants/$variant/eval/temp_cleanup.json" <<'PY'
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
  echo "EVAL_DONE $(date -Is) variant=$variant gpu=$gpu result=$result_path"
}

variants=( "$@" )
declare -a pids=()
for index in 0 1 2 3; do
  run_one "${variants[$index]}" "$index" > "$RUN_ROOT/worker_logs/eval_${SLURM_JOB_ID}_${variants[$index]}.stdout" 2>&1 &
  pids+=( "$!" )
done

pack_rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || pack_rc=1
done
if [[ "$pack_rc" -ne 0 ]]; then
  echo "EVAL_PACK_FAILED job=$SLURM_JOB_ID" >&2
  exit "$pack_rc"
fi

"$V/bin/python" "$PROJECT_SCRIPTS/reduce_vbench_history.py" --run-dir "$RUN_ROOT" --history-path "$HISTORY_PATH" --mode "$SUE_MODE" --run-id "$SUE_RUN_ID" --variants "${variants[@]}"

"$V/bin/python" - "$RUN_ROOT/run_state/eval_complete.json" "$SUE_MODE" "$SLURM_JOB_ID" "${variants[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, mode, job_id, *variants = sys.argv[1:]
Path(path).write_text(json.dumps({
    "stage": "vbench_long_eval",
    "mode": mode,
    "job_id": job_id,
    "variants": variants,
    "prompt_count": 1 if mode == "dryrun" else 128,
    "protocol": {"num_output_frames": 123, "derived_internal_processing_counts": {"raw_decoded_frames": 489, "eval_frames": 480}, "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts", "fps": 16, "duration_seconds": 30},
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY

if [[ "$SUE_MODE" == dryrun ]]; then
  "$V/bin/python" - "$SUE_EXP/readiness/readiness_nm5.json" "$SUE_RUN_ID" "${variants[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, run_id, *variants = sys.argv[1:]
Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(path).write_text(json.dumps({
    "backend": "nm5",
    "status": "ready",
    "fullrun_ready": True,
    "dryrun_run_id": run_id,
    "variants": variants,
    "protocol": {"num_output_frames": 123, "derived_internal_processing_counts": {"raw_decoded_frames": 489, "eval_frames": 480}, "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts", "fps": 16, "duration_seconds": 30, "seed": 0, "use_ema": True},
    "evidence": ["all four generation tasks completed", "all four official VBench-Long evaluations completed", "CSV ledger and labeled history updated"],
    "verified_at": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY
fi
echo "EVAL_PACK_DONE $(date -Is) job=$SLURM_JOB_ID variants=${variants[*]}"
