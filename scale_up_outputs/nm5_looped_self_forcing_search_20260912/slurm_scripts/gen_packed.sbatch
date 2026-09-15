#!/bin/bash -l
#SBATCH --partition=acc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=06:00:00
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
export SUE_STAGE=generation

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
[[ -x "$SUE_LOCAL_BIN/ffmpeg" ]] || { echo "MEDIA PREFLIGHT ERROR: libx264 ffmpeg is missing" >&2; exit 2; }
command -v ffprobe >/dev/null 2>&1 || { echo "MEDIA PREFLIGHT ERROR: ffprobe is missing" >&2; exit 2; }
sue_tools
sue_preflight
verify_allocation

PROMPT_COUNT=128
if [[ "$SUE_MODE" == dryrun ]]; then
  PROMPT_COUNT="${SUE_DRYRUN_PROMPT_COUNT:-1}"
fi
[[ "$PROMPT_COUNT" =~ ^[0-9]+$ && "$PROMPT_COUNT" -ge 1 && "$PROMPT_COUNT" -le 128 ]] || {
  echo "PROMPT ERROR: invalid prompt count $PROMPT_COUNT" >&2
  exit 2
}
PROMPT_MANIFEST_ABS="$SUE_EXP/config/prompts_selected_128.txt"
[[ -f "$PROMPT_MANIFEST_ABS" ]] || { echo "PREFLIGHT ERROR: missing $PROMPT_MANIFEST_ABS" >&2; exit 2; }

run_one() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"
  local config_abs="$SUE_EXP/config/${variant}.yaml"
  local rel_exp="scale_up_outputs/nm5_looped_self_forcing_search_20260912"
  local rel_run="$rel_exp/$SUE_RUN_ID"
  local rel_var="$rel_run/variants/$variant"
  local vid="$WS/$rel_var"
  local task_id="${variant}:generation"
  local config_rel_ws="$rel_exp/config/${variant}.yaml"
  local prompt_rel="$rel_var/prompts.txt"
  local rank_dir="$vid/videos/rank_000"
  local rc=0

  export SUE_VARIANT="$variant"
  export SUE_CONFIG="$config_abs"
  [[ "$variant" =~ ^[a-z0-9_]+$ ]] || { echo "VARIANT ERROR: $variant" >&2; return 2; }
  [[ -f "$config_abs" ]] || { echo "PREFLIGHT ERROR: missing $config_abs" >&2; return 2; }

  mkdir -p "$vid"
  if [[ "$SUE_MODE" == dryrun ]]; then
    head -n "$PROMPT_COUNT" "$PROMPT_MANIFEST_ABS" > "$vid/prompts.txt"
  else
    cp "$PROMPT_MANIFEST_ABS" "$vid/prompts.txt"
  fi
  local actual_prompts
  actual_prompts="$(wc -l < "$vid/prompts.txt")"
  [[ "$actual_prompts" -eq "$PROMPT_COUNT" ]] || { echo "PROMPT ERROR: $variant has $actual_prompts prompts, expected $PROMPT_COUNT" >&2; return 2; }
  cp "$config_abs" "$vid/config_snapshot.yaml"

  eval "$("$V/bin/python" - "$config_abs" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
loop = cfg.get("temporal_loop", {}) or {}
latent = int(cfg.get("num_output_frames", 0))
raw = 4 * latent - 3
eval_frames = 16 * (raw // 16)
print("LATENT=" + str(latent))
print("RAW=" + str(raw))
print("EVAL_FRAMES=" + str(eval_frames))
print("K_MIN=" + str(int(loop.get("k_min", 0))))
print("K_MAX=" + str(int(loop.get("k_max", 0))))
print("LAYER_START=" + str(int(loop.get("layer_start", 0))))
print("LAYER_END=" + str(int(loop.get("layer_end", 0))))
PY
  )"
  [[ "$LATENT" -eq 123 && "$RAW" -eq 489 && "$EVAL_FRAMES" -eq 480 ]] || {
    echo "FRAME ERROR: $variant num_output_frames=$LATENT; derived_internal_raw=$RAW derived_internal_eval=$EVAL_FRAMES; fixed num_output_frames=123 required" >&2
    return 2
  }
  [[ "$K_MIN" -ge 0 && "$K_MAX" -le 6 && "$K_MIN" -le "$K_MAX" ]] || {
    echo "K ERROR: $variant K=[$K_MIN,$K_MAX]" >&2
    return 2
  }
  [[ "$LAYER_START" -ge 0 && "$LAYER_START" -le "$LAYER_END" ]] || {
    echo "LAYER ERROR: $variant layers=$LAYER_START-$LAYER_END" >&2
    return 2
  }
  verify_worker_gpu "$gpu"

  "$V/bin/python" "$DEEPRESEARCH_SCRIPTS/eval/vbench_long/prepare_assets.py" --root "$WS" --prompt-file "$prompt_rel" --out-dir "$rel_var/work" --video-dir "$rel_var/videos/final" --world-size 1 --seed 0 --latent-frames "$LATENT" --frames-per-block 3 --raw-decoded-frames "$RAW" --eval-frames "$EVAL_FRAMES" --fps 16 --checkpoint Self-Forcing/checkpoints/self_forcing_dmd.pt --config "$config_rel_ws"

  "$V/bin/python" - "$vid/work/manifest.json" "$config_abs" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf

manifest_path = Path(sys.argv[1])
cfg = OmegaConf.load(sys.argv[2])
loop = cfg.get("temporal_loop", {}) or {}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["temporal_loop"] = {
    "k_min": int(loop.get("k_min", 0)),
    "k_max": int(loop.get("k_max", 0)),
    "layer_start": int(loop.get("layer_start", 0)),
    "layer_end": int(loop.get("layer_end", 0)),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY

  mkdir -p "$rank_dir"
  task_update "$task_id" --state running --processed 0 --expected "$PROMPT_COUNT" --slurm-job-id "$SLURM_JOB_ID"
  task_heartbeat "$task_id" "$PROMPT_COUNT" &
  local heartbeat_pid="$!"
  START_EPOCH="$(date +%s)"
  export START_EPOCH
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
      task_update "$task_id" --state done --processed "$PROMPT_COUNT" --expected "$PROMPT_COUNT" --slurm-job-id "$SLURM_JOB_ID" --result-path "$vid/generation_metadata.json" || true
      record_timing COMPLETED || true
    else
      task_update "$task_id" --state failed --last-error "generation worker failed (rc=$status)" --slurm-job-id "$SLURM_JOB_ID" || true
      record_timing FAILED || true
    fi
    exit "$status"
  }
  trap finish_one EXIT
  echo "GENERATION_START $(date -Is) variant=$variant gpu=$gpu mode=$SUE_MODE prompts=$PROMPT_COUNT num_output_frames=$LATENT derived_internal_raw=$RAW derived_internal_eval=$EVAL_FRAMES K=[$K_MIN,$K_MAX] layers=$LAYER_START-$LAYER_END job=$SLURM_JOB_ID"
  # inference.py:59 loads configs/default_config.yaml by relative path, so the
  # process must start from $SF. The checkpoint is passed absolute for the same
  # reason. prepare_assets/postprocess resolve everything from --root "$WS" and
  # are unaffected by this subshell.
  (
    cd "$SF"
    CUDA_VISIBLE_DEVICES="$gpu" "$V/bin/python" -u "$SF/inference.py" --config_path "$config_abs" --checkpoint_path "$SF/checkpoints/self_forcing_dmd.pt" --data_path "$vid/work/prompt_shards/rank_000.txt" --output_folder "$rank_dir" --num_output_frames "$LATENT" --num_samples 1 --seed 0 --use_ema --save_with_index
  ) > "$RUN_ROOT/worker_logs/generation_${SLURM_JOB_ID}_${variant}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$V/bin/python" "$DEEPRESEARCH_SCRIPTS/eval/vbench_long/postprocess_and_validate.py" --root "$WS" --manifest "$rel_var/work/manifest.json" --raw-dir "$rel_var/videos/rank_000" --final-dir "$rel_var/videos/final" --ffmpeg "$SUE_LOCAL_BIN/ffmpeg" --ffprobe "$(command -v ffprobe)"
  local valid_count
  valid_count="$(find "$vid/videos/final" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  [[ "$valid_count" -eq "$PROMPT_COUNT" ]] || { echo "VALIDATION ERROR: $variant final=$valid_count expected=$PROMPT_COUNT" >&2; return 2; }
  "$V/bin/python" - "$vid/generation_metadata.json" "$config_abs" "$PROMPT_COUNT" "$LATENT" "$RAW" "$EVAL_FRAMES" "$K_MIN" "$K_MAX" "$LAYER_START" "$LAYER_END" <<'PY'
import json
import os
import sys
from pathlib import Path

path, config, prompts, latent, raw, eval_frames, k_min, k_max, layer_start, layer_end = sys.argv[1:]
Path(path).write_text(json.dumps({
    "stage": "generation",
    "config": config,
    "prompt_count": int(prompts),
    "num_output_frames": int(latent),
    "derived_internal_processing_counts": {"raw_decoded_frames": int(raw), "eval_frames": int(eval_frames)},
    "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts",
    "fps": 16,
    "duration_seconds": 30,
    "seed": 0,
    "use_ema": True,
    "k_min": int(k_min),
    "k_max": int(k_max),
    "layer_start": int(layer_start),
    "layer_end": int(layer_end),
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY
  echo "GENERATION_DONE $(date -Is) variant=$variant gpu=$gpu valid_videos=$valid_count"
}

variants=( "$@" )
declare -a pids=()
for index in 0 1 2 3; do
  run_one "${variants[$index]}" "$index" > "$RUN_ROOT/worker_logs/generation_${SLURM_JOB_ID}_${variants[$index]}.stdout" 2>&1 &
  pids+=( "$!" )
done

pack_rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || pack_rc=1
done
if [[ "$pack_rc" -ne 0 ]]; then
  echo "GENERATION_PACK_FAILED job=$SLURM_JOB_ID" >&2
  exit "$pack_rc"
fi

"$V/bin/python" - "$RUN_ROOT/run_state/generation_complete.json" "$SUE_MODE" "$SLURM_JOB_ID" "${variants[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, mode, job_id, *variants = sys.argv[1:]
Path(path).write_text(json.dumps({
    "stage": "generation",
    "mode": mode,
    "job_id": job_id,
    "variants": variants,
    "prompt_count": 1 if mode == "dryrun" else 128,
    "num_output_frames": 123,
    "derived_internal_processing_counts": {"raw_decoded_frames": 489, "eval_frames": 480},
    "frame_count_policy": "only num_output_frames is configurable; raw/eval counts are derived internal processing counts",
    "fps": 16,
    "duration_seconds": 30,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
PY
echo "GENERATION_PACK_DONE $(date -Is) job=$SLURM_JOB_ID variants=${variants[*]}"
