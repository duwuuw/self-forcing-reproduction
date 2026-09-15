#!/bin/bash -l
# NOTE: the canonical submitter overrides this with the user.yaml-resolved
# <username>- prefix; the default is kept prefixed so a direct `sbatch` never
# emits an unprefixed job name.
#SBATCH --job-name=silly-nm5-env-smoke
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
set -Eeuo pipefail
: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
WS_ROOT="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
EXP="$WS_ROOT/scale_up_outputs/nm5_looped_self_forcing_search_20260912"
source "$EXP/slurm_scripts/_env.sh"
export EXP
cd "$SF"
# `-l` above is required for the module system. ffprobe comes from the NM5
# module (imageio-ffmpeg ships no ffprobe) while ffmpeg must resolve to the
# static libx264 build, so load the module before sue_tools.
module load ffmpeg/7.0.1-gcc
sue_tools
RESULT="$EXP/readiness/nm5_env_smoke_result.json"
IMPORT_RESULT="$EXP/readiness/nm5_env_import_smoke_20260913.json"
SMOKE_DIR="$EXP/readiness/nm5_env_smoke_20260913"
LOG_HINT="$EXP/logs/nm5_env_smoke_${SLURM_JOB_ID}.log"
STEP=init
FAILED_COMMAND=""
write_result() {
  local status="$1"; local code="$2"; local message="$3"
  "$V/bin/python" - "$RESULT" "$status" "$code" "$message" "$SLURM_JOB_ID" "$WS_ROOT" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, status, code, message, job_id, ws = sys.argv[1:]
Path(out).write_text(json.dumps({
  "status": status, "success": status == "success", "exit_code": int(code),
  "message": message, "job_id": job_id, "log_path": f"scale_up_outputs/nm5_looped_self_forcing_search_20260912/logs/nm5_env_smoke_{job_id}.out",
  "fixed_num_output_frames": 123, "derived_raw_decoded_frames": 489,
  "checkpoint": "Self-Forcing/checkpoints/self_forcing_dmd.pt", "use_ema": True,
  "active_bundle": "scale_up_outputs/nm5_looped_self_forcing_search_20260912",
  "updated_at": datetime.now(timezone.utc).isoformat()
}, indent=2) + chr(10))
PY
}
on_err() { FAILED_COMMAND="$BASH_COMMAND"; }
on_exit() {
  local code=$?
  if [[ "$code" -ne 0 ]]; then write_result failure "$code" "step=$STEP command=${FAILED_COMMAND:-unknown}" || true; fi
  exit "$code"
}
trap on_err ERR
trap on_exit EXIT
mkdir -p "$SMOKE_DIR" "$EXP/logs"
STEP=import_backend
nvidia-smi -L
"$V/bin/python" - "$IMPORT_RESULT" <<'PY'
import json, sys, torch, av, diffusers, transformers
import vbench2_beta_long
result = {
  "success": bool(torch.cuda.is_available() and torch.cuda.device_count() == 1),
  "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
  "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
  "av": getattr(av, "__version__", "unknown"), "diffusers": diffusers.__version__,
  "transformers": transformers.__version__, "vbench2_beta_long": "available",
  "fixed_num_output_frames": 123
}
if not result["success"]: raise SystemExit(json.dumps(result))
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + chr(10))
print("IMPORT_SMOKE=OK cuda=true visible_gpus=1 num_output_frames=123")
PY
STEP=generation
PROMPT="$SMOKE_DIR/prompt.txt"
sed -n '1p' "$EXP/config/prompts_selected_128.txt" > "$PROMPT"
test -s "$PROMPT"
OUT="$SMOKE_DIR/videos"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 "$V/bin/python" -u "$SF/inference.py" --config_path "$EXP/config/k1_full_19_26.yaml" --checkpoint_path "$WS_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt" --data_path "$PROMPT" --output_folder "$OUT" --num_output_frames 123 --num_samples 1 --seed 0 --use_ema --save_with_index
STEP=validate_generation_output
VIDEO=$(find "$OUT" -type f -name '*.mp4' -print -quit)
test -n "$VIDEO"
test -s "$VIDEO"
command -v ffprobe >/dev/null 2>&1
RAW_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$VIDEO" | tr -d '\r\n')
[[ "$RAW_FRAMES" == "489" ]]
STEP=success
"$V/bin/python" - "$RESULT" "$VIDEO" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
out, video = sys.argv[1:]
p = Path(video)
Path(out).write_text(json.dumps({
  "status": "success", "success": True, "exit_code": 0, "job_id": os.environ.get("SLURM_JOB_ID"),
  "video_path": str(p), "video_bytes": p.stat().st_size, "raw_frames": 489,
  "fixed_num_output_frames": 123, "derived_raw_decoded_frames": 489,
  "checkpoint": "Self-Forcing/checkpoints/self_forcing_dmd.pt", "use_ema": True,
  "active_bundle": "scale_up_outputs/nm5_looped_self_forcing_search_20260912",
  "updated_at": datetime.now(timezone.utc).isoformat()
}, indent=2) + chr(10))
PY
echo "SMOKE_SUCCESS job=$SLURM_JOB_ID raw_frames=$RAW_FRAMES num_output_frames=123 use_ema=true"
