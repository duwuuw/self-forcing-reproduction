#!/usr/bin/env bash
set -Eeuo pipefail
WS_ROOT="$1"
REMOTE_LOG="$2"
ACCOUNT="$3"
PREFIX="$4"
NM5_ROOT="$5"
EXP="$WS_ROOT/scale_up_outputs/nm5_looped_self_forcing_search_20260912"
ENV_ROOT="$WS_ROOT/scale_up_outputs/envs"
CONDA_ROOT="$ENV_ROOT/miniconda3"
ENV_PREFIX="$CONDA_ROOT/envs/looped-self-forcing"
RESULT="$EXP/readiness/nm5_env_install_20260913.json"
SMOKE_RESULT="$EXP/readiness/nm5_env_smoke_result.json"
SMOKE_IMPORT_RESULT="$EXP/readiness/nm5_env_import_smoke_20260913.json"
SMOKE_SCRIPT="$EXP/slurm_scripts/nm5_env_smoke_20260913.sbatch"
SMOKE_JOB_ID=""
STEP="start"
FAILED_COMMAND=""
mkdir -p "$(dirname "$REMOTE_LOG")" "$ENV_ROOT" "$EXP/readiness" "$EXP/logs"
exec > >(tee -a "$REMOTE_LOG") 2>&1
record_result() {
  local status="$1"; local code="$2"; local message="$3"
  python3 - "$RESULT" "$status" "$code" "$STEP" "$message" "$SMOKE_JOB_ID" "$REMOTE_LOG" "$EXP" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, status, code, step, message, job_id, log_path, exp = sys.argv[1:]
payload = {
    "status": status,
    "success": status == "success",
    "exit_code": int(code),
    "failed_step": None if status == "success" else step,
    "message": message,
    "smoke_job_id": job_id or None,
    "log_path": log_path,
    "active_bundle": "scale_up_outputs/nm5_looped_self_forcing_search_20260912",
    "fixed_num_output_frames": 123,
    "use_ema": True,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
Path(out).write_text(json.dumps(payload, indent=2) + chr(10), encoding="utf-8")
PY
}
on_err() { FAILED_COMMAND="$BASH_COMMAND"; }
on_exit() {
  local code=$?
  if [[ "$code" -ne 0 ]]; then
    record_result failure "$code" "step=$STEP command=${FAILED_COMMAND:-unknown}" || true
  fi
  exit "$code"
}
trap on_err ERR
trap on_exit EXIT
echo "INSTALL_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "INSTALL_SESSION=${SESSION:-unset}"
echo "ACTIVE_BUNDLE=nm5_looped_self_forcing_search_20260912"
STEP=wait_for_transfer
required=(looped-self-forcing.tar.gz nm5_miniconda3_base.tar.gz nm5_overlay.tar.gz nm5_cache.tar.gz nm5_generation_assets.tar.gz nm5_env_manifest.json)
while true; do
  if [[ -s "$ENV_ROOT/nm5_env_manifest.json" ]] && python3 - "$ENV_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
required = ["looped-self-forcing.tar.gz", "nm5_miniconda3_base.tar.gz", "nm5_overlay.tar.gz", "nm5_cache.tar.gz", "nm5_generation_assets.tar.gz"]
try:
    data = json.loads((root / "nm5_env_manifest.json").read_text())
    files = data["files"]
    ok = True
    for name in required:
        expected = int(files[name]["size"])
        actual = (root / name).stat().st_size
        if actual != expected or actual <= 0:
            ok = False
    raise SystemExit(0 if ok else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  present=()
  for name in "${required[@]}"; do [[ -s "$ENV_ROOT/$name" ]] && present+=("$name") || true; done
  echo "TRANSFER_WAIT present=${#present[@]}/${#required[@]}"
  sleep 30
done
echo "TRANSFER_SIZE_GATE=OK"
STEP=verify_manifest
python3 - "$ENV_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
required = ["looped-self-forcing.tar.gz", "nm5_miniconda3_base.tar.gz", "nm5_overlay.tar.gz", "nm5_cache.tar.gz", "nm5_generation_assets.tar.gz"]
data = json.loads((root / "nm5_env_manifest.json").read_text())
if data.get("fixed_num_output_frames") != 123:
    raise SystemExit("manifest fixed_num_output_frames is not 123")
for name in required:
    meta = data.get("files", {}).get(name)
    if not meta:
        raise SystemExit(f"manifest missing {name}")
    path = root / name
    size = path.stat().st_size
    if size != int(meta["size"]):
        raise SystemExit(f"size mismatch {name}: {size} != {meta['size']}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    if digest.hexdigest() != meta["sha256"]:
        raise SystemExit(f"sha256 mismatch {name}")
print("MANIFEST_VERIFY=OK archives=5 fixed_num_output_frames=123")
PY
STEP=unpack_miniconda_base
mkdir -p "$CONDA_ROOT"
tar --no-same-owner --no-same-permissions -xzf "$ENV_ROOT/nm5_miniconda3_base.tar.gz" -C "$CONDA_ROOT"
test -x "$CONDA_ROOT/bin/conda"
STEP=unpack_runtime_env
mkdir -p "$ENV_PREFIX"
tar --no-same-owner --no-same-permissions -xzf "$ENV_ROOT/looped-self-forcing.tar.gz" -C "$ENV_PREFIX"
test -x "$ENV_PREFIX/bin/python"
test -f "$ENV_PREFIX/bin/conda-unpack"
"$ENV_PREFIX/bin/python" "$ENV_PREFIX/bin/conda-unpack"
# flash_attn is a hard requirement of wan/modules/attention.py:flash_attention
# on the T2V path, but it is absent from Self-Forcing/requirements.txt and is
# not installable on NM5 (no egress). Ship the prebuilt wheel and install it
# from the local file. The wheel ABI must match the env: torch 2.5.1+cu124 is
# built with _GLIBCXX_USE_CXX11_ABI=0, hence the cxx11abiFALSE wheel.
STEP=install_flash_attn
FLASH_ATTN_WHEEL="$(find "$ENV_ROOT/downloads" -maxdepth 1 -type f -name 'flash_attn-*cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' -print -quit 2>/dev/null || true)"
[[ -n "$FLASH_ATTN_WHEEL" ]] || { echo "INSTALL ERROR: flash_attn wheel missing under $ENV_ROOT/downloads" >&2; exit 1; }
if ! "$ENV_PREFIX/bin/python" -c 'import flash_attn' >/dev/null 2>&1; then
  "$ENV_PREFIX/bin/python" -m pip install --no-deps --no-build-isolation -q "$FLASH_ATTN_WHEEL"
fi
"$ENV_PREFIX/bin/python" -c 'import flash_attn; print("FLASH_ATTN_INSTALL=OK version=%s" % flash_attn.__version__)'
# wandb is required by the workspace's record_vbench_result.py, and the eval
# launcher hard-fails when WANDB_ENTITY/WANDB_API_KEY are unset. It is also
# absent from the curated core requirements. It refuses protobuf major 7, so the
# wheelhouse carries protobuf 6.x as well; every reverse-dependency on protobuf
# in this env is an optional extra, so the downgrade is inert for the rest of
# the stack.
STEP=install_wandb_stack
WANDB_WH="$(find "$ENV_ROOT/downloads" -maxdepth 1 -type d -name 'wandb_wheelhouse' -print -quit 2>/dev/null || true)"
[[ -n "$WANDB_WH" ]] || { echo "INSTALL ERROR: wandb wheelhouse missing under $ENV_ROOT/downloads" >&2; exit 1; }
if ! "$ENV_PREFIX/bin/python" -c 'import wandb' >/dev/null 2>&1; then
  # --no-deps: the curated env already provides click/packaging/platformdirs/
  # pyyaml/requests/urllib3/typing-extensions/psutil, and letting pip resolve
  # would upgrade pydantic and protobuf past their recorded pins.
  "$ENV_PREFIX/bin/python" -m pip install --no-index --no-deps --no-build-isolation -q \
    "$WANDB_WH"/wandb-*.whl \
    "$WANDB_WH"/pydantic-2.10.6-*.whl \
    "$WANDB_WH"/pydantic_core-2.27.2-*.whl \
    "$WANDB_WH"/annotated_types-*.whl \
    "$WANDB_WH"/gitpython-*.whl \
    "$WANDB_WH"/gitdb-*.whl \
    "$WANDB_WH"/smmap-*.whl \
    "$WANDB_WH"/sentry_sdk-*.whl \
    "$WANDB_WH"/setproctitle-*.whl \
    "$WANDB_WH"/protobuf-6.*.whl
fi
"$ENV_PREFIX/bin/python" -c 'import wandb; print("WANDB_INSTALL=OK version=%s" % wandb.__version__)'
STEP=unpack_overlay_cache_assets
tar --no-same-owner --no-same-permissions -xzf "$ENV_ROOT/nm5_overlay.tar.gz" -C "$WS_ROOT"
tar --no-same-owner --no-same-permissions -xzf "$ENV_ROOT/nm5_cache.tar.gz" -C "$WS_ROOT"
tar --no-same-owner --no-same-permissions -xzf "$ENV_ROOT/nm5_generation_assets.tar.gz" -C "$WS_ROOT"
STEP=verify_active_inputs
test -s "$EXP/slurm_scripts/_env.sh"
grep -q 'SUE_NUM_OUTPUT_FRAMES=123' "$EXP/slurm_scripts/_env.sh"
CHECKPOINT="$WS_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt"
WAN_ROOT="$WS_ROOT/Self-Forcing/wan_models/Wan2.1-T2V-1.3B"
test -s "$CHECKPOINT"
test -d "$WAN_ROOT"
find "$WAN_ROOT" -type f -print -quit | grep -q .
actual_ckpt_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
expected_ckpt_sha=a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3
[[ "$actual_ckpt_sha" == "$expected_ckpt_sha" ]]
if find "$WAN_ROOT" -type l ! -exec test -e {} \; -print -quit | grep -q .; then
  echo "broken symlink under Wan base" >&2
  exit 1
fi
test -d "$ENV_ROOT/vbench_cache"
test -d "$WS_ROOT/.cache/torch"
test -s "$WS_ROOT/.cache/torch/checkpoints/dinov2_vitb14_pretrain.pth"
test -s "$WS_ROOT/.cache/home/.cache/ensemble_lora/adapter_config.json"
echo "ACTIVE_INPUTS_VERIFY=OK checkpoint_use_ema=true num_output_frames=123"
STEP=write_env_ready_manifest
python3 - "$EXP/readiness/nm5_env_manifest_verified_20260913.json" "$ENV_ROOT/nm5_env_manifest.json" "$REMOTE_LOG" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, manifest, log = sys.argv[1:]
data = json.loads(Path(manifest).read_text())
Path(out).write_text(json.dumps({
  "status": "verified", "success": True, "verified_at": datetime.now(timezone.utc).isoformat(),
  "manifest": "scale_up_outputs/envs/nm5_env_manifest.json", "fixed_num_output_frames": 123,
  "archive_count": len(data.get("files", {})), "log_path": log
}, indent=2) + chr(10))
PY
STEP=write_smoke_launcher
cat > "$SMOKE_SCRIPT" <<'SMOKE'
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
# inference.py:59 loads configs/default_config.yaml by relative path, so the
# process must start from $SF. The checkpoint is already absolute.
cd "$SF"
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
SMOKE
chmod +x "$SMOKE_SCRIPT"
# A stale result from an earlier attempt would short-circuit wait_for_smoke
# below, so clear both result files before submitting.
rm -f "$SMOKE_RESULT" "$SMOKE_IMPORT_RESULT"
STEP=submit_smoke
SMOKE_JOB_RAW=$(sbatch --parsable --chdir="$WS_ROOT" --account="$ACCOUNT" --partition=acc --qos=acc_debug --gres=gpu:1 --cpus-per-task=20 --time=00:30:00 --job-name="${PREFIX}nm5-env-smoke-20260913" --output="$EXP/logs/nm5_env_smoke_%j.out" --error="$EXP/logs/nm5_env_smoke_%j.err" --export="ALL,SUE_DEEPRESEARCH_ROOT=$NM5_ROOT" "$SMOKE_SCRIPT")
SMOKE_JOB_ID="${SMOKE_JOB_RAW%%;*}"
[[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
echo "SMOKE_SUBMITTED job=$SMOKE_JOB_ID job_name=${PREFIX}nm5-env-smoke-20260913"
STEP=wait_for_smoke
deadline=$(( $(date +%s) + 2400 ))
while true; do
  if [[ -s "$SMOKE_RESULT" ]]; then
    if python3 - "$SMOKE_RESULT" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if data.get("success") is True else 1)
PY
    then
      STEP=success
      record_result success 0 "environment unpacked; manifest verified; import and generation smoke passed"
      echo "INSTALL_SUCCESS job=$SMOKE_JOB_ID"
      exit 0
    else
      echo "SMOKE_RESULT_FAILURE" >&2
      exit 1
    fi
  fi
  state=$(squeue -h -j "$SMOKE_JOB_ID" -o '%T' 2>/dev/null | head -n 1 || true)
  if [[ -z "$state" ]]; then
    acct_state=$(sacct -X -j "$SMOKE_JOB_ID" --format=State%20,ExitCode --parsable2 -n 2>/dev/null | head -n 1 || true)
    if [[ -n "$acct_state" ]]; then
      echo "SMOKE_TERMINAL state=$acct_state"
      case "$acct_state" in COMPLETED\|0:0*) ;; *) echo "SMOKE_JOB_FAILED" >&2; exit 1 ;; esac
    fi
  fi
  if (( $(date +%s) >= deadline )); then echo "SMOKE_TIMEOUT job=$SMOKE_JOB_ID" >&2; exit 124; fi
  echo "SMOKE_WAIT state=${state:-UNKNOWN} job=$SMOKE_JOB_ID"
  sleep 30
done
