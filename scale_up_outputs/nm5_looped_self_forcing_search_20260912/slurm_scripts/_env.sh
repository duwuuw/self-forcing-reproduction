#!/usr/bin/env bash
set -euo pipefail

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
# Self-Forcing source tree. Defaults to the original tree; override with
# SUE_SF_DIR to run an alternative checkout (e.g. the layer-wise
# selected-layer stack loop tree) without touching the default one.
SF="${SUE_SF_DIR:-$WS/Self-Forcing}"
ENV_ROOT="$WS/scale_up_outputs/envs"
V="$ENV_ROOT/miniconda3/envs/looped-self-forcing"
OVERLAY="$ENV_ROOT/overlay310"
VBENCH_CACHE_DIR="$ENV_ROOT/vbench_cache"
SUE_LOCAL_BIN="$ENV_ROOT/bin"
CACHE_ROOT="$WS/.cache"

export WS SF ENV_ROOT V OVERLAY VBENCH_CACHE_DIR SUE_LOCAL_BIN CACHE_ROOT
export HOME="$CACHE_ROOT/home"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$CACHE_ROOT/torch"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export XDG_CACHE_HOME="$HOME/.cache"
export WANDB_MODE=offline
export SUE_NUM_OUTPUT_FRAMES=123
export PYTHONNOUSERSITE=1
export PYTHONHOME=''
export PYTHONUNBUFFERED=1
export PATH="$V/bin:$SUE_LOCAL_BIN:$PATH"
export PYTHONPATH="$SF:$SF/VBench:$OVERLAY:${PYTHONPATH:-}"

mkdir -p "$HOME/.cache" "$HF_HUB_CACHE" "$TORCH_HOME" "$MODELSCOPE_CACHE" \
  "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$SUE_LOCAL_BIN"

static_ffmpeg="$(find "$OVERLAY/imageio_ffmpeg/binaries" -maxdepth 1 -type f -name 'ffmpeg-linux-x86_64*' -print -quit 2>/dev/null || true)"
if [[ -n "$static_ffmpeg" ]]; then
  ln -sfn "$static_ffmpeg" "$SUE_LOCAL_BIN/ffmpeg"
fi

# Sourced by every launcher.  Call `module load ffmpeg/7.0.1-gcc` first: the
# NM5 module has no libx264, so ffmpeg must resolve to the static
# imageio-ffmpeg build while ffprobe still comes from the module
# (imageio-ffmpeg ships no ffprobe).  Rebuilding the link here makes the
# launchers self-healing across env re-unpacks.
sue_tools() {
  local static_ffmpeg
  static_ffmpeg="$(find "$OVERLAY/imageio_ffmpeg/binaries" -maxdepth 1 -type f -name 'ffmpeg-linux-x86_64*' -print -quit 2>/dev/null || true)"
  [[ -n "$static_ffmpeg" ]] || {
    echo "PREFLIGHT ERROR: no static ffmpeg under $OVERLAY/imageio_ffmpeg/binaries" >&2
    return 2
  }
  chmod +x "$static_ffmpeg" 2>/dev/null || true
  mkdir -p "$SUE_LOCAL_BIN"
  ln -sfn "$static_ffmpeg" "$SUE_LOCAL_BIN/ffmpeg"
  export PATH="$SUE_LOCAL_BIN:$PATH"
  # Collect the encoder list first.  Do NOT pipe ffmpeg into `grep -q` under
  # `set -o pipefail`: grep exits at the first match, ffmpeg then dies on
  # SIGPIPE, and pipefail reports that as a failure even though libx264 is
  # present.
  local encoders
  encoders="$("$SUE_LOCAL_BIN/ffmpeg" -hide_banner -encoders 2>/dev/null || true)"
  grep -q libx264 <<<"$encoders" \
    || { echo "PREFLIGHT ERROR: static ffmpeg lacks libx264 ($(command -v ffmpeg))" >&2; return 2; }
  command -v ffprobe >/dev/null \
    || { echo "PREFLIGHT ERROR: ffprobe not on PATH (module load ffmpeg first)" >&2; return 2; }
}

# Fail loudly on the allocation, before spending GPU time, when a required
# asset or runtime import is missing.  flash_attn is a hard requirement of
# wan/modules/attention.py:flash_attention on the T2V path.
sue_preflight() {
  local rc=0
  for f in \
    "$SF/inference.py" \
    "$SF/configs/default_config.yaml" \
    "$SF/checkpoints/self_forcing_dmd.pt"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT ERROR: missing $f" >&2; rc=1; }
  done
  [[ -d "$SF/wan_models/Wan2.1-T2V-1.3B" ]] \
    || { echo "PREFLIGHT ERROR: missing $SF/wan_models/Wan2.1-T2V-1.3B" >&2; rc=1; }
  [[ -d "$OVERLAY/av" ]] \
    || { echo "PREFLIGHT ERROR: missing overlay av at $OVERLAY/av" >&2; rc=1; }
  [[ -x "$V/bin/python" ]] \
    || { echo "PREFLIGHT ERROR: missing env interpreter $V/bin/python" >&2; rc=1; }
  [[ $rc -eq 0 ]] || return 2
  "$V/bin/python" - <<'PY'
import sys
import torch

try:
    import av
except Exception as exc:  # PyAV is required by torchvision.io.write_video
    print(f"PREFLIGHT ERROR: import av failed: {exc}", file=sys.stderr)
    raise SystemExit(2)
try:
    import flash_attn
except Exception as exc:
    print(f"PREFLIGHT ERROR: import flash_attn failed: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not torch.cuda.is_available():
    print("PREFLIGHT ERROR: CUDA unavailable", file=sys.stderr)
    raise SystemExit(2)
print(
    "PREFLIGHT ok torch=%s cuda=%s gpus=%d av=%s flash_attn=%s"
    % (
        torch.__version__,
        torch.version.cuda,
        torch.cuda.device_count(),
        av.__version__,
        getattr(flash_attn, "__version__", "available"),
    ),
    flush=True,
)
PY
}
