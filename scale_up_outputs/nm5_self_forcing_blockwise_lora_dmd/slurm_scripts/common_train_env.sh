#!/usr/bin/env bash
# Training-side environment for the block-wise selected-block LoRA DMD run.
#
# Sourced by train_lora_dmd.sbatch. Selects the Self-Forcing-blockwise worktree
# through SUE_SF_DIR *before* sourcing the base env, so _sf_env.sh runs its
# shared-asset guard for the alternative checkout.
set -euo pipefail

: "${SUE_BASE_ENV_SCRIPT:?SUE_BASE_ENV_SCRIPT is required}"
: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
: "${SUE_EXP:?SUE_EXP is required}"
: "${SUE_RUN_ID:?SUE_RUN_ID is required}"
: "${SUE_DATASET_PATH:?SUE_DATASET_PATH is required}"

# Must be exported before the base env script so SF resolves to the block-wise
# worktree and the VBench/checkpoints/wan_models symlink guard fires.
export SUE_SF_DIR="${SUE_SF_DIR:-$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching/Self-Forcing-blockwise}"
source "$SUE_BASE_ENV_SCRIPT"

EXPECTED_WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
[[ "${WS:-}" == "$EXPECTED_WS" ]] || {
  echo "PATH ERROR: base environment resolved an unexpected workspace root" >&2; exit 2; }
case "$SUE_EXP" in
  "$EXPECTED_WS/scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd") ;;
  *) echo "PATH ERROR: experiment root is outside the selected SUE bundle" >&2; exit 2 ;;
esac
[[ "$SF" == "$EXPECTED_WS/Self-Forcing-blockwise" ]] || {
  echo "PATH ERROR: SF did not resolve to the Self-Forcing-blockwise worktree ($SF)" >&2; exit 2; }

export EXP="$SUE_EXP"
export SUE_EXP_DIR="$SUE_EXP"
export PROJECT_SCRIPTS="$EXPECTED_WS/scripts"
export PATH="$V/bin:$PATH"
export PYTHONPATH="$SF:$SF/VBench:$OVERLAY:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONHOME=''
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# NM5's H100 exposes 63.29 GiB usable and the backward reduce-scatter needs a
# transient shard-sized buffer. Expandable segments reclaim allocator
# fragmentation without changing the model or loop semantics.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------- training cfg
export SUE_TRAIN_CONFIG="${SUE_TRAIN_CONFIG:-configs/self_forcing_dmd_temporal_loop_lora_train.yaml}"
export SUE_TRAIN_LOG_ITERS="${SUE_TRAIN_LOG_ITERS:-10}"
export SUE_TRAIN_SECONDS="${SUE_TRAIN_SECONDS:-1200}"
export SUE_TRAIN_MASTER_PORT="${SUE_TRAIN_MASTER_PORT:-29531}"
# Frame/latent contract of the training preset (21 latent frames == 81 pixel
# frames == ~5 s @ 16 fps). Must NOT be confused with the 123-latent VBench-Long
# evaluation protocol used by the search bundle.
export SUE_TRAIN_LATENT_FRAMES=21
# Sample-video logging and the step cap are implemented in the branch-external
# wrapper (scripts/train_blockwise_lora_dmd.py). Native Self-Forcing DMD does not
# sample videos from its training save path, so sampling is opt-in here.
# SUE_MAX_STEPS=0 means "no cap", which leaves the wall-clock bound as the only
# stop condition.
export SUE_MAX_STEPS="${SUE_MAX_STEPS:-0}"
export SUE_SAMPLE_EVERY="${SUE_SAMPLE_EVERY:-0}"
export SUE_SAMPLE_PROMPTS="${SUE_SAMPLE_PROMPTS:-2}"
# Memory probe: on this step the wrapper resets the peak counter and traces the
# rollout / teacher KL / DMD loss separately. SUE_MEM_PROBE=0 disables it.
export SUE_MEM_PROBE="${SUE_MEM_PROBE:-0}"
export SUE_MEM_PROBE_STEP="${SUE_MEM_PROBE_STEP:-3}"

# ------------------------------------------------------------------- wandb/off
# NM5 has no outbound internet. W&B runs must be offline; the offline run
# directory is durable under the run root and synced later from a host with
# egress. WANDB_API_KEY only has to be non-empty; it is ignored while offline.
export WANDB_MODE=offline
export WANDB_API_KEY="${WANDB_API_KEY:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-${NM5_WANDB_ENTITY:-}}"
export WANDB_PROJECT="${WANDB_PROJECT:-looped-self-forcing-blockwise-lora-dmd}"
export WANDB_GROUP="$SUE_RUN_ID"
export WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-blockwise-lora-dmd-$SUE_RUN_ID}"
export WANDB_DIR="$SUE_EXP/$SUE_RUN_ID/wandb"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$EXP/env/wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$EXP/env/wandb_data}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$EXP/env/wandb_config}"
export WANDB_INIT_TIMEOUT=60

RUN_ROOT="$SUE_EXP/$SUE_RUN_ID"
case "$RUN_ROOT" in
  "$SUE_EXP"/*) ;;
  *) echo "PATH ERROR: run root escaped SUE_EXP_DIR" >&2; exit 2 ;;
esac
export RUN_ROOT
export SUE_CKPT_DIR="$RUN_ROOT/checkpoints"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/worker_logs" "$RUN_ROOT/run_state" \
         "$SUE_CKPT_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" \
         "$WANDB_CONFIG_DIR" "$SUE_EXP/logs" "$SUE_EXP/artifacts"

# Sample GPU utilisation into the run root so a post-mortem can tell "the job
# was waiting on the teacher forward" from "the job was idle".
gpu_sample() {
  local label="$1"
  {
    printf '%s,' "$(date -Is)"
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used \
      --format=csv,noheader,nounits 2>/dev/null || true
  } >> "$RUN_ROOT/worker_logs/gpu_util_${SLURM_JOB_ID:-local}_${label}.csv"
}

# --------------------------------------------------------------- preflight
# Fail loudly before the job burns GPU time; every required asset and import
# is asserted here.
sue_train_preflight() {
  local rc=0
  # The branch resolves configs/default_config.yaml and wan_models/... relative
  # to CWD. The prompt corpus is an explicit SUE dataset path so the source
  # checkout and the run bundle cannot silently diverge.
  cd "$SF"
  # SUE_TRAIN_CONFIG is relative for a fullrun (the branch preset) and absolute
  # for a smoke (the derived config under the run root); resolve both without
  # doubling the path.
  local train_cfg="$SUE_TRAIN_CONFIG"
  [[ "$train_cfg" = /* ]] || train_cfg="$SF/$train_cfg"
  for f in \
    "$SF/train.py" \
    "$train_cfg" \
    "$SF/configs/default_config.yaml" \
    "$SUE_DATASET_PATH"; do
    [[ -f "$f" ]] || { echo "TRAIN PREFLIGHT ERROR: missing $f" >&2; rc=1; }
  done
  for d in \
    "$SF/wan_models/Wan2.1-T2V-1.3B" \
    "$SF/wan_models/Wan2.1-T2V-14B"; do
    [[ -d "$d" ]] || { echo "TRAIN PREFLIGHT ERROR: missing $d" >&2; rc=1; }
  done
  [[ -f "$SF/checkpoints/self_forcing_dmd.pt" ]] \
    || { echo "TRAIN PREFLIGHT ERROR: missing $SF/checkpoints/self_forcing_dmd.pt" >&2; rc=1; }
  [[ -x "$V/bin/python" ]] \
    || { echo "TRAIN PREFLIGHT ERROR: missing env interpreter $V/bin/python" >&2; rc=1; }
  [[ $rc -eq 0 ]] || return 2

  # Resolve the generator init checkpoint exactly as the preset does, so a
  # missing init weight fails here rather than after the models are built.
  "$V/bin/python" - <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/default_config.yaml")
cfg = OmegaConf.merge(cfg, OmegaConf.load(__import__("os").environ["SUE_TRAIN_CONFIG"]))
gen = Path(str(cfg.generator_ckpt)).expanduser()
print(f"TRAIN_PREFLIGHT generator_ckpt={gen}")
if not gen.exists():
    print(f"TRAIN PREFLIGHT ERROR: generator checkpoint does not exist: {gen}", file=sys.stderr)
    raise SystemExit(2)
teacher = Path(str(cfg.teacher_checkpoint)).expanduser()
print(f"TRAIN_PREFLIGHT teacher_checkpoint={teacher} real_name={cfg.real_name}")
if not teacher.exists():
    print(f"TRAIN PREFLIGHT ERROR: teacher checkpoint does not exist: {teacher}", file=sys.stderr)
    raise SystemExit(2)

# Training corpus must be a text-to-video prompt corpus, never the evaluation
# prompt sets. VBench / VBench-Long prompts are eval-only in this workspace; a
# training run that trained on them would not be comparable to the search-round
# numbers and would leak the eval set.
data_path = Path(str(cfg.data_path)).expanduser()
print(f"TRAIN_PREFLIGHT data_path={data_path} (cwd-relative to $SF)")
if not data_path.exists():
    print(f"TRAIN PREFLIGHT ERROR: training prompt file does not exist: {data_path}", file=sys.stderr)
    raise SystemExit(2)
low = str(data_path).lower()
for banned in ("vbench", "vbench_long", "vbench-long"):
    if banned in low:
        print(f"TRAIN PREFLIGHT ERROR: training corpus points at the evaluation set ({banned}): {data_path}",
              file=sys.stderr)
        raise SystemExit(2)
lines = sum(1 for ln in data_path.open(encoding="utf-8") if ln.strip())
print(f"TRAIN_PREFLIGHT corpus_prompts={lines}")
if lines < 1000:
    print(f"TRAIN PREFLIGHT ERROR: training corpus has only {lines} non-empty prompts; "
          "DistributedSampler(drop_last=True) on 4 ranks can yield zero batches and hang in cycle()",
          file=sys.stderr)
    raise SystemExit(2)
hits = 0
with data_path.open(encoding="utf-8", errors="replace") as fh:
    for ln in fh:
        if "vbench" in ln.lower():
            hits += 1
print(f"TRAIN_PREFLIGHT corpus_vbench_mentions={hits}")

# Frame contract: num_training_frames drives the sampled-noise shape, but the
# rollout length itself comes from image_or_video_shape[1]. They must agree, and
# the count must tile into whole autoregressive blocks of num_frame_per_block.
n_tf = int(cfg.num_training_frames)
shape_frames = int(list(cfg.image_or_video_shape)[1])
block = int(cfg.num_frame_per_block)
print(f"TRAIN_PREFLIGHT frames: num_training_frames={n_tf} image_or_video_shape[1]={shape_frames} "
      f"num_frame_per_block={block} grad_window={cfg.training_gradient_window_frames}")
if n_tf != shape_frames:
    print(f"TRAIN PREFLIGHT ERROR: num_training_frames ({n_tf}) != image_or_video_shape[1] ({shape_frames})",
          file=sys.stderr)
    raise SystemExit(2)
if n_tf % block != 0:
    print(f"TRAIN PREFLIGHT ERROR: num_training_frames ({n_tf}) is not divisible by "
          f"num_frame_per_block ({block})", file=sys.stderr)
    raise SystemExit(2)
# The branch rejects the reverse inequality at pipeline/self_forcing_training.py:93.
# Upstream keeps window == frames, so start_gradient_frame_index == 0 and every AR
# chunk is gradient-enabled; keep that relationship.
if int(cfg.training_gradient_window_frames) > n_tf:
    print(f"TRAIN PREFLIGHT ERROR: training_gradient_window_frames "
          f"({cfg.training_gradient_window_frames}) > num_training_frames ({n_tf}); the branch "
          "rejects this at self_forcing_training.py:93", file=sys.stderr)
    raise SystemExit(2)
if int(cfg.training_gradient_window_frames) != n_tf:
    print(f"TRAIN PREFLIGHT WARNING: training_gradient_window_frames "
          f"({cfg.training_gradient_window_frames}) != num_training_frames ({n_tf}); upstream "
          "keeps them equal so every AR chunk is graded", file=sys.stderr)
if int(cfg.min_training_frames) != n_tf:
    print(f"TRAIN PREFLIGHT ERROR: min_training_frames ({cfg.min_training_frames}) must equal "
          f"num_training_frames ({n_tf}) to keep the rollout length fixed", file=sys.stderr)
    raise SystemExit(2)
# The preset defaults these to literal placeholders; a DMD/LoRA run must not
# start with an unresolved host/key.
for key in ("wandb_host", "wandb_key", "wandb_entity", "wandb_project"):
    val = str(cfg.get(key, ""))
    if val in ("WANDB_HOST", "WANDB_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        print(f"TRAIN PREFLIGHT WARNING: {key} is still the unresolved preset placeholder", file=sys.stderr)
print(f"TRAIN_PREFLIGHT temporal_loop={OmegaConf.to_container(cfg.temporal_loop, resolve=True)}")
print(f"TRAIN_PREFLIGHT lora={OmegaConf.to_container(cfg.lora, resolve=True)}")
PY

  "$V/bin/python" - <<'PY'
import sys
try:
    import torch, peft, wandb
except Exception as exc:
    print(f"TRAIN PREFLIGHT ERROR: import failed: {exc}", file=sys.stderr)
    raise SystemExit(2)
try:
    import flash_attn
except Exception as exc:
    print(f"TRAIN PREFLIGHT ERROR: import flash_attn failed: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not torch.cuda.is_available():
    print("TRAIN PREFLIGHT ERROR: CUDA unavailable", file=sys.stderr)
    raise SystemExit(2)
print("TRAIN_PREFLIGHT ok torch=%s cuda=%s gpus=%d peft=%s wandb=%s flash_attn=%s"
      % (torch.__version__, torch.version.cuda, torch.cuda.device_count(),
         peft.__version__, wandb.__version__,
         getattr(flash_attn, "__version__", "available")), flush=True)
PY
}
