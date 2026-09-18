#!/bin/bash -l
#SBATCH --partition=acc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
# Qualitative three-way rollout for one prompt: same 8..15 window, same K=2 for
# both looped arms, so the only intended differences are the loop operator and
# whether the loop engages at all.
#
#   baseline   K=1  -> temporal_loop fast path (loop_k==1, strength==1.0): no looping
#   blockwise  K=2  -> Self-Forcing/            per-block loop
#   layerwise  K=2  -> Self-Forcing-layerwise/  contiguous stack as one loop body
#
# Writes exactly one mp4 per arm under $SUE_EXP/$SUE_RUN_ID/<arm>/.
set -euo pipefail

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
: "${SUE_EXP:?SUE_EXP is required}"
: "${SUE_RUN_ID:?SUE_RUN_ID is required}"
: "${SUE_BASE_ENV_SCRIPT:?SUE_BASE_ENV_SCRIPT is required}"

SCRIPT_DIR="${SUE_SCRIPTS_DIR:-}"
if [[ -z "$SCRIPT_DIR" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[[ -r "$SCRIPT_DIR/common_env.sh" ]] || {
  echo "PATH ERROR: common_env.sh not found under SCRIPT_DIR=$SCRIPT_DIR" >&2
  exit 2
}
source "$SCRIPT_DIR/common_env.sh"

module load ffmpeg/7.0.1-gcc
sue_tools
sue_preflight
verify_allocation

LAYERWISE_TREE="$WS/Self-Forcing-layerwise"
# The stacked-loop checkout is a git clone carrying only tracked source; its
# large assets stay in the default tree. Mirror the _env.sh guard so a missing
# symlink fails here with the fix rather than deep inside inference.py.
for asset in VBench checkpoints wan_models; do
  if [[ ! -e "$LAYERWISE_TREE/$asset" && -e "$WS/Self-Forcing/$asset" ]]; then
    echo "SF ASSET ERROR: $LAYERWISE_TREE/$asset is not reachable." >&2
    echo "  ln -sfn ../Self-Forcing/$asset \"$LAYERWISE_TREE/$asset\"" >&2
    exit 2
  fi
done
unset asset

VIZ_ROOT="$SUE_EXP/$SUE_RUN_ID"
PROMPT_FILE="$VIZ_ROOT/prompt.txt"
mkdir -p "$VIZ_ROOT/logs"

# One dynamic VBench-Long prompt, refined so the action stays continuous for the
# whole 30 s: a loop that freezes motion shows up as a visibly static clip.
cat > "$PROMPT_FILE" <<'PROMPT'
A dynamic, continuous action scene of two medieval knights duelling in a spacious, dimly lit stone arena. Both knights are fully armoured in shining steel plate and wield large ornate swords. They circle each other and exchange rapid, forceful blows without pause, parrying, lunging, sidestepping and swinging in a steady back-and-forth rhythm, with bright sparks scattering from every clash of the blades. One knight presses forward while the other retreats and counterattacks, their cloaks and armour plates shifting with every stride. In the background, spectators in period clothing cheer from the stands and long banners hang from the stone walls. Medium shot tracking the fighters from the side, keeping both knights and their swinging swords continuously in frame.
PROMPT

prompt_lines="$(wc -l < "$PROMPT_FILE")"
[[ "$prompt_lines" -eq 1 ]] || { echo "PROMPT ERROR: expected 1 line, found $prompt_lines" >&2; exit 2; }

run_arm() {
  local arm="$1" gpu="$2" tree="$3" config="$4"
  local sf="$WS/$tree"
  local out="$VIZ_ROOT/$arm"
  [[ -f "$SUE_EXP/config/$config" ]] || { echo "CONFIG ERROR: missing $SUE_EXP/config/$config" >&2; return 2; }
  mkdir -p "$out"
  echo "VIZ_START $(date -Is) arm=$arm gpu=$gpu tree=$tree config=$config num_output_frames=123"
  # inference.py loads configs/default_config.yaml by relative path, so the
  # process has to start from $SF.
  (
    cd "$sf"
    CUDA_VISIBLE_DEVICES="$gpu" "$V/bin/python" -u "$sf/inference.py" \
      --config_path "$SUE_EXP/config/$config" \
      --checkpoint_path "$sf/checkpoints/self_forcing_dmd.pt" \
      --data_path "$PROMPT_FILE" \
      --output_folder "$out" \
      --num_output_frames 123 \
      --num_samples 1 --seed 0 --use_ema --save_with_index
  ) > "$VIZ_ROOT/logs/${arm}.log" 2>&1
  echo "VIZ_DONE $(date -Is) arm=$arm"
}

run_arm baseline  0 Self-Forcing           viz_baseline.yaml &
pid_baseline=$!
run_arm blockwise 1 Self-Forcing           k2_l08_15.yaml &
pid_blockwise=$!
run_arm layerwise 2 Self-Forcing-layerwise k2_s08_15.yaml &
pid_layerwise=$!

rc=0
for pair in "baseline:$pid_baseline" "blockwise:$pid_blockwise" "layerwise:$pid_layerwise"; do
  arm="${pair%%:*}"
  pid="${pair##*:}"
  if wait "$pid"; then
    echo "VIZ_OK arm=$arm"
  else
    status=$?
    echo "VIZ_FAIL arm=$arm rc=$status (see $VIZ_ROOT/logs/${arm}.log)" >&2
    rc=1
  fi
done

for arm in baseline blockwise layerwise; do
  count="$(find "$VIZ_ROOT/$arm" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  if [[ "$count" -ne 1 ]]; then
    echo "VALIDATION ERROR: arm=$arm produced $count mp4 file(s), expected 1" >&2
    rc=1
  else
    echo "VIZ_ARTIFACT arm=$arm $(find "$VIZ_ROOT/$arm" -maxdepth 1 -type f -name '*.mp4' -print -quit)"
  fi
done

if [[ "$rc" -eq 0 ]]; then
  echo "VIZ_ALL_DONE $(date -Is) job=$SLURM_JOB_ID root=$VIZ_ROOT"
else
  echo "VIZ_ERROR $(date -Is) job=$SLURM_JOB_ID root=$VIZ_ROOT" >&2
fi
exit "$rc"
