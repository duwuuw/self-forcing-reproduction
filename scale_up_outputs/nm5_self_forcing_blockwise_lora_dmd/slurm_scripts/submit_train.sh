#!/bin/bash -l
# Submitter for the block-wise LoRA DMD training job.
#
# Requires: SUE_DEEPRESEARCH_ROOT, SUE_RUN_ID, SUE_ACCOUNT, SUE_PARTITION,
#           SUE_QOS.  Normally invoked through nm5_train_submit.sh, which
#           resolves the private values on the execution plane.
#
# Usage: submit_train.sh <smoke|fullrun>
set -euo pipefail

: "${SUE_DEEPRESEARCH_ROOT:?SUE_DEEPRESEARCH_ROOT is required}"
: "${SUE_RUN_ID:?SUE_RUN_ID is required}"
: "${SUE_ACCOUNT:?SUE_ACCOUNT is required}"
: "${SUE_PARTITION:?SUE_PARTITION is required}"
: "${SUE_QOS:?SUE_QOS is required}"
: "${SUE_DATASET_PATH:?SUE_DATASET_PATH is required}"

[[ "$#" -eq 1 ]] || { echo "SUBMISSION ERROR: pass exactly one mode (smoke|fullrun)" >&2; exit 2; }
SUE_MODE="$1"
case "$SUE_MODE" in
  smoke|fullrun) ;;
  *) echo "MODE ERROR: expected smoke or fullrun, got $SUE_MODE" >&2; exit 2 ;;
esac
[[ "$SUE_RUN_ID" =~ ^[a-z0-9_]+$ ]] || { echo "RUN_ID ERROR: $SUE_RUN_ID" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUE_BASE_ENV_SCRIPT="${SUE_BASE_ENV_SCRIPT:-$SCRIPT_DIR/_sf_env.sh}"
export SUE_SF_DIR="${SUE_SF_DIR:-$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching/Self-Forcing-blockwise}"
export SUE_EXP="${SUE_EXP:-$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching/scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd}"
export SUE_SCRIPTS_DIR="$SCRIPT_DIR"

EXPECTED_WS="$SUE_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
EXPECTED_WS_REAL="$(realpath -e "$EXPECTED_WS")" || {
  echo "PATH ERROR: workspace root does not resolve: $EXPECTED_WS" >&2; exit 2; }
[[ "$(pwd -P)" == "$EXPECTED_WS_REAL" ]] || {
  echo "PATH ERROR: submit from the workspace root ($EXPECTED_WS) only" >&2; exit 2; }

# Resolves $WS/$SF/$V, the shared caches, and runs the alternative-checkout
# asset guard (fails loudly here rather than inside the allocation).
# shellcheck disable=SC1090
source "$SUE_BASE_ENV_SCRIPT"
[[ -n "${V:-}" && -x "$V/bin/python" ]] || {
  echo "PATH ERROR: base env did not provide an interpreter at \$V/bin/python" >&2; exit 2; }

# ------------------------------------------------------------------ defaults
if [[ "$SUE_MODE" == smoke ]]; then
  SUE_TRAIN_SECONDS="${SUE_TRAIN_SECONDS:-2400}"
  SUE_MAX_STEPS="${SUE_MAX_STEPS:-30}"
  # Dryrun scale: cap the real training loop at 30 optimizer steps. The
  # objective is untouched; callers may add further whitelisted overrides (see
  # ALLOWED below) to isolate a single variable.
  SUE_TRAIN_OVERRIDES="${SUE_TRAIN_OVERRIDES:-log_iters=10}"
  TIME_LIMIT="${SUE_TIME_LIMIT:-02:00:00}"
else
  # The measured steady-state probe rate is about 30 s/step; 600 steps need
  # roughly 5.1 h before startup/checkpoint overhead.  Keep the launcher-side
  # timeout below the six-hour Slurm allocation so it can write evidence.
  SUE_TRAIN_SECONDS="${SUE_TRAIN_SECONDS:-19800}"
  SUE_MAX_STEPS="${SUE_MAX_STEPS:-600}"
  # A fullrun uses the branch preset's training semantics and runs 600 steps
  # unless the operator supplies an explicit cap.
  SUE_TRAIN_OVERRIDES="${SUE_TRAIN_OVERRIDES:-}"
  TIME_LIMIT="${SUE_TIME_LIMIT:-06:00:00}"
fi
# Keep the submit-time export contract identical to the training-side
# defaults.  These variables are consumed by the sbatch wrapper, but must also
# be defined here because this script runs with `set -u` on the login node.
export SUE_SAMPLE_EVERY="${SUE_SAMPLE_EVERY:-0}"
export SUE_SAMPLE_PROMPTS="${SUE_SAMPLE_PROMPTS:-2}"
export SUE_MEM_PROBE="${SUE_MEM_PROBE:-0}"
export SUE_MEM_PROBE_STEP="${SUE_MEM_PROBE_STEP:-3}"
export SUE_TRAIN_SECONDS SUE_MAX_STEPS SUE_TRAIN_OVERRIDES

RUN_ROOT="$SUE_EXP/$SUE_RUN_ID"
case "$RUN_ROOT" in
  "$SUE_EXP"/*) ;;
  *) echo "PATH ERROR: run root escaped SUE_EXP" >&2; exit 2 ;;
esac
mkdir -p "$RUN_ROOT/config" "$RUN_ROOT/state" "$SUE_EXP/logs" "$SUE_EXP/artifacts"

# Canonical Slurm prefix resolver: user.yaml only; missing/invalid -> silly-.
SUE_SLURM_JOB_PREFIX="$(SUE_WORKSPACE="$EXPECTED_WS" "$V/bin/python" - <<'PY'
import os, re, sys
from pathlib import Path
workspace = Path(os.environ["SUE_WORKSPACE"])
username = ""
try:
    import yaml
    loaded = yaml.safe_load((workspace / "user.yaml").read_text(encoding="utf-8")) or {}
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
case "$SUE_SLURM_JOB_PREFIX" in
  *-) ;;
  *) echo "JOB NAME ERROR: prefix must end with '-'" >&2; exit 2 ;;
esac

# ------------------------------------------------------- effective train config
# The preset is the contract. Only keys on the explicit ALLOWED whitelist below
# may be derived, and the derivation fails closed unless the diff is exactly the
# requested overrides. The derived file is written into the run root so the exact
# configuration used is recorded next to the run.
PRESET="$SUE_SF_DIR/configs/self_forcing_dmd_temporal_loop_lora_train.yaml"
[[ -f "$PRESET" ]] || { echo "CONFIG ERROR: missing preset $PRESET" >&2; exit 2; }
SUE_TRAIN_CONFIG_REL="configs/self_forcing_dmd_temporal_loop_lora_train.yaml"
if [[ -n "${SUE_TRAIN_OVERRIDES// /}" || -n "${SUE_DATASET_PATH}" ]]; then
  "$V/bin/python" - \
      "$PRESET" "$RUN_ROOT/config/train_config.yaml" "$SUE_MODE" \
      $SUE_TRAIN_OVERRIDES "data_path=$SUE_DATASET_PATH" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Only keys that cannot change the optimisation objective are derivable here.
# Anything touching training semantics (layer range, K, LoRA target, the
# generator/teacher checkpoints) is deliberately NOT on this list and must be
# approved and edited in the branch preset itself.
def _as_bool(raw: str) -> bool:
    # A plain str() here would produce the STRING 'false', which is truthy in
    # Python and would silently leave activation checkpointing ENABLED while the
    # derived config claimed otherwise.
    low = raw.strip().lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    raise SystemExit(f"CONFIG ERROR: {raw!r} is not a boolean")


ALLOWED = {
    "batch_size": int,              # bs_per_gpu; one prompt per GPU is the current preset
    "data_path": str,               # canonical SUE dataset root
    "log_iters": int,                # checkpoint / W&B cadence only
    "gradient_checkpointing": _as_bool,   # activation recompute; memory-vs-speed only
    "text_encoder_cpu_offload": _as_bool,  # where the frozen T5 lives; placement only
}

preset_path, out_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
overrides = {}
for item in sys.argv[4:]:
    if "=" not in item:
        raise SystemExit(f"CONFIG ERROR: override {item!r} is not key=value")
    key, _, raw = item.partition("=")
    if key in overrides:
        raise SystemExit(f"CONFIG ERROR: duplicate override for {key}")
    if key not in ALLOWED:
        raise SystemExit(
            f"CONFIG ERROR: {key!r} is not a derivable key; allowed={sorted(ALLOWED)}"
        )
    overrides[key] = ALLOWED[key](raw)

preset = OmegaConf.load(preset_path)
before = OmegaConf.to_container(preset, resolve=False)
derived = OmegaConf.create(OmegaConf.to_container(preset, resolve=False))
applied = {}
for key, value in overrides.items():
    if before.get(key) == value:
        continue
    derived[key] = value
    applied[key] = {"from": before.get(key), "to": value}

if not applied:
    print(f"CONFIG {mode}: overrides already match the preset, using it as-is")
    raise SystemExit(0)

after = OmegaConf.to_container(derived, resolve=False)
diff = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
if diff != set(applied):
    raise SystemExit(
        f"CONFIG ERROR: derived {mode} config changed unexpected keys: {sorted(diff - set(applied))}"
    )

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(derived, out_path)
print(f"CONFIG derived_{mode}_config={out_path}")
for key, change in applied.items():
    print(f"CONFIG   {key}: {change['from']} -> {change['to']}")
PY
  SUE_TRAIN_CONFIG_REL="$RUN_ROOT/config/train_config.yaml"
  echo "SUBMIT_NOTE: derived $SUE_MODE config from the preset; overrides=[$SUE_TRAIN_OVERRIDES]"
fi
export SUE_TRAIN_CONFIG="$SUE_TRAIN_CONFIG_REL"

# --------------------------------------------------------------------- job name
job_body="${WANDB_EXPERIMENT_NAME:-blockwise-lora-dmd}-${SUE_RUN_ID}-${SUE_MODE}"
job_body="${job_body:0:70}"
JOB_NAME="${SUE_SLURM_JOB_PREFIX}${job_body}"

export_args="ALL,SUE_SCRIPTS_DIR=$SCRIPT_DIR,SUE_BASE_ENV_SCRIPT=$SUE_BASE_ENV_SCRIPT,SUE_DEEPRESEARCH_ROOT=$SUE_DEEPRESEARCH_ROOT,SUE_EXP=$SUE_EXP,SUE_RUN_ID=$SUE_RUN_ID,SUE_SF_DIR=$SUE_SF_DIR,SUE_DATASET_PATH=$SUE_DATASET_PATH,SUE_TRAIN_CONFIG=$SUE_TRAIN_CONFIG,SUE_TRAIN_SECONDS=$SUE_TRAIN_SECONDS,SUE_MAX_STEPS=$SUE_MAX_STEPS,SUE_SAMPLE_EVERY=$SUE_SAMPLE_EVERY,SUE_SAMPLE_PROMPTS=$SUE_SAMPLE_PROMPTS,SUE_MEM_PROBE=$SUE_MEM_PROBE,SUE_MEM_PROBE_STEP=$SUE_MEM_PROBE_STEP,SUE_TRAIN_MASTER_PORT=${SUE_TRAIN_MASTER_PORT:-29531},SUE_SLURM_JOB_PREFIX=$SUE_SLURM_JOB_PREFIX,WANDB_MODE=offline,WANDB_PROJECT=${WANDB_PROJECT:-looped-self-forcing-blockwise-lora-dmd},WANDB_GROUP=$SUE_RUN_ID"

echo "SUBMIT_START run_id=$SUE_RUN_ID mode=$SUE_MODE qos=$SUE_QOS time=$TIME_LIMIT name=$JOB_NAME"
if [[ "${SUE_PLAN_ONLY:-0}" == "1" ]]; then
  echo "PLAN-ONLY: would run sbatch --account=$SUE_ACCOUNT --partition=$SUE_PARTITION --qos=$SUE_QOS --gres=gpu:4 --cpus-per-task=80 --time=$TIME_LIMIT --job-name=$JOB_NAME $SCRIPT_DIR/train_lora_dmd.sbatch"
  echo "PLAN-ONLY export=$export_args"
  exit 0
fi

job_id="$(sbatch --parsable \
  --chdir="$EXPECTED_WS" \
  --account="$SUE_ACCOUNT" \
  --partition="$SUE_PARTITION" \
  --qos="$SUE_QOS" \
  --gres=gpu:4 \
  --cpus-per-task=80 \
  --time="$TIME_LIMIT" \
  --job-name="$JOB_NAME" \
  --output="$SUE_EXP/logs/${SUE_RUN_ID}_${SUE_MODE}_%j.out" \
  --error="$SUE_EXP/logs/${SUE_RUN_ID}_${SUE_MODE}_%j.err" \
  --export="$export_args" \
  "$SCRIPT_DIR/train_lora_dmd.sbatch")"
job_id="${job_id%%;*}"

"$V/bin/python" - \
    "$RUN_ROOT/state/launch_manifest.json" "$SUE_RUN_ID" "$SUE_MODE" "$job_id" \
    "$JOB_NAME" "$SUE_QOS" "$TIME_LIMIT" "$SUE_TRAIN_SECONDS" "$SUE_TRAIN_CONFIG" \
    "$SUE_ACCOUNT" "$SUE_PARTITION" "$SUE_SF_DIR" "$SUE_TRAIN_OVERRIDES" <<'PY'
import json, sys
from pathlib import Path
(manifest, run_id, mode, job_id, job_name, qos, time_limit,
 seconds, config_path, account, partition, sf_dir, train_overrides) = sys.argv[1:]
payload = {
    "run_id": run_id, "mode": mode, "slurm_job_id": job_id, "slurm_job_name": job_name,
    "qos": qos, "time_limit": time_limit, "train_seconds": int(seconds),
    "train_config": config_path, "account": account, "partition": partition,
    "sf_dir": sf_dir, "train_overrides": train_overrides,
}
p = Path(manifest)
p.parent.mkdir(parents=True, exist_ok=True)
existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
existing.append(payload)
p.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "SUBMIT_OK job_id=$job_id name=$JOB_NAME"
echo "SUBMIT_LOG=$SUE_EXP/logs/${SUE_RUN_ID}_${SUE_MODE}_${job_id}.out"
