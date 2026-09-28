#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/launcher_common.sh"

usage() {
  printf 'usage: %s <pair_id> scale=smoke [overrides...] | <pair_id> scale=full --after-smoke <smoke_pair_id> train.timeout_seconds=<seconds> [overrides...] | --verify-smoke <smoke_pair_id>\n' "${0##*/}" >&2
  exit 2
}

bootstrap_nm5_environment() {
  : "${NM5_DEEPRESEARCH_ROOT:?load the private NM5 configuration first}"
  : "${NM5_ACCOUNT:?load the private NM5 configuration first}"
  : "${NM5_WORKSPACE_ROOT:?load the private NM5 configuration first}"
  [[ -d "$NM5_DEEPRESEARCH_ROOT" && -d "$NM5_WORKSPACE_ROOT" ]] || {
    printf 'NM5 roots from the private backend config are unavailable\n' >&2
    exit 2
  }
  NM5_DEEPRESEARCH_ROOT="$(realpath -e -- "$NM5_DEEPRESEARCH_ROOT")"
  NM5_WORKSPACE_ROOT="$(realpath -e -- "$NM5_WORKSPACE_ROOT")"
  workspace="$NM5_DEEPRESEARCH_ROOT/workspace/looped-flow-matching"
  pipeline="$workspace/scripts/looped_self_forcing_pipeline/pipeline.py"
  [[ -d "$workspace" && -f "$pipeline" ]] || {
    printf 'configured NM5 workspace or pipeline entrypoint is unavailable\n' >&2
    exit 2
  }
  if [[ -z "${SUE_EXP_DIR:-}" ]]; then
    SUE_EXP_DIR="$workspace/scale_up_outputs/looped_self_forcing_pipeline"
  elif [[ "$SUE_EXP_DIR" != /* ]]; then
    SUE_EXP_DIR="$workspace/$SUE_EXP_DIR"
  fi
  [[ -f "$SUE_EXP_DIR/config/config.yaml" && -f "$SUE_EXP_DIR/config/runtime.yaml" ]] || {
    printf 'selected SUE_EXP_DIR lacks the pipeline config contract\n' >&2
    exit 2
  }
  SUE_EXP_DIR="$(realpath -e -- "$SUE_EXP_DIR")"
  configure_nm5_runtime_environment "$NM5_WORKSPACE_ROOT" "$SUE_EXP_DIR"
  SUE_SCRIPTS_DIR="$(realpath -e -- "$script_dir")"
  export NM5_DEEPRESEARCH_ROOT NM5_WORKSPACE_ROOT NM5_ACCOUNT SUE_EXP_DIR SUE_PYTHON SUE_ASSET_ROOT
  export SUE_SCRIPTS_DIR SUE_PYTHONPATH PYTHONPATH
  load_nm5_cache_environment "$SUE_EXP_DIR" "$workspace"
}

if [[ "${1:-}" == --verify-smoke ]]; then
  [[ $# == 2 ]] || usage
  smoke_pair_id=$2
  [[ "$smoke_pair_id" =~ ^[a-z0-9][a-z0-9_-]{0,52}$ ]] || usage
  bootstrap_nm5_environment
  cd "$workspace"
  "$SUE_PYTHON" "$pipeline" record-smoke-readiness --exp-dir "$SUE_EXP_DIR" \
    --smoke-pair-id "$smoke_pair_id"
  exit 0
fi

[[ $# -ge 2 ]] || usage
pair_id=$1
[[ "$pair_id" =~ ^[a-z0-9][a-z0-9_-]{0,52}$ ]] || {
  printf 'pair_id must be 1–53 lowercase letters, digits, _ or -\n' >&2
  exit 2
}
shift
mode=""
after_smoke=""
overrides=()
while (($#)); do
  case "$1" in
    --after-smoke)
      [[ -z "$after_smoke" && $# -ge 2 ]] || usage
      after_smoke=$2
      shift 2
      ;;
    --*)
      usage
      ;;
    *)
      overrides+=("$1")
      shift
      ;;
  esac
done
validate_hydra_overrides train "${overrides[@]}"
for override in "${overrides[@]}"; do
  key=${override%%=*}
  value=${override#*=}
  case "$key" in
    method)
      printf 'method overrides are fixed by this paired comparison\n' >&2
      exit 2
      ;;
    scale)
      mode=$value
      ;;
  esac
done
[[ "$mode" == smoke || "$mode" == full ]] || {
  printf 'paired launch requires an explicit scale=smoke or scale=full\n' >&2
  exit 2
}
if [[ "$mode" == smoke ]]; then
  [[ -z "$after_smoke" ]] || usage
else
  [[ "$after_smoke" =~ ^[a-z0-9][a-z0-9_-]{0,52}$ && "$after_smoke" != "$pair_id" ]] || {
    printf 'full scale requires a distinct --after-smoke pair ID\n' >&2
    exit 2
  }
  has_explicit_timeout=0
  for override in "${overrides[@]}"; do
    [[ "${override%%=*}" == train.timeout_seconds ]] && has_explicit_timeout=1
    if [[ "${override%%=*}" == train.max_steps ]]; then
      printf 'full scale uses the versioned 600-step config; train.max_steps override is not allowed\n' >&2
      exit 2
    fi
  done
  (( has_explicit_timeout == 1 )) || {
    printf 'full scale requires train.timeout_seconds computed from the completed smoke throughput\n' >&2
    exit 2
  }
fi

baseline_method=layerwise_l23_30_k3_lr5gen
low_lr_method=layerwise_l23_30_k2_lr5both
baseline_run_id="${pair_id}-k3-lr5gen"
low_lr_run_id="${pair_id}-k2-lr5both"
[[ "$baseline_run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ && "$low_lr_run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || usage

bootstrap_nm5_environment
cd "$workspace"
if [[ "$mode" == full ]]; then
  "$SUE_PYTHON" "$pipeline" check-smoke-readiness --exp-dir "$SUE_EXP_DIR" \
    --smoke-pair-id "$after_smoke"
  SUE_MEM_PROBE=0
else
  SUE_MEM_PROBE=1
fi
SUE_MEM_PROBE_STEP=3
export SUE_MEM_PROBE SUE_MEM_PROBE_STEP

pair_evidence_paths="$("$SUE_PYTHON" - "$SUE_EXP_DIR" "$workspace" "$pair_id" 2>/dev/null <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
from looped_self_forcing_pipeline.config import resolve_runtime_paths

roots = resolve_runtime_paths(sys.argv[1])
pair_id = sys.argv[3]
print(
    str(roots["artifacts_root"] / "pairs" / pair_id / "submission.json")
    + "\t"
    + str(roots["readiness_root"] / f"layerwise_23_30_{pair_id}.json")
)
PY
)" || {
  printf 'paired evidence roots could not be resolved from runtime.yaml\n' >&2
  exit 2
}
IFS=$'\t' read -r receipt_path readiness_path <<<"$pair_evidence_paths"
[[ ! -e "$receipt_path" && ! -e "$readiness_path" ]] || {
  printf 'paired run ID already has submission or readiness evidence; choose a fresh pair ID\n' >&2
  exit 2
}

# Per-run preflights validate both immutable configs, exact assets, tracking, and
# Slurm MaxTime. The workspace adapter records the redacted static-gate report.
bash "$script_dir/nm5_submit.sh" train "$baseline_run_id" \
  --preflight-only "method=$baseline_method" "${overrides[@]}"
bash "$script_dir/nm5_submit.sh" train "$low_lr_run_id" \
  --preflight-only "method=$low_lr_method" "${overrides[@]}"

preflight_smoke_args=()
if [[ "$mode" == full ]]; then
  preflight_smoke_args+=(--smoke-pair-id "$after_smoke")
fi
bash "$NM5_DEEPRESEARCH_ROOT/scripts/preflight.sh" \
  --workspace-root "$workspace" \
  --adapter scripts/looped_self_forcing_pipeline/preflight.py -- \
  --pair-id "$pair_id" "${preflight_smoke_args[@]}" "${overrides[@]}"

baseline_output="$(bash "$script_dir/nm5_submit.sh" train "$baseline_run_id" \
  "method=$baseline_method" "${overrides[@]}")"
baseline_job_id="${baseline_output##*$'\n'}"
baseline_job_id="${baseline_job_id%%;*}"
[[ "$baseline_job_id" =~ ^[1-9][0-9]*$ ]] || {
  printf 'NM5 baseline submitter did not return a numeric job ID\n' >&2
  exit 1
}
baseline_submitted_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

low_lr_output="$(bash "$script_dir/nm5_submit.sh" train "$low_lr_run_id" \
  --dependency-afterany "$baseline_job_id" "method=$low_lr_method" "${overrides[@]}")"
low_lr_job_id="${low_lr_output##*$'\n'}"
low_lr_job_id="${low_lr_job_id%%;*}"
[[ "$low_lr_job_id" =~ ^[1-9][0-9]*$ ]] || {
  printf 'NM5 low-LR submitter did not return a numeric job ID\n' >&2
  exit 1
}
low_lr_submitted_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$SUE_PYTHON" - "$SUE_EXP_DIR" "$workspace" "$pair_id" \
  "$baseline_method" "$baseline_run_id" "$baseline_job_id" "$baseline_submitted_at" \
  "$low_lr_method" "$low_lr_run_id" "$low_lr_job_id" "$low_lr_submitted_at" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
from looped_self_forcing_pipeline.config import resolve_runtime_paths

exp_dir = Path(sys.argv[1]).resolve()
pair_id = sys.argv[3]
jobs = []
for offset in (4, 8):
    jobs.append(
        {
            "profile": sys.argv[offset],
            "run_id": sys.argv[offset + 1],
            "job_id": sys.argv[offset + 2],
            "submitted_at": sys.argv[offset + 3],
        }
    )
for job in jobs:
    datetime.fromisoformat(job["submitted_at"].replace("Z", "+00:00"))
target = resolve_runtime_paths(exp_dir)["artifacts_root"] / "pairs" / pair_id / "submission.json"
target.parent.mkdir(parents=True, exist_ok=True)
record = {
    "schema_version": 1,
    "pair_id": pair_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "jobs": jobs,
}
encoded = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
descriptor, temporary_name = tempfile.mkstemp(prefix=".submission.", suffix=".tmp", dir=target.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise SystemExit("paired submission receipt already exists; choose a fresh pair ID") from error
finally:
    temporary.unlink(missing_ok=True)
PY

printf 'k3_lr5gen run_id=%s job_id=%s\n' "$baseline_run_id" "$baseline_job_id"
printf 'k2_lr5both run_id=%s job_id=%s dependency=afterany:%s\n' \
  "$low_lr_run_id" "$low_lr_job_id" "$baseline_job_id"
