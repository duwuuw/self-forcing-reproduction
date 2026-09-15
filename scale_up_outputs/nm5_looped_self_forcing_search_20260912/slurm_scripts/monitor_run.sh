#!/bin/bash -l
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: monitor_run.sh <exp> <run_id> <generation_job> <evaluation_job> <mode> <interval_seconds>" >&2
  exit 2
fi
EXP="$1"
RUN_ID="$2"
GEN_JOB="$3"
EVAL_JOB="$4"
MODE="$5"
INTERVAL="$6"
RUN_ROOT="$EXP/$RUN_ID"
mkdir -p "$RUN_ROOT/monitor_blockers"

state_for() {
  local job_id="$1"
  local state
  state="$(sacct -j "$job_id" --format=State%30 -n -X 2>/dev/null | head -n 1 | awk '{print $1}' | tr -d ' ' || true)"
  if [[ -z "$state" ]]; then
    if squeue -j "$job_id" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "$job_id" -h -o '%T' 2>/dev/null)" ]]; then
      printf 'RUNNING'
    else
      printf 'UNKNOWN'
    fi
  else
    printf '%s' "$state" | sed 's/+.*$//; s/[^A-Z_].*$//'
  fi
}

terminal_failure() {
  case "$1" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|BOOT_FAIL|NODE_FAIL|DEPENDENCYNEVERSATISFIED|PREEMPTED) return 0 ;;
    *) return 1 ;;
  esac
}

evidence_line() {
  local strict="$1"
  local ledger="$RUN_ROOT/state/tasks.jsonl"
  local variants_root="$RUN_ROOT/variants"
  local csv_path="$EXP/artifacts/experiment_results.csv"
  local history_path="$EXP/artifacts/eval_results_history.md"
  local readiness_path="$EXP/readiness/readiness_nm5.json"
  if ! command -v python3 >/dev/null 2>&1; then
    printf 'verdict=unknown reason=python3_unavailable'
    return 0
  fi
  python3 - "$ledger" "$variants_root" "$csv_path" "$history_path" "$readiness_path" "$RUN_ID" "$MODE" "$strict" <<'PY'
import csv
import json
import sys
import time
from pathlib import Path

ledger_path, variants_root, csv_path, history_path, readiness_path, run_id, mode, strict = sys.argv[1:]
rows = []
path = Path(ledger_path)
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
counts = {}
for row in rows:
    state = str(row.get("state", "unknown"))
    counts[state] = counts.get(state, 0) + 1
run_metrics = []
for metrics_path in Path(variants_root).glob("*/eval/metrics.json"):
    try:
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if value.get("run_id") == run_id:
        run_metrics.append(metrics_path)
csv_rows = 0
csv_file = Path(csv_path)
if csv_file.exists():
    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("wandb_group") == run_id:
                csv_rows += 1
marker = f"<!-- eval:{run_id}:"
history_markers = 0
history_file = Path(history_path)
if history_file.exists():
    history_markers = sum(
        1
        for line in history_file.read_text(encoding="utf-8").splitlines()
        if line.startswith(marker)
    )
readiness_ok = False
readiness_file = Path(readiness_path)
if mode != "dryrun":
    readiness_ok = True
elif readiness_file.exists():
    try:
        readiness_ok = bool(
            json.loads(readiness_file.read_text(encoding="utf-8")).get("fullrun_ready")
        )
    except (OSError, json.JSONDecodeError):
        readiness_ok = False
now = time.time()
stale_active = 0
for row in rows:
    if row.get("state") not in {"queued", "running"}:
        continue
    heartbeat = row.get("last_heartbeat")
    if not heartbeat:
        continue
    try:
        stamp = heartbeat.replace("Z", "+00:00")
        from datetime import datetime
        age = now - datetime.fromisoformat(stamp).timestamp()
        stale_active += age > 900
    except (TypeError, ValueError):
        stale_active += 1
ready = (
    counts.get("done", 0) == 8
    and len(run_metrics) == 4
    and csv_rows >= 4
    and history_markers >= 4
    and readiness_ok
    and stale_active == 0
)
line = (
    f"verdict={'ready' if ready else 'incomplete'} "
    f"tasks_done={counts.get('done', 0)} tasks_failed={counts.get('failed', 0)} "
    f"tasks_total={len(rows)} metrics={len(run_metrics)} csv_rows={csv_rows} "
    f"history_markers={history_markers} readiness={'ok' if readiness_ok else 'missing'} "
    f"stale_active={stale_active}"
)
print(line)
if strict == "1" and not ready:
    raise SystemExit(1)
PY
}

while true; do
  now="$(date -Is)"
  gen_state="$(state_for "$GEN_JOB")"
  eval_state="$(state_for "$EVAL_JOB")"
  sample_count="$(find "$RUN_ROOT/worker_logs" -maxdepth 1 -type f -name 'gpu_util_*.csv' 2>/dev/null | wc -l)"
  evidence="$(evidence_line 0)"
  printf '%s mode=%s generation=%s evaluation=%s gpu_sample_files=%s %s\n' "$now" "$MODE" "$gen_state" "$eval_state" "$sample_count" "$evidence"
  if terminal_failure "$gen_state" || terminal_failure "$eval_state"; then
    blocker="$RUN_ROOT/monitor_blockers/$(date -u +%Y%m%dT%H%M%SZ)_scheduler_failure.json"
    "${PYTHON:-python3}" - "$blocker" "$GEN_JOB" "$EVAL_JOB" "$gen_state" "$eval_state" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, gen_job, eval_job, gen_state, eval_state = sys.argv[1:]
Path(path).write_text(json.dumps({
    "kind": "scheduler_failure",
    "generation_job": gen_job,
    "evaluation_job": eval_job,
    "generation_state": gen_state,
    "evaluation_state": eval_state,
    "detected_at": datetime.now(timezone.utc).isoformat(),
    "action_required": "diagnose raw logs and resubmit failed stage only within retry budget",
}, indent=2) + chr(10), encoding="utf-8")
PY
    exit 1
  fi
  if [[ "$eval_state" == COMPLETED ]]; then
    if ! final_evidence="$(evidence_line 1)"; then
      blocker="$RUN_ROOT/monitor_blockers/$(date -u +%Y%m%dT%H%M%SZ)_completion_evidence_missing.json"
      "${PYTHON:-python3}" - "$blocker" "$GEN_JOB" "$EVAL_JOB" "$final_evidence" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, gen_job, eval_job, evidence = sys.argv[1:]
Path(path).write_text(json.dumps({
    "kind": "completion_evidence_missing",
    "generation_job": gen_job,
    "evaluation_job": eval_job,
    "evidence": evidence,
    "detected_at": datetime.now(timezone.utc).isoformat(),
    "action_required": "inspect task ledger, metrics, CSV ledger, history, and readiness artifacts",
}, indent=2) + chr(10), encoding="utf-8")
PY
      printf '%s monitor_exit=blocked reason=completion_evidence_missing %s\n' "$(date -Is)" "$final_evidence"
      exit 1
    fi
    printf '%s monitor_exit=completed run_id=%s %s\n' "$(date -Is)" "$RUN_ID" "$final_evidence"
    exit 0
  fi
  sleep "$INTERVAL"
done
