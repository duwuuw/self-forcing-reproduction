# Looped Self-Forcing Pipeline Hardening Plan

## Goal

Make the newly added training and inference pipeline usable on both NM5 and
AutoDL, preserve the existing model computation, and make each run's inputs,
checkpoint provenance, and completion evidence auditable.

The current workspace-wide `scale_up_outputs/runtime.yaml` selects another
experiment. This change keeps that pointer untouched and makes the new
experiment launchers resolve their own runtime bundle explicitly.

## Design

- Give the new experiment a backend-neutral bundle name and derive its absolute
  root from the selected runtime configuration / `SUE_EXP_DIR`.
- Keep `pipeline.py` as a small CLI dispatcher. Move configuration composition,
  checkpoint/provenance checks, worker execution/evidence, and result-ledger I/O
  into separate modules with narrow interfaces.
- Compose versioned Hydra config groups for method, backend, stage, tracking,
  and run scale. Keep SUE resource/path policy in `config/runtime.yaml`.
- Keep NM5's Slurm submission path. Add an AutoDL direct-run launcher that
  checks its local environment and GPU shape, starts the worker in detached
  `tmux`, and writes durable logs/state beneath the same resolved experiment
  root. It must not read credentials from tracked files.
- Make adapter metadata portable: identify and hash the base model, resolve the
  base path from the active backend's config, and verify its bytes before
  loading the adapter. Require explicit inference checkpoint provenance.
- Fingerprint the full resolved run configuration. Reject reuse of a `run_id`
  with changed training or inference inputs, and reject stale/incomplete output
  directories rather than treating old videos as current evidence.
- Gate successful training on finite saved model tensors and successful
  inference on the expected number of decodable videos. Check `ffprobe` before
  GPU work. Record backend, seed, measured training speed, adapter parameter
  count, checkpoint step, and output paths in the ledger.
- Preserve the workspace's checkpoint and metric cadence: hourly resumable
  checkpoints with a final checkpoint marker; scalar W&B logs every 100 steps,
  including the first/final step.

## Implementation Tasks

1. Add focused regression tests for Hydra composition, immutable run identity,
   portable checkpoint export/load, both backend command plans, finite training
   evidence, exact inference output evidence, and ledger fields. Run them first
   and confirm each fails for its intended reason.
2. Refactor the new pipeline tree and runtime bundle to the backend-neutral
   layout. Add Hydra config groups and a thin dispatcher while preserving the
   copied training/inference computation.
3. Add NM5 and AutoDL launch adapters. NM5 remains Slurm-based; AutoDL uses
   direct execution under detached `tmux`. Both must resolve paths from the
   runtime contract and record the selected backend.
4. Fix checkpoint export/load provenance, run manifest coverage, checkpoint
   validation, output freshness/count checks, `ffprobe` preflight, logging
   cadence, and ledger measurements.
5. Update the new pipeline README and workspace references so commands and
   paths match the implementation. Do not edit unrelated older experiment
   bundles or the active root bootstrap runtime.
6. Run targeted and full local test suites, inspect the full diff, request an
   independent code review, and run the requested `student-codebase-audit`.
   Do not launch remote or GPU jobs; record those checks as not run.
7. Run the privacy audit, stage only the pipeline change set (including the
   newly added pipeline files), create one commit in the project repository,
   and push it to the configured GitHub remote. Check whether an outer submodule
   pin actually changed before applying the parent repository push rule.

## Verification

- `uv run pytest tests/test_looped_self_forcing_pipeline.py -q`
- `uv run pytest -q`
- Static YAML/config composition checks and Bash syntax checks for both launchers
- `student-codebase-audit` traceability and verification report
- Privacy audit and staged-file review before commit

Remote AutoDL credentials/configuration are absent from this local checkout, so
AutoDL remote execution and GPU smoke checks are explicitly outside this local
verification pass.
