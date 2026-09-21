# agent.md — looped-flow-matching / stable code map

This file records stable repository boundaries and implementation constraints. It is
not a training-run ledger; run-specific logs, checkpoints, metrics, and scheduler
records belong under the ignored experiment bundle.

## Repository map

- `Self-Forcing/` is the upstream causal Self-Forcing checkout and owns the shared
  large assets.
- `Self-Forcing-blockwise/` is the selected-block LoRA DMD checkout. Keep its source
  pinned and unmodified.
- `Self-Forcing-layerwise/` is the separate stack-loop checkout. Do not mix its loop
  semantics with the block-wise training checkout.
- `scripts/train_blockwise_lora_dmd.py` is the branch-external training entry point.
  It imports the upstream training code and applies only runtime integration fixes.
- `scripts/self_forcing_runtime_alignment.py` contains the non-loop checkpoint
  alignment helper. The loop's cache snapshots and cursor semantics stay untouched.
- `scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd/` contains the durable
  runtime config and Slurm launcher source.

## Loop contract

The temporal loop is user-owned research logic. Preserve `mode: block`, layers
`8..15`, `k_min == k_max == 2`, and `strength: 1.0`. Memory, placement, allocator,
checkpoint, and W&B integration fixes must not alter the loop implementation or its
schedule.

## Native Self-Forcing alignment

The native entry point is `Self-Forcing/train.py`. Apart from the loop, keep the
student/teacher/critic construction, checkpointing, optimizer, loss, and tracking
path aligned with that entry point. Runtime-only fixes belong in the wrapper and
must assert that each patch landed.

## Launcher and artifact rules

- Resolve NM5 accounts, usernames, routes, and storage roots only from ignored
  backend configuration on the execution plane.
- Slurm job names use the canonical `user.yaml` prefix resolver.
- Keep checkpoints, model weights, logs, W&B runs, and generated evaluation output
  out of Git.
- Run syntax/unit checks and inspect the staged file list before publishing.

## Git publishing

Publish experiment code on a dedicated branch. Do not force-update unrelated
branches, and never commit private backend values or generated artifacts.
