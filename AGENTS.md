# looped-flow-matching — workspace agent rules

## Current handoff

Read `agent.md` first for the live handoff; `HANDOFF_NEXT.md` carries the
current-stage detail and `SCALE_UP.md` the lessons. **Do not interpret the
historical 40-second experiment summary below as the current protocol.**

Two stages, both settled as of 2026-09-20:

1. **Search (sealed).** R1–R4, sixteen candidate configs, all finished and reconciled.
   Conclusion: `layer_start=8, layer_end=15` with `K=2` is the least motion-damaging
   depth range, and it holds under both per-block (R3) and stack (R4) loop semantics.
   **Do not submit any further search-stage jobs.** The eval protocol is `num_output_frames`
   fixed at `123` → derived 489 raw / 480 eval frames @ 16 fps (30 s final), EMA weights;
   only `k_min`, `k_max`, and the loop-layer range ever varied.
2. **Training (aligned formal run submitted).** The upstream branch
   `block-wise-temporal-loop` (`168d4de`) was brought up on NM5 and its aligned training
   dryrun passed — job `46239326`, `COMPLETED`, zero OOM, zero CheckpointError. The
   current 600-step formal run `formal_align_r1_20260921` is job `46240819` and is
   currently pending with Slurm; it has not produced completion evidence yet. The older
   15-frame attempt that OOMed is historical. See `agent.md` §「第五轮」for the root
   causes and the branch-external fixes.

Parent context: this workspace is under `deepresearch/workspace/`. On start, load the
parent repo memory and skills before acting — `memory/project.md`, root `AGENTS.md`,
`memory/sue/SCALE_UP.md`, `.codex/skills/AGENTS.md`, and the active backend's private
`deepresearch-sandbox/config_<sandbox>.txt`.

## What this workspace is

`looped-flow-matching` is the upstream paper repo for *Training-Free Hidden-State
Refinement for Flow-Matching Image Generators*. The **`Self-Forcing/` subdirectory** is a
nested checkout of `guandeh17/Self-Forcing` carrying a port of the paper's **Dense Token
Loop** into Self-Forcing's causal autoregressive rollout. That port is what we call
**`looped-self-forcing`** — the subject of the scale-up run.

- Method config: `Self-Forcing/configs/self_forcing_dmd_temporal_loop.yaml`
- Model-side loop: `Self-Forcing/wan/modules/temporal_loop.py`
- Schedule/config: `Self-Forcing/utils/temporal_loop.py` (K driven by AR rollout
  progress only — never by diffusion timestep)
- Inference entry: `Self-Forcing/inference.py`
- Eval orchestration: `Self-Forcing/eval/vbench_long/`
- Design notes: `Self-Forcing/docs/temporal_loop.md`, `Self-Forcing/handoff.md`, `1.md`

**Three checkout trees, three different loop semantics — do not mix them up:**

| tree | semantics | role |
| --- | --- | --- |
| `Self-Forcing/` | **per-block** (upstream `guandeh17/Self-Forcing`) | default tree; also owns the shared large assets |
| `Self-Forcing-layerwise/` | **stack**: the whole contiguous layer range repeats as one loop body (submodule, pin `e1df619`) | R4 |
| `Self-Forcing-blockwise/` | **block-wise residual loop + selected-block LoRA DMD training** (worktree, branch `block-wise-temporal-loop`, pin `168d4de`) | training stage |

`K=1 && strength=1.0` takes a no-loop fast path, so `k1_full_19_26` is the true
loop-off baseline. Select a non-default tree with `SUE_SF_DIR`; the alternative
trees are plain git checkouts and carry only tracked source, so `VBench`,
`checkpoints/` and `wan_models/` must be symlinked from the default tree —
`_env.sh` fails loudly with the exact `ln -sfn` remediation if they are missing.

## Rules

- Do not edit `Self-Forcing/VBench/` (vendored eval upstream) or `preliminary_*` material.
- Keep SUE-owned launchers in
  `scale_up_outputs/nm5_looped_self_forcing_search_20260912/slurm_scripts/`; they are durable run-bundle
  source, not disposable output.
- `Self-Forcing/checkpoints/` and `Self-Forcing/wan_models/` are git-ignored weight
  locations; verify their active workspace-local reachability from the current bundle
  and private NM5 configuration before smoke or launch.
- Slurm job names must carry the `<username>-` prefix resolved from `user.yaml`.
  Resolve usernames, accounts, SSH routes, and storage roots only from the permitted
  ignored configuration files; do not hardcode them here.
- Report metric semantics precisely: this run uses the fixed 123-frame input protocol,
  derives 489 raw and 480 eval frames internally, and keeps only a final 30-second
  video at 16 fps, with EMA weights. Historical 40-second results are separate.

### Training stage (`Self-Forcing-blockwise/`)

- **Never change the loop.** `temporal_loop` and its layer range are a standing user
  constraint: `mode: block`, `layer_start: 8`, `layer_end: 15`, `k_min == k_max == 2`,
  `strength: 1.0`. Fix memory or throughput problems anywhere else — FSDP wrap
  granularity, CPU offload, allocator settings — never in the loop itself.
- **Keep the branch pristine.** `Self-Forcing-blockwise/` must stay at `168d4de` with
  its code unmodified; only the preset YAML under `configs/` carries deliberate,
  commented deviations. Every runtime fix belongs in the branch-external wrapper
  `scripts/train_blockwise_lora_dmd.py`, which patches and then asserts each patch
  landed. It is the shared entry point for smoke and fullrun alike — a dryrun must not
  take a path the fullrun cannot.
- The wrapper prints one audit line per FSDP unit
  (`WRAPPER: fsdp_wrap <module> params=X.XXB cpu_offload=… min_num_params=…`). Read
  those lines to confirm what is actually on the GPU; do not infer placement from config.
- Training launchers are durable run-bundle source and live in
  `scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd/slurm_scripts/`.
  `nm5_train_submit.sh <run_id> <smoke|fullrun>` resolves the private backend values on
  the execution plane; only the submitter may derive a config, only for the whitelisted
  non-semantic keys, and it fails closed on any other difference.
- The branch rejects `training_gradient_window_frames > num_training_frames`
  (`pipeline/self_forcing_training.py:93`). Upstream keeps them equal, which makes
  `start_gradient_frame_index == 0` and grades every AR chunk; preserve that relationship
  if the frame count ever changes.
- Training and evaluation use **different frame protocols**: training runs 21 latent
  frames, evaluation runs 123. Their numbers are not comparable — label which is which.
- NM5's H100 exposes **63.29 GiB usable** (`nvidia-smi` reports `65247 MiB`), not 80 GB.
  Budget GPU memory against that, and against the fact that a 4-way FSDP shard of the
  frozen 14B teacher is ~14 GiB/rank.

## SUE experiment summary

| field | value |
| --- | --- |
| experiment | `nm5_looped_self_forcing_search_20260912` |
| backend | NM5 (resolve partition/account/QoS from ignored backend config) |
| exp_dir | `scale_up_outputs/nm5_looped_self_forcing_search_20260912/` |
| config | active bundle YAMLs (K 1..6, candidate loop layers, `temporal_uniform`) |
| checkpoint | `checkpoints/self_forcing_dmd.pt`, `--use_ema` |
| generation | 128 selected prompts, fixed `num_output_frames=123` → derived 489 raw → 480 eval frames @ 16 fps (30 s final) |
| eval | VBench-Long, 6 dims, `--mode long_custom_input --dev_flag` |

And the training-stage run bundle:

| field | value |
| --- | --- |
| experiment | `nm5_self_forcing_blockwise_lora_dmd` |
| backend | NM5 (partition/account/QoS resolved from ignored backend config on the execution plane) |
| exp_dir | `scale_up_outputs/nm5_self_forcing_blockwise_lora_dmd/` |
| entry | `scripts/train_blockwise_lora_dmd.py` → the branch's own `train.py` |
| config | `Self-Forcing-blockwise/configs/self_forcing_dmd_temporal_loop_lora_train.yaml` |
| student init | `checkpoints/self_forcing_dmd_ema_as_generator.pt` (rewrap of the released Self-Forcing DMD EMA weights) |
| teacher | `wan_models/Wan2.1-T2V-14B` (frozen, CPU-offloaded) |
| training protocol | 21 latent frames, LoRA on `blocks.{8..15}.self_attn.{q,v}` (32 tensors), K=2 |
| aligned dryrun | job `46239326`, `COMPLETED`, 30 steps, `TRAIN_EVIDENCE_OK` (2026-09-21) |
| formal run | job `46240819`, `PENDING` at handoff; 600 steps, completion not yet verified |
