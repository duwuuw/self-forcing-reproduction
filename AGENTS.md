# looped-flow-matching — workspace agent rules

## Workspace guide

Read `agent.md` for stable code boundaries and the parent SUE memory before making
changes. The evaluation contract uses a fixed `num_output_frames` of `123`, derives
internal raw/eval counts `489/480`, and keeps only a final 30-second video at 16 fps.
Do not use this file or `agent.md` as a run ledger.

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
