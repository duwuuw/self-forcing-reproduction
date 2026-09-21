# looped-flow-matching — SUE / scale-up lessons

Workspace-specific lessons. Generic cross-workspace lessons live in
`${<SANDBOX>_DEEPRESEARCH_ROOT}/memory/sue/SCALE_UP.md`; read those first.

## Run identity

- Method: **looped-self-forcing** (Self-Forcing + training-free Dense Token Loop).
- First validated end-to-end on NM5, 2026-09-09 (dryrun job 45638025).
- Measured cost: **100.5 s per 40 s video** (162 latent frames, K 1..6 on layers 19..26)
  on one H100, after ~110 s of process start + model load.
- Frame contract: `4 * latent_frames - 3 = 645` raw frames → trimmed to 640 @ 16 fps.
  Dryrun ffprobe confirmed 645 frames / 40.31 s before trimming.

### NM5 asset reuse for this workspace
- `checkpoints/self_forcing_dmd.pt` SHA256 `a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3`
  was already on NM5 from a prior Self-Forcing run — verify the hash before re-transferring
  5.7 GB.
- `wan_models/Wan2.1-T2V-1.3B` (17 GB) likewise already present; symlink instead of copying.
- VBench 1.0 weights for the six long-video dims live in the prior run's
  `eval/vbench_cache/cache/`; only the LAION aesthetic linear head, the dinov2 hub repo,
  and the dreamsim ensemble had to be staged.

### Runtime env
- Base: NM5 `scale-rae` conda-pack env (py3.10, torch 2.5.1+cu124, transformers 4.37.0).
- Missing pieces are supplied by an offline wheel overlay at
  `scale_up_outputs/nm5_looped_self_forcing/env/overlay310` (PyAV for
  `torchvision.io.write_video`, plus dreamsim/moviepy/scenedetect/pyiqa/facexlib/filterpy/future
  for VBench-Long). It goes on `PYTHONPATH`; the base env's torch/transformers/numpy are
  never mutated.
- `torchvision 0.20.1` still ships `torchvision.io.write_video`, so the PyAV shim needed on
  the RunPod pod (torchvision 0.26) is **not** required here — only PyAV itself.

### Known traps
- The NM5 login node has a stray `/tmp/six.py`; never run helper scripts from `/tmp`.
- `eval/vbench_long/run_vbench_eval.sh` as committed is broken — it passes `--prompt_file`
  with `--mode long_custom_input`, which `eval_long.py:236` rejects. The launcher in
  `scale_up_outputs/nm5_looped_self_forcing/slurm_scripts/` drops it.
- `eval/vbench_long/check_ready.py` does **not** check VBench weights or the vendored
  VBench checkout; a green `ready_for_official_eval` is not sufficient for eval.

### `num_output_frames` must be divisible by `num_frame_per_block` (3)
- **date**: 2026-09-10
- **trigger**: A 30-second config (`num_output_frames: 121`) failed 19 s into the job with
  `ValueError: latent-frames must be divisible by frames-per-block` from
  `eval/vbench_long/prepare_assets.py:79`. `pipeline/causal_inference.py:99` asserts the
  same thing, so this breaks `inference.py` too, not just the eval helper.
- **wrong**: Picking a latent-frame count by duration alone. The real constraint chain is:
  - `latent % 3 == 0` (block size)
  - `raw  = 4*latent - 3` (Wan VAE 4x temporal compression, first frame special)
  - `eval = 16 * (raw // 16)` (trim to whole seconds at 16 fps)
- **correct**: Choose latent frames from this table, not from the target duration:
  | latent | seconds |
  |---|---|
  | 120 | 29.0 |
  | **123** | **30.0** |
  | 126 | 31.0 |
  | 162 | 40.0 (main 946-prompt run) |
  | 81 | 20.0 |
  `123` is the unique value that is divisible by 3 *and* yields exactly 30.00 s.
- **source**: workspace/looped-flow-matching dyn30 runs 45670430 / 45670432, 2026-09-10.

## Dryrun

### Resolve Slurm script paths from the durable experiment bundle
- **sandbox**: NM5
- **session**: dryrun
- **date**: 2026-09-10
- **trigger**: Slurm jobs failed immediately because Slurm staged the submitted script and `$0` no longer pointed into the workspace bundle.
- **wrong**: Deriving sibling paths with `dirname "$0"` inside an `sbatch` script.
- **correct**: Require the launcher to pass the durable `SUE_EXP` path and source `"$SUE_EXP/slurm_scripts/common_env.sh"`; keep generated logs and scripts under the configured experiment directory.
- **source**: workspace/looped-flow-matching

### Brace Slurm variable expansions when appending suffixes
- **sandbox**: NM5
- **session**: dryrun
- **date**: 2026-09-10
- **trigger**: Bash interpreted `SLURM_JOB_ID_none` and `SLURM_JOB_ID_` as variable names, producing malformed timing and worker-log paths.
- **wrong**: Writing `$SLURM_JOB_ID_none` or `$SLURM_JOB_ID_$SUE_VARIANT` when a variable is immediately followed by an identifier character.
- **correct**: Use `${SLURM_JOB_ID}_none` and `${SLURM_JOB_ID}_${SUE_VARIANT}` for concatenated path components, then run `bash -n` and a real one-job dryrun before scale-up.
- **source**: workspace/looped-flow-matching

### Validate VBench aggregate JSON by dimension keys
- **sandbox**: NM5
- **session**: dryrun
- **date**: 2026-09-10
- **trigger**: VBench `long_custom_input` emits one aggregate result JSON plus one `full_info` JSON for the requested six dimensions; requiring six JSON files caused an otherwise successful eval job to exit 2.
- **wrong**: Assume one result file per dimension and gate evaluation on `result_json_count >= 6`.
- **correct**: Require `*_eval_results.json` and `*_full_info.json`, then validate all six requested dimension keys in the aggregate JSON before writing metrics and ledger records.
- **source**: workspace/looped-flow-matching

## Fullrun

### Make multi-file source sync fail closed
- **sandbox**: NM5
- **session**: fullrun pre-submit
- **date**: 2026-09-10
- **trigger**: A pre-submit sync used one incorrect run-root path; without shell fail-fast, later file syncs still ran and obscured the failed artifact transfer.
- **wrong**: Run a sequence of sync commands without `set -euo pipefail`, or create a fullrun root before verifying the exact artifact target.
- **correct**: Enable fail-fast for every pre-submit sync, verify the exact run root is empty or absent, use `rmdir` only for an accidentally created empty directory, then resync and checksum every required source file.
- **source**: workspace/looped-flow-matching

## Search round 1 — 30 s / 123-frame protocol (2026-09-13)

This round replaced the `scale-rae` env with a fresh workspace-local conda-pack
env built on the pod, transferred to NM5, and unpacked. Two lessons below are
**regressions** of lessons already recorded above — the new launchers were
written without re-reading this file.

### Slurm script staging — REGRESSION, see the 2026-09-10 entry above
- **sandbox**: NM5
- **session**: dryrun
- **date**: 2026-09-13
- **trigger**: `dryrun_r1_20260913` generation job 45808484 died in 35 s with
  `<SLURM_SPOOL_ROOT>/job<N>/common_env.sh: No such file or directory`. Slurm
  copies the batch script into a per-job spool directory before running it, so
  `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` resolves to the
  spool dir, not the durable bundle.
- **wrong**: Re-deriving the launcher directory from `BASH_SOURCE[0]` inside
  `*_packed.sbatch`. The 2026-09-10 lesson said not to do this; the new bundle
  did it anyway. It also differs from the older bundle, which hardcoded an
  absolute `source` path and therefore worked.
- **correct**: Have the submitter pass the real directory explicitly —
  `submit_search.sh` adds `SUE_SCRIPTS_DIR=$SCRIPT_DIR` to `--export`, and each
  `.sbatch` uses `SCRIPT_DIR="${SUE_SCRIPTS_DIR:-$(...BASH_SOURCE...)}"` then
  asserts `common_env.sh` is readable and exits 2 with an explicit message.
- **source**: workspace/looped-flow-matching, dryrun_r1 → dryrun_r2

### `nvidia-smi -L` cannot verify a packed `CUDA_VISIBLE_DEVICES` binding
- **sandbox**: NM5
- **session**: dryrun
- **date**: 2026-09-13
- **trigger**: `dryrun_r2_20260913` generation job 45808791 preflight passed,
  then all four workers exited in 35 s with
  `GPU WORKER ERROR: CUDA binding 0 exposes 4 GPUs` (and 1/2/3 likewise).
- **wrong**: Counting devices with
  `CUDA_VISIBLE_DEVICES="$gpu" nvidia-smi -L | wc -l` and asserting the count is
  1. `nvidia-smi` enumerates physical devices and **ignores**
  `CUDA_VISIBLE_DEVICES`; only the CUDA runtime honours it. On a 4-GPU node the
  assertion can never pass, so every packed worker failed before doing work.
- **correct**: Assert through the CUDA runtime instead —
  `CUDA_VISIBLE_DEVICES=$gpu python -c 'import torch; assert torch.cuda.device_count() == 1'`.
  Keep `nvidia-smi -L` only for the allocation-wide "4 GPUs visible" check,
  where no binding is set.
- **source**: workspace/looped-flow-matching, dryrun_r2 job 45808791

### `flash_attn` is a hard T2V requirement but is not in any requirements file
- **sandbox**: NM5
- **session**: environment
- **date**: 2026-09-13
- **trigger**: Smoke job 45807818 loaded the model and ran the rollout, then died
  at `wan/modules/attention.py:118` with `assert FLASH_ATTN_2_AVAILABLE`.
- **wrong**: Treating the curated core requirements as sufficient.
  `Self-Forcing/requirements.txt` does not list `flash-attn` (only `README.md`
  mentions it), and `wan/modules/model.py` calls `flash_attention()` directly, so
  the `attention()` SDPA fallback never rescues it.
- **correct**: Ship the prebuilt wheel and install it offline. torch 2.5.1+cu124
  is built with `_GLIBCXX_USE_CXX11_ABI=0`, so the wheel must be
  `flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
  (an `abiTRUE` wheel loads but crashes). NM5 has no egress, so download on the
  build host and `pip install --no-deps --no-build-isolation` from the local
  file. `sue_preflight()` now asserts the import so it fails before GPU time.
- **source**: workspace/looped-flow-matching, smoke jobs 45807818 / 45808311

### `--dev_flag` makes every VBench dimension a `[aggregate, details]` list
- **sandbox**: NM5
- **session**: dryrun
- **date**: 2026-09-13
- **trigger**: `dryrun_r6_20260913` evaluation job 45810270 computed every score,
  then died in `scripts/record_vbench_result.py:42` with
  `ValueError: subject_consistency is not a scalar: [0.9971…, [{...per-clip...}]]`.
- **wrong**: Assuming `*_eval_results.json` maps each dimension to a bare float,
  or (as `scalar()` did) only to a single-element list. With `--dev_flag` the
  aggregate JSON holds `[aggregate, [per-clip dicts]]` for `subject_consistency`
  and `background_consistency` (len 2), and `[aggregate, [...], [...]]` for
  `motion_smoothness`, `dynamic_degree`, `aesthetic_quality` and
  `imaging_quality` (len 3). The failing job had already produced correct
  scores — only the ledger/W&B write rejected them.
- **correct**: Take the leading element and discard the detail lists, which is
  what `eval_packed.sbatch`'s inline dimension guard already did
  (`data[key][0] if isinstance(data[key], list) else data[key]`). `scalar()` now
  does the same. Do not key off `len(value) == 1`.
- **source**: workspace/looped-flow-matching, dryrun_r6 job 45810270

### `wandb` is required by the ledger writer but missing from the curated env
- **sandbox**: NM5
- **session**: environment
- **date**: 2026-09-13
- **trigger**: `dryrun_r5_20260913` evaluation job 45809443 failed in all four
  workers with `ModuleNotFoundError: No module named 'wandb'`, raised from
  `record_vbench_result.py:15`. Separately, `wandb` refuses to import against
  protobuf 7.x: `Failed to import protobufs for protobuf version 7.36.1. wandb
  only works with major versions 3, 4, 5, and 6`.
- **wrong**: Assuming the curated core requirements cover the tracking path.
  `wandb` is in `Self-Forcing/requirements.txt` but was omitted from
  `requirements-nm5-core.txt`; `protobuf` was listed unpinned, so pip installed
  7.x, which wandb rejects.
- **correct**: Pin `protobuf<7` and add `wandb` to the curated requirements
  (done). On the execution plane, install from the prebuilt `wandb_wheelhouse`
  with `--no-index --no-deps` so the resolver cannot upgrade pydantic past its
  `==2.10.6` pin. Safe here because every reverse-dependency on protobuf in this
  env (`diffusers`, `peft`, `transformers`) is an optional extra, not a runtime
  requirement — verify that before downgrading a shared library.
- **source**: workspace/looped-flow-matching, dryrun jobs 45809443 / 45810581

### An over-long wall-time request blocks its own backfill
- **sandbox**: NM5
- **session**: fullrun
- **date**: 2026-09-13
- **trigger**: The round-1 VBench-Long evaluation job 45810878 stayed PENDING for
  45+ min behind four 12 h jobs from other workspaces. It requested
  `--time=24:00:00` (from `runtime.yaml`'s `eval_walltime`) for work that
  finishes in about two hours.
- **wrong**: Copying a generous `eval_walltime` straight into `sbatch --time`.
  Slurm backfill only starts a job early when it can still meet every
  higher-priority job's *reservation*; a 24 h reservation is very hard to fit, so
  the job blocks itself as effectively as a busy partition would.
- **correct**: Request a realistic multiple of the expected runtime and expose
  it as `SUE_EVAL_TIME` / `SUE_GEN_TIME`. For an already-queued job, lower it in
  place rather than resubmitting —
  `scontrol update jobid=<id> TimeLimit=06:00:00` keeps queue seniority, and the
  job's `Reason` changes from `Priority` to `None` immediately. Do not go so low
  that the job can time out: `VBenchLong.evaluate` accumulates every dimension
  in memory and writes only after the last one, so a timeout loses all scores.
- **source**: workspace/looped-flow-matching, fullrun job 45810878

### Variant names must be lowercase — `submit_search.sh` rejects uppercase
- **sandbox**: NM5
- **session**: fullrun
- **date**: 2026-09-14
- **trigger**: A depth-sweep round named its arms `k2_L00_07`, `k2_L08_15`, … and
  the submitter exited 2 with `VARIANT ERROR: k2_L00_07` before any sbatch ran.
- **wrong**: Encoding a readable layer window with an uppercase letter
  (`L00_07`). `submit_search.sh` validates every variant against `^[a-z0-9_]+$`,
  because the name becomes a config filename, a W&B group component, and a Slurm
  job-name fragment. `sbatch`/`squeue` are case-sensitive and the config lookup
  is exact, so a mixed-case name fails closed rather than silently.
- **correct**: Keep variant ids strictly `[a-z0-9_]` — `k2_l00_07`, `k2_l22_29`.
  Rename the config file *and* every `search_candidates.yaml` id together, then
  re-sync; a stale uppercase file left on the remote is harmless but confusing.
- **source**: workspace/looped-flow-matching, round-3 depth sweep

### NM5 operational details that cost a round trip each
- **sandbox**: NM5
- **date**: 2026-09-13
- **`nm5-transfer` is the data-mover cluster, not the HPC cluster.** Its Slurm
  controller has `GresTypes = (null)` and only `projects`/`tapes`/`archive`
  partitions, so `sbatch --gres=gpu:1` fails with
  `Invalid generic resource (gres) specification`. Use `nm5-transfer` for rsync
  and read-only file inspection only; submit with `sbatch` on `nm5`.
- **`tmux` on NM5 needs its prerequisite**: `module load ncurses/6.4 tmux/3.3a`.
  Loading `tmux/3.3a` alone fails with `Lmod ... prereq_any("ncurses/6.4")`.
  Because `submit_search.sh` starts the monitor session with tmux, an unmet
  prerequisite silently downgrades the run to "no monitor session".
- **Batch scripts that call `module load` need `#!/bin/bash -l`.** A plain
  `#!/usr/bin/env bash` shebang is not a login shell, so the module function is
  undefined and `module load ffmpeg/7.0.1-gcc` has no effect.
- **`ffprobe` comes from the module, not from imageio-ffmpeg.**
  `imageio-ffmpeg` ships only `ffmpeg` (the static libx264 build). Any script
  that runs `ffprobe` must `module load ffmpeg/7.0.1-gcc` first, even though
  `ffmpeg` itself resolves to the static overlay build.
- **`inference.py` must run with `CWD=Self-Forcing/`.** `inference.py:59` does
  `OmegaConf.load("configs/default_config.yaml")`, a relative path, and the
  workspace root has no such file. Run it in a `( cd "$SF" && ... )` subshell and
  pass `--checkpoint_path` absolutely; `prepare_assets.py` and
  `postprocess_and_validate.py` are CWD-independent because they resolve
  everything from `--root`.
- **`torch.hub` looks under `$TORCH_HOME/hub/`, the bundle stages one level up.**
  `eval_packed.sbatch` bridges the two layouts with idempotent symlinks
  (`$TORCH_HOME/<repo>` → `$TORCH_HOME/hub/<repo>`) before its `require_eval_*`
  assertions, keeping the fail-loud checks meaningful.
- **`user.yaml` lives on the execution plane.** The local workspace copy may be
  absent, so the prefix can resolve to the default `silly-` on the control
  plane while NM5's `workspace/looped-flow-matching/user.yaml` yields the real
  per-user prefix. Do not conclude the prefix is wrong from the control plane:
  check the actual job name in `squeue` on the execution plane. Never copy the
  resolved prefix value into tracked docs.
- **source**: workspace/looped-flow-matching

## Search round 4 — layer-wise selected-layer stack loop (2026-09-15)

### An alternative Self-Forcing checkout needs the default tree's non-git assets symlinked

- **sandbox**: NM5
- **date**: 2026-09-15

Running a second Self-Forcing tree via `SUE_SF_DIR` means running a **git
clone**, and a clone carries only tracked source. The large non-git assets stay
in the default tree and are all resolved relative to `$SF`:

- `VBench/` (~1.1 GB) — the eval launcher asserts
  `$SF/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml` and then runs
  `$SF/VBench/vbench2_beta_long/eval_long.py`.
- `checkpoints/self_forcing_dmd.pt` (5.7 GB) — passed as
  `--checkpoint_path "$SF/checkpoints/..."`.
- `wan_models/Wan2.1-T2V-1.3B` — loaded by relative path, so generation must
  `cd "$SF"` first.

**Cost of getting this wrong:** the R4 dryrun *generation* succeeded in 4:47,
then the *eval* stage died in 37 s with
`EVAL PREFLIGHT ERROR: missing evaluator file .../AMT-S.yaml`, and all four
`metrics.json` were missing. A passing generation stage does **not** validate
the eval stage when `$SF` has moved — the two stages assert on different files.

**Fix — symlink, never copy** (zero duplication, both trees left untouched, so
the default tree's checkpoint stays the single source of truth):

```bash
cd "$WS/Self-Forcing-layerwise"
ln -sfn ../Self-Forcing/VBench      VBench
ln -sfn ../Self-Forcing/checkpoints checkpoints
ln -sfn ../Self-Forcing/wan_models  wan_models
```

**Folded into durable code:** `_env.sh` now asserts those three assets are
reachable whenever `SUE_SF_DIR` differs from the default tree, and exits with
the exact `ln -sfn` remediation. It runs at source time, i.e. before the job
burns its queue slot. Verified all three branches: alternative tree passes,
default tree passes, missing asset exits 2 with the fix printed.

- **source**: workspace/looped-flow-matching
