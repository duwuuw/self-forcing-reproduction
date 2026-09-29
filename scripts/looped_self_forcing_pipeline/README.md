# Backend-neutral looped Self-Forcing pipeline

This entry point composes the run configuration from the active SUE bundle.
The default bundle is `scale_up_outputs/looped_self_forcing_pipeline`; set
`SUE_EXP_DIR` to select another compatible bundle. Its Hydra groups live under
`<SUE_EXP_DIR>/config/`.

The `source/` tree is the code-only runtime copied from the previous NM5
pipeline. Model and temporal-loop math remain unchanged. The training memory
probe adds rank labels to its diagnostic lines so smoke evidence can be checked
per GPU. `COPY_MANIFEST.json` records the source hashes. Model weights, datasets,
generated videos, run outputs, and Python caches are not part of this source
migration.

The copied upstream source is licensed under Apache License 2.0. Its original
`LICENSE` is preserved at `source/LICENSE`; the source checkout had no
`NOTICE` file.

## Runtime inputs

For NM5, `config/runtime.yaml:environment.asset_root` selects the
workspace-relative `Self-Forcing-blockwise-layerwise` checkout under the
private `NM5_WORKSPACE_ROOT`. The resolved root must contain `checkpoints/` and
`wan_models/`; preflight checks the configured checkpoint and both model trees.
`SUE_ASSET_ROOT` and `SUE_PYTHON` are optional private overrides. By default,
the NM5 launcher resolves the interpreter and asset root from `runtime.yaml`.
It adds `environment.python_overlay` to the Python import path when configured.
Do not put resolved backend paths in tracked files or CLI overrides.

### Model and checkpoint roles

The selected method's `generator_ckpt` is the rewrapped released Self-Forcing
DMD EMA generator checkpoint. Training first constructs the causal Wan2.1
T2V 1.3B architecture from `wan_models/Wan2.1-T2V-1.3B`, then loads this full
generator checkpoint strictly before injecting the selected-block LoRA
adapters. If that load succeeds, the EMA checkpoint replaces the initial
generator weights. The separate Wan1.3B fake-score model, T5 encoder, and VAE
still use the Wan1.3B asset tree; `teacher_checkpoint` supplies the Wan2.1
T2V 14B real-score teacher.

Actual train and inference worker subprocesses use the configured asset root as
their working directory because the copied upstream code opens `wan_models/`
and related assets with relative paths. Both entrypoints load
`source/configs/default_config.yaml` relative to their own source file, so
their defaults do not depend on that working directory.

The NM5 runtime maps cache paths from ignored `NM5_HF_HOME`,
`NM5_HF_HUB_CACHE`, `NM5_MODELSCOPE_CACHE`, `NM5_TORCH_HOME`,
`NM5_MPLCONFIGDIR`, and `NM5_WANDB_CACHE_DIR` values. Launch preflight requires
all six writable directories and rejects cache paths under `HOME`; their
resolved values stay private and are exported to the Slurm worker. Partition,
QoS, GPU, node, and CPU requests come from top-level `sandbox_resources.nm5`.

The bundle's `config/runtime.yaml:paths.datasets_root` is a bundle-relative
directory. `train.prompt_file` and `infer.prompt_file` select relative files
under it; the defaults are `vidprom_filtered_extended.txt` and
`inference_prompts.txt`. Place each selected, non-empty prompt file under that
directory before running preflight, or choose another bundle-relative filename
in the bundle config. The pipeline does not download prompt data and this
repository does not specify an upstream prompt URL.
`paths.config_root` is fixed to `config`, where `config.yaml`, Hydra groups,
and `runtime.yaml` form the bundle bootstrap contract. `SUE_EXP_DIR` already
identifies this bundle; `paths.scale_up_outputs_root` and `paths.exp_dir` are
bootstrap identity fields. Worker artifacts resolve from the named roots below
that bundle.

## Configuration

```bash
SUE_EXP_DIR="$PWD/scale_up_outputs/looped_self_forcing_pipeline" \
  uv run python scripts/looped_self_forcing_pipeline/pipeline.py \
    --show-config backend=autodl stage=infer scale=smoke
```

The group selectors are `method`, `backend`, `stage`, `tracking`, and `scale`.
Defaults use `layerwise_l16_23_k2`, four-GPU training, one-GPU inference, the
offline Hydra tracking group, and a 600-step training cap. NM5 uses offline
tracking by default. `train.log_iters` is the copied
trainer's native save/checkpoint opportunity cadence: 50 steps by default and
10 for `scale=smoke`. The timed checkpoint policy writes a checkpoint when due
(one hour by default) or at the final step. This cadence is separate from W&B
scalar logging every 100 steps and console progress every 10 steps.
AutoDL and NM5 are configuration choices; host paths and credentials do not
belong in tracked config.
Training uses the reproducible non-zero seed `1` by default; training overrides
of `seed=0` are rejected because the copied trainer treats zero as a request for
a fresh random seed. Inference may use seed zero. The copied T2V inference
profile uses 123 latent frames and must decode 489 video frames at 16 fps;
preflight rejects other latent-frame settings and worker verification counts the
decoded frames rather than applying the separate VBench crop size.

Backend launchers require `train|infer <run_id>` and accept validated optional
Hydra `key=value` overrides. Accepted keys are `method`, `tracking`, `scale`,
`seed`, `train.max_steps`, `train.log_iters`, `train.timeout_seconds`,
`train.checkpoint_interval_seconds`, `train.resume_checkpoint`,
`infer.checkpoint`, `infer.num_samples`, `infer.num_output_frames`,
`infer.timeout_seconds`, `infer.seed`, and `infer.use_ema`. Checkpoint overrides must be relative to
`SUE_EXP_DIR`. Backend, stage, run ID, assets, unknown keys, absolute paths,
and malformed values are rejected.

When `tracking=online`, load `WANDB_API_KEY` and `WANDB_ENTITY` from ignored
private backend configuration. The launchers pass those variables through the
process environment; never put their values in CLI overrides or tracked files.
The comparison project comes from the selected Hydra `tracking.project` group.
The AutoDL launcher selects online tracking by default only when both private
credentials are present; otherwise it keeps offline mode. An explicit
`tracking=offline|online` override takes precedence.
The training ledger leaves total `parameter_count` blank when unavailable and
records the gathered LoRA adapter size as `trainable_parameter_count`.

## NM5 Slurm launch

Load the ignored private NM5 environment and choose a fresh run_id:

~~~bash
bash scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/nm5_submit.sh \
  train loop-train-001 scale=smoke train.max_steps=20 train.log_iters=5
~~~

The launcher reads NM5_DEEPRESEARCH_ROOT, NM5_WORKSPACE_ROOT, NM5_ACCOUNT,
and cache values from the ignored backend environment. It resolves SUE_PYTHON
and SUE_ASSET_ROOT from runtime unless the operator supplies private overrides.
Partition and QoS come from sandbox_resources.nm5. Before submission, scontrol
must confirm that the selected Hydra train.timeout_seconds or
infer.timeout_seconds fits the partition MaxTime. The launcher requests four
GPUs for training or one GPU for inference, and writes Slurm stdout/stderr under
paths.logs_root. The allocated NM5 worker checks every visible CUDA device
against runtime gpu_type=H100 before entering the pipeline; the controller does
not claim hardware proof from partition metadata. Slurm --time adds runtime
timeout_kill_after_seconds and finalization_grace_seconds after Hydra's timeout,
leaving time to record ledger and W&B status without changing model duration.
When the selected workspace has Git metadata, launchers record its commit and
dirty state automatically. An archive deployment without `.git` must set
`SUE_GIT_COMMIT` to the source commit SHA and `SUE_GIT_DIRTY` to `clean`,
`dirty`, or `unknown`; both values are forwarded to the worker and recorded in
the W&B run config.
NM5 submissions disable Slurm email notifications. The launcher clears inherited
`SBATCH_MAIL_USER` and `SBATCH_MAIL_TYPE` settings and submits with
`--mail-type=NONE`; it does not require or use `SUE_SLURM_MAIL_USER`. Durable
stdout and stderr remain under `paths.logs_root`.
The Slurm job name uses the username from workspace user.yaml;
invalid or missing values use the silly- prefix. Its remaining name matches the
W&B experiment name after resolving method, run ID, and stage. Only selected
runtime variables are exported to the batch job.

## Paired layerwise 23–30 comparison

The two versioned method groups use 0-based layer indices 22..29 (paper
layers 23–30) and identical LoRA/loss settings. layerwise_l23_30_k3_lr5gen
uses fixed K=3, lr=4e-7 (one fifth of the original generator LR), and keeps
lr_critic=4e-7. layerwise_l23_30_k2_lr5both uses fixed K=2 and scales both
learning rates to lr=4e-7, lr_critic=8e-8.

Before each pair stage, run the SUE capacity query and place its fresh result at
SUE_EXP_DIR/<pair_id>/max_parallel.json using the standard sue-job-max-parallel
schema. It must match the runtime NM5 partition, allow max_parallel >= 1, and be
no more than 10 minutes old. The workspace preflight adapter separately queries
association/QoS submit limits and active pending/running jobs; it fails closed
unless two pending-submit slots remain for the dependency pair. It records only
redacted summaries under artifacts/pairs/<pair_id>/pending_submit_preflight.json
and artifacts/pairs/<pair_id>/preflight.json. Raw queue output, account names,
hosts, and absolute paths do not belong in these reports.

Choose a fresh smoke pair ID and use it for both jobs:

~~~bash
SMOKE_PAIR_ID="<fresh-smoke-pair-id>" # replace with a unique lowercase ID
bash scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/nm5_submit_pair_23_30.sh \
  "$SMOKE_PAIR_ID" scale=smoke
~~~

The launcher runs both stage preflights and the parent preflight dispatcher
before the first sbatch. The workspace adapter writes its redacted report under
artifacts/pairs/<pair_id>/preflight.json. It enables SUE_MEM_PROBE=1 at step 3
only for smoke. A pair ID is single-use after any preflight attempt; if
preflight or submission fails, start again with a fresh pair ID and capacity
artifact. The versioned smoke scale uses a 1,800-second worker timeout; Slurm
adds the 300-second kill-after and 600-second finalization grace, for a
2,700-second (45-minute) walltime request per job.
Each job requests four GPUs; the second uses an afterany dependency on the first,
so they cannot overlap. After both jobs finish, validate their scheduler state,
final checkpoint, ledger rows, and four step-3 memory peaks with no probe
warning or OOM evidence. W&B acceptance requires more than an offline run ID:
inspect the real offline event history, verify the resolved run config, and
require finite loss records at the expected steps. For the 10-step K3 smoke
this means generator and critic loss at step 1 and critic loss at step 10. Then
record the immutable readiness stamp:

~~~bash
bash scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/nm5_submit_pair_23_30.sh \
  --verify-smoke "$SMOKE_PAIR_ID"
~~~

Complete the manual W&B run-config and offline loss-history inspection before
invoking `--verify-smoke`. The stamp is stored under paths.readiness_root and
records bundle-relative references and hashes for checkpoints, manifests,
ledger rows, Slurm logs, the submission receipt, and the fresh smoke capacity
artifact. Among W&B fields, the current stamp records only the run-ID identity
marker, path, and hash; it does not record or hash the manually inspected run
config or event history. It
also records the allocated worker's H100 check and step-3 reserved-memory
headroom per CUDA rank. The runtime records 63.29 GiB usable H100 memory and a
7 GiB non-PyTorch reserve, so every rank's step-3 PyTorch reserved peak must be
at most 56.29 GiB before the full pair is allowed. A peak above that threshold
blocks full scale even if no OOM was reported.

This paired smoke is training-only; it does not exercise inference. Validate
inference separately from a completed full EMA checkpoint using the inference
stage and its decoded-frame checks.

For full scale, use a new pair ID, refresh its capacity artifact, and set
train.timeout_seconds from observed smoke throughput with a conservative
margin. The launcher requires the readiness stamp, rechecks smoke evidence and
hashes, compares each full manifest with its smoke manifest, disables the
memory probe, and uses the requested timeout for the worker's internal timer.
Slurm `--time` adds the configured kill-after and finalization grace:

Only the scale group, max_steps, log_iters, timeout_seconds, run ID, and
resolved config digest may differ. Loss weights, LoRA, model/checkpoint hashes,
prompt hashes, tracking project/mode, and seed must match the smoke manifest.

~~~bash
FULL_PAIR_ID="<fresh-full-pair-id>" # replace with a different unique lowercase ID
bash scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/nm5_submit_pair_23_30.sh \
  "$FULL_PAIR_ID" scale=full --after-smoke "$SMOKE_PAIR_ID" \
  train.timeout_seconds=<smoke-derived-seconds>
~~~

Full scale uses the versioned 600-step cap; train.max_steps cannot be
overridden. A smoke-derived timeout is required and must fit the runtime
partition MaxTime.

The fixed K/LR comparison is the recorded autotune waiver in
config/runtime.yaml:autotune_hyperparam; no hyperparameter sweep is implied.

## AutoDL direct tmux launch

Set the explicit AutoDL checkout root and prepared interpreter before running
the corresponding launcher:

```bash
AUTODL_DEEPRESEARCH_ROOT=/path/to/deepresearch \
SUE_PYTHON=/path/to/autodl/python \
  bash scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/autodl_run.sh \
    infer loop-infer-001 scale=smoke infer.num_samples=2
```

The launcher also requires `SUE_ASSET_ROOT` from the private backend
environment. Configure the inference adapter in the selected bundle with a
path relative to `SUE_EXP_DIR` before starting inference. It requires `tmux`,
verifies that four CUDA GPUs are visible for training or one for inference, and
runs the pipeline preflight before creating a detached session on a dedicated
tmux socket with a restricted worker environment. It writes each worker log
under the bundle-relative `paths.logs_root` and prints the exact attach command.
If `SUE_EXP_DIR` is not set, both backend launchers select the bundle shown above.

## Export a training checkpoint for inference

After a successful training run, export its full checkpoint into the selected
bundle. The exporter verifies the run config, manifest, method, base model, and
checkpoint before writing an inference adapter and provenance sidecar.
Replace the angle-bracket placeholders with the selected bundle/interpreter,
training run ID, method group, and backend before running the commands.

```bash
export SUE_EXP_DIR="<selected_bundle>"
export SUE_PYTHON="<prepared_interpreter>"
TRAIN_RUN_ID="<training_run_id>"
METHOD_GROUP="<method_group>"
BACKEND="<nm5_or_autodl>"
"$SUE_PYTHON" scripts/looped_self_forcing_pipeline/export_checkpoint.py \
  --exp-dir "$SUE_EXP_DIR" \
  --run-id "$TRAIN_RUN_ID" \
  --method "$METHOD_GROUP" \
  --backend "$BACKEND" \
  --full-checkpoint "ckpt/$TRAIN_RUN_ID/latest.pt" \
  --destination "ckpt/$TRAIN_RUN_ID/inference.pt"
```

The default `ckpt_root` is bundle-relative `ckpt/`. Use `infer.checkpoint` with
the exported adapter's bundle-relative name when launching inference; use a new
`run_id` for the inference run:

```bash
bash scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/autodl_run.sh \
  infer inference-run-001 \
  method="$METHOD_GROUP" infer.checkpoint="ckpt/$TRAIN_RUN_ID/inference.pt"
```

For NM5, use the matching `nm5_submit.sh` launcher with the same stage, run ID,
and overrides. The checkpoint override remains relative to `SUE_EXP_DIR`.

`config/runtime.yaml` is this bundle's SUE runtime contract. The older active
experiment's `scale_up_outputs/runtime.yaml` remains independent.
