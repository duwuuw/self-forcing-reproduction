# Backend-neutral looped Self-Forcing pipeline

This entry point composes the run configuration from the active SUE bundle.
The default bundle is `scale_up_outputs/looped_self_forcing_pipeline`; set
`SUE_EXP_DIR` to select another compatible bundle. Its Hydra groups live under
`<SUE_EXP_DIR>/config/`.

The `source/` tree is the code-only runtime copied from the previous NM5
pipeline. Its Python modules and default YAML are preserved byte-for-byte,
including the temporal-loop and model code. `COPY_MANIFEST.json` records the
source hashes. Model weights, datasets, generated videos, run outputs, and
Python caches are not part of this source migration.

The copied upstream source is licensed under Apache License 2.0. Its original
`LICENSE` is preserved at `source/LICENSE`; the source checkout had no
`NOTICE` file.

## Runtime inputs

Set `SUE_ASSET_ROOT` through the ignored, private backend environment. It must
name an existing absolute directory containing `checkpoints/` and
`wan_models/`; the pipeline resolves its configured model and checkpoint names
relative to that root. Keep the root value and host-specific asset locations
out of tracked config and command-line overrides. Both launchers forward the
environment variable to preflight and the worker without placing its value in
Slurm or tmux arguments.

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
`infer.seed`, and `infer.use_ema`. Checkpoint overrides must be relative to
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

Load the private NM5 environment in the normal operator shell, set
`SUE_PYTHON` to the prepared interpreter, and choose a fresh `run_id`:

```bash
SUE_PYTHON=/path/to/nm5/python \
  bash scale_up_outputs/looped_self_forcing_pipeline/slurm_scripts/nm5_submit.sh \
    train loop-train-001 scale=smoke train.max_steps=20 train.log_iters=5
```

The launcher requires `NM5_DEEPRESEARCH_ROOT`, `NM5_ACCOUNT`, `NM5_PARTITION`,
`NM5_QOS`, and `SUE_ASSET_ROOT`. It runs the stage preflight before submission,
requests four GPUs for training or one GPU for inference, and writes Slurm
stdout and stderr under the bundle-relative `paths.logs_root`. The Slurm job
name uses the username from workspace `user.yaml`; invalid or missing values
use the required `silly-` prefix. Its remaining name matches the W&B experiment
name after resolving the selected method, run ID, and stage. The batch job receives only the runtime
environment variables it needs, not the entire submitting shell environment.

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
