# Reproduction guide

## 1. Validate the archive

After installing the environment, run:

```bash
python scripts/validate_release.py
python -m unittest discover tests
```

These CPU-only checks validate every preset and every supported method
composition, exercise Dense/Sparse Token Loop with and without Loop Guidance,
verify all 2,418 ordered prompt snapshots and checksums, check paper table
shapes, and scan for retired method names, identity information,
machine-specific paths, weights, and generated outputs.

## 2. Resolve a method configuration

Method selection is compositional. Choose the base using `--token-loop` and
add `--loop-guidance` independently:

```bash
# Dense Token Loop
python scripts/resolve_config.py \
  --config configs/raev2.json --token-loop dense

# Sparse Token Loop
python scripts/resolve_config.py \
  --config configs/raev2.json --token-loop sparse

# Dense Token Loop + Loop Guidance
python scripts/resolve_config.py \
  --config configs/raev2.json --token-loop dense --loop-guidance

# Sparse Token Loop + Loop Guidance
python scripts/resolve_config.py \
  --config configs/raev2.json --token-loop sparse --loop-guidance
```

The resolver prints the final configuration and whether it is still an exact
paper preset. It does not load a model or generate an image.

## 3. Adjust the paper hyperparameters

The following example changes all requested experiment controls in one place:

```bash
python scripts/resolve_config.py \
  --config configs/raev2.json \
  --token-loop sparse \
  --loop-guidance \
  --outer-steps 37 \
  --loop-count 6 \
  --lambda-loop 0.75 \
  --loop-active 0.20 0.80 \
  --loop-layers 3 7 \
  --loop-guidance-weight 5.5 \
  --selector random \
  --selection-ratio 0.60
```

Layer endpoints are inclusive, so `--loop-layers 3 7` expands to
`[3,4,5,6,7]`. `lambda_loop` is the total strength and the runtime uses
`lambda_loop / K` for each of the `K` inner Euler updates. Sampling progress is
`i/(N-1)` and both interval endpoints are inclusive.

The same controls are available from Python:

```python
import json
from pathlib import Path

from diffusion_loop import resolve_run_config

payload = json.loads(Path("configs/raev2.json").read_text())
resolved = resolve_run_config(
    payload,
    token_loop="sparse",
    loop_guidance=True,
    outer_steps=37,
    num_loops=6,
    lambda_loop=0.75,
    loop_active=(0.20, 0.80),
    loop_layers=(3, 7),
    loop_guidance_weight=5.5,
)
generation = resolved.generation
loop_config = resolved.loop_config
```

Alternatively, construct a base loop directly and add guidance without
changing its token scope:

```python
from diffusion_loop import LoopConfig

sparse = LoopConfig(
    block_indices=list(range(15, 23)),
    num_loops=4,
    lambda_value=1.0,
    start_frac=0.0,
    end_frac=0.5,
    token_operator="sparse",
    token_domain="image_prefix_only",
    selector="random",
    selection_ratio=0.5,
)
sparse_with_guidance = sparse.with_loop_guidance(weight=6.0)
```

If a backend supports a composition that was not reported, no paper default is
invented. For example, Scale-RAE Sparse Token Loop + Loop Guidance requires an
explicit `--loop-guidance-weight`; its resolved record is marked custom.

## 4. Apply the backend adapter

Use `resolved.loop_config` with the adapter for the chosen backend:

- Scale-RAE: `apply_scale_rae_loop_patch`,
  `set_scale_rae_sampler_step_count`, `apply_scale_rae_sampler_step_hook`, and
  `set_scale_rae_loopguidance_config`.
- RAEv2 proxy: `apply_raev2_loop_patch` and `wrap_raev2_model_fn`.
- PixArt-alpha: `apply_pixart_loop_patch` and `set_pixart_step_metadata`.
- FLUX.2: `apply_flux2_loop_patch` and `set_flux2_step_metadata`.

Pass `generation["num_inference_steps"]` to the official outer sampler and to
the adapter's step tracker. Call the corresponding `restore_*` function before
switching configurations on a resident model. Reset the Scale-RAE sampler hook
or RAEv2 wrapped model function before every new image. For Diffusers pipelines,
bind `total_steps` immediately before each generation call.

The executable hook sequences for already loaded public Scale-RAE and RAEv2
checkpoints are in `examples/scale_rae_integration.py` and
`examples/raev2_integration.py`. They keep checkpoint loading in the public
backend while binding the exact resolved loop configuration to its sampler.

## 5. Freeze a generation plan

Before a full run, materialize one JSONL row per expected image. This catches
prompt-order, seed, method-name, and hyperparameter mismatches without loading
a model:

```bash
python scripts/prepare_benchmark.py \
  --config configs/raev2.json \
  --token-loop sparse \
  --loop-guidance \
  --prompt-file prompts/geneval_553.txt \
  --benchmark geneval \
  --output-root /tmp/token_loop_images \
  --plan /tmp/sparse_lg_geneval_plan.jsonl
```

Use `--max-prompts 2` for a preflight. Omit it for the standard paper split.
The standard snapshots contain 553 GenEval, 1,065 DPG-Bench, and 800 GenEval2
prompts; every method uses one image per prompt and seed 42.

## 6. Diffusers generation entry point

The included entry point runs PixArt-alpha or FLUX.2 with the same shared
arguments. It downloads nothing unless `--allow-download` is supplied and
refuses to overwrite an existing output.

```bash
python scripts/run_diffusers.py \
  --config configs/pixart.json \
  --token-loop dense \
  --loop-guidance \
  --model-path /path/to/local/model \
  --prompt 'your prompt' \
  --output /tmp/example.png
```

PixArt-alpha and FLUX.2 use Dense Token Loop in the reported transfer study;
their capability declarations reject Sparse Token Loop before model loading.

The same process loads the model once and generates a complete ordered split:

```bash
python scripts/run_diffusers.py \
  --config configs/pixart.json \
  --token-loop dense \
  --loop-guidance \
  --model-path /path/to/local/model \
  --prompt-file prompts/geneval_553.txt \
  --benchmark geneval \
  --output-dir /tmp/token_loop_images/loop_guidance_dense_token_loop/geneval \
  --records /tmp/pixart_geneval_records.jsonl
```

For each prompt, the entry point resets a CPU generator to seed 42, rebinds the
outer-step tracker, writes `promptNNNNN.png`, and appends a self-contained
record with the fully resolved configuration. Existing images or records are
never overwritten.

## 7. Models, prompts, and evaluators

The public model identifiers are in `configs/*.json`. Obtain checkpoints using
the official backend tooling or point the loader at a local snapshot. This
archive intentionally excludes pretrained weights, benchmark images, generated
answers, evaluator outputs, and credentials.

Use the prompt files under `prompts/`; their counts and SHA-256 digests are
fixed in `prompts/manifest.json`. Ground-truth annotations and evaluator
answers are intentionally absent and must be obtained from the official
GenEval, DPG-Bench, and GenEval2 releases. GenEval is evaluated at native
generation resolution without pre-upscaling. DPG-Bench uses its corrected CSV
parser, one image per prompt, and the public mPLUG visual-question-answering
checkpoint. Auxiliary scores are sample-count-weighted as defined in the
supplement.

Validate records and build each official evaluator's input layout. For
GenEval, pass the official `evaluation_metadata.jsonl`; DPG-Bench uses the
included public item-ID mapping; GenEval2 creates `image_paths.json`:

```bash
python scripts/export_benchmark.py \
  --records /tmp/pixart_geneval_records.jsonl \
  --prompt-file prompts/geneval_553.txt \
  --benchmark geneval \
  --method-id loop_guidance_dense_token_loop \
  --seed 42 \
  --annotations /path/to/official/evaluation_metadata.jsonl \
  --out-dir /tmp/geneval_export \
  --copy
```

Run the corresponding official evaluator on that export. Preserve its raw
output outside this archive. To reproduce common aggregation and baseline
deltas, normalize per-item scores to a CSV with columns
`method_id,benchmark,prompt_id,metric,value`, then run:

```bash
python scripts/aggregate_metrics.py \
  --per-item-csv /tmp/per_item_scores.csv \
  --baseline without_loop \
  --output /tmp/aggregated_scores.json
```

The aggregator rejects duplicate/non-finite rows, uses actual sample counts,
reports absolute values and baseline deltas, and omits percentage deltas for
ImageReward as required by the paper protocol.

Generate every benchmark image with the matching resolved config, run the
official evaluators, and compare aggregate scores with `data/*.csv`. Hardware
timing includes generation plus device synchronization, but excludes model
loading, tokenization, decoding, and file writing. Compare absolute latency
only on matched hardware and software.
