# Training-Free Hidden-State Refinement for Flow-Matching Image Generators

Official implementation and data supplement for:

> **Training-Free Hidden-State Refinement for Flow-Matching Image Generators**<br>
> Yuanyi Yan, Xinzhe Rao, Canyu Shen, Yang Chen, Yunlu Chen, Meng Tang,
> Teng Long, and Vincent Tao Hu<br>
> [Paper](https://arxiv.org/abs/2608.29160) ·
> [PDF](https://arxiv.org/pdf/2608.29160) ·
> [Project page](https://yuanyiyan.com/projects/looped-flow-matching)

This repository provides the inference-time implementation, exact paper
presets, complete ordered benchmark prompt snapshots,
generation/export/aggregation utilities, aggregate paper results, tests, and
reproduction instructions. The method is organized as two independent choices:

1. the base token loop is **Dense Token Loop** or **Sparse Token Loop**; and
2. **Loop Guidance** is an optional prediction-space modifier on either base
   loop.

The Scale-RAE and RAEv2 adapters implement the complete 2-by-2 composition.
The PixArt-alpha and FLUX.2 transfer adapters implement the Dense Token Loop
column, with or without Loop Guidance; Sparse Token Loop was not evaluated for
those generators in the paper and is not claimed by their adapters.

The release is deliberately minimal. It contains no private cluster paths, job
metadata, credentials, pretrained weights, generated answers, generated image
grids, evaluator caches, or logs. The accompanying media supplement contains
selected prompt-matched visual comparisons.

Original code is provided under the MIT license in `LICENSE`. Third-party
prompt text and external model/evaluator components retain their upstream
terms; see `THIRD_PARTY.md`.

## Method matrix

| Base token loop | Without Loop Guidance | With Loop Guidance |
|---|---|---|
| Dense Token Loop | `dense_token_loop` | `loop_guidance_dense_token_loop` |
| Sparse Token Loop | `sparse_token_loop` | `loop_guidance_sparse_token_loop` |

`without_loop` is the ordinary denoiser path.  Loop Guidance never changes the
token-loop type: it combines an ordinary prediction with the prediction from
the selected Dense or Sparse Token Loop.

## Adjustable paper parameters

Every runtime parameter requested by the paper protocol is exposed both in
`configs/*.json` and through the shared command-line resolver:

| Paper quantity | JSON field | CLI override |
|---|---|---|
| Outer sampler steps `N` | `generation.num_inference_steps` | `--outer-steps` |
| Loop count `K` | `num_loops` | `--loop-count` |
| Total loop strength `lambda_loop` | `lambda_value` | `--lambda-loop` |
| Loop-active sampling interval | `start_frac`, `end_frac` | `--loop-active START END` |
| Inclusive loop layer range `[a,b]` | `block_indices` | `--loop-layers FIRST LAST` |
| Dense/Sparse token scope | `token_operator` | `--token-loop dense|sparse` |
| Loop Guidance and scale `g_lg` | `loopguidance_*` | `--loop-guidance`, `--loop-guidance-weight` |

Sparse routing can additionally be changed with `--selector`,
`--selection-ratio`, `--selector-seed`, and `--token-domain`.

The JSON files are immutable paper presets. CLI overrides are applied to a
copy, and the resolved record says `"paper_preset": false`.  This prevents a
custom setting from being mistaken for a reported result.

## Layout

```text
configs/       Exact reported presets and backend capability declarations
data/          Aggregate paper table values only
examples/      Scale-RAE and RAEv2 public-loader integration hooks
prompts/       Exact GenEval, DPG-Bench, and GenEval2 generation snapshots
requirements/ Backend-specific inference dependencies
scripts/       Resolve, generate, export, aggregate, and validate entry points
src/           Backend-neutral runtime and four backend adapters
tests/         CPU-only runtime, composition, and release-integrity tests
```

Start with [ENVIRONMENT.md](ENVIRONMENT.md), then follow
[REPRODUCE.md](REPRODUCE.md). A lightweight check that loads no generator is:

```bash
python scripts/validate_release.py
python -m unittest discover tests
```

Inspect an exact Sparse Token Loop + Loop Guidance preset:

```bash
python scripts/resolve_config.py \
  --config configs/raev2.json \
  --token-loop sparse \
  --loop-guidance
```

See `python scripts/resolve_config.py --help` for all overrides. The end-to-end
prompt-plan, batch-generation, evaluator-export, and score-aggregation commands
are in `REPRODUCE.md`.

## Citation

If this work is useful in your research, please cite:

```bibtex
@misc{yan2026trainingfree,
  title={Training-Free Hidden-State Refinement for Flow-Matching Image Generators},
  author={Yan, Yuanyi and Rao, Xinzhe and Shen, Canyu and Chen, Yang and Chen, Yunlu and Tang, Meng and Long, Teng and Hu, Vincent Tao},
  year={2026},
  eprint={2608.29160},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.29160}
}
```
