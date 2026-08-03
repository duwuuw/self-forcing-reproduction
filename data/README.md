# Aggregate paper data

This directory contains the numerical values reported in the paper and
supplement, in machine-readable form. It intentionally contains only aggregate
metrics and public experiment settings. Generated images, model outputs,
per-prompt evaluator answers, cluster logs, model weights, and private paths are
not included.

- `main_results.csv`: the main RAE comparison, Loop Guidance comparison, and
  VAE-transfer diagnostic.
- `scale_rae_1p5b_ablations.csv`: outer-step, sampling-progress, loop-count,
  layer-range, and Loop Guidance sweeps.
- `raev2_ablations.csv`: layer-range and Loop Guidance sweeps for the public
  RAE-SigLIP2-B/DDT proxy.
- `sparsifier_results.csv`: the paper's sparsifier comparison.

All method labels use the terminology in the submitted paper. Scores are
rounded exactly as displayed in the tables; use the paper for evaluator
definitions and uncertainty/interpretation.
