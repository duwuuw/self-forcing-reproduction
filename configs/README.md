# Configuration semantics

Each JSON file contains two deliberately separate records:

- `methods` contains only exact configurations reported in the paper.
- `capabilities` states which base token loops and Loop Guidance compositions
  the released backend adapter can execute.

The method IDs are unambiguous:

- `without_loop`
- `dense_token_loop`
- `sparse_token_loop`
- `loop_guidance_dense_token_loop`
- `loop_guidance_sparse_token_loop`

Scale-RAE and RAEv2 support Dense and Sparse Token Loop, each with optional
Loop Guidance. The paper reports both guided variants for RAEv2, but reports
only the dense guided variant for Scale-RAE. Consequently, a custom Scale-RAE
sparse guided run must supply `loop_guidance_weight` explicitly. PixArt-alpha
and FLUX.2 report and expose Dense Token Loop only, with optional Loop Guidance.

Use `scripts/resolve_config.py` rather than editing a preset in place. The
resolver accepts overrides for outer steps, `K`, `lambda_loop`, active sampling
interval, layer range, Dense/Sparse selection, Loop Guidance, and sparse routing,
then prints the complete record that should accompany a custom run.
