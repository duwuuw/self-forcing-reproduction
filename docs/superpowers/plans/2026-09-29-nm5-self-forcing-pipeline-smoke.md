# NM5 Self-Forcing paired pipeline smoke recovery plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` for implementation and independent review. Keep all edits limited to the files listed below.

**Goal:** Prove the K=2/K=3 layerwise 23–30 pipeline starts from the released Self-Forcing DMD EMA student checkpoint, runs both training and inference from the configured assets, and passes a fresh NM5 smoke before any fullrun.

**Architecture:** Keep the student as the causal Wan 1.3B architecture initialized from the rewrapped Self-Forcing EMA generator state, then train LoRA adapters. Use the configured Wan 14B real-score teacher and Wan 1.3B fake-score model. Resolve relative model assets from the selected asset root and resolve copied-source configs from the copied source tree.

**Tech Stack:** Python 3.10–3.12, PyTorch/FSDP, Hydra/OmegaConf, Weights & Biases offline mode, Slurm on NM5.

**Spec:** User’s paired K=2/K=3 experiment instructions; `scale_up_outputs/looped_self_forcing_pipeline/config/runtime.yaml`; the two `layerwise_l23_30_*` method configs; pipeline README.

## Global Constraints

- Target NM5 only; each active training job requests exactly one node with 4 H100 GPUs.
- Run K=3 then K=2 serially with `afterany`; never run both allocations concurrently.
- Both variants use paper layers 23–30 (indices 22–29), LoRA rank 8 / alpha 16 / dropout 0, seed 1, and `lr=4e-7`; K=3 keeps `lr_critic=4e-7`, while K=2 uses `lr_critic=8e-8`.
- Student generator initialization is `checkpoints/self_forcing_dmd_ema_as_generator.pt`; the resolved checkpoint must be recorded by SHA-256.
- W&B remains offline on NM5 and must record the run config and loss metrics.
- Do not submit a fullrun until both jobs in a fresh smoke pair complete and the immutable readiness stamp passes checkpoint, ledger, W&B, H100, and per-rank memory checks.
- Do not modify vendored files under `preliminary_code/` or reuse either failed smoke pair ID.

## Review Focus

1. The exact selected Self-Forcing checkpoint and its wrapped `generator` tensors match the documented released EMA source.
2. The Wan 1.3B name is treated as architecture/auxiliary/fake-score context; the full Self-Forcing state is loaded into the causal generator before LoRA insertion.
3. Both train and inference resolve `wan_models/...` from the selected asset root, independent of the source checkout directory.
4. Training/inference default configs resolve from the copied source tree, while relative `wan_models/...` reads resolve from the configured asset-root CWD.
5. Preflight and runtime consume the same copied-source default config and validate the selected model/checkpoint assets before allocating GPUs.

---

### Task 1: Verify student checkpoint identity without GPU work

**Files:**
- Read: `scale_up_outputs/looped_self_forcing_pipeline/config/config.yaml`
- Read: `scale_up_outputs/looped_self_forcing_pipeline/config/method/layerwise_l23_30_k2_lr5both.yaml`
- Read: `scale_up_outputs/looped_self_forcing_pipeline/config/method/layerwise_l23_30_k3_lr5gen.yaml`
- Read: `scripts/looped_self_forcing_pipeline/source/model/base.py`
- Read: `scripts/looped_self_forcing_pipeline/source/trainer/distillation.py`

- [ ] Confirm both profiles select `self_forcing_dmd_ema_as_generator.pt` and that the loader applies it with `strict=True` before LoRA insertion.
- [ ] Through the configured NM5 login route, record the selected checkpoint hash and metadata-only wrapper shape; if the original release file is present, compare its `generator_ema` tensors to the wrapped `generator` tensors without printing tensor values.
- [ ] If the original release file is unavailable, record that limitation and do not claim tensor-level provenance; keep the configured Self-Forcing artifact unchanged and record its actual hash in the run manifest.

### Task 2: Make train and inference path resolution explicit

**Files:**
- Modify: `scripts/looped_self_forcing_pipeline/worker.py`
- Modify: `scripts/looped_self_forcing_pipeline/source/train.py`
- Modify: `scripts/looped_self_forcing_pipeline/source/inference.py`
- Modify: `scripts/looped_self_forcing_pipeline/COPY_MANIFEST.json`
- Test: `tests/test_looped_self_forcing_worker.py`
- Test: `tests/test_looped_self_forcing_entrypoint_config.py`

- [ ] Keep the failing regression proving both train and inference subprocesses use `prepared["assets"]["asset_root"]` as CWD, not `SOURCE_ROOT`.
- [ ] Review the existing uncommitted CWD change and parameterized test before making any further edits.
- [ ] Add failing coverage proving copied train/inference `load_config()` uses each file’s own `configs/default_config.yaml` when the process CWD is unrelated.
- [ ] Use the validated asset root as CWD for actual train/infer subprocesses so relative `wan_models/...` reads resolve locally; keep stage preflight on the copied source root.
- [ ] Anchor each copied entrypoint’s default config at `Path(__file__).resolve().parent` and update the corresponding `COPY_MANIFEST.json` hashes.
- [ ] Run the focused preflight/CWD tests, then the complete looped pipeline suite and full workspace suite.

### Task 3: Clarify the documented model roles

**Files:**
- Modify: `scripts/looped_self_forcing_pipeline/README.md`

- [ ] State that the causal generator starts from the Self-Forcing DMD EMA checkpoint and LoRA is applied afterward.
- [ ] Label Wan 1.3B as the compatible causal architecture plus fake-score/auxiliary assets, and Wan 14B as the real-score teacher.
- [ ] Explain that both preflight and runtime use the selected asset root for relative model/config files.
- [ ] State that a smoke is not accepted until the Self-Forcing checkpoint load, training step, offline W&B identity, ledger row, final checkpoint, and four-rank memory evidence are verified.

### Task 4: Review, acceptance, and durable delivery

**Files:**
- Review the exact staged diff only.

- [ ] Run `PYTHONPATH=. uv run pytest -q` and record the complete result.
- [ ] Run independent code review, `student-codebase-audit`, and `sue-doc-code-consistency`; fix only findings tied to this plan.
- [ ] Run privacy audit and stage only the files named above plus the workspace lesson for the confirmed asset-root/CWD failure.
- [ ] Commit and push the nested workspace branch; sync only changed production files to NM5 and verify SHA-256 per file.

### Task 5: Retry smoke, then fullrun

- [ ] Query fresh NM5 capacity for a new smoke pair ID and store matching local/remote `max_parallel.json` artifacts.
- [ ] Run the parent and per-profile preflights, then submit the two 10-step jobs serially on 4×H100.
- [ ] Monitor through terminal states. Require both jobs `COMPLETED`, final checkpoint and ledger evidence, offline W&B run IDs, four H100 markers, and all four rank-3 reserved-memory peaks at or below 56.29 GiB.
- [ ] Write and verify the immutable smoke readiness stamp.
- [ ] Refresh capacity for a distinct full pair, derive `train.timeout_seconds` from measured smoke throughput, and submit the 600-step full pair only after readiness passes.
