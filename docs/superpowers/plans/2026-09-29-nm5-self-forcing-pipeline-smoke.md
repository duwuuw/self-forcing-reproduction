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
- [ ] Monitor through terminal states. Require both jobs `COMPLETED`, final checkpoint and ledger evidence, offline W&B run IDs, four H100 markers, and all four rank-3 reserved-memory peaks at or below 56.29 GiB. Manually inspect each profile's resolved W&B run config and actual offline event history before `--verify-smoke`; require finite losses at the expected steps, including K3 generator and critic losses at step 1 and critic loss at step 10. The readiness stamp records W&B identity only and does not replace this manual history check.
- [ ] Write and verify the immutable smoke readiness stamp.
- [ ] Refresh capacity for a distinct full pair, derive `train.timeout_seconds` from measured smoke throughput, and submit the 600-step full pair only after readiness passes.

### Task 6: Repair the empty LoRA checkpoint and retain the fresh-smoke gate

**Evidence from smoke C:** The K3 job reached its 10-step cap and wrote a checkpoint marked `generator_format=lora_adapter`, `metadata.step=10`, and `metadata.final=true`. Metadata-only checkpoint inspection found an empty `generator` mapping, no generator EMA, and 825 critic entries. Training logs had three LoRA-name markers, each listing 32 unique adapter names; the generator optimizer had 32 state entries and 32 parameters. This rules out LoRA injection being absent and points to adapter extraction/serialization after optimizer setup. The worker then failed closed because there were no generator weights. This is separate from the earlier import-shadowing and asset-root CWD failures.

**Historical evidence gap:** The resolved `latest.pt` target and SHA-256 were confirmed, and the runtime imported all three training modules from the copied source tree under torch `2.5.1+cu124` / PEFT `0.21.0`, but smoke C did not capture the intermediate raw and PEFT-filtered state-dict key counts. A faithful nested-wrapper regression later reproduced an empty PEFT-filtered mapping despite the expected LoRA parameters. Treat the missing smoke-C counts as a diagnostic limitation, not as an unfinished pre-fix investigation. The source confirms K3 should log generator and critic losses at step 1, then critic losses at step 10; the save runs before the step-10 `wandb.log`, so this failed save can explain a missing final loss and skipped `wandb.finish()`. A missing numeric `wandb-summary.json` does not establish that offline event history is empty. The actual history remains unverified.

**Implemented fault boundary repair:** LoRA injection checks the selected trainable names before FSDP wrapping and the optimizer rejects an empty or non-LoRA trainable set. Checkpoint save now calls `fsdp_lora_state_dict()` with those expected names, collects only trainable `lora_A`/`lora_B` tensors from `named_parameters()` inside `summon_full_params()`, normalizes nested FSDP/checkpoint/default-adapter path segments, and rejects any key-set mismatch against the injected names. `build_lora_checkpoint_payload()` rejects empty or non-LoRA generator mappings before publication; the exporter and worker independently enforce the exact selected adapter scope.

**Regression evidence and remaining gate:** The original regression reproduced an empty PEFT-filtered mapping for a faithful 32-tensor nested-wrapper layout. Focused coverage now drives the actual `fsdp_lora_state_dict()` control flow with a summon-context harness, verifies wrapper extraction plus all 32 names and values, and exports, serializes, reloads, and restores every tensor into a fresh equivalent model through `load_lora_state_dict()`. Local torch lacks PEFT and a CPU accelerator for a real FSDP instance, so live four-rank FSDP and PEFT 0.21 compatibility remain required acceptance checks in the fresh NM5 K2/K3 smoke pair.

**Smoke acceptance status:** The worker finalizer now reuses the inference scope validator to reject wrong-layer or incomplete generator adapters independently of the trainer. The next smoke remains blocked on live FSDP/PEFT export and load evidence plus real offline W&B event-history inspection. Acceptance requires the resolved W&B run config and finite recorded losses, not only a run-ID marker or summary file; K3 must contain generator and critic losses at step 1 and critic loss at step 10. History parsing remains a manual smoke acceptance check in this task.

**Files:**
- Reviewed and modified: `scripts/looped_self_forcing_pipeline/source/model/lora.py`
- Reviewed and modified: `scripts/looped_self_forcing_pipeline/source/utils/distributed.py`
- Reviewed and modified: `scripts/looped_self_forcing_pipeline/source/trainer/distillation.py`
- Further implementation changes require a new reproduced failure and a scoped review.
- Tests: focused LoRA export tests and `tests/test_looped_self_forcing_worker.py`.

- [x] Reconcile the smoke checkpoint pointer with its concrete file and confirm metadata/counts before modifying the exporter. The resolved target is `checkpoint_model_000010/model.pt`; SHA-256 and metadata/counts are recorded above.
- [x] Confirm the runtime module origins and torch/PEFT versions; all three modules resolved from the copied source tree.
- [x] Preserve the smoke-C diagnostic limitation: the checkpoint had no generator tensors despite 32 injected/optimized adapters, but intermediate raw and PEFT-filtered outputs and key counts were not captured. The controlled nested-wrapper regression later reproduced empty PEFT-filtered output; the fresh smoke remains the live compatibility check.
- [x] Reproduce the failing boundary with a tiny model or controlled training save path. The faithful nested-wrapper regression reproduced an empty filtered mapping despite 32 trainable A/B tensors.
- [ ] Reconcile W&B offline event history directly and verify real loss records at K3 step 1 and step 10 after the next successful smoke; require the actual run history, not only a run-ID marker or summary JSON.
- [x] Add a failing regression for the demonstrated cause before changing implementation. It requires the exact selected-block A/B key scope of 32 tensors and their values.
- [x] Use direct trainable-parameter extraction and exact expected-name comparison after the nested-prefix hypothesis reproduced.
- [x] Reject empty or non-LoRA adapter mappings at payload construction/save time; the exporter checks exact injected names and the worker independently validates the selected layer scope.
- [x] Add save/load coverage proving a valid exported adapter restores every tensor into a fresh model and an empty adapter fails before a checkpoint is published.
- [x] Run focused and workspace suites and complete independent code review; results are recorded in the implementation report.
- [x] Run `student-codebase-audit`; the current result remains partially conforming because live NM5 FSDP/PEFT, W&B history, and readiness evidence are pending.
- [x] Rerun focused `sue-doc-code-consistency` after correcting this section; the Task 6 plan, pipeline README, and implementation contracts are consistent.
- [ ] Do not start the next smoke until the reviewer confirms the extraction fix is supported by the reproduced cause. Do not submit fullrun until a fresh K2/K3 smoke pair and readiness checks pass.
