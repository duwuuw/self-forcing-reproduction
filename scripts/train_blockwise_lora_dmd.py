#!/usr/bin/env python
"""Out-of-tree launch wrapper for the block-wise LoRA DMD training entry point.

Runtime differences are corrected here rather than by editing the
`Self-Forcing-blockwise` checkout, so the branch keeps its pinned commit:

1. **W&B cannot go offline.** `trainer/distillation.py:58-67` hardcodes
   ``wandb.init(..., mode="online", ...)`` and always calls
   ``wandb.login(host=config.wandb_host, key=config.wandb_key)``. NM5 has no
   outbound internet, so the run hangs; ``WANDB_MODE=offline`` does not help
   because an explicit ``mode=`` kwarg outranks the env var, and the preset's
   literal ``WANDB_HOST`` placeholder raises a pydantic ``ValidationError``.

2. **The checkpoint/cache boundary diverges from Self-Forcing.** The runtime
   alignment helper restores the upstream cross-attention boundary while
   preserving the branch's detached KV replay snapshots for every cached
   checkpointed block.

Usage is identical to train.py:
    python scripts/train_blockwise_lora_dmd.py \\
        --config_path configs/self_forcing_dmd_temporal_loop_lora_train.yaml \\
        --logdir <dir> --wandb-save-dir <dir>
"""

from __future__ import annotations

import os
import sys

# Must be set before anything imports wandb.
_REQUESTED_MODE = os.environ.get("SUE_WANDB_MODE", "offline")
os.environ.setdefault("WANDB_MODE", _REQUESTED_MODE)


def _install_offline_wandb() -> None:
    """Force offline W&B on the branch's trainer without editing the branch."""

    import wandb

    # Validate the assumption this wrapper depends on: an explicit mode kwarg
    # outranks WANDB_MODE, so we must rewrite the kwarg itself.
    original_init = wandb.init
    original_login = wandb.login

    def offline_init(*args, **kwargs):
        kwargs["mode"] = _REQUESTED_MODE
        return original_init(*args, **kwargs)

    def offline_login(*args, **kwargs):
        # Offline runs need no credentials and must not touch the network; the
        # preset's placeholder host would raise ValidationError.
        return True

    wandb.init = offline_init
    wandb.login = offline_login

    # The trainer does `import wandb` at module scope, so patching the module
    # attribute is enough -- but assert it rather than assume.
    import trainer.distillation as distillation

    if getattr(distillation, "wandb", None) is not wandb:
        raise SystemExit(
            "WRAPPER ERROR: trainer.distillation.wandb is not the wandb module; "
            "the offline patch would not take effect"
        )
    if distillation.wandb.init is not offline_init:
        raise SystemExit(
            "WRAPPER ERROR: trainer.distillation.wandb.init is not the patched "
            "callable; offline mode would not be applied"
        )
    print(f"WRAPPER: wandb forced to mode={_REQUESTED_MODE!r}, login skipped", flush=True)


def _install_native_checkpoint_alignment():
    """Restore Self-Forcing's cross-attention boundary around checkpoints."""

    # The wrapper is executed as a script from ``scripts/`` rather than
    # imported as a package, so this sibling import is intentional.
    from self_forcing_runtime_alignment import install_native_checkpoint_alignment

    return install_native_checkpoint_alignment()


def _install_teacher_cpu_offload() -> None:
    """Keep the 14B teacher and the critic in host memory.

    ``trainer/distillation.py`` only forwards a ``cpu_offload`` flag for the text
    encoder, so ``real_score`` (frozen Wan2.1-T2V-14B) and ``fake_score`` (the
    trainable 1.42B critic) both keep their flat shard resident on the GPU.

    * Teacher: ~14 GiB/rank under a 4-way FSDP shard, against ~0.9 GiB on the
      64-GPU reference run. It is frozen and only used under ``torch.no_grad()``
      in ``_compute_kl_grad``, so host residency costs transfer, not correctness.
    * Critic: ~5.7 GiB/rank once its fp32 shard, gradient and two Adam moments
      are counted. Offloading a *trainable* unit is supported but slower: the
      shard moves host<->device around each forward/backward, and its optimizer
      state lives on the host.

    Both are needed on reduced-memory nodes: keeping only the teacher offloaded
    can still leave the teacher forward without enough free device memory.

    Applied from outside the branch; the clean fix is a ``real_score_cpu_offload``
    / ``fake_score_cpu_offload`` config key in the branch.

    Identification: the teacher by parameter count (>10B). The critic is the
    other 1.4B ``WanDiffusionWrapper`` whose inner model is a non-causal
    ``WanModel`` -- the 1.42B student is a ``CausalWanModel`` and the 5.68B T5 is
    a ``WanTextEncoder``, so this selects the critic uniquely.
    """

    import trainer.distillation as distillation

    original = distillation.fsdp_wrap
    if getattr(original, "_sue_teacher_offload", False):
        print("WRAPPER: teacher CPU offload already installed", flush=True)
        return

    def fsdp_wrap_with_teacher_offload(module, *args, **kwargs):
        try:
            n_params = sum(p.numel() for p in module.parameters())
        except Exception:
            n_params = 0
        inner = getattr(module, "model", None)
        inner_name = type(inner).__name__ if inner is not None else ""
        reason = ""
        if n_params > 10_000_000_000:
            reason = "teacher"
        elif n_params < 2_000_000_000 and inner_name == "WanModel":
            reason = "critic"
        forced = False
        if reason and not kwargs.get("cpu_offload"):
            kwargs["cpu_offload"] = True
            forced = True
        # FSDP wrap granularity. The branch calls fsdp_wrap with wrap_strategy
        # "size" and no min_num_params, so it takes the default 5e7. This model's
        # transformer blocks are ~4.7e7 params, i.e. *just* under that, so the
        # whole 1.42B student lands in ONE FSDP unit and its per-forward
        # all-gather asks for a multi-gigabyte single allocation. Lowering the
        # threshold to 4e7 wraps each block separately, so the gather is ~94 MB.
        # It should also let all-frozen (non-LoRA) blocks become units whose
        # parameters do not require grad, collapsing their autograd graph instead
        # of retaining activations for them.
        # Granularity is a memory/communication choice; it does not change the
        # math. Applied only where the caller left the default in place.
        if (
            1_000_000_000 < n_params < 2_000_000_000
            and kwargs.get("wrap_strategy", "size") == "size"
            and "min_num_params" not in kwargs
        ):
            kwargs["min_num_params"] = 40_000_000
        result = original(module, *args, **kwargs)
        # One audit line per wrapped unit: this is the evidence that the teacher
        # and the T5 text encoder really are running with host-resident params
        # rather than just being configured to.
        print(
            f"WRAPPER: fsdp_wrap {type(module).__name__} params={n_params / 1e9:.2f}B "
            f"cpu_offload={bool(kwargs.get('cpu_offload'))} "
            f"min_num_params={kwargs.get('min_num_params', 'default')}"
            + (f" (forced by wrapper: {reason})" if forced else ""),
            flush=True,
        )
        return result

    fsdp_wrap_with_teacher_offload._sue_teacher_offload = True
    distillation.fsdp_wrap = fsdp_wrap_with_teacher_offload

    if distillation.fsdp_wrap is not fsdp_wrap_with_teacher_offload:
        raise SystemExit("WRAPPER ERROR: could not install teacher CPU offload")
    print("WRAPPER: teacher CPU offload installed", flush=True)


_PROBE = {"active": False, "step": 0}


class _TrainingStepCapReached(Exception):
    """Normal wrapper-level termination after a real checkpoint boundary."""


def _install_memory_probe() -> None:
    """Log GPU memory at named points so the accounting stops being guesswork.

    The probe instruments the real execution path instead of relying on static
    estimates.

    Two levels:
      * every step, a one-line total (allocated / reserved / peak) to stdout and a
        couple of scalars to W&B;
      * on one designated step (``SUE_MEM_PROBE_STEP``), a nested trace through
        the rollout, the teacher KL term and the DMD loss, with the peak counter
        reset first so the numbers are that step's, not the run's high-water mark.

    Everything is wrapped in try/except: a probe must never take down training.
    """

    import os as _os

    import torch

    import model.dmd as dmd_mod
    import pipeline.self_forcing_training as sfp
    import trainer.distillation as distillation

    gib = float(2 ** 30)

    def _snap(tag: str, force: bool = False, to_wandb: bool = False, step=None) -> None:
        if not force and not _PROBE["active"]:
            return
        try:
            allocated = torch.cuda.memory_allocated() / gib
            reserved = torch.cuda.memory_reserved() / gib
            peak = torch.cuda.max_memory_allocated() / gib
            # reserved-but-unallocated has been the recurring story in the OOMs,
            # so log it explicitly rather than making the reader subtract.
            frag = reserved - allocated
            print(
                f"MEMPROBE step={_PROBE['step']} {tag}: allocated={allocated:.2f}GiB "
                f"reserved={reserved:.2f}GiB frag={frag:.2f}GiB peak={peak:.2f}GiB",
                flush=True,
            )
            if to_wandb:
                import wandb

                # The native trainer creates one W&B run on rank 0.  Keep the
                # probe's console telemetry on every rank, but do not call
                # wandb.log from the other ranks (they have no active run).
                if getattr(wandb, "run", None) is not None:
                    wandb.log(
                        {
                            f"mem/{tag}_allocated": allocated,
                            f"mem/{tag}_reserved": reserved,
                            f"mem/{tag}_frag": frag,
                            "mem/peak": peak,
                        },
                        step=step,
                    )
        except Exception as exc:  # noqa: BLE001 - probing must never abort training
            print(f"MEMPROBE WARNING: {tag}: {type(exc).__name__}: {exc}", flush=True)

    probe_step = int(_os.environ.get("SUE_MEM_PROBE_STEP", "3"))

    # --- static footprint once the trainer is fully built -------------------
    original_init = distillation.Trainer.__init__

    def init_probe(self, config):
        original_init(self, config)
        _PROBE["step"] = 0
        _snap("after_construction", force=True, to_wandb=True, step=0)

    distillation.Trainer.__init__ = init_probe

    # --- nested trace on the designated step --------------------------------
    def make_wrapper(cls, name, label):
        original = getattr(cls, name)

        def wrapped(self, *args, **kwargs):
            _snap(label + "_enter")
            out = original(self, *args, **kwargs)
            _snap(label + "_exit")
            return out

        setattr(cls, name, wrapped)

    for cls, name, label in (
        (sfp.SelfForcingTrainingPipeline, "inference_with_trajectory", "rollout"),
        (dmd_mod.DMD, "_compute_kl_grad", "teacher_kl"),
        (dmd_mod.DMD, "compute_distribution_matching_loss", "dmd_loss"),
        (dmd_mod.DMD, "critic_loss", "critic_loss"),
    ):
        make_wrapper(cls, name, label)

    # --- per-step totals, and the switch for the deep trace ------------------
    original_step = distillation.Trainer.fwdbwd_one_step

    def step_probe(self, batch, train_generator):
        _PROBE["step"] = self.step
        deep = self.step == probe_step
        if deep:
            torch.cuda.reset_peak_memory_stats()
            _PROBE["active"] = True
            print(f"===== MEMPROBE deep trace at step {self.step} "
                  f"(train_generator={train_generator}) =====", flush=True)
            _snap("step_enter", to_wandb=True, step=self.step)
        out = original_step(self, batch, train_generator)
        if deep:
            _snap("step_exit", to_wandb=True, step=self.step)
            _PROBE["active"] = False
        else:
            _snap("step_exit", to_wandb=True, step=self.step)
        return out

    distillation.Trainer.fwdbwd_one_step = step_probe
    print(
        f"WRAPPER: memory probe installed (deep trace at step {probe_step})",
        flush=True,
    )


def _install_training_callbacks() -> None:
    """Attach periodic sample-video logging and an optional step cap.

    Hooked onto ``Trainer.save`` because that is the only per-N-steps callback the
    branch exposes: ``train()`` is a ``while True:`` loop that calls ``save()``
    every ``log_iters`` iterations and writes checkpoints only on rank 0, while the
    surrounding state-dict gather runs on every rank. Sampling must likewise run on
    **every** rank, because the rollout goes through the FSDP-wrapped generator and
    therefore issues collectives; only the W&B write is rank-0-gated.

    Sampling reuses the branch's own inference path rather than
    ``Trainer.generate_video``, which is dead code on this path: it calls
    ``pipeline.inference(...)``, a method ``SelfForcingTrainingPipeline`` does not
    have, and scales the decoded frames by ``255.0`` alone even though
    ``WanVAEWrapper.decode_to_pixel`` returns ``clamp_(-1, 1)`` values. The
    ``255.0 * (x * 0.5 + 0.5)`` in ``trainer/ode.py:199-204`` is the correct range.

    Environment knobs:
        SUE_MAX_STEPS       stop after this many optimizer steps (0 = no cap)
        SUE_SAMPLE_EVERY    optional sample cadence; 0 (the native default)
                            disables video sampling
        SUE_SAMPLE_PROMPTS  how many prompts to render per sample
    """

    import os as _os
    import random

    import numpy as np
    import torch
    import wandb

    import trainer.distillation as distillation
    from utils.dataset import TextDataset

    max_steps = int(_os.environ.get("SUE_MAX_STEPS", "0"))
    sample_every = int(_os.environ.get("SUE_SAMPLE_EVERY", "0"))
    n_prompts = int(_os.environ.get("SUE_SAMPLE_PROMPTS", "2"))

    original_save = distillation.Trainer.save
    if getattr(original_save, "_sue_callbacks", False):
        print("WRAPPER: training callbacks already installed", flush=True)
        return

    def _sample_videos(self) -> None:
        """Render a couple of prompts with the current generator and log to W&B."""
        # The pipeline needs a valid batch shape on every rank; keep it identical
        # across ranks so the FSDP collectives line up.
        frames = int(self.model.num_training_frames)
        prompts = [self._sue_prompt_pool[i] for i in
                   random.Random(self.step).sample(range(len(self._sue_prompt_pool)), n_prompts)]
        print(f"WRAPPER: sampling {n_prompts} videos at step {self.step}: {prompts}", flush=True)

        torch.cuda.empty_cache()
        with torch.no_grad():
            conditional = self.model.text_encoder(text_prompts=prompts)
            noise = torch.randn(
                [len(prompts), frames, 16, 60, 104],
                device=self.device, dtype=self.dtype,
            )
            latent, _, _ = self.model.inference_pipeline.inference_with_trajectory(
                noise=noise, **conditional
            )
            # decode_to_pixel clamps to [-1, 1]; map to 0-255 uint8 for the encoder.
            pixel = self.model.vae.decode_to_pixel(latent)
            frames_u8 = (255.0 * (pixel.float().cpu().numpy() * 0.5 + 0.5)).clip(0, 255).astype(np.uint8)
        torch.cuda.empty_cache()

        if self.is_main_process:
            payload = {}
            for i, prompt in enumerate(prompts):
                # frames_u8[i] is [F, C, H, W], which is what wandb.Video expects.
                payload[f"sample/prompt_{i}"] = wandb.Video(
                    frames_u8[i], caption=prompt[:120], fps=16, format="mp4"
                )
            wandb.log(payload, step=self.step)
            print(f"WRAPPER: logged {n_prompts} sample videos to wandb at step {self.step}", flush=True)

    def save_with_callbacks(self):
        original_save(self)

        if sample_every > 0 and self.step > 0 and self.step % sample_every == 0:
            # A monitoring failure must not take down a multi-hour run; report it
            # loudly and keep training.
            try:
                if getattr(self, "_sue_prompt_pool", None) is None:
                    self._sue_prompt_pool = [
                        p.strip() for p in TextDataset(self.config.data_path).prompt_list
                        if p.strip()
                    ]
                    print(f"WRAPPER: prompt pool loaded ({len(self._sue_prompt_pool)} prompts) "
                          f"from {self.config.data_path}", flush=True)
                _sample_videos(self)
            except Exception as exc:
                print(f"WRAPPER WARNING: sample-video logging failed at step {self.step}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()

        if max_steps > 0 and self.step >= max_steps:
            print(f"WRAPPER: step cap {max_steps} reached; finishing cleanly", flush=True)
            if self.is_main_process:
                try:
                    wandb.finish()
                except Exception:
                    pass
            # train.py only calls wandb.finish() after trainer.train() returns, and
            # trainer.train() never returns; unwind here instead.  main() catches
            # this wrapper-only signal so the distributed process group can be
            # closed cleanly before torchrun exits.
            raise _TrainingStepCapReached

    save_with_callbacks._sue_callbacks = True
    distillation.Trainer.save = save_with_callbacks
    print(
        f"WRAPPER: training callbacks installed "
        f"(max_steps={max_steps or 'none'}, sample_every={sample_every or 'off'}, "
        f"prompts_per_sample={n_prompts})",
        flush=True,
    )


def main() -> int:
    import train

    if "--preflight" in sys.argv[1:]:
        # Preflight constructs no models and never touches wandb; run the
        # branch's own code path unmodified.
        return train.main()

    # Patch before train.main() constructs the trainer. train.main() imports
    # `from trainer import ...` lazily, and trainer.distillation is already in
    # sys.modules by then, so it picks up the patched module.
    _install_offline_wandb()
    _install_native_checkpoint_alignment()
    _install_teacher_cpu_offload()
    _install_training_callbacks()
    if os.environ.get("SUE_MEM_PROBE") == "1":
        _install_memory_probe()
    try:
        return train.main()
    except _TrainingStepCapReached:
        # The native train loop is intentionally untouched.  All ranks reach
        # the checkpoint boundary together; close their process groups before
        # returning success to avoid a false NCCL teardown warning.
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask success
            print(f"WRAPPER WARNING: distributed cleanup failed: {exc}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
