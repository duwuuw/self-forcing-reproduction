"""Keep the branch's checkpoint boundary aligned with Self-Forcing.

Self-Forcing does not pass a mutable cross-attention cache into a checkpointed
block.  The block-wise branch does, because its causal KV replay transaction is
needed by the gradient-enabled multi-chunk training path.  This small wrapper
matches the upstream cross-attention boundary without changing that branch
cache contract or its temporal-loop executor.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class NativeCheckpointPolicy:
    """Runtime counters for the one upstream-compatible boundary we patch."""

    cross_attention_bypasses: int = 0

    def disable_cross_attention_cache(self, checkpoint_active: bool) -> bool:
        """Return whether a checkpointed block should receive no cache."""
        if checkpoint_active:
            self.cross_attention_bypasses += 1
        return checkpoint_active


_CHECKPOINT_ACTIVE: ContextVar[bool] = ContextVar(
    "sue_checkpoint_active", default=False
)


def install_native_checkpoint_alignment() -> NativeCheckpointPolicy:
    """Install the upstream checkpoint/cross-attention boundary once.

    The branch's ``KVCacheCheckpointState`` is deliberately left untouched:
    its detached-prefix snapshot is what makes backward replay of earlier AR
    chunks safe.  Only the mutable cross-attention cache is hidden while the
    checkpoint body runs, matching ``Self-Forcing/wan/modules/causal_model.py``.
    """
    import torch.utils.checkpoint as checkpoint

    from wan.modules import causal_model
    from wan.modules import model as wan_model

    existing = getattr(causal_model, "_sue_native_checkpoint_alignment", None)
    if existing is not None:
        print("WRAPPER: native checkpoint alignment already installed", flush=True)
        return existing

    policy = NativeCheckpointPolicy()
    original_checkpoint = checkpoint.checkpoint
    original_cross_attention = wan_model.WanT2VCrossAttention.forward

    def checkpoint_with_context(function, *args, **kwargs):
        def run_in_checkpoint(*inner_args, **inner_kwargs):
            token = _CHECKPOINT_ACTIVE.set(True)
            try:
                return function(*inner_args, **inner_kwargs)
            finally:
                _CHECKPOINT_ACTIVE.reset(token)

        return original_checkpoint(
            run_in_checkpoint,
            *args,
            **kwargs,
        )

    checkpoint_with_context._sue_native_checkpoint_alignment = True
    checkpoint.checkpoint = checkpoint_with_context

    def cross_attention_without_checkpoint_cache(
        self,
        x,
        context,
        context_lens,
        crossattn_cache=None,
    ):
        if _CHECKPOINT_ACTIVE.get() and crossattn_cache is not None:
            policy.disable_cross_attention_cache(True)
            crossattn_cache = None
        return original_cross_attention(
            self,
            x,
            context,
            context_lens,
            crossattn_cache,
        )

    cross_attention_without_checkpoint_cache._sue_native_checkpoint_alignment = True
    wan_model.WanT2VCrossAttention.forward = cross_attention_without_checkpoint_cache

    causal_model._sue_native_checkpoint_alignment = policy
    print(
        "WRAPPER: native checkpoint alignment installed "
        "(cross-attn cache=off inside checkpoint; KV replay snapshots unchanged)",
        flush=True,
    )
    return policy


def alignment_policy_from_modules() -> NativeCheckpointPolicy | None:
    """Return the installed policy for optional probe reporting."""
    try:
        from wan.modules import causal_model
    except ImportError:
        return None
    policy = getattr(causal_model, "_sue_native_checkpoint_alignment", None)
    return policy if isinstance(policy, NativeCheckpointPolicy) else None
