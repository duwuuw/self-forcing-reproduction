"""Small transactional helpers for causal self-attention KV caches."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheInterval:
    """Half-open token interval used for logical or physical cache ranges."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _scalar(cache, name, default=0):
    value = cache.get(name, default)
    return int(value.item()) if torch.is_tensor(value) else int(value)


def _set_scalar(cache, name, value):
    current = cache.get(name)
    if torch.is_tensor(current):
        current.fill_(value)
    else:
        device = cache["k"].device
        cache[name] = torch.tensor([value], dtype=torch.long, device=device)


def create_kv_cache(
    batch_size,
    num_heads,
    head_dim,
    capacity=None,
    *,
    frame_seqlen=None,
    num_frames=None,
    local_attn_size=-1,
    sink_size=0,
    cache_start=0,
    dtype=None,
    device=None,
):
    """Create one detached cache entry with explicit logical/physical metadata."""
    if frame_seqlen is None:
        raise ValueError("frame_seqlen is required for a causal KV cache")
    if local_attn_size != -1:
        capacity = local_attn_size * frame_seqlen if capacity is None else capacity
    elif capacity is None:
        if num_frames is None:
            raise ValueError("global cache requires num_frames or capacity")
        capacity = num_frames * frame_seqlen
    if capacity <= 0:
        raise ValueError("cache capacity must be positive")
    if sink_size * frame_seqlen >= capacity:
        raise ValueError("attention sink must leave room for cached tokens")

    if dtype is None:
        dtype = torch.get_default_dtype()
    key = torch.zeros(
        (batch_size, capacity, num_heads, head_dim), dtype=dtype, device=device
    )
    value = torch.zeros_like(key)
    zero = torch.tensor([0], dtype=torch.long, device=key.device)
    return {
        "k": key,
        "v": value,
        "global_end_index": zero.clone(),
        "local_end_index": zero.clone(),
        "absolute_start_index": zero.clone(),
        "absolute_end_index": zero.clone(),
        "physical_start_index": zero.clone(),
        "physical_end_index": zero.clone(),
        "physical_origin": torch.tensor(
            [cache_start], dtype=torch.long, device=key.device
        ),
        "local_attn_size": local_attn_size,
        "sink_size": sink_size,
        "frame_seqlen": frame_seqlen,
    }


class KVCacheTransaction:
    """Stage one logical chunk and commit its detached K/V to physical storage.

    ``cache_start`` is an initialization-time absolute origin for physical slot
    zero. Once a cache has content, stored absolute and physical metadata own
    placement; later ``cache_start`` values are intentionally not reused as a
    cursor.
    """

    def __init__(
        self,
        cache,
        *,
        current_start,
        num_tokens,
        cache_start=None,
        max_attention_size=None,
        frame_seqlen=None,
        local_attn_size=None,
        sink_size=None,
    ):
        self.cache = cache
        self.current_start = int(current_start)
        self.num_tokens = int(num_tokens)
        if self.current_start < 0 or self.num_tokens <= 0:
            raise ValueError("current_start must be non-negative and chunk non-empty")

        self.current_end = self.current_start + self.num_tokens
        self.old_global_end = _scalar(cache, "global_end_index")
        self.old_local_end = _scalar(cache, "local_end_index")
        self.capacity = cache["k"].shape[1]
        self.frame_seqlen = int(frame_seqlen or cache.get("frame_seqlen", 1))
        self.local_attn_size = (
            cache.get("local_attn_size", -1)
            if local_attn_size is None else local_attn_size
        )
        self.sink_size = (
            cache.get("sink_size", 0) if sink_size is None else sink_size
        )
        self.sink_tokens = int(self.sink_size) * self.frame_seqlen
        self.max_attention_size = max_attention_size or self.capacity
        self.absolute_interval = CacheInterval(self.current_start, self.current_end)
        self.current_chunk_interval = self.absolute_interval

        if self.old_global_end == 0:
            origin = self.current_start if cache_start is None else int(cache_start)
            if self.current_start != origin:
                raise ValueError("initial current_start must equal cache_start")
            _set_scalar(cache, "physical_origin", origin)
            self.is_replay = False
        else:
            previous_start = _scalar(
                cache, "absolute_start_index", self.old_global_end - self.num_tokens
            )
            previous_end = _scalar(
                cache, "absolute_end_index", self.old_global_end
            )
            self.is_replay = (
                self.current_start == previous_start
                and self.current_end == previous_end
            )
            if not self.is_replay and self.current_start != self.old_global_end:
                raise ValueError(
                    "causal KV chunks must replay the current interval or start "
                    "at the previous logical end"
                )

        if self.is_replay:
            self.physical_start = _scalar(
                cache,
                "physical_start_index",
                self.old_local_end - self.num_tokens,
            )
            self.physical_end = _scalar(cache, "physical_end_index", self.old_local_end)
            self.num_evicted_tokens = 0
            self.num_rolled_tokens = 0
        else:
            requested_end = self.old_local_end + self.num_tokens
            self.num_evicted_tokens = max(0, requested_end - self.capacity)
            if self.num_evicted_tokens and self.local_attn_size == -1:
                raise ValueError("global KV cache capacity is insufficient")
            if self.num_evicted_tokens:
                self.num_rolled_tokens = (
                    self.old_local_end - self.num_evicted_tokens - self.sink_tokens
                )
                if self.num_rolled_tokens < 0:
                    raise ValueError("local KV cache cannot fit its sink and current chunk")
            else:
                self.num_rolled_tokens = 0
            self.physical_start = self.old_local_end - self.num_evicted_tokens
            self.physical_end = self.physical_start + self.num_tokens

        if not 0 <= self.physical_start <= self.physical_end <= self.capacity:
            raise ValueError("KV cache physical write interval exceeds capacity")
        self.physical_write_interval = CacheInterval(
            self.physical_start, self.physical_end
        )
        self._staged_key = None
        self._staged_value = None
        self._committed = False

    def _virtual_prefix(self, tensor):
        if not self.num_evicted_tokens:
            return tensor[:, :self.physical_start]
        sink_end = self.sink_tokens
        active_start = sink_end + self.num_evicted_tokens
        active = tensor[:, active_start:active_start + self.num_rolled_tokens]
        if sink_end:
            return torch.cat((tensor[:, :sink_end], active), dim=1)
        return active

    def stage(self, key, value):
        """Return prior-prefix plus current-call K/V without mutating the cache."""
        if self._staged_key is not None:
            raise RuntimeError("KV cache transaction was already staged")
        if key.shape[1] != self.num_tokens or value.shape[1] != self.num_tokens:
            raise ValueError("staged K/V length does not match the current chunk")
        prior_key = self._virtual_prefix(self.cache["k"])
        prior_value = self._virtual_prefix(self.cache["v"])
        total_key = torch.cat((prior_key, key), dim=1)
        total_value = torch.cat((prior_value, value), dim=1)
        if total_key.shape[1] > self.max_attention_size:
            total_key = total_key[:, -self.max_attention_size:]
            total_value = total_value[:, -self.max_attention_size:]
        self._staged_key = key
        self._staged_value = value
        return total_key, total_value

    def commit_staged(self, key, value):
        """Install already-staged K/V without rereading the persistent prefix.

        Checkpointed causal attention stages against a detached prefix snapshot.
        Its forward pass must commit that exact staged pair after the checkpoint
        returns; calling :meth:`stage` again here would read whatever cursor a
        later AR chunk has left in the persistent cache.
        """
        if self._committed:
            raise RuntimeError("KV cache transaction was already committed")
        if self._staged_key is not None:
            raise RuntimeError("KV cache transaction was already staged")
        if key.shape[1] != self.num_tokens or value.shape[1] != self.num_tokens:
            raise ValueError("staged K/V length does not match the current chunk")
        self._staged_key = key
        self._staged_value = value

    def commit(self, key=None, value=None):
        """Commit detached K/V and metadata under ``torch.no_grad``."""
        if self._committed:
            raise RuntimeError("KV cache transaction was already committed")
        if self._staged_key is None:
            if key is None or value is None:
                raise ValueError("commit requires staged K/V")
            self.stage(key, value)
        elif key is not None or value is not None:
            raise RuntimeError("cannot replace K/V after staging")

        with torch.no_grad():
            if self.num_evicted_tokens:
                sink_end = self.sink_tokens
                active_start = sink_end + self.num_evicted_tokens
                if sink_end:
                    source_key = self.cache["k"][:, active_start:active_start + self.num_rolled_tokens].clone()
                    source_value = self.cache["v"][:, active_start:active_start + self.num_rolled_tokens].clone()
                    self.cache["k"][:, sink_end:sink_end + self.num_rolled_tokens].copy_(source_key)
                    self.cache["v"][:, sink_end:sink_end + self.num_rolled_tokens].copy_(source_value)
                else:
                    source_key = self.cache["k"][:, active_start:active_start + self.num_rolled_tokens].clone()
                    source_value = self.cache["v"][:, active_start:active_start + self.num_rolled_tokens].clone()
                    self.cache["k"][:, :self.num_rolled_tokens].copy_(source_key)
                    self.cache["v"][:, :self.num_rolled_tokens].copy_(source_value)
            self.cache["k"][:, self.physical_start:self.physical_end].copy_(
                self._staged_key.detach()
            )
            self.cache["v"][:, self.physical_start:self.physical_end].copy_(
                self._staged_value.detach()
            )
            if not self.is_replay:
                _set_scalar(self.cache, "global_end_index", self.current_end)
                _set_scalar(self.cache, "local_end_index", self.physical_end)
            _set_scalar(self.cache, "absolute_start_index", self.current_start)
            _set_scalar(self.cache, "absolute_end_index", self.current_end)
            _set_scalar(self.cache, "physical_start_index", self.physical_start)
            _set_scalar(self.cache, "physical_end_index", self.physical_end)
        self._committed = True


class KVCacheCheckpointState:
    """Pure checkpoint-call view over one mutable causal-cache transaction.

    The persistent cache is read once before a checkpointed block executes.
    The checkpoint body only uses the detached prefix snapshot and records the
    current call's K/V.  The owning model commits the original transaction
    after the forward checkpoint returns, while backward recomputation can only
    update this temporary Python object and never mutate persistent storage.
    """

    def __init__(
        self,
        cache,
        *,
        current_start,
        num_tokens,
        cache_start=None,
        max_attention_size=None,
        frame_seqlen=None,
        local_attn_size=None,
        sink_size=None,
    ):
        self.transaction = KVCacheTransaction(
            cache,
            current_start=current_start,
            num_tokens=num_tokens,
            cache_start=cache_start,
            max_attention_size=max_attention_size,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        prefix_key = self.transaction._virtual_prefix(cache["k"])
        prefix_value = self.transaction._virtual_prefix(cache["v"])
        prefix_limit = max(
            0, self.transaction.max_attention_size - self.transaction.num_tokens
        )
        if prefix_key.shape[1] > prefix_limit:
            if prefix_limit:
                prefix_key = prefix_key[:, -prefix_limit:]
                prefix_value = prefix_value[:, -prefix_limit:]
            else:
                prefix_key = prefix_key[:, :0]
                prefix_value = prefix_value[:, :0]
        self.prefix_key = prefix_key.detach().clone()
        self.prefix_value = prefix_value.detach().clone()
        self.max_attention_size = self.transaction.max_attention_size
        self._pending_key = None
        self._pending_value = None

    def stage(self, key, value):
        """Build attention inputs from the frozen prefix without cache writes."""
        if key.shape[1] != self.transaction.num_tokens or value.shape[1] != self.transaction.num_tokens:
            raise ValueError("staged K/V length does not match the current chunk")
        total_key = torch.cat((self.prefix_key, key), dim=1)
        total_value = torch.cat((self.prefix_value, value), dim=1)
        if total_key.shape[1] > self.max_attention_size:
            total_key = total_key[:, -self.max_attention_size:]
            total_value = total_value[:, -self.max_attention_size:]
        self._pending_key = key
        self._pending_value = value
        return total_key, total_value

    def commit(self):
        """Commit the forward pass's staged K/V to the original cache."""
        if self._pending_key is None or self._pending_value is None:
            raise RuntimeError("checkpoint cache state has no staged K/V")
        self.transaction.commit_staged(self._pending_key, self._pending_value)
        self.transaction.commit()
        self.transaction._staged_key = None
        self.transaction._staged_value = None
        self._pending_key = None
        self._pending_value = None
