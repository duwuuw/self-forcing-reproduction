from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def load_fsdp_state_dict(model, state_dict, *, strict=True):
    """Load a full state dict using the matching FSDP state-dict context."""

    if isinstance(model, FSDP):
        full_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
            return model.load_state_dict(state_dict, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)


def optimizer_state_dict_for_checkpoint(model, optimizer):
    """Return an FSDP-compatible or ordinary optimizer state dict."""

    if isinstance(model, FSDP):
        return FSDP.optim_state_dict(model, optimizer)
    return optimizer.state_dict()


def load_optimizer_state_dict_for_checkpoint(model, optimizer, state_dict):
    """Load an optimizer state saved by the matching model wrapper."""

    if isinstance(model, FSDP):
        state_dict = FSDP.optim_state_dict_to_load(model, optimizer, state_dict)
    optimizer.load_state_dict(state_dict)


def _lora_model_from_wrapper(model):
    module = getattr(model, "module", model)
    candidate = getattr(module, "model", module)
    if not hasattr(candidate, "peft_config"):
        raise ValueError("model does not contain an injected LoRA adapter")
    return candidate


def fsdp_lora_state_dict(model):
    """Gather adapter-only state from an FSDP or ordinary generator wrapper."""

    from model.lora import extract_lora_state_dict

    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, writeback=False):
            return extract_lora_state_dict(_lora_model_from_wrapper(model))
    return extract_lora_state_dict(_lora_model_from_wrapper(model))


def load_fsdp_lora_state_dict(model, state_dict):
    """Load adapter-only state into an FSDP or ordinary generator wrapper."""

    from model.lora import load_lora_state_dict

    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, writeback=True):
            load_lora_state_dict(_lora_model_from_wrapper(model), state_dict)
        return
    load_lora_state_dict(_lora_model_from_wrapper(model), state_dict)


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        decay: float = 0.999,
        parameter_filter=None,
    ):
        self.decay = decay
        self.parameter_filter = parameter_filter
        self.shadow = {}
        self._init_shadow(fsdp_module)

    def _named_parameters(self, fsdp_module):
        module = getattr(fsdp_module, "module", fsdp_module)
        for name, parameter in module.named_parameters():
            if self.parameter_filter is None or self.parameter_filter(name, parameter):
                yield name, parameter

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in self._named_parameters(fsdp_module):
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in self._named_parameters(fsdp_module):
                if n not in self.shadow:
                    continue
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in self._named_parameters(fsdp_module):
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
