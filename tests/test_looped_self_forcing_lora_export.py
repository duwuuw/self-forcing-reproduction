"""Regression for adapter export through nested FSDP module names."""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


SOURCE = Path(__file__).resolve().parents[1] / "scripts/looped_self_forcing_pipeline/source/model/lora.py"
DISTRIBUTED_SOURCE = SOURCE.parents[1] / "utils/distributed.py"


def _lora_module():
    spec = importlib.util.spec_from_file_location("copied_lora_export", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _distributed_module():
    spec = importlib.util.spec_from_file_location(
        "copied_distributed_export", DISTRIBUTED_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _nested_generator():
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})

    model = nn.Module()
    model.blocks = nn.ModuleList()
    for block_index in range(30):
        block = nn.Module()
        block.self_attn = nn.Module()
        if 22 <= block_index <= 29:
            for target in ("q", "v"):
                wrapper = nn.Module()
                wrapper._fsdp_wrapped_module = Adapter()
                setattr(block.self_attn, target, wrapper)
        model.blocks.append(block)
    model.peft_config = {"default": object()}
    expected = {
        f"blocks.{block}.self_attn.{target}.lora_{side}.weight"
        for block in range(22, 30)
        for target in ("q", "v")
        for side in ("A", "B")
    }
    return model, expected


def test_fsdp_export_summons_and_unwraps_exact_selected_32_tensors(monkeypatch):
    lora = _lora_module()
    model, expected = _nested_generator()
    expected_values = {}
    for index, (name, parameter) in enumerate(model.named_parameters(), start=1):
        parameter.data.fill_(index)
        portable = lora._portable_lora_name(name)
        expected_values[portable] = parameter.detach().clone()

    package = ModuleType("model")
    package.lora = lora
    monkeypatch.setitem(sys.modules, "model", package)
    monkeypatch.setitem(sys.modules, "model.lora", lora)
    distributed = _distributed_module()
    events = []

    class FakeFSDP(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.module = wrapped

        @staticmethod
        @contextmanager
        def summon_full_params(candidate, writeback):
            events.append(("enter", candidate, writeback))
            yield
            events.append(("exit", candidate, writeback))

    generator_wrapper = nn.Module()
    generator_wrapper.model = model
    fsdp_generator = FakeFSDP(generator_wrapper)
    monkeypatch.setattr(distributed, "FSDP", FakeFSDP)

    exported = distributed.fsdp_lora_state_dict(fsdp_generator, expected)

    assert len(expected) == 32
    assert set(exported) == expected
    assert events == [
        ("enter", fsdp_generator, False),
        ("exit", fsdp_generator, False),
    ]
    for name in expected:
        torch.testing.assert_close(exported[name], expected_values[name])


def test_empty_adapter_payload_fails_before_serialization():
    lora = _lora_module()

    with pytest.raises(ValueError, match="generator adapter"):
        lora.build_lora_checkpoint_payload(
            generator={}, critic={}, generator_ema=None,
            generator_optimizer=None, critic_optimizer=None, metadata={},
        )


def test_out_of_scope_adapter_payload_fails_before_serialization():
    lora = _lora_module()

    with pytest.raises(ValueError, match="out-of-scope"):
        lora.build_lora_checkpoint_payload(
            generator={"blocks.0.weight": torch.ones(1)}, critic={},
            generator_ema=None, generator_optimizer=None,
            critic_optimizer=None, metadata={},
        )


def test_dmd_lora_payload_keeps_existing_critic_checkpoint_schema():
    lora = _lora_module()
    generator = {"blocks.2.self_attn.q.lora_A.weight": torch.ones(1)}
    critic = {"weight": torch.tensor([2.0])}
    critic_optimizer = {"state": {}}

    payload = lora.build_lora_checkpoint_payload(
        generator=generator,
        critic=critic,
        generator_ema=None,
        generator_optimizer={"state": {}},
        critic_optimizer=critic_optimizer,
        metadata={"step": 12},
    )

    assert payload == {
        "checkpoint_version": 1,
        "generator_format": "lora_adapter",
        "generator": generator,
        "critic": critic,
        "generator_ema": None,
        "generator_optimizer": {"state": {}},
        "critic_optimizer": critic_optimizer,
        "metadata": {"step": 12},
    }


def test_supervised_lora_payload_allows_explicitly_absent_critic():
    lora = _lora_module()

    payload = lora.build_lora_checkpoint_payload(
        generator={"blocks.2.self_attn.q.lora_A.weight": torch.ones(1)},
        critic=None,
        generator_ema=None,
        generator_optimizer={"state": {}},
        critic_optimizer=None,
        metadata={
            "step": 12,
            "training_objective": "supervised_flow_matching",
        },
    )

    assert payload["critic"] is None
    assert payload["critic_optimizer"] is None
    assert payload["metadata"]["training_objective"] == "supervised_flow_matching"


@pytest.mark.parametrize(
    ("critic", "critic_optimizer", "metadata"),
    [
        (None, None, {"step": 1}),
        (None, None, {"training_objective": "dmd"}),
        (None, {}, {"training_objective": "supervised_flow_matching"}),
        ({}, None, {"training_objective": "supervised_flow_matching"}),
    ],
)
def test_critic_free_payload_requires_supervised_objective_and_paired_fields(
    critic, critic_optimizer, metadata
):
    lora = _lora_module()

    with pytest.raises(ValueError, match="critic|supervised"):
        lora.build_lora_checkpoint_payload(
            generator={"blocks.2.self_attn.q.lora_A.weight": torch.ones(1)},
            critic=critic,
            generator_ema=None,
            generator_optimizer={"state": {}},
            critic_optimizer=critic_optimizer,
            metadata=metadata,
        )


def test_exported_adapter_round_trip_restores_fresh_model(monkeypatch):
    lora = _lora_module()
    source, expected = _nested_generator()
    for index, parameter in enumerate(source.parameters(), start=1):
        parameter.data.fill_(index)
    generator = lora.extract_lora_state_dict(source, expected)
    payload = lora.build_lora_checkpoint_payload(
        generator=generator, critic={}, generator_ema=None,
        generator_optimizer=None, critic_optimizer={}, metadata={"step": 10},
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    reloaded = torch.load(buffer, weights_only=False)

    destination, _ = _nested_generator()
    for parameter in destination.parameters():
        parameter.data.zero_()

    def set_adapter_state(model, state):
        by_portable_name = {
            lora._portable_lora_name(name): parameter
            for name, parameter in model.named_parameters()
        }
        unexpected = []
        for name, value in state.items():
            parameter = by_portable_name.get(name)
            if parameter is None:
                unexpected.append(name)
            else:
                parameter.data.copy_(value)
        missing = sorted(set(by_portable_name) - set(state))
        return SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)

    peft = ModuleType("peft")
    peft.set_peft_model_state_dict = set_adapter_state
    monkeypatch.setitem(sys.modules, "peft", peft)
    lora.load_lora_state_dict(destination, reloaded["generator"])

    assert set(reloaded["generator"]) == expected
    assert reloaded["metadata"]["step"] == 10
    restored = lora.extract_lora_state_dict(destination, expected)
    for name in expected:
        torch.testing.assert_close(restored[name], generator[name])
