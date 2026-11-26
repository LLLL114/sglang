# SPDX-License-Identifier: Apache-2.0
"""Local RoPE declarations must avoid repeated whole-model buffer traversal."""

import pytest
import torch

from sglang.srt.weight_cache.weight_layout import (
    WeightLayoutBuilder,
    registered_tensors,
)
from sglang.srt.weight_cache.weight_runtime_manifest import WeightManifestError


class RotaryModel(torch.nn.Module):
    def __init__(self, layers=48):
        super().__init__()
        self.buffer_scans = 0
        self.weight = torch.nn.Parameter(torch.arange(16).reshape(4, 4).float())
        self.layers = torch.nn.ModuleList()
        for index in range(layers):
            layer = torch.nn.Module()
            layer.rotary_emb = torch.nn.Module()
            layer.rotary_emb.register_buffer("cos_sin_cache", torch.arange(8) + index)
            layer.rotary_emb.register_buffer("inv_freq", torch.arange(4) + index)
            self.layers.append(layer)

    def named_buffers(self, *args, **kwargs):
        self.buffer_scans += 1
        return super().named_buffers(*args, **kwargs)


def _declare(model, builder):
    builder.region(model.weight, "weight", global_shape=(4, 4))
    for index in range(len(model.layers)):
        for suffix in ("cos_sin_cache", "inv_freq"):
            builder.local_buffer(f"layers.{index}.rotary_emb.{suffix}")


def test_rotary_declarations_scan_buffers_a_constant_number_of_times():
    model = RotaryModel()
    before = {
        name: (id(tensor), tensor.clone())
        for name, tensor in registered_tensors(model).items()
    }
    model.buffer_scans = 0
    builder = WeightLayoutBuilder(model)
    _declare(model, builder)
    plan = builder.build()
    scans = model.buffer_scans
    after = registered_tensors(model)
    assert set(after) == set(before)
    for name, tensor in after.items():
        assert id(tensor) == before[name][0]
        assert torch.equal(tensor, before[name][1])
    assert len(plan.local_buffers) == 96
    assert scans <= 3


def test_parameter_cannot_be_declared_as_local_buffer():
    model = RotaryModel(layers=1)
    builder = WeightLayoutBuilder(model)
    with pytest.raises(WeightManifestError, match="must name a buffer"):
        builder.local_buffer("weight")


def test_buffer_changed_to_parameter_before_declaration_is_rejected():
    model = RotaryModel(layers=1)
    builder = WeightLayoutBuilder(model)
    rotary = model.layers[0].rotary_emb
    del rotary.inv_freq
    rotary.register_parameter("inv_freq", torch.nn.Parameter(torch.ones(4)))
    with pytest.raises(WeightManifestError, match="must name a buffer"):
        builder.local_buffer("layers.0.rotary_emb.inv_freq")


def test_none_buffer_is_rejected():
    model = RotaryModel(layers=1)
    model.register_buffer("optional_cache", None)
    with pytest.raises(WeightManifestError, match="must name a buffer"):
        WeightLayoutBuilder(model).local_buffer("optional_cache")


@pytest.mark.parametrize("change", ["added", "removed"])
def test_buffer_inventory_mutation_after_snapshot_is_rejected_at_build(change):
    model = RotaryModel(layers=1)
    builder = WeightLayoutBuilder(model)
    _declare(model, builder)
    if change == "added":
        model.register_buffer("extra_cache", torch.ones(1))
    else:
        del model.layers[0].rotary_emb.inv_freq
    with pytest.raises(WeightManifestError, match="registered tensor names changed"):
        builder.build()


def test_new_buffer_can_be_declared_after_builder_construction():
    model = RotaryModel(layers=1)
    first = WeightLayoutBuilder(model)
    model.register_buffer("extra_cache", torch.ones(1))
    second = WeightLayoutBuilder(model)
    _declare(model, first)
    first.local_buffer("extra_cache")
    assert "extra_cache" in first.build().local_buffers
    _declare(model, second)
    second.local_buffer("extra_cache")
    assert "extra_cache" in second.build().local_buffers
