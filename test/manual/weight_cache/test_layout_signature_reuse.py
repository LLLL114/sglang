# SPDX-License-Identifier: Apache-2.0
"""Signature reuse stays local to each public layout operation."""

import pytest
import torch

from sglang.srt.weight_cache import weight_layout as layout
from sglang.srt.weight_cache.weight_layout import WeightLayoutBuilder
from sglang.srt.weight_cache.weight_runtime_manifest import WeightManifestError


def _model():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32))
    return model


def _regions(builder, tensor):
    for index in range(16):
        builder.region(
            tensor,
            f"weight.{index}",
            global_shape=(1,),
            local_shape=(1,),
            byte_offset=index * tensor.element_size(),
        )


@pytest.mark.parametrize(
    "operation,expected", [("region", 16), ("build", 3), ("validate", 1)]
)
def test_public_operations_compute_signatures_once_per_parameter(
    monkeypatch, operation, expected
):
    model = _model()
    original = layout._layout_storage_key
    calls = []

    def counted(tensor):
        calls.append(id(tensor))
        return original(tensor)

    monkeypatch.setattr(layout, "_layout_storage_key", counted)
    builder = WeightLayoutBuilder(model)
    calls.clear()
    _regions(builder, model.weight)
    if operation != "region":
        calls.clear()
        plan = builder.build()
        if operation == "validate":
            calls.clear()
            plan.validate_bindings(model)
            assert len(calls) == expected
            calls.clear()
            plan.validate_bindings(model)
    # Build reads grouping, binding construction and fresh validation separately.
    assert len(calls) == expected
    assert torch.equal(model.weight, torch.arange(16, dtype=torch.float32))


@pytest.mark.parametrize("mutation", ["data", "set"])
def test_later_validation_rejects_changed_storage(mutation):
    model = _model()
    builder = WeightLayoutBuilder(model)
    _regions(builder, model.weight)
    plan = builder.build()
    plan.validate_bindings(model)
    with torch.no_grad():
        if mutation == "data":
            model.weight.data = model.weight.detach().clone()
        else:
            model.weight.set_(model.weight.detach().clone())
    with pytest.raises(WeightManifestError, match="weight storage changed"):
        plan.validate_bindings(model)


def test_replacement_tensor_with_same_storage_preserves_binding():
    model = _model()
    builder = WeightLayoutBuilder(model)
    _regions(builder, model.weight)
    plan = builder.build()
    previous = model.weight
    model.weight = torch.nn.Parameter(previous.detach())
    assert model.weight is not previous
    plan.validate_bindings(model)


def test_build_rejects_storage_changed_after_declaration():
    model = _model()
    builder = WeightLayoutBuilder(model)
    _regions(builder, model.weight)
    model.weight.data = model.weight.detach().clone()
    with pytest.raises(WeightManifestError):
        builder.build()


@pytest.mark.parametrize("rebound", [False, True])
def test_public_declare_validates_registered_storage(rebound):
    model = _model()
    builder = WeightLayoutBuilder(model)
    tensor = torch.empty_like(model.weight)
    if rebound:
        model.weight.data = tensor
        tensor = model.weight
    with pytest.raises(WeightManifestError, match="registered storage"):
        builder.declare(tensor, "weight", global_shape=(16,))


def test_exact_aliases_and_multiple_regions_are_accepted():
    model = _model()
    model.alias = torch.nn.Parameter(model.weight.detach())
    builder = WeightLayoutBuilder(model)
    _regions(builder, model.weight)
    _regions(builder, model.alias)
    plan = builder.build()
    assert len(plan.views) == 16
    assert all(v.parameter_names == ("alias", "weight") for v in plan.views)
    plan.validate_bindings(model)


def test_partial_storage_overlap_remains_rejected():
    model = _model()
    arena = torch.empty(24)
    model.weight = torch.nn.Parameter(arena[:16])
    model.other = torch.nn.Parameter(arena[8:])
    builder = WeightLayoutBuilder(model)
    builder.region(model.weight, "weight", global_shape=(16,))
    builder.region(model.other, "other", global_shape=(16,))
    with pytest.raises(WeightManifestError, match="overlapping registered"):
        builder.build()


def test_conflicting_repeated_region_remains_rejected():
    model = _model()
    builder = WeightLayoutBuilder(model)
    builder.region(model.weight, "weight", global_shape=(16,))
    builder.region(model.weight, "conflict", global_shape=(16,))
    with pytest.raises(WeightManifestError, match="overlap or leave a gap"):
        builder.build()


def test_initialization_range_and_coverage_remain_checked():
    model = _model()
    builder = WeightLayoutBuilder(model)
    builder.region(model.weight, "weight", global_shape=(8,), local_shape=(8,))
    with pytest.raises(WeightManifestError, match="out of bounds"):
        builder.initialize(model.weight, byte_offset=32, nbytes=36)
    builder.initialize(model.weight, byte_offset=32, nbytes=32, value=0)
    plan = builder.build()
    assert len(plan.initializations) == 1
    plan.initialize_local_regions()
    assert torch.equal(model.weight[8:], torch.zeros(8))
    plan.validate_bindings(model)
