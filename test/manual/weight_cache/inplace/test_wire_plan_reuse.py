# SPDX-License-Identifier: Apache-2.0
"""Stopped epochs reuse compact wire geometry with fresh physical validation."""

import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from msgspec.structs import replace as replace_struct

from sglang.srt.weight_cache.inplace import arena_weights, binding, block_wire, wire
from sglang.srt.weight_cache.weight_layout import WeightLayoutBuilder
from sglang.srt.weight_cache.weight_runtime_manifest import (
    ImmutableWeightRuntimeManifestBuilder,
    WeightManifestError,
    WeightParallelTopology,
)

from . import test_wire as fixtures


def test_meta_template_keeps_no_parameter_attributes():
    model = torch.nn.Linear(4, 4, bias=False)
    model.weight.weight_loader = lambda: model
    builder = WeightLayoutBuilder(model)
    builder.region(model.weight, "weight", global_shape=(4, 4))
    plan = builder.build()
    template, tensors = plan.as_meta(model)
    assert template.views[0].parameter is tensors["weight"]
    assert tensors["weight"].is_meta and not vars(tensors["weight"])
    previous = weakref.ref(model.weight)
    del plan, builder
    enabled = gc.isenabled()
    gc.disable()
    try:
        model.weight = torch.nn.Parameter(torch.ones(4, 4))
        assert previous() is None
        rebound = template.rebind(model, tensors)
        assert rebound.views[0].parameter is model.weight
        rebound.validate_bindings(model)
    finally:
        if enabled:
            gc.enable()


def _export(model, plan, previous, args):
    return ImmutableWeightRuntimeManifestBuilder(
        model=model,
        layout_plan=plan,
        topology=WeightParallelTopology(
            tp_rank=previous.rank.tp,
            tp_size=8 // args.ep_size,
            ep_rank=previous.rank.ep,
            ep_size=args.ep_size,
        ),
        allowed_devices=("cpu",),
    ).build(
        model_id=previous.model_id,
        revision=previous.revision,
        instance_id=previous.instance_id,
        worker_id=previous.worker_id,
        endpoint=previous.endpoint,
    )


def _same_manifest(first, second):
    assert {t.tensor_id: t for t in first.tensors} == {
        t.tensor_id: t for t in second.tensors
    }
    order = lambda f: (f.tensor_id, f.global_offset, f.address)
    assert sorted(first.fragments, key=order) == sorted(second.fragments, key=order)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "source,target",
    [(s, t) for s in fixtures.LAYOUTS for t in fixtures.LAYOUTS if s != t],
)
def test_public_reader_all_bytes_with_rebound_wire_plan(
    monkeypatch, source, target, ordered
):
    templates = {}
    original_fixture = fixtures.fixture
    original_derive = wire.derive_wire_manifest
    config = SimpleNamespace(num_experts=8, hidden_size=8, moe_intermediate_size=8)

    def fixture(args, rank, *, initialized):
        model, plan, previous = original_fixture(args, rank, initialized=initialized)
        peer_layout = target if initialized else source
        peer = SimpleNamespace(
            tp_size=8, ep_size=peer_layout[1], dp_size=peer_layout[0]
        )
        small = wire.build_wire_plan(model, plan, args, peer, config)
        templates[id(plan)] = (model, *small.as_meta(model))
        return model, plan, previous

    def derive(previous, canonical, args, peer, config, *, exclude=()):
        model, template, tensors = templates[id(canonical)]
        rebound = template.rebind(model, tensors)
        value = wire.exclude_parameters(
            _export(model, rebound, previous, args), rebound, exclude
        )
        _same_manifest(
            value,
            original_derive(previous, canonical, args, peer, config, exclude=exclude),
        )
        return value

    monkeypatch.setattr(fixtures, "fixture", fixture)
    monkeypatch.setattr(wire, "derive_wire_manifest", derive)
    fixtures.test_wire_reader_matches_canonical_bytes(source, target, ordered)


@pytest.mark.parametrize("mutation", ["shape", "stride", "dtype", "names", "alias"])
def test_rebind_rejects_changed_contract(mutation):
    args = SimpleNamespace(tp_size=8, ep_size=1, dp_size=1)
    model, plan, _ = fixtures.fixture(args, 0, initialized=True)
    small = wire.build_wire_plan(
        model,
        plan,
        args,
        args,
        SimpleNamespace(num_experts=8, hidden_size=8, moe_intermediate_size=8),
    )
    template, tensors = small.as_meta(model)
    original = model.experts.w2_weight
    if mutation == "shape":
        model.experts.w2_weight = torch.nn.Parameter(original.reshape(8, 1, 8))
    elif mutation == "stride":
        model.experts.w2_weight = torch.nn.Parameter(
            torch.empty_strided(original.shape, (8, 1, 8), dtype=original.dtype)
        )
    elif mutation == "dtype":
        model.experts.w2_weight = torch.nn.Parameter(original.float())
    elif mutation == "names":
        model.experts.other = model.experts.w2_weight
        del model.experts.w2_weight
    else:
        model.experts.alias = model.experts.w2_weight
    with pytest.raises(WeightManifestError):
        template.rebind(model, tensors)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_padding_only_declarations_buffers_and_derived_state_survive(device):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(8, device=device))
    model.register_buffer("rope", torch.ones(4))
    model.ready = True
    builder = WeightLayoutBuilder(model, allow_meta=device == "meta")
    builder.declare(model.weight, "padding_only", global_shape=(8,))
    builder.initialize(model.weight, byte_offset=0, nbytes=32, value=0)
    builder.local_buffer("rope")
    builder.derived("ready")
    plan = builder.build()
    args = SimpleNamespace(tp_size=8, ep_size=1, dp_size=1)
    small = wire.build_wire_plan(model, plan, args, args, SimpleNamespace())
    expected = tuple(
        replace_struct(t, layout_fingerprint="sglang:inplace-grouped-block-v1")
        for t in plan.tensors
    )
    assert small.tensors == expected
    template, tensors = small.as_meta(model)
    if device == "meta":
        model.weight = torch.nn.Parameter(torch.empty(8))
    rebound = template.rebind(model, tensors)
    rebound.initialize_local_regions()
    assert torch.equal(model.weight, torch.zeros(8))
    assert rebound.tensors == expected
    assert rebound.local_buffers == frozenset({"rope"})
    assert rebound.derived_state == ("ready",)
    model.ready = None
    with pytest.raises(WeightManifestError, match="derived"):
        template.rebind(model, tensors)


def test_rebind_checks_partial_overlap_and_initialization_range():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.empty(8))
    model.b = torch.nn.Parameter(torch.empty(8))
    builder = WeightLayoutBuilder(model)
    builder.region(model.a, "a", global_shape=(4,), local_shape=(4,))
    builder.initialize(model.a, byte_offset=16, nbytes=16, value=0)
    builder.region(model.b, "b", global_shape=(8,))
    template, tensors = builder.build().as_meta(model)
    bad = replace(
        template,
        initializations=(replace(template.initializations[0], nbytes=20),),
    )
    with pytest.raises(WeightManifestError, match="out of bounds"):
        bad.rebind(model, tensors)
    arena = torch.empty(12)
    model.a = torch.nn.Parameter(arena[:8])
    model.b = torch.nn.Parameter(arena[4:])
    with pytest.raises(WeightManifestError, match="overlapping registered"):
        template.rebind(model, tensors)


def test_target_meta_becomes_physical():
    model = torch.nn.Linear(4, 4, bias=False, device="meta")
    builder = WeightLayoutBuilder(model, allow_meta=True)
    builder.region(model.weight, "weight", global_shape=(4, 4))
    args = SimpleNamespace(tp_size=8, ep_size=1, dp_size=1)
    plan = wire.build_wire_plan(model, builder.build(), args, args, SimpleNamespace())
    template, tensors = plan.as_meta(model)
    model.weight = torch.nn.Parameter(torch.ones(4, 4))
    rebound = template.rebind(model, tensors)
    assert rebound.views[0].parameter is model.weight
    rebound.validate_bindings(model)


def test_alias_filter_preserves_other_parameters_of_the_same_tensor():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(4))
    model.b = torch.nn.Parameter(torch.ones(4))
    builder = WeightLayoutBuilder(model)
    builder.region(model.a, "shared", global_shape=(8,), local_shape=(4,))
    builder.region(
        model.b, "shared", global_shape=(8,), global_offset=(4,), local_shape=(4,)
    )
    plan = builder.build()
    manifest = ImmutableWeightRuntimeManifestBuilder(
        model=model,
        layout_plan=plan,
        topology=WeightParallelTopology(),
        allowed_devices=("cpu",),
    ).build(model_id="m", revision="r", instance_id="i", worker_id="w", endpoint="e")
    remaining = wire.exclude_parameters(manifest, plan, {"a"})
    assert len(remaining.fragments) == 1
    assert remaining.fragments[0].tensor_id == "shared"
    assert remaining.fragments[0].address == model.b.data_ptr()


def test_full_padding_rank_preserves_declarations_and_existing_participation_limit():
    from sglang.srt.weight_cache.mooncake_weight_adapter import (
        build_mooncake_transfer_groups,
    )
    from sglang.srt.weight_cache.weight_runtime_manifest import WeightPartition

    args = SimpleNamespace(tp_size=8, ep_size=1, dp_size=1, pp_size=1)
    manifests, canonical_manifests = [], []
    for rank in range(8):
        model = torch.nn.Linear(1, 1, bias=False)
        builder = WeightLayoutBuilder(model)
        partition = WeightPartition(rank=rank, size=8, dim=0)
        builder.declare(
            model.weight, "weight", global_shape=(1, 1), partition=partition
        )
        if rank == 0:
            builder.region(
                model.weight, "weight", global_shape=(1, 1), partition=partition
            )
        else:
            builder.initialize(model.weight, byte_offset=0, nbytes=4, value=0)
        canonical = builder.build()
        canonical_manifests.append(
            ImmutableWeightRuntimeManifestBuilder(
                model=model,
                layout_plan=canonical,
                topology=WeightParallelTopology(tp_rank=rank, tp_size=8),
                allowed_devices=("cpu",),
            ).build(
                model_id="m",
                revision="r",
                instance_id=f"i-{rank}",
                worker_id=f"worker-{rank}",
                endpoint=f"host:{rank}",
            )
        )
        small = wire.build_wire_plan(model, canonical, args, args, SimpleNamespace())
        template, tensors = small.as_meta(model)
        rebound = template.rebind(model, tensors)
        manifest = ImmutableWeightRuntimeManifestBuilder(
            model=model,
            layout_plan=rebound,
            topology=WeightParallelTopology(tp_rank=rank, tp_size=8),
            allowed_devices=("cpu",),
        ).build(
            model_id="m",
            revision="r",
            instance_id=f"i-{rank}",
            worker_id=f"worker-{rank}",
            endpoint=f"host:{rank}",
        )
        manifests.append(manifest)
    assert all(
        m.tensors[0].layout_fingerprint == "sglang:inplace-grouped-block-v1"
        for m in manifests
    )
    for values in (canonical_manifests, manifests):
        with pytest.raises(
            ValueError, match="missing or unexpected split-axis participant"
        ):
            build_mooncake_transfer_groups(
                values, values, source_parallel_layout=args, target_parallel_layout=args
            )


def test_controlled_replacements_preserve_layer_semantics_and_release_old_tensors():
    source_args = SimpleNamespace(tp_size=8, ep_size=8, dp_size=1)
    target_args = SimpleNamespace(tp_size=8, ep_size=1, dp_size=1)
    config = SimpleNamespace(num_experts=8, hidden_size=8, moe_intermediate_size=8)
    source, source_plan, _ = fixtures.fixture(source_args, 0, initialized=True)
    target, target_plan, _ = fixtures.fixture(target_args, 0, initialized=False)
    for model, args in ((source, source_args), (target, target_args)):
        model.config = config
        model.experts.quant_method = SimpleNamespace(
            runner=SimpleNamespace(runner_backend=object()),
            with_bias=False,
            w13_swiglu_interleaved=False,
        )
        model.experts.moe_tp_size = 8 // args.ep_size
        model.experts.moe_ep_rank = 0
        model.experts.layer_id = 0
        model.experts.tp_group = object()
        model.experts.w13_weight.weight_loader = lambda *args: None

    def semantics(model):
        method = model.experts.quant_method
        return (
            tuple(
                (name, type(module), id(module))
                for name, module in model.named_modules()
            ),
            id(model.config),
            id(method),
            id(method.runner),
            id(method.runner.runner_backend),
            method.with_bias,
            method.w13_swiglu_interleaved,
            model.experts.moe_tp_size,
            model.experts.moe_ep_rank,
            model.experts.layer_id,
            id(model.experts.tp_group),
            id(model.experts.w13_weight.weight_loader),
        )

    source_semantics, target_semantics = semantics(source), semantics(target)
    aliases = wire.alias_names(source_plan, target_plan)
    source_template, source_tensors = wire.build_wire_plan(
        source, source_plan, source_args, target_args, config
    ).as_meta(source)
    target_template, target_tensors = wire.build_wire_plan(
        target, target_plan, target_args, source_args, config
    ).as_meta(target)
    del source_plan, target_plan
    parameters = dict(source.named_parameters())
    references = [
        weakref.ref(p) for name, p in parameters.items() if name not in aliases
    ]
    cuda = SimpleNamespace(
        synchronize=lambda: None,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
        memory_snapshot=lambda: [],
        empty_cache=lambda: None,
    )
    cpu = SimpleNamespace(
        cuda=cuda, empty=torch.empty, no_grad=torch.no_grad, nn=torch.nn
    )
    profile, owners = {}, []
    arena_weights.compact_source(
        cpu, source, parameters, set(parameters) - aliases, profile, owners
    )
    assert all(reference() is None for reference in references)
    block_wire.convert(parameters, 8, pack=True)
    allocated = {
        name: parameters[name] if name in aliases else torch.empty_like(value)
        for name, value in target.named_parameters()
    }
    binding._install_target(torch, target, allocated, source)
    assert semantics(source) == source_semantics
    assert semantics(target) == target_semantics
    source_template.rebind(source, source_tensors)
    target_template.rebind(target, target_tensors)


def test_48_layer_meta_moe_target_retains_all_packed_blocks():
    from sglang.srt.weight_cache.weight_runtime_manifest import WeightPartition

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for _ in range(48):
        layer = torch.nn.Module()
        layer.experts = torch.nn.Module()
        layer.experts.w13_weight = torch.nn.Parameter(
            torch.empty(16, 32, 16, device="meta", dtype=torch.bfloat16)
        )
        layer.experts.w2_weight = torch.nn.Parameter(
            torch.empty(16, 16, 16, device="meta", dtype=torch.bfloat16)
        )
        model.layers.append(layer)
    builder = WeightLayoutBuilder(model, allow_meta=True)
    for index, layer in enumerate(model.layers):
        for expert in range(16):
            for half, projection in enumerate(("gate_proj", "up_proj")):
                builder.region(
                    layer.experts.w13_weight,
                    f"layers.{index}.experts.{expert}.{projection}.weight",
                    global_shape=(16, 16),
                    local_shape=(16, 16),
                    byte_offset=(expert * 2 + half) * 512,
                    expert_id=expert,
                    partition=WeightPartition(kind="expert", dim=0),
                )
            builder.region(
                layer.experts.w2_weight,
                f"layers.{index}.experts.{expert}.down_proj.weight",
                global_shape=(16, 16),
                local_shape=(16, 16),
                byte_offset=expert * 512,
                expert_id=expert,
                partition=WeightPartition(kind="expert", dim=1),
            )
    canonical = builder.build()
    args = SimpleNamespace(tp_size=8, ep_size=8, dp_size=1)
    peer = SimpleNamespace(tp_size=8, ep_size=1, dp_size=1)
    small = wire.build_wire_plan(
        model,
        canonical,
        args,
        peer,
        SimpleNamespace(num_experts=128, hidden_size=16, moe_intermediate_size=16),
    )
    assert len(canonical.views) == 2304 and len(small.views) == 432
    template, tensors = small.as_meta(model)
    for layer in model.layers:
        layer.experts.w13_weight = torch.nn.Parameter(
            torch.empty_like(layer.experts.w13_weight, device="cpu")
        )
        layer.experts.w2_weight = torch.nn.Parameter(
            torch.empty_like(layer.experts.w2_weight, device="cpu")
        )
    rebound = template.rebind(model, tensors)
    assert len(rebound.views) == 432
    rebound.validate_bindings(model)
