# SPDX-License-Identifier: Apache-2.0
"""Grouped wire geometry must reproduce canonical loader bytes across layouts."""

import ctypes
import zlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from mooncake.reshard.weight.inplace import InplaceWeightResharder
from mooncake.reshard.weight.te import MooncakeTransferEngineReader

from sglang.srt.weight_cache.inplace import binding as bridge
from sglang.srt.weight_cache.inplace import block_wire, wire
from sglang.srt.weight_cache.inplace.binding import CollectiveScope, execute
from sglang.srt.weight_cache.inplace.endpoint_order import make_executor
from sglang.srt.weight_cache.mooncake_weight_adapter import (
    build_mooncake_transfer_groups,
)
from sglang.srt.weight_cache.weight_layout import WeightLayoutBuilder
from sglang.srt.weight_cache.weight_runtime_manifest import (
    ImmutableWeightRuntimeManifestBuilder,
    WeightManifestError,
    WeightParallelTopology,
    WeightPartition,
)


class HostEngine:
    def register_memory(self, address, size):
        return 0

    def unregister_memory(self, address):
        return 0

    def batch_transfer_sync_read(self, endpoint, targets, sources, sizes):
        for target, source, size in zip(targets, sources, sizes):
            ctypes.memmove(target, source, size)
        return 0


def fixture(args, rank, *, initialized):
    expert_tp = 8 // args.ep_size
    expert_rank, ep_rank = rank % expert_tp, rank // expert_tp
    local_i, local_e, hidden = 8 // expert_tp, 8 // args.ep_size, 8
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.w13_weight = torch.nn.Parameter(
        torch.empty(local_e, 2 * local_i, hidden, dtype=torch.bfloat16)
    )
    model.experts.w2_weight = torch.nn.Parameter(
        torch.empty(local_e, hidden, local_i, dtype=torch.bfloat16)
    )
    model.self_attn = torch.nn.Module()
    model.self_attn.o_proj = torch.nn.Linear(
        8 // (8 // args.dp_size), 8, bias=False, dtype=torch.bfloat16
    )
    builder = WeightLayoutBuilder(model)
    for slot in range(local_e):
        expert = ep_rank * local_e + slot
        for half, name in enumerate(("gate_proj", "up_proj")):
            builder.region(
                model.experts.w13_weight,
                f"experts.{expert}.{name}.weight",
                global_shape=(8, hidden),
                global_offset=(expert_rank * local_i, 0),
                local_shape=(local_i, hidden),
                expert_id=expert,
                byte_offset=(slot * 2 + half) * local_i * hidden * 2,
                partition=WeightPartition(
                    kind="expert", rank=expert_rank, size=expert_tp, dim=0
                ),
            )
        builder.region(
            model.experts.w2_weight,
            f"experts.{expert}.down_proj.weight",
            global_shape=(hidden, 8),
            global_offset=(0, expert_rank * local_i),
            local_shape=(hidden, local_i),
            expert_id=expert,
            byte_offset=slot * local_i * hidden * 2,
            partition=WeightPartition(
                kind="expert", rank=expert_rank, size=expert_tp, dim=1
            ),
        )
    attn_tp = 8 // args.dp_size
    width = 8 // attn_tp
    builder.region(
        model.self_attn.o_proj.weight,
        "self_attn.o_proj.weight",
        global_shape=(8, 8),
        global_offset=(0, (rank % attn_tp) * width),
        partition=WeightPartition(rank=rank % attn_tp, size=attn_tp, dim=1),
    )
    plan = builder.build()
    with torch.no_grad():
        for view in plan.views:
            canonical = (
                torch.arange(64).reshape(8, 8)
                + zlib.crc32(view.tensor_id.encode()) % 127
            ) % 251
            slices = tuple(
                slice(start, start + size)
                for start, size in zip(view.global_offset, view.local_shape)
            )
            local = canonical[slices].contiguous().flatten().to(torch.bfloat16)
            view.parameter.flatten().narrow(
                0, view.byte_offset // 2, local.numel()
            ).copy_(local if initialized else torch.zeros_like(local))
    manifest = ImmutableWeightRuntimeManifestBuilder(
        model=model,
        layout_plan=plan,
        topology=WeightParallelTopology(
            tp_rank=expert_rank,
            tp_size=expert_tp,
            ep_rank=ep_rank,
            ep_size=args.ep_size,
        ),
        allowed_devices=("cpu",),
    ).build(
        model_id="test",
        revision="test",
        instance_id=f"{'s' if initialized else 't'}:{rank}",
        worker_id=f"worker-{rank}",
        endpoint=f"host:{rank}",
    )
    return model, plan, manifest


LAYOUTS = ((1, 1), (1, 2), (1, 4), (1, 8), (2, 8), (8, 8))


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "source_layout,target_layout",
    [(source, target) for source in LAYOUTS for target in LAYOUTS if source != target],
)
def test_wire_reader_matches_canonical_bytes(source_layout, target_layout, ordered):
    def args(layout):
        return SimpleNamespace(
            tp_size=8, dp_size=layout[0], ep_size=layout[1], pp_size=1
        )

    old, new = args(source_layout), args(target_layout)
    config = SimpleNamespace(num_experts=8, hidden_size=8, moe_intermediate_size=8)
    sources, targets, owners, expected, original_sources = [], [], [], [], []
    for rank in range(8):
        smodel, splan, sm = fixture(old, rank, initialized=True)
        tmodel, tplan, tm = fixture(new, rank, initialized=False)
        oracle, _, _ = fixture(new, rank, initialized=True)
        expected.append({n: p.detach().clone() for n, p in oracle.named_parameters()})
        aliases = wire.alias_names(splan, tplan)
        original_sources.append(
            {n: p.detach().clone() for n, p in smodel.named_parameters()}
        )
        with torch.no_grad():
            for name in aliases:
                dict(tmodel.named_parameters())[name].copy_(
                    dict(smodel.named_parameters())[name]
                )
        block_wire.convert(
            dict(smodel.named_parameters()),
            block_wire.parts(old, new),
            pack=True,
            attention_blocks=max(1, old.dp_size // new.dp_size),
        )
        sources.append(wire.derive_wire_manifest(sm, splan, old, new, config))
        targets.append(
            wire.derive_wire_manifest(tm, tplan, new, old, config, exclude=aliases)
        )
        owners.append((smodel, tmodel))
    sources, targets = wire.close_inventory(sources, targets)
    contracts = build_mooncake_transfer_groups(
        sources,
        targets,
        source_parallel_layout=old,
        target_parallel_layout=new,
        participant_ids={f"worker-{i}": f"worker-{i}" for i in range(8)},
    )
    for sm, bindings, tm, destinations in contracts.values():
        assert len(sm.tensors) <= 3
        scope = CollectiveScope((*bindings, *destinations), owners)
        for rank, destination in enumerate(destinations):
            barriers = []
            engine = HostEngine()
            executor = (
                make_executor(
                    engine,
                    rank=rank,
                    target_attention_tp=8 // new.dp_size,
                    local_endpoint=f"host:{rank}",
                    endpoints=tuple(f"host:{i}" for i in range(8)),
                    phase_barrier=lambda: barriers.append(True),
                )
                if ordered
                else None
            )
            execute(
                InplaceWeightResharder(
                    MooncakeTransferEngineReader(engine, transfer_executor=executor)
                ),
                sm,
                bindings,
                tm,
                destination,
                scope,
            )
            if ordered:
                assert barriers == [True] * 8
                assert executor.profiles[-1]["completed"]
    for rank, (source, model) in enumerate(owners):
        block_wire.convert(
            dict(model.named_parameters()),
            block_wire.parts(new, old),
            pack=False,
            attention_blocks=max(1, new.dp_size // old.dp_size),
        )
        for name, value in model.named_parameters():
            assert torch.equal(value, expected[rank][name]), (
                source_layout,
                target_layout,
                rank,
                name,
            )
        block_wire.convert(
            dict(source.named_parameters()),
            block_wire.parts(old, new),
            pack=False,
            attention_blocks=max(1, old.dp_size // new.dp_size),
        )
        for name, value in source.named_parameters():
            assert torch.equal(value, original_sources[rank][name])


def test_process_cache_hits_with_fresh_physical_bindings(monkeypatch):
    monkeypatch.setattr(bridge, "_PLAN_CACHE", None)
    monkeypatch.setenv("MC_RESHARD_PLAN_CACHE_ENABLED", "1")
    monkeypatch.setenv("MC_RESHARD_PLAN_CACHE_MAX_ENTRIES", "2")
    cache = bridge._plan_cache()
    old = SimpleNamespace(tp_size=8, dp_size=1, ep_size=1, pp_size=1)
    new = SimpleNamespace(tp_size=8, dp_size=1, ep_size=8, pp_size=1)
    config = SimpleNamespace(num_experts=8, hidden_size=8, moe_intermediate_size=8)
    previous = None
    held = []
    for _ in range(2):
        sources, targets, owners = [], [], []
        for rank in range(8):
            source, sp, sm = fixture(old, rank, initialized=True)
            target, tp, tm = fixture(new, rank, initialized=False)
            sources.append(wire.derive_wire_manifest(sm, sp, old, new, config))
            targets.append(wire.derive_wire_manifest(tm, tp, new, old, config))
            owners.append((source, target))
        contracts = build_mooncake_transfer_groups(
            sources,
            targets,
            source_parallel_layout=old,
            target_parallel_layout=new,
            participant_ids={f"worker-{i}": f"worker-{i}" for i in range(8)},
        )
        for sm, bindings, tm, destinations in contracts.values():
            if not any(
                t.expert_id is None and t.tensor_id.endswith(".grouped")
                for t in sm.tensors
            ):
                continue
            destination = destinations[0]
            if previous is not None:
                assert previous.fragments[0].address != destination.fragments[0].address
            execute(
                InplaceWeightResharder(
                    MooncakeTransferEngineReader(HostEngine()),
                    plan_cache=bridge._plan_cache(),
                ),
                sm,
                bindings,
                tm,
                destination,
                CollectiveScope((*bindings, *destinations), owners),
            )
            previous = destination
        target = owners[0][1]
        block_wire.convert(dict(target.named_parameters()), 8, pack=False)
        oracle, _, _ = fixture(new, 0, initialized=True)
        for name in ("experts.w13_weight", "experts.w2_weight"):
            assert torch.equal(
                dict(target.named_parameters())[name],
                dict(oracle.named_parameters())[name],
            )
        held.append(owners)
    assert cache.cache_info().hits == 1
    assert cache.cache_info().misses == 1
    monkeypatch.setenv("MC_RESHARD_PLAN_CACHE_ENABLED", "0")
    with pytest.raises(ValueError, match="cannot change"):
        bridge._plan_cache()


def test_install_rebinds_direct_frozen_runtime_consumers():
    @dataclass(frozen=True)
    class Runtime:
        weight: torch.Tensor

    source = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    target = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16, device="meta")
    target.runtime = Runtime(target.weight)
    old_parameter = target.weight
    allocated = torch.arange(16).reshape(4, 4).to(torch.bfloat16)
    bridge._install_target(torch, target, {"weight": allocated}, source)
    assert target.runtime.weight is target.weight
    assert target.runtime.weight is not old_parameter
    value = torch.ones(4, dtype=torch.bfloat16)
    assert torch.equal(target.runtime.weight @ value, allocated @ value)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "permuted"])
def test_stable_worker_mapping_rejects_incomplete_or_changed_fleet(kind):
    old = SimpleNamespace(tp_size=8, dp_size=1, ep_size=1, pp_size=1)
    new = SimpleNamespace(tp_size=8, dp_size=1, ep_size=8, pp_size=1)
    owners = [fixture(old, rank, initialized=True) for rank in range(8)]
    targets = [fixture(new, rank, initialized=False) for rank in range(8)]
    identities = {f"worker-{i}": f"worker-{i}" for i in range(8)}
    if kind == "missing":
        identities.pop("worker-7")
    elif kind == "duplicate":
        identities["worker-7"] = "worker-0"
    else:
        from msgspec.structs import replace

        targets[0] = (*targets[0][:2], replace(targets[0][2], worker_id="worker-1"))
        targets[1] = (*targets[1][:2], replace(targets[1][2], worker_id="worker-0"))
    with pytest.raises(WeightManifestError, match="matching workers"):
        build_mooncake_transfer_groups(
            [item[2] for item in owners],
            [item[2] for item in targets],
            source_parallel_layout=old,
            target_parallel_layout=new,
            participant_ids=identities,
        )


def test_default_daemon_participants_keep_topology_coordinates():
    args = SimpleNamespace(tp_size=8, dp_size=1, ep_size=8, pp_size=1)
    owners = [fixture(args, rank, initialized=True) for rank in range(8)]
    manifests = [item[2] for item in owners]
    groups = build_mooncake_transfer_groups(
        manifests, manifests, source_parallel_layout=args, target_parallel_layout=args
    )
    for _, bindings, _, _ in groups.values():
        assert all(
            binding.participant_id.startswith("dp0:pp0:ep") for binding in bindings
        )


def test_preflight_rejects_native_group_rank_mismatch():
    args = SimpleNamespace(tp_size=8, dp_size=1, ep_size=8, pp_size=1)
    model, plan, _ = fixture(args, 7, initialized=True)
    config = SimpleNamespace(num_experts=8)
    bridge._validate_plan_topology(plan, args, 7, config)
    before = {
        name: parameter.data_ptr() for name, parameter in model.named_parameters()
    }
    with pytest.raises(ValueError, match="groups differ"):
        bridge._validate_plan_topology(plan, args, 0, config)
    assert before == {
        name: parameter.data_ptr() for name, parameter in model.named_parameters()
    }


def test_current_server_args_use_resolved_attention_dp_and_ep_fields():
    args = SimpleNamespace(
        tp_size=8,
        attn_dp_size=1,
        ep_size=1,
        pp_size=1,
        attn_cp_size=1,
        moe_dp_size=1,
        enable_dp_lm_head=False,
        init_expert_location="trivial",
        enable_eplb=False,
        ep_num_redundant_experts=0,
        elastic_ep_backend=None,
        ep_join_mode=None,
        _resolved_overrides=[("test", {"attn_dp_size": 2, "ep_size": 8})],
    )
    normalized = bridge._normalize_args(args)
    assert (normalized.dp_size, normalized.ep_size) == (2, 8)


def test_oversized_meta_target_is_rejected_before_source_mutation():
    source = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    target = torch.nn.Linear(
        4, (1 << 29) + 1, bias=False, dtype=torch.bfloat16, device="meta"
    )
    config = SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        to_dict=lambda: {"model_type": "qwen3_moe"},
    )
    source.config = target.config = config
    args = SimpleNamespace(
        tp_size=8,
        dp_size=1,
        ep_size=1,
        pp_size=1,
        enable_dp_lm_head=False,
        init_expert_location="trivial",
        enable_eplb=False,
        ep_num_redundant_experts=0,
        elastic_ep_backend=None,
        ep_join_mode=None,
    )
    pointer = source.weight.data_ptr()
    with pytest.raises(ValueError, match="1 GiB"):
        bridge._validate_models(source, target, args, args, 0)
    assert source.weight.data_ptr() == pointer
    assert target.weight.is_meta


def test_public_default_group_reads_current_runtime_world(monkeypatch):
    """The default entry must use the current world CPU group before preflight."""
    from sglang.srt.runtime_context import get_parallel

    group = object()
    observed = []
    args = SimpleNamespace(
        tp_size=8,
        attn_dp_size=1,
        ep_size=1,
        pp_size=1,
        attn_cp_size=1,
        moe_dp_size=1,
        enable_dp_lm_head=False,
        init_expert_location="trivial",
        enable_eplb=False,
        ep_num_redundant_experts=0,
        elastic_ep_backend=None,
        ep_join_mode=None,
    )
    monkeypatch.delenv("MC_INTRANODE_NVLINK", raising=False)
    monkeypatch.setattr(bridge, "_PARKED", [])
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda value: observed.append(value) or 8
    )
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda value: observed.append(value) or 0
    )
    monkeypatch.setitem(
        get_parallel()._overrides, "world_group", SimpleNamespace(cpu_group=group)
    )
    with pytest.raises(ValueError, match="MC_INTRANODE_NVLINK"):
        bridge.bind_weights(None, None, args, args, 0)
    assert observed == [group, group]


@pytest.mark.parametrize("phase", ["source", "target"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("init_expert_location", '{"physical_to_logical_map":[[1,0]]}'),
        ("enable_eplb", True),
        ("ep_num_redundant_experts", 1),
        ("elastic_ep_backend", "mooncake"),
        ("ep_join_mode", "scale"),
    ],
)
def test_public_preflight_rejects_nonstatic_source_or_target(
    monkeypatch, phase, field, value
):
    args = dict(
        tp_size=8,
        attn_dp_size=1,
        ep_size=1,
        pp_size=1,
        attn_cp_size=1,
        moe_dp_size=1,
        enable_dp_lm_head=False,
        init_expert_location="trivial",
        enable_eplb=False,
        ep_num_redundant_experts=0,
        elastic_ep_backend=None,
        ep_join_mode=None,
    )
    old, new = SimpleNamespace(**args), SimpleNamespace(**args)
    selected = old if phase == "source" else new
    selected._resolved_overrides = [("test", {field: value})]
    source = torch.nn.Linear(4, 4, bias=False)
    target = torch.nn.Linear(4, 4, bias=False, device="meta")
    before = source.weight.detach().clone()
    pointer = source.weight.data_ptr()
    monkeypatch.setenv("MC_INTRANODE_NVLINK", "1")
    monkeypatch.setattr(bridge, "_PARKED", [])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    with pytest.raises(ValueError, match=f"{phase} requires static trivial"):
        bridge.bind_weights(source, target, old, new, 0, group=object())
    assert source.weight.data_ptr() == pointer
    assert torch.equal(source.weight, before)
    assert target.weight.is_meta


def test_source_mapping_fields_are_required_without_default_acceptance():
    args = SimpleNamespace(
        tp_size=8,
        attn_dp_size=1,
        ep_size=1,
        pp_size=1,
        attn_cp_size=1,
        moe_dp_size=1,
        enable_dp_lm_head=False,
    )
    with pytest.raises(AttributeError, match="init_expert_location"):
        bridge._normalize_args(args)
