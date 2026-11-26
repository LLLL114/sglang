# SPDX-License-Identifier: Apache-2.0
"""Explicit stopped-epoch bridge over SGLang's canonical layout contracts."""

import os
import socket
from types import SimpleNamespace
from uuid import uuid4

import msgspec
from mooncake.reshard.weight.inplace import InplaceWeightResharder
from mooncake.reshard.weight.lifetime import (
    AcquiredWeightBinding,
    weight_allocation_fence,
)
from mooncake.reshard.weight.plan_cache import WeightTransferPlanCache
from mooncake.reshard.weight.te import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
)

from ..model_adapters import WeightAdapterContext, get_weight_layout_adapter
from ..mooncake_weight_adapter import build_mooncake_transfer_groups
from ..tensor_aliases import capture_tensor_aliases, rebind_tensor_aliases
from ..weight_runtime_manifest import (
    ImmutableWeightRuntimeManifestBuilder,
    WeightParallelTopology,
    model_identity_from_config,
)
from . import arena_weights, block_wire, endpoint_order, wire


def _normalize_args(args):
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(args)
    return SimpleNamespace(
        tp_size=cfg.tp_size,
        dp_size=cfg.attn_dp_size,
        ep_size=cfg.ep_size,
        pp_size=cfg.pp_size,
        attn_cp_size=cfg.attn_cp_size,
        moe_dp_size=cfg.moe_dp_size,
        enable_dp_lm_head=cfg.enable_dp_lm_head,
        init_expert_location=cfg.init_expert_location,
        enable_eplb=cfg.enable_eplb,
        ep_num_redundant_experts=cfg.ep_num_redundant_experts,
        elastic_ep_backend=cfg.elastic_ep_backend,
        ep_join_mode=cfg.ep_join_mode,
    )


def _layout(args):
    if args.tp_size != 8 or (args.dp_size, args.ep_size) not in {
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (2, 8),
        (8, 8),
    }:
        raise ValueError("requires a supported same-eight-worker DP/EP layout")
    for field in ("pp_size", "attn_cp_size", "moe_dp_size"):
        if getattr(args, field, 1) != 1:
            raise ValueError(f"unsupported topology: {field}")
    return SimpleNamespace(tp_size=8, pp_size=1, ep_size=args.ep_size)


def _build_plan(model, *, meta=False):
    context = WeightAdapterContext(model.config, allow_meta=meta)
    adapter = get_weight_layout_adapter(model, context)
    return adapter.build_layout(model, context)


def _validate_plan_topology(plan, args, rank, config):
    expert_tp, attention_tp = 8 // args.ep_size, 8 // args.dp_size
    local_experts = config.num_experts // args.ep_size
    first_expert = (rank // expert_tp) * local_experts
    for tensor in plan.tensors:
        partition = tensor.partition
        if tensor.expert_id is not None:
            if (
                partition.size != expert_tp
                or partition.rank != rank % expert_tp
                or not first_expert <= tensor.expert_id < first_expert + local_experts
            ):
                raise ValueError("native expert groups differ from requested topology")
        elif tensor.tensor_id.endswith((".q_proj.weight", ".o_proj.weight")):
            if partition.size != attention_tp or partition.rank != rank % attention_tp:
                raise ValueError(
                    "native attention groups differ from requested topology"
                )


def _manifest(model, plan, args, rank, endpoint, epoch, phase):
    moe_tp = 8 // args.ep_size
    return ImmutableWeightRuntimeManifestBuilder(
        model=model,
        layout_plan=plan,
        topology=WeightParallelTopology(
            tp_rank=rank % moe_tp,
            tp_size=moe_tp,
            ep_rank=rank // moe_tp,
            ep_size=args.ep_size,
        ),
    ).build(
        model_id=model_identity_from_config(model.config),
        revision="live-weights",
        instance_id=f"{phase}:{epoch}:{rank}",
        worker_id=f"worker-{rank}",
        endpoint=endpoint,
    )


def _validate_models(old_model, new_model, old_args, target_args, rank):
    _layout(old_args)
    _layout(target_args)
    if not 0 <= rank < 8:
        raise ValueError("rank outside the same fleet")
    for phase, args in (("source", old_args), ("target", target_args)):
        if (
            args.init_expert_location != "trivial"
            or args.enable_eplb
            or args.ep_num_redundant_experts != 0
            or args.elastic_ep_backend is not None
            or args.ep_join_mode is not None
        ):
            raise ValueError(
                f"{phase} requires static trivial expert placement without EPLB, "
                "redundant experts or elastic EP"
            )
    if old_args.enable_dp_lm_head != target_args.enable_dp_lm_head:
        raise ValueError("changing DP LM-head mode is unsupported")
    if old_model.config.architectures != ["Qwen3MoeForCausalLM"]:
        raise ValueError("same-fleet bridge supports Qwen3 MoE only")
    if model_identity_from_config(old_model.config) != model_identity_from_config(
        new_model.config
    ):
        raise ValueError("source and target model identities differ")
    sources = dict(old_model.named_parameters(remove_duplicate=False))
    targets = dict(new_model.named_parameters(remove_duplicate=False))
    if sources.keys() != targets.keys():
        raise ValueError("source and target parameter inventories differ")
    arena_weights.chunks(targets, targets)
    devices = {p.device for p in sources.values()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("source parameters must share the local CUDA device")
    if any(not p.is_meta for p in targets.values()):
        raise ValueError("target parameters must be meta before the stopped epoch")
    if len({id(p) for p in targets.values()}) != len(targets):
        raise ValueError("tied target storage is unsupported")
    spans = sorted(
        (p.data_ptr(), p.data_ptr() + p.numel() * p.element_size())
        for p in sources.values()
    )
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("tied or overlapping source storage is unsupported")
    if any(
        not p.is_contiguous() or p.numel() * p.element_size() > arena_weights.CAP_BYTES
        for p in sources.values()
    ):
        raise ValueError("source parameters must be contiguous and fit arena chunks")
    buffers = dict(old_model.named_buffers(remove_duplicate=False))
    for name, value in new_model.named_buffers(remove_duplicate=False):
        original = buffers.get(name)
        if (
            original is None
            or value.shape != original.shape
            or value.dtype != original.dtype
            or (not value.is_meta and value is not original)
        ):
            raise ValueError("target buffers must match local source buffers")


def _install_target(torch, model, allocated, source_model):
    aliases = capture_tensor_aliases(
        model,
        {
            **dict(model.named_parameters(remove_duplicate=False)),
            **dict(model.named_buffers(remove_duplicate=False)),
        },
    )
    replacements = {}
    for name, value in allocated.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        previous = parent._parameters[leaf]
        replacement = torch.nn.Parameter(value, requires_grad=previous.requires_grad)
        replacement.__dict__.update(previous.__dict__)
        parent._parameters[leaf] = replacement
        replacements[name] = replacement
    source_buffers = dict(source_model.named_buffers(remove_duplicate=False))
    for name, _ in model.named_buffers(remove_duplicate=False):
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._buffers[leaf] = source_buffers[name]
        replacements[name] = source_buffers[name]
    rebind_tensor_aliases(aliases, replacements)


def _domain_presence(targets):
    def domain(name):
        if ".experts." in name:
            return "moe"
        if name.endswith((".k_proj.weight", ".v_proj.weight")):
            return "kv"
        return "attention"

    presence = {name: [] for name in ("attention", "kv", "moe")}
    for target in targets:
        local = {domain(fragment.tensor_id) for fragment in target.fragments}
        for name in presence:
            presence[name].append(name in local)
    if any(any(values) and not all(values) for values in presence.values()):
        raise ValueError("phased transfer refuses mixed empty target domains")
    return {
        name: "all-nonempty" if values and all(values) else "globally-empty-skip"
        for name, values in presence.items()
    }


def bind_weights(
    old_model, new_model, old_args, target_args, rank, *, engine=None, group=None
):
    """Transfer a drained fleet; any failure after mutation requires restart.

    The caller drains admission, requests and external weight readers, clears KV,
    drops graphs and stale views, and prevents new manifest snapshots throughout
    this call. Target parameters are meta until this stopped epoch. After success
    the caller rebuilds execution state and runtime manifests before admission.
    Module types, configuration, backends and process-group metadata stay fixed;
    only this helper's tensor replacements and packed weight bytes may change.
    """
    global _STATE, _EPOCH
    import torch
    import torch.distributed as dist

    old_args, target_args = _normalize_args(old_args), _normalize_args(target_args)

    if group is None:
        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().world_group.cpu_group
    if dist.get_world_size(group) != 8 or dist.get_rank(group) != rank:
        raise ValueError("requires eight global-rank-ordered workers")
    if _PARKED:
        raise RuntimeError("unresolved transition; restart the complete fleet")
    if os.environ.get("MC_INTRANODE_NVLINK", "").lower() not in ("1", "true"):
        raise ValueError("MC_INTRANODE_NVLINK must select the native route")
    _validate_models(old_model, new_model, old_args, target_args, rank)
    if next(old_model.parameters()).device.index != torch.cuda.current_device():
        raise ValueError("source model must use the current device")
    source_plan = _build_plan(old_model)
    target_plan = _build_plan(new_model, meta=True)
    _validate_plan_topology(source_plan, old_args, rank, old_model.config)
    _validate_plan_topology(target_plan, target_args, rank, new_model.config)
    for model, args, peer in (
        (old_model, old_args, target_args),
        (new_model, target_args, old_args),
    ):
        block_wire.validate(
            dict(model.named_parameters()),
            block_wire.parts(args, peer),
            attention_blocks=max(1, args.dp_size // peer.dp_size),
        )
    aliases = wire.alias_names(source_plan, target_plan)
    source_parameters = dict(old_model.named_parameters(remove_duplicate=False))
    arena_weights.source_chunks(source_parameters, set(source_parameters) - aliases)
    del source_parameters
    source_template, source_tensors = wire.build_wire_plan(
        old_model, source_plan, old_args, target_args, old_model.config
    ).as_meta(old_model)
    target_template, target_tensors = wire.build_wire_plan(
        new_model, target_plan, target_args, old_args, new_model.config
    ).as_meta(new_model)
    # Layout plans own tensor references and must not pin replaced allocations.
    del source_plan, target_plan
    if engine is None:
        if _STATE is None:
            from mooncake.engine import TransferEngine

            candidate = TransferEngine()
            if candidate.initialize("127.0.0.1", "P2PHANDSHAKE", "nvlink_intra", ""):
                raise RuntimeError("native initialize failed")
            _STATE = candidate
        engine = _STATE
    if any(
        not callable(getattr(engine, name, None))
        for name in (
            "register_memory",
            "unregister_memory",
            "get_rpc_port",
            "batch_transfer_sync_read",
            "reset_reshard_peers_quiescent",
        )
    ):
        raise TypeError("engine lacks same-fleet native APIs")

    def gather(value):
        result = [None] * 8
        dist.all_gather_object(result, value, group=group)
        return result

    _EPOCH += 1
    epoch = _EPOCH
    endpoint = f"127.0.0.1:{engine.get_rpc_port()}"
    cache = _plan_cache()
    startup = gather(
        (
            endpoint,
            cache.enabled,
            cache.cache_info().maxsize,
            socket.gethostname(),
            str(torch.cuda.get_device_properties().uuid),
        )
    )
    if len({item[1:3] for item in startup}) != 1:
        raise ValueError("plan cache settings must agree across workers")
    if (
        len({item[3] for item in startup}) != 1
        or len({item[4] for item in startup}) != 8
    ):
        raise ValueError("same-fleet NVLink requires eight distinct GPUs on one host")
    endpoints = [item[0] for item in startup]
    if len(set(endpoints)) != 8:
        raise ValueError("workers must publish distinct native endpoints")
    parameters = dict(old_model.named_parameters(remove_duplicate=False))
    profile, owners, registered, groups = {}, [], [], []
    retained = dict(
        source=old_model,
        target=new_model,
        parameters=parameters,
        owners=owners,
        engine=engine,
        registered=registered,
        groups=groups,
    )
    dist.barrier(group=group)
    _PARKED.append(retained)
    arena_weights.compact_source(
        torch, old_model, parameters, set(parameters) - aliases, profile, owners
    )
    source_wire = block_wire.convert(
        parameters,
        block_wire.parts(old_args, target_args),
        pack=True,
        attention_blocks=max(1, old_args.dp_size // target_args.dp_size),
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)
    allocated = arena_weights.allocate_target(
        torch, new_model, parameters, aliases, profile
    )
    retained["allocated"] = allocated
    _install_target(torch, new_model, allocated, old_model)
    # Native layer semantics were validated before mutation and stay fixed during
    # this controlled stop. Recheck materialized adapter/storage prerequisites.
    get_weight_layout_adapter(old_model, WeightAdapterContext(old_model.config))
    get_weight_layout_adapter(new_model, WeightAdapterContext(new_model.config))
    source_plan = source_template.rebind(old_model, source_tensors)
    target_plan = target_template.rebind(new_model, target_tensors)
    del source_template, target_template, source_tensors, target_tensors
    target_plan.initialize_local_regions()
    sm = _manifest(old_model, source_plan, old_args, rank, endpoint, epoch, "source")
    tm = wire.exclude_parameters(
        _manifest(new_model, target_plan, target_args, rank, endpoint, epoch, "target"),
        target_plan,
        aliases,
    )
    sent = (len(sm.fragments), len(tm.fragments))
    gathered = gather((sm, tm))
    sources, targets = wire.close_inventory(*zip(*gathered))
    domain_presence = _domain_presence(targets)
    del source_plan, target_plan, gathered
    executor = endpoint_order.make_executor(
        engine,
        rank=rank,
        target_attention_tp=8 // target_args.dp_size,
        local_endpoint=endpoint,
        endpoints=endpoints,
        phase_barrier=lambda: dist.barrier(group=group),
    )
    resharder = InplaceWeightResharder(
        MooncakeTransferEngineReader(engine, transfer_executor=executor),
        plan_cache=cache,
    )
    retained["resharder"] = resharder
    contracts = (
        build_mooncake_transfer_groups(
            sources,
            targets,
            source_parallel_layout=_layout(old_args),
            target_parallel_layout=_layout(target_args),
            participant_ids={f"worker-{i}": f"worker-{i}" for i in range(8)},
        )
        if sources
        else {}
    )
    all_bindings = []
    for source, bindings, target, destinations in contracts.values():
        presence = [bool(b.fragments) for b in destinations]
        if any(presence) and not all(presence):
            raise ValueError("phased transfer refuses mixed empty target groups")
        groups.append((source, bindings, target, destinations[rank]))
        all_bindings.extend((*bindings, *destinations))
    scope = CollectiveScope(all_bindings, retained)
    retained["scope"] = scope
    spans = registration_spans(groups, rank)
    profile["registered_cuda_spans"] = len(spans)
    for address, size in sorted(spans):
        if engine.register_memory(address, size):
            raise RuntimeError("native registration failed")
        registered.append(address)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    receipts = []
    for source, bindings, target, destination in groups:
        receipts.extend(
            execute(
                resharder,
                source,
                bindings,
                target,
                destination,
                scope,
                target_pre_registered=True,
            )
        )
    torch.cuda.synchronize()
    dist.barrier(group=group)
    target_wire = block_wire.convert(
        allocated,
        block_wire.parts(target_args, old_args),
        pack=False,
        attention_blocks=max(1, target_args.dp_size // old_args.dp_size),
    )
    source_restore = block_wire.convert(
        parameters,
        block_wire.parts(old_args, target_args),
        pack=False,
        attention_blocks=max(1, old_args.dp_size // target_args.dp_size),
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)
    for address in registered:
        if engine.unregister_memory(address):
            raise RuntimeError("native unregister failed")
    dist.barrier(group=group)
    if engine.reset_reshard_peers_quiescent(
        [e for i, e in enumerate(endpoints) if i != rank]
    ):
        raise RuntimeError("native peer reset failed")
    dist.barrier(group=group)
    scope.active, scope.owners = False, None
    _PARKED.remove(retained)
    return dict(
        arena_profile=profile,
        collective_epoch=epoch,
        endpoint_order_profiles=executor.profiles,
        wire_profile=dict(
            source_pack=source_wire,
            target_unpack=target_wire,
            source_restore=source_restore,
        ),
        inventory_profile=dict(
            sent_source_spec_count=sent[0], sent_target_spec_count=sent[1]
        ),
        plan_cache=dict(enabled=cache.enabled, **cache.cache_info()._asdict()),
        actual_plan_cache=dict(
            enabled=cache.enabled, info=cache.cache_info()._asdict()
        ),
        domain_presence=domain_presence,
        transfer_groups=tuple(contracts),
        group_presence={
            key: tuple(bool(b.fragments) for b in value[3])
            for key, value in contracts.items()
        },
        wire_bytes=sum(r.nbytes for r in receipts),
        operation_count=sum(r.operation_count for r in receipts),
        alias_names=sorted(aliases),
    )


def registration_leases(bindings):
    return tuple(
        MemoryRegistrationLease(
            **{
                name: getattr(f, name)
                for name in (
                    "fragment_id",
                    "address",
                    "nbytes",
                    "worker_id",
                    "device",
                    "itemsize",
                    "local_shape",
                    "strides_bytes",
                    "storage_offset_bytes",
                    "storage_address",
                    "storage_nbytes",
                )
            },
            lease_generation=b.generation,
            runtime_lease_id=b.lease_id,
        )
        for b in bindings
        for f in b.fragments
    )


class EpochPin(msgspec.Struct):
    fence: object
    epoch_scope: object

    def release_after_terminal(self, terminal_state):
        # Framework collective scope, not this token, owns all source tensors.
        self.epoch_scope = None


class EpochGuard(msgspec.Struct):
    binding: object
    scope: object

    def acquire(self, *, transfer_id, expected_binding, required_fragment_ids):
        self.scope.assert_active(self.binding)
        if expected_binding != self.binding:
            raise ValueError("collective epoch binding changed")
        return AcquiredWeightBinding(
            self.binding,
            EpochPin(
                weight_allocation_fence(
                    self.binding, required_fragment_ids, token_id=uuid4().hex
                ),
                self.scope,
            ),
        )


def execute(
    resharder,
    source,
    sources,
    target,
    destination,
    scope,
    *,
    target_pre_registered=False,
):
    """Scope must pin local owners until successful world barrier; failure parks.

    Remote guards attest the collective epoch protocol, not remote object
    ownership. Caller registers whole CUDA allocation spans before this call,
    and unregisters/resets native peers only after world completion.
    """
    guards = {(b.instance_id, b.participant_id): EpochGuard(b, scope) for b in sources}
    return resharder.execute(
        source,
        sources,
        target,
        destination,
        source_registrations=registration_leases(sources),
        target_pre_registered=target_pre_registered,
        target_registrations=(
            registration_leases((destination,)) if target_pre_registered else None
        ),
        source_allocation_guards=guards,
        target_allocation_guards={
            (destination.instance_id, destination.participant_id): EpochGuard(
                destination, scope
            )
        },
    )


_PARKED = []
_STATE = None
_EPOCH = 0
_PLAN_CACHE = None


def _plan_cache():
    global _PLAN_CACHE
    settings = WeightTransferPlanCache.from_environment()
    if _PLAN_CACHE is None:
        _PLAN_CACHE = settings
    elif (_PLAN_CACHE.enabled, _PLAN_CACHE.cache_info().maxsize) != (
        settings.enabled,
        settings.cache_info().maxsize,
    ):
        raise ValueError("plan cache settings cannot change during the process")
    return _PLAN_CACHE


class CollectiveScope:
    """Attest frozen rank snapshots while each rank pins its own model."""

    def __init__(self, bindings, owners):
        self.bindings, self.owners = tuple(bindings), owners
        self.active = True

    def assert_active(self, binding):
        if not self.active or binding not in self.bindings:
            raise ValueError("inactive collective epoch")


def _cuda_span(address):
    import ctypes

    driver = ctypes.CDLL("libcuda.so.1")
    base, size = ctypes.c_uint64(), ctypes.c_size_t()
    query = driver.cuMemGetAddressRange_v2
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint64,
    ]
    query.restype = ctypes.c_int
    if query(ctypes.byref(base), ctypes.byref(size), address):
        raise RuntimeError("CUDA allocation span query failed")
    return base.value, size.value


def registration_spans(groups, rank, span_query=_cuda_span):
    """Register every bound local backing allocation, including needed aliases."""
    spans, storage_spans = set(), {}
    for _, sources, _, destination in groups:
        local = [b for b in sources if b.participant_id == f"worker-{rank}"]
        for binding_manifest in (*local, destination):
            for fragment in binding_manifest.fragments:
                if fragment.storage_address not in storage_spans:
                    storage_spans[fragment.storage_address] = span_query(
                        fragment.address
                    )
                address, size = storage_spans[fragment.storage_address]
                if (
                    fragment.storage_address < address
                    or fragment.storage_address + fragment.storage_nbytes
                    > address + size
                ):
                    raise RuntimeError(
                        "bound storage extends beyond native registration span"
                    )
                spans.add((address, size))
    return spans
