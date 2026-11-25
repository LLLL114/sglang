"""Rank-local attention layouts and storage I/O for HiCache v1."""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import socket
import uuid
from importlib import import_module

import torch

logger = logging.getLogger(__name__)


def get_reshard_reader_type(name="gather"):
    try:
        sdk = import_module("mooncake.reshard.kv_cache")
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "mooncake",
            "mooncake.reshard",
            "mooncake.reshard.kv_cache",
        ):
            raise
        return None
    lookup = getattr(sdk, "get_store_page_reader", None)
    return (
        lookup(name) if lookup is not None and hasattr(sdk, "StoreReadTarget") else None
    )


def _store_formats():
    from mooncake.reshard.kv_cache import KVCacheStoreFormat

    return {
        "page_first": KVCacheStoreFormat.PLHD,
        "page_first_direct": KVCacheStoreFormat.LPHD,
        "layer_first": KVCacheStoreFormat.LPHD,
        "page_head": KVCacheStoreFormat.HPLD,
    }


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def collect_reshard_metadata(controller, config):
    """Collect actual rank ownership on the existing CPU groups at attach time."""
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
    from sglang.srt.runtime_context import get_server_args, process_model_config

    formats = _store_formats()
    options = config.extra_config["kv_reshard"]
    pool = controller.storage_host_pool
    args = get_server_args()
    model = process_model_config()
    if not isinstance(options, dict) or not options.get("model_revision"):
        raise ValueError("kv_reshard requires an explicit model_revision")
    if options.get("storage_mode", "rank_gather") != "rank_gather":
        raise ValueError("KV gather requires rank_gather storage mode")
    mla = isinstance(pool, MLATokenToKVPoolHost) and config.is_mla_model
    entries = getattr(controller.mem_pool_host, "entry_map", {})
    if (
        not (mla or isinstance(pool, MHATokenToKVPoolHost))
        or config.attn_cp_size != 1
        or config.dcp_size != 1
        or len(getattr(controller.mem_pool_host, "entries", ())) > 1
        or PoolName.MAMBA in entries
        or getattr(args, "dp_size", 1) != 1
        or getattr(args, "enable_lora", False)
        or getattr(model, "is_multimodal", False)
        or getattr(pool, "mtp_draft_device_pools", ())
        or config.tp_lcm_size
        or config.extra_config.get("enable_group_semantics", False)
    ):
        raise ValueError(
            "KV Store reshard requires MHA/GQA or dense MLA, DP1/CP1, no LoRA, side pools, or legacy splitting"
        )
    if pool.layout not in formats or pool.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("unsupported KV Store host layout or dtype")
    if mla and (
        pool.layout != "page_first_direct"
        or not config.is_rank_replicated
        or pool.kv_buffer is None
        or pool.dsa_kv_cache_store_fp8
    ):
        raise ValueError("expected a dense, TP-replicated MLA latent buffer")
    if (
        not mla and pool.v_head_dim != pool.head_dim
    ) or pool.layer_num != pool.target_layer_num:
        raise ValueError(
            "KV Store reshard requires matching K/V dimensions and plain model layers"
        )
    layers = list(range(pool.start_layer, pool.end_layer))
    local = {
        "tp": config.tp_rank,
        "pp": config.pp_rank,
        "layers": layers,
        "heads": 1 if mla else pool.head_num,
        "head_dim": pool.kv_cache_dim if mla else pool.head_dim,
        "dtype": str(pool.dtype).removeprefix("torch."),
        "page_size": pool.page_size,
        "format": formats[pool.layout].value,
        "writer_instance": uuid.uuid4().hex,
        "hostname": socket.gethostname(),
    }
    parts = [local]
    # TP first gathers one PP stage; PP then gathers every stage for each TP lane.
    for group in (controller.tp_group, controller.pp_group):
        if group is not None and torch.distributed.get_world_size(group) > 1:
            gathered = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(gathered, parts, group=group)
            parts = [part for group_parts in gathered for part in group_parts]
    parts = sorted(parts, key=lambda part: (part["pp"], part["tp"]))
    expected = {
        (pp, tp) for pp in range(config.pp_size) for tp in range(config.tp_size)
    }
    if len(parts) != len(expected) or {(p["pp"], p["tp"]) for p in parts} != expected:
        raise ValueError("incomplete TP/PP ownership for KV Store reshard")
    for part in parts:
        for field in ("head_dim", "dtype", "page_size", "format"):
            if part[field] != local[field]:
                raise ValueError(f"inconsistent KV Store descriptor: {field}")
    if len({p["hostname"] for p in parts}) != 1:
        raise ValueError("rank gather requires all TP/PP ranks on one host")
    semantic = model.hf_config.to_dict()
    for field in ("_name_or_path", "transformers_version", "_commit_hash"):
        semantic.pop(field, None)
    return {
        "parts": parts,
        "writer_id": _digest([(p["pp"], p["tp"], p["writer_instance"]) for p in parts]),
        "total_kv_heads": 1 if mla else model.get_total_num_kv_heads(),
        "component": "latent" if mla else None,
        "num_layers": model.num_hidden_layers,
        "model_name": config.model_name,
        "model_revision": options["model_revision"],
        "semantic": _digest(semantic),
        "tp_size": config.tp_size,
        "pp_size": config.pp_size,
    }


def make_placement(metadata, component):
    """Build the existing Mooncake placement from collected framework ownership."""
    from mooncake.reshard.contracts import (
        ParticipantId,
        PlacementSetId,
        ResourceId,
        RevisionId,
    )
    from mooncake.reshard.kv_cache import (
        KVCacheComponent,
        KVCacheDescriptor,
        KVCachePlacementManifest,
        KVCachePlacementPart,
        KVCacheRank,
        KVCacheTopology,
        KVCacheTopologyParticipant,
    )

    ranks = metadata["parts"]
    local = ranks[0]
    total_heads = metadata["total_kv_heads"]
    tp_size = metadata["tp_size"]
    descriptor = KVCacheDescriptor(
        global_layer_ids=tuple(range(metadata["num_layers"])),
        components=(KVCacheComponent(component),),
        dtype=local["dtype"],
        itemsize={"float16": 2, "bfloat16": 2, "float32": 4}[local["dtype"]],
        page_size=local["page_size"],
        total_kv_heads=total_heads,
        key_head_dim=local["head_dim"],
        value_head_dim=local["head_dim"],
    )
    participants = tuple(
        KVCacheTopologyParticipant(
            ParticipantId(f"pp{p['pp']}-tp{p['tp']}"),
            KVCacheRank(dp=0, pp=p["pp"], tp=p["tp"]),
        )
        for p in ranks
    )
    topology = KVCacheTopology(
        dp_size=1,
        pp_size=metadata["pp_size"],
        tp_size=tp_size,
        participants=participants,
    )
    resource = ResourceId(
        "sglang-kv:"
        + _digest(
            [metadata["model_name"], metadata["model_revision"], metadata["semantic"]]
        )
    )
    revision = RevisionId(metadata["model_revision"])
    placement_set = PlacementSetId("sglang:" + _digest(ranks))
    parts = []
    for record, participant in zip(ranks, participants):
        if total_heads >= tp_size and total_heads % tp_size == 0:
            head_count, replicas = total_heads // tp_size, 1
            head_start, ordinal = record["tp"] * head_count, 0
        elif tp_size > total_heads and tp_size % total_heads == 0:
            head_count, replicas = 1, tp_size // total_heads
            head_start, ordinal = record["tp"] // replicas, record["tp"] % replicas
        else:
            raise ValueError("unsupported TP/KV-head ratio")
        if record["heads"] != head_count:
            raise ValueError("runtime head ownership differs from model topology")
        parts.append(
            KVCachePlacementPart(
                resource_id=resource,
                revision=revision,
                placement_set_id=placement_set,
                topology_id=topology.topology_id,
                participant_id=participant.participant_id,
                rank=participant.rank,
                descriptor=descriptor,
                layer_ids=tuple(record["layers"]),
                head_start=head_start,
                head_count=head_count,
                replica_ordinal=ordinal,
                replica_count=replicas,
            )
        )
    return KVCachePlacementManifest(
        resource_id=resource,
        revision=revision,
        placement_set_id=placement_set,
        topology=topology,
        descriptor=descriptor,
        parts=tuple(parts),
    )


class _ComponentIO:
    """Source writer and rank-local buffer binding for one component."""

    def __init__(self, native_store, pool, config, component):
        from mooncake.reshard.contracts import ParticipantId
        from mooncake.reshard.kv_cache import (
            StoreReadTarget,
            KVCacheStoreManifest,
            plan_kv_cache_store_upload,
        )

        self.pool = pool
        metadata = config.kv_reshard_metadata
        self._component = component
        self.placement = make_placement(metadata, component)
        self.participant = ParticipantId(f"pp{config.pp_rank}-tp{config.tp_rank}")
        upload_plan = plan_kv_cache_store_upload(
            self.placement,
            object_format=_store_formats()[pool.layout],
        )
        manifest = KVCacheStoreManifest(
            str(config.extra_config.get("extra_backend_tag", "sglang")),
            metadata["model_name"],
            metadata["model_revision"],
            metadata["semantic"],
            upload_plan.layout,
        )
        from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE

        from .rank_gather import RankGatherWriter

        writers = upload_plan.part_writers
        if len(writers) != len(self.placement.parts) and self._component != "latent":
            raise ValueError("unsupported replicated writer for this resource")
        self.writer = None
        if self.participant in writers:
            self.writer = RankGatherWriter(
                native_store,
                manifest,
                writers.index(self.participant),
                _digest([metadata["writer_id"], component]),
                STORAGE_BATCH_SIZE,
            )
        self._pool_signature = self._current_pool_signature()
        paged = pool.layout in ("page_first_direct", "page_head")
        self._slot_unit = pool.page_size if paged else 1
        slot_axis = 1 if pool.layout == "layer_first" else 0
        tensor = self._tensor
        self._slot_stride = tensor.stride(slot_axis) * tensor.element_size()
        binding = self._page_binding()
        (self._read_region,) = binding.regions
        self.target = StoreReadTarget(self.placement, binding, manifest.model_domain)
        format_tag = {"key": "rsrg4_k", "value": "rsrg4_v", "latent": "rsrg2"}[
            component
        ]
        self._key_suffix = f"_m{manifest.model_domain}_{format_tag}"
        logger.info(
            "Mooncake KV reshard registered layout=%s participant=%s manifest=%s part=%d/%d",
            manifest.layout.layout_id,
            self.participant,
            manifest.manifest_key,
            self.writer.rank if self.writer is not None else -1,
            len(manifest.layout.shards),
        )

    def object_keys(self, tagged_keys):
        return tuple(f"{key}{self._key_suffix}" for key in tagged_keys)

    @property
    def _tensor(self):
        if self._component == "latent":
            return self.pool.kv_buffer
        return self.pool.k_buffer if self._component == "key" else self.pool.v_buffer

    def _page_binding(self):
        """Describe page zero. Every other slot is a byte translation of this template."""
        from mooncake.reshard.kv_cache import (
            KVCacheRegisteredRegion,
            KVCacheResolvedRange,
            KVCacheStoreRuntimeBinding,
        )

        pool = self.pool
        part = self.placement.part(self.participant)
        # Indices of the layer, token and head axes in a single K or V allocation.
        layer_axis, token_axis, head_axis = {
            "layer_first": (0, 1, 2),
            "page_first": (1, 0, 2),
            "page_first_direct": (1, 2, 3),
            "page_head": (3, 2, 1),
        }[pool.layout]
        # In page_head, a single head's tokens/layers precede the next head.
        heads_per_range = 1 if pool.layout == "page_head" else part.head_count
        tensor = self._tensor
        (component,) = self.placement.descriptor.components
        stride = tuple(s * tensor.element_size() for s in tensor.stride())
        region = KVCacheRegisteredRegion(
            component.value,
            "hicache-host",
            tensor.data_ptr(),
            (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())))
            * tensor.element_size(),
        )
        ranges = []
        for index, layer in enumerate(part.layer_ids):
            for head in range(0, part.head_count, heads_per_range):
                ranges.append(
                    KVCacheResolvedRange(
                        layer,
                        component,
                        0,
                        pool.page_size,
                        part.head_start + head,
                        heads_per_range,
                        component.value,
                        index * stride[layer_axis] + head * stride[head_axis],
                        stride[token_axis],
                        stride[head_axis],
                    )
                )
        instance = uuid.uuid4().hex
        return KVCacheStoreRuntimeBinding(
            "page-template:" + instance,
            self.placement.resource_id,
            self.placement.placement_id,
            self.placement.digest,
            instance,
            self.placement.revision,
            self.participant,
            (region,),
            tuple(ranges),
        )

    def close(self) -> None:
        """Release adapter-owned resources after HiCache has drained pending I/O."""
        if self.writer is not None:
            self.writer.close()

    def _current_pool_signature(self):
        t = self._tensor
        return (
            self.pool.layout,
            self.pool.page_size,
            self.pool.size,
            t.data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
        )

    def _page_slots(self, keys, indices):
        if self._current_pool_signature() != self._pool_signature:
            raise ValueError(
                "KV Store pool changed; reattach the backend before transfer"
            )
        page_size = self.pool.page_size
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("HiCache indices must be a one-dimensional integer tensor")
        if indices.numel() != len(keys) * page_size:
            raise ValueError("transfer keys and host indices differ")
        tokens = indices.tolist()
        slots = tokens[::page_size]
        for page, slot in enumerate(slots):
            if slot < 0 or slot + page_size > self.pool.size:
                raise ValueError("HiCache page outside host pool")
            if slot % page_size or tokens[
                page * page_size : (page + 1) * page_size
            ] != list(range(slot, slot + page_size)):
                raise ValueError("HiCache page slots must be aligned and contiguous")
        return slots

    def upload(self, physical, indices):
        if self.writer is None:
            raise ValueError("only canonical latent replicas publish")
        slots = self._page_slots(physical, indices)
        if self._component == "latent":
            tensor = self._tensor
            if self.pool.layout != "page_first_direct":
                raise ValueError("single resource requires page_first_direct")
            pages = slots
            src = []
            lengths = []
            for slot in pages:
                blocks = tensor[slot // self.pool.page_size]
                if blocks.is_contiguous():
                    blocks = [blocks]
                else:
                    blocks = blocks.unbind(0)
                if not all(b.is_contiguous() for b in blocks):
                    raise ValueError("noncontiguous resource layer payload")
                src.append([b.data_ptr() for b in blocks])
                lengths.append([b.numel() * b.element_size() for b in blocks])
            return self.writer.upload(physical, src, lengths)
        pointers, sizes = self.pool.get_page_buffer_meta(indices)
        # Base MHA pools return alternating K/V spans, one pair per layer for
        # layer_first and one pair per page for the other layouts.
        width = 2 * self.pool.layer_num if self.pool.layout == "layer_first" else 2
        if len(pointers) != width * len(physical) or len(sizes) != len(pointers):
            raise ValueError("HiCache page buffers differ from gather payload coverage")

        def pages(values):
            rows = []
            for start in range(0, len(values), width):
                spans = values[start : start + width]
                ordered = spans[int(self._component == "value") :: 2]
                row = []
                for span in ordered:
                    if isinstance(span, (list, tuple)):
                        row.extend(span)
                    else:
                        row.append(span)
                rows.append(row)
            return rows

        return self.writer.upload(physical, pages(pointers), pages(sizes))

    def read_buffers(self, keys, indices):
        region = self._read_region
        offsets = [
            (slot // self._slot_unit) * self._slot_stride
            for slot in self._page_slots(keys, indices)
        ]
        return [(region.address + delta, region.nbytes - delta) for delta in offsets]


class ReshardManager:
    """Read and write the logical pages selected by HiCache."""

    def __init__(self, store, pool, config):
        name = config.extra_config["kv_reshard"].get("reader", "gather")
        reader_type = get_reshard_reader_type(name)
        if reader_type is None:
            raise ImportError(
                "kv_reshard requires a Mooncake SDK with registered batch_get_into readers"
            )
        names = (
            ("latent",)
            if config.kv_reshard_metadata.get("component") == "latent"
            else ("key", "value")
        )
        from mooncake.reshard.kv_cache import GATHER_HEADER_BYTES
        from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE

        self._store = store
        self._scratch_ptr = None
        self.components = []
        try:
            for name in names:
                self.components.append(_ComponentIO(store, pool, config, name))
            first, *others = [c.target for c in self.components]
            self._scratch = bytearray(STORAGE_BATCH_SIZE * (GATHER_HEADER_BYTES + 32))
            self.reader = reader_type(
                store, *first, self._scratch, additional_targets=others
            )
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._scratch))
            if store.register_buffer(ptr, len(self._scratch)) != 0:
                raise RuntimeError("failed to register GatherPageReader scratch")
            self._scratch_ptr = ptr
        except BaseException:
            self.close()
            raise

    def object_keys(self, keys):
        # The ordinary batch_exists contract counts the components of each page.
        return [
            k
            for page in zip(
                *(c.object_keys(keys) for c in self.components), strict=True
            )
            for k in page
        ]

    def prepare_read(self, keys, host_indices):
        """Prepare v1 vectors in component order: all K pages, then all V pages."""
        key_strs, ptrs, sizes = [], [], []
        for c in self.components:
            key_strs.extend(c.object_keys(keys))
            for ptr, size in c.read_buffers(keys, host_indices):
                ptrs.append(ptr)
                sizes.append(size)
        return key_strs, ptrs, sizes

    def batch_set_v1(self, keys, host_indices):
        per_component = [
            c.upload(c.object_keys(keys), host_indices) for c in self.components
        ]
        return [
            all(code == 0 for code in page) for page in zip(*per_component, strict=True)
        ]

    def close(self):
        for c in self.components:
            c.close()
        if self._scratch_ptr is not None:
            if self._store.unregister_buffer(self._scratch_ptr) != 0:
                raise RuntimeError("failed to unregister GatherPageReader scratch")
            self._scratch_ptr = None
