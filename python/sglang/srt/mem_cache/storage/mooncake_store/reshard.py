"""HiCache host-pool adapter for Mooncake source-native KV Store resharding."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from dataclasses import replace

import torch
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
    KVCacheRegisteredRegion,
    KVCacheResolvedRange,
    KVCacheResolvedRuntimeBinding,
    KVCacheStoreFormat,
    KVCacheStoreManifest,
    KVCacheTopology,
    KVCacheTopologyParticipant,
    MultipartKVCacheStore,
    MultipartReadContext,
    plan_kv_cache_store_upload,
)

logger = logging.getLogger(__name__)

_FORMATS = {
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
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.runtime_context import get_server_args, process_model_config

    options = config.extra_config["kv_reshard"]
    pool = controller.storage_host_pool
    args = get_server_args()
    model = process_model_config()
    if not isinstance(options, dict) or not options.get("model_revision"):
        raise ValueError("kv_reshard requires an explicit model_revision")
    if (
        not isinstance(pool, MHATokenToKVPoolHost)
        or config.is_mla_model
        or config.attn_cp_size != 1
        or getattr(args, "dp_size", 1) != 1
        or getattr(args, "enable_lora", False)
        or getattr(model, "is_multimodal", False)
        or getattr(pool, "mtp_draft_device_pools", ())
        or config.tp_lcm_size
        or config.extra_config.get("enable_group_semantics", False)
    ):
        raise ValueError(
            "KV Store reshard requires plain MHA/GQA, DP1/CP1, no LoRA, side pools, or legacy splitting"
        )
    if pool.layout not in _FORMATS or pool.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("unsupported KV Store host layout or dtype")
    device = pool.device_pool
    if device.v_head_dim != device.head_dim or pool.layer_num != device.layer_num:
        raise ValueError(
            "KV Store reshard requires matching K/V dimensions and plain model layers"
        )
    local = {
        "tp": config.tp_rank,
        "pp": config.pp_rank,
        "layers": list(range(device.start_layer, device.end_layer)),
        "heads": pool.head_num,
        "head_dim": pool.head_dim,
        "dtype": str(pool.dtype).removeprefix("torch."),
        "page_size": pool.page_size,
        "format": _FORMATS[pool.layout].value,
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
    semantic = model.hf_config.to_dict()
    for field in ("_name_or_path", "transformers_version", "_commit_hash"):
        semantic.pop(field, None)
    return {
        "parts": parts,
        "total_kv_heads": model.get_total_num_kv_heads(),
        "num_layers": model.num_hidden_layers,
        "model_name": config.model_name,
        "model_revision": options["model_revision"],
        "semantic": _digest(semantic),
        "tp_size": config.tp_size,
        "pp_size": config.pp_size,
    }


def make_placement(metadata):
    """Build the existing Mooncake placement from collected framework ownership."""
    ranks = metadata["parts"]
    local = ranks[0]
    total_heads = metadata["total_kv_heads"]
    tp_size = metadata["tp_size"]
    descriptor = KVCacheDescriptor(
        global_layer_ids=tuple(range(metadata["num_layers"])),
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


class MooncakeKVReshardAdapter:
    """Own metadata only; HiCache owns and protects the registered host buffers."""

    def __init__(self, native_store, pool, config):
        if config.extra_config.get("kv_reshard", {}).get("multipart") is False:
            raise ValueError(
                "legacy KV reshard was removed; use the measured snapshot for comparisons"
            )
        self.pool = pool
        self.metadata = config.kv_reshard_metadata
        self.placement = make_placement(self.metadata)
        self.participant = ParticipantId(f"pp{config.pp_rank}-tp{config.tp_rank}")
        self.instance = str(uuid.uuid4())
        self.format = _FORMATS[pool.layout]
        self.upload_plan = plan_kv_cache_store_upload(
            self.placement, object_format=self.format
        )
        manifest = KVCacheStoreManifest(
            str(config.extra_config.get("extra_backend_tag", "sglang")),
            self.metadata["model_name"],
            self.metadata["model_revision"],
            self.metadata["semantic"],
            self.upload_plan.layout,
        )
        self.native_store = native_store
        if len(manifest.layout.shards) != len(self.placement.parts):
            raise ValueError("multipart requires one distinct KV shard per rank")
        self.store = MultipartKVCacheStore(native_store, manifest)
        self._part_index = self.upload_plan.part_writers.index(self.participant)
        # Ownership was gathered across all ranks before backend attachment.
        # Only the request coordinator publishes; serving starts after rank init.
        if config.tp_rank == 0 and config.pp_rank == 0:
            self.store.register_layout()
        self._model_domain = manifest.model_domain
        self._pool_signature = self._current_pool_signature()
        axis = 1 if pool.layout == "layer_first" else 0
        self._translation_unit = (
            pool.page_size if pool.layout in ("page_first_direct", "page_head") else 1
        )
        self._translation_steps = tuple(
            (component.value, tensor.stride(axis) * tensor.element_size())
            for component, tensor in (
                (KVCacheComponent.KEY, pool.k_buffer),
                (KVCacheComponent.VALUE, pool.v_buffer),
            )
        )
        self.reader = self.store.prepare_page_reader(
            self.placement,
            self.binding(
                torch.arange(pool.page_size), "page-template:" + self.instance
            ),
        )
        logger.info(
            "Mooncake KV reshard registered layout=%s participant=%s manifest=%s part=%d/%d",
            manifest.layout.layout_id,
            self.participant,
            manifest.manifest_key,
            self._part_index,
            len(manifest.layout.shards),
        )

    def binding(self, indices, operation_id):
        """Map the current page allocation to registered K/V tensors without copies."""
        pool, placement = self.pool, self.placement
        part = placement.part(self.participant)
        tokens = indices.tolist()
        page_size = pool.page_size
        if len(tokens) % page_size:
            raise ValueError("KV Store transfer must contain complete HiCache pages")
        regions, ranges = [], []
        for component, tensor in (
            (KVCacheComponent.KEY, pool.k_buffer),
            (KVCacheComponent.VALUE, pool.v_buffer),
        ):
            region_id = component.value
            regions.append(
                KVCacheRegisteredRegion(
                    region_id,
                    "hicache-host",
                    tensor.data_ptr(),
                    tensor.numel() * tensor.element_size(),
                )
            )
            strides = [s * tensor.element_size() for s in tensor.stride()]
            for page in range(len(tokens) // page_size):
                slot = tokens[page * page_size]
                if slot % page_size or tokens[
                    page * page_size : (page + 1) * page_size
                ] != list(range(slot, slot + page_size)):
                    raise ValueError(
                        "HiCache page slots must be aligned and contiguous"
                    )
                if slot < 0 or slot + page_size > pool.size:
                    raise ValueError("HiCache page outside host pool")
                if pool.layout == "page_first":
                    base, ls, ts, hs = (
                        slot * strides[0],
                        strides[1],
                        strides[0],
                        strides[2],
                    )
                elif pool.layout == "layer_first":
                    base, ls, ts, hs = (
                        slot * strides[1],
                        strides[0],
                        strides[1],
                        strides[2],
                    )
                elif pool.layout == "page_first_direct":
                    base, ls, ts, hs = (
                        slot // page_size * strides[0],
                        strides[1],
                        strides[2],
                        strides[3],
                    )
                else:
                    base, ls, ts, hs = (
                        slot // page_size * strides[0],
                        strides[3],
                        strides[2],
                        strides[1],
                    )
                for local_layer, layer in enumerate(part.layer_ids):
                    for head in (
                        range(part.head_count)
                        if self.format is KVCacheStoreFormat.HPLD
                        else (0,)
                    ):
                        ranges.append(
                            KVCacheResolvedRange(
                                layer,
                                component,
                                page * page_size,
                                page_size,
                                part.head_start + head,
                                (
                                    1
                                    if self.format is KVCacheStoreFormat.HPLD
                                    else part.head_count
                                ),
                                region_id,
                                base + local_layer * ls + head * hs,
                                ts,
                                hs,
                            )
                        )
        return KVCacheResolvedRuntimeBinding(
            operation_id,
            placement.resource_id,
            placement.placement_id,
            placement.digest,
            self.instance,
            placement.revision,
            self.participant,
            None,
            None,
            tuple(regions),
            tuple(ranges),
        )

    def _current_pool_signature(self):
        return (
            self.pool.layout,
            self.pool.page_size,
            self.pool.size,
            self.pool.kv_buffer.data_ptr(),
            tuple(
                (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
                for t in (self.pool.k_buffer, self.pool.v_buffer)
            ),
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

    def _page_keys(self, keys):
        # Same HiCache hash under different weights/RoPE settings is different KV.
        return tuple(f"{key}_m{self._model_domain}_mp1" for key in keys)

    def upload_meta(self, keys, indices):
        """One page key and one K-then-V scatter list for this rank's Part."""
        self._page_slots(keys, indices)
        pointers, sizes = self.pool.get_page_buffer_meta(indices)
        if self.pool.layout == "layer_first":
            layers = self.pool.layer_num
            pointers = [
                pointers[start + component : start + 2 * layers : 2]
                for start in range(0, len(pointers), 2 * layers)
                for component in (0, 1)
            ]
            sizes = [
                sizes[start + component : start + 2 * layers : 2]
                for start in range(0, len(sizes), 2 * layers)
                for component in (0, 1)
            ]
        if len(pointers) != 2 * len(keys) or len(sizes) != len(pointers):
            raise ValueError("HiCache page buffers differ from Part coverage")

        def row(value):
            return list(value) if isinstance(value, (tuple, list)) else [value]

        return (
            self._page_keys(keys),
            [row(pointers[2 * i]) + row(pointers[2 * i + 1]) for i in range(len(keys))],
            [row(sizes[2 * i]) + row(sizes[2 * i + 1]) for i in range(len(keys))],
        )

    def upload_parts(self, keys, indices):
        pages, pointers, sizes = self.upload_meta(keys, indices)
        results = self.native_store.batch_put_parts_from(
            pages,
            self._part_index,
            len(self.upload_plan.part_writers),
            self.store.manifest.manifest_key,
            pointers,
            sizes,
        )
        return [status == 0 for status in results]

    def discover(self, keys, operation_id):
        started = time.perf_counter()
        context = self.store.discover(self._page_keys(keys), operation_id=operation_id)
        if os.getenv("MOONCAKE_RESHARD_TIMING") == "1":
            logger.info(
                "RESHARD_TIMING discover_ms=%.6f operation=%s participant=%s pages=%d multipart=True",
                (time.perf_counter() - started) * 1000,
                operation_id,
                self.participant,
                len(context.page_keys),
            )
        logger.info(
            "Mooncake KV reshard discover pages=%d/%d sources=%s",
            len(context.page_keys),
            len(keys),
            sorted(set(context.layout_ids)),
        )
        return context

    def pack_prefetch_context(self, context):
        # Every rank already hashes the same request. Keep a digest to detect
        # divergent request/order, rather than broadcasting all tagged keys.
        return pickle.dumps(
            (
                2,
                context.operation_id,
                context.model_domain,
                context.layout_ids,
                context.manifests,
                _digest(context.page_keys),
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def unpack_prefetch_context(self, payload, keys):
        version, operation_id, domain, layout_ids, manifests, digest = pickle.loads(
            payload
        )
        if version != 2 or domain != self._model_domain:
            raise ValueError("prefetch context version or model domain differs")
        if len(layout_ids) > len(keys):
            raise ValueError("prefetch context exceeds local request")
        pages = self._page_keys(keys[: len(layout_ids)])
        if _digest(pages) != digest:
            raise ValueError("prefetch context does not match local page keys")
        return MultipartReadContext(operation_id, domain, pages, layout_ids, manifests)

    def load(self, keys, indices, context, page_offset, cancelled):
        if context is None:
            raise ValueError("KV reshard load requires this prefetch's read context")
        keys = self._page_keys(keys)
        selected_keys = context.page_keys[page_offset : page_offset + len(keys)]
        if (
            tuple(keys) != selected_keys
            or indices.numel() != len(keys) * self.pool.page_size
        ):
            raise ValueError("prefetch context does not match this batch")
        selected_ids = context.layout_ids[page_offset : page_offset + len(keys)]
        batch_context = context
        if page_offset != 0 or len(keys) != len(context.page_keys):
            batch_context = replace(
                context,
                page_keys=selected_keys,
                layout_ids=selected_ids,
                manifests=tuple(
                    m for m in context.manifests if m.layout.layout_id in selected_ids
                ),
            )
        slots = self._page_slots(keys, indices)
        offsets = [
            {
                key: (slot // self._translation_unit) * step
                for key, step in self._translation_steps
            }
            for slot in slots
        ]
        started = time.perf_counter()
        count = self.reader.load(batch_context, offsets, cancelled=cancelled)
        if os.getenv("MOONCAKE_RESHARD_TIMING") == "1":
            logger.info(
                "RESHARD_TIMING read_ms=%.6f operation=%s participant=%s pages=%d multipart=True",
                (time.perf_counter() - started) * 1000,
                context.operation_id,
                self.participant,
                count,
            )
        return [i < count for i in range(len(keys))]
