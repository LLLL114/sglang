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

    formats = {
        "page_first": KVCacheStoreFormat.PLHD,
        "page_first_direct": KVCacheStoreFormat.LPHD,
        "layer_first": KVCacheStoreFormat.LPHD,
        "page_head": KVCacheStoreFormat.HPLD,
    }
    if hasattr(KVCacheStoreFormat, "PAGE_UNIFIED"):
        formats["page_unified"] = KVCacheStoreFormat.PAGE_UNIFIED
    return formats


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _local_reshard_metadata(controller, config):
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
        pool.layout not in ("page_first_direct", "page_unified")
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
        "format": formats["page_first_direct" if mla else pool.layout].value,
        "writer_instance": uuid.uuid4().hex,
        "hostname": socket.gethostname(),
    }
    if pool.layout == "page_unified":
        local["heads_per_group"] = pool.page_unified_layout.heads_per_group
    semantic = model.hf_config.to_dict()
    for field in ("_name_or_path", "transformers_version", "_commit_hash"):
        semantic.pop(field, None)
    local["model_info"] = {
        "total_kv_heads": 1 if mla else model.get_total_num_kv_heads(),
        "component": "latent" if mla else None,
        "num_layers": model.num_hidden_layers,
        "model_name": config.model_name,
        "model_revision": options["model_revision"],
        "semantic": _digest(semantic),
    }
    return local


def collect_reshard_metadata(controller, config):
    """Exchange either metadata or a local error on both axes before returning."""
    try:
        local = _local_reshard_metadata(controller, config)
    except Exception as exc:
        local = {
            "pp": config.pp_rank,
            "tp": config.tp_rank,
            "error": f"{type(exc).__name__}: {exc}",
        }
    parts = [local]
    # TP first gathers one PP stage; PP then gathers every stage for each TP lane.
    for group in (controller.tp_group, controller.pp_group):
        if group is not None and torch.distributed.get_world_size(group) > 1:
            gathered = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(gathered, parts, group=group)
            parts = [part for group_parts in gathered for part in group_parts]
    parts = sorted(parts, key=lambda part: (part["pp"], part["tp"]))
    errors = [f"PP{p['pp']}/TP{p['tp']}: {p['error']}" for p in parts if "error" in p]
    if errors:
        raise ValueError("KV Store reshard attach failed: " + "; ".join(errors))
    model_info = parts[0]["model_info"]
    expected = {
        (pp, tp) for pp in range(config.pp_size) for tp in range(config.tp_size)
    }
    if len(parts) != len(expected) or {(p["pp"], p["tp"]) for p in parts} != expected:
        raise ValueError("incomplete TP/PP ownership for KV Store reshard")
    for part in parts:
        if part.pop("model_info") != model_info:
            raise ValueError("inconsistent KV Store model identity across ranks")
        if part.get("heads_per_group") != local.get("heads_per_group"):
            raise ValueError(
                "inconsistent unified KV head grouping across source ranks"
            )
        for field in ("head_dim", "dtype", "page_size", "format"):
            if part[field] != local[field]:
                raise ValueError(f"inconsistent KV Store descriptor: {field}")
    if len({p["hostname"] for p in parts}) != 1:
        raise ValueError("rank gather requires all TP/PP ranks on one host")
    return {
        **model_info,
        "parts": parts,
        "writer_id": _digest([(p["pp"], p["tp"], p["writer_instance"]) for p in parts]),
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
        components=(KVCacheComponent.KEY, KVCacheComponent.VALUE)
        if component == "kv"
        else (KVCacheComponent(component),),
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
            KVCacheStoreManifest,
            StoreReadTarget,
            plan_kv_cache_store_upload,
        )

        self.pool = pool
        metadata = config.kv_reshard_metadata
        self._component = component
        self.placement = make_placement(metadata, component)
        self.participant = ParticipantId(f"pp{config.pp_rank}-tp{config.tp_rank}")
        group = (
            {"head_group_size": pool.page_unified_layout.heads_per_group}
            if component == "kv"
            else {}
        )
        upload_plan = plan_kv_cache_store_upload(
            self.placement,
            object_format=_store_formats()[
                "page_first_direct" if component == "latent" else pool.layout
            ],
            **group,
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
        slot_axis = 1 if pool.layout == "layer_first" else 0
        tensor = self._tensor
        self._slot_stride = tensor.stride(slot_axis) * tensor.element_size()
        binding = self._page_binding()
        (self._read_region,) = binding.regions
        self._write_spans = None
        self.target = StoreReadTarget(self.placement, binding, manifest.model_domain)
        format_tag = {
            "key": "rsrg4_k",
            "value": "rsrg4_v",
            "latent": "rsrg2",
            "kv": "rsrg_unified_kv",
        }[component]
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
        if self._component == "latent" and self.pool.layout == "page_unified":
            # Unified MLA elides the singleton head axis of the LPHD latent view.
            return self.pool.kv_buffer.unsqueeze(-2)
        if self._component in ("latent", "kv"):
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
        tensor = self._tensor
        stride = tuple(s * tensor.element_size() for s in tensor.stride())
        region = KVCacheRegisteredRegion(
            self._component,
            "hicache-host",
            tensor.data_ptr(),
            (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())))
            * tensor.element_size(),
        )
        ranges = []
        if self._component == "kv":
            layout = pool.page_unified_layout
            group_heads = layout.heads_per_group
            if (
                tuple(tensor.shape) != layout.page_dims(tensor.shape[0])
                or not tensor.is_contiguous()
            ):
                raise ValueError("unexpected unified KV page storage")
            for group in range(layout.head_group_num):
                for layer_index, layer in enumerate(part.layer_ids):
                    for plane, component in enumerate(
                        self.placement.descriptor.components
                    ):
                        ranges.append(
                            KVCacheResolvedRange(
                                layer,
                                component,
                                0,
                                pool.page_size,
                                part.head_start + group * group_heads,
                                group_heads,
                                self._component,
                                group * stride[1]
                                + layer_index * stride[2]
                                + plane * stride[3],
                                stride[4],
                                stride[5],
                            )
                        )
        else:
            # Indices of the layer, token and head axes in a single K or V allocation.
            layer_axis, token_axis, head_axis = {
                "layer_first": (0, 1, 2),
                "page_first": (1, 0, 2),
                "page_first_direct": (1, 2, 3),
                "page_unified": (1, 2, 3),  # dense MLA latent view
                "page_head": (3, 2, 1),
            }[pool.layout]
            # In page_head, a single head's tokens/layers precede the next head.
            heads_per_range = 1 if pool.layout == "page_head" else part.head_count
            (component,) = self.placement.descriptor.components
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
                            self._component,
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

    def validate_binding(self):
        if self._current_pool_signature() != self._pool_signature:
            raise ValueError(
                "KV Store pool changed; reattach the backend before transfer"
            )

    def bind_write_page(self, page_meta, last_slot):
        """Bind page-zero spans once; the pool signature guards later translations."""
        if self.writer is None:
            return
        if self._component in ("latent", "kv"):
            page = self._tensor[0]
            blocks = (page,) if page.is_contiguous() else page.unbind(0)
            if not all(block.is_contiguous() for block in blocks):
                raise ValueError("noncontiguous resource layer payload")
            spans = tuple((block.data_ptr(), block.nbytes) for block in blocks)
        elif page_meta is not None:
            pointers, sizes = self._component_buffers(1, page_meta)
            spans = tuple(zip(pointers[0], sizes[0], strict=True))
        else:
            return
        self._validate_write_spans(spans, last_slot * self._slot_stride)
        self._write_spans = spans

    def _validate_write_spans(self, spans, last_delta=0):
        region = self._read_region
        end = region.address + region.nbytes
        for address, size in spans:
            if size < 0 or (
                size and (address < region.address or address + last_delta + size > end)
            ):
                raise ValueError("source page spans exceed the bound host region")

    def _component_buffers(self, count, page_meta):
        pointers, sizes = page_meta
        width = 2 * self.pool.layer_num if self.pool.layout == "layer_first" else 2
        if len(pointers) != width * count or len(sizes) != len(pointers):
            raise ValueError("HiCache page buffers differ from gather payload coverage")
        plane = int(self._component == "value")

        def rows(values):
            result = []
            for i in range(0, len(values), width):
                row = []
                for v in values[i + plane : i + width : 2]:
                    row.extend(v if isinstance(v, (list, tuple)) else (v,))
                result.append(row)
            return result

        return rows(pointers), rows(sizes)

    def upload(self, physical, slots, page_meta):
        if self.writer is None:
            raise ValueError("only canonical Gather writers publish")
        if self._write_spans is not None:
            return self.writer.upload_slots(
                physical, slots, self._write_spans, self._slot_stride
            )
        pointers, sizes = self._component_buffers(len(physical), page_meta)
        for row, lengths in zip(pointers, sizes, strict=True):
            self._validate_write_spans(zip(row, lengths, strict=True))
        return self.writer.upload(physical, pointers, sizes)

    def read_buffers(self, slots):
        region = self._read_region
        offsets = [slot * self._slot_stride for slot in slots]
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
            else ("kv",)
            if pool.layout == "page_unified"
            else ("key", "value")
        )
        from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE

        self.pool = pool
        paged = pool.layout in ("page_first_direct", "page_head", "page_unified")
        self._slot_unit = pool.page_size if paged else 1
        self._store = store
        self._scratch_ptr = None
        self.components = []
        try:
            for name in names:
                self.components.append(_ComponentIO(store, pool, config, name))
            from sglang.srt.mem_cache.pool_host.mha import (
                AsymmetricMHATokenToKVPoolHost,
                MHATokenToKVPoolHost,
            )

            page_meta = None
            needs_meta = any(
                c.writer is not None and c._component in ("key", "value")
                for c in self.components
            )
            getter = pool.get_page_buffer_meta if needs_meta else None
            # Custom getters may use a non-affine mapping. Keep their per-batch lookup.
            affine = getattr(getter, "__func__", None) in (
                MHATokenToKVPoolHost.get_page_buffer_meta,
                AsymmetricMHATokenToKVPoolHost.get_page_buffer_meta,
            )
            affine = affine and all(c._tensor.is_contiguous() for c in self.components)
            self._needs_batch_write_meta = needs_meta and not affine
            if affine:
                page_meta = getter(torch.arange(pool.page_size, device="cpu"))
            last_slot = (pool.size - pool.page_size) // self._slot_unit
            for component in self.components:
                component.bind_write_page(page_meta, last_slot)
            first, *others = [c.target for c in self.components]
            size_for = getattr(reader_type, "scratch_bytes", None)
            if size_for is None:
                from mooncake.reshard.kv_cache import GATHER_HEADER_BYTES

                scratch_bytes = STORAGE_BATCH_SIZE * (GATHER_HEADER_BYTES + 32)
            else:
                scratch_bytes = size_for(STORAGE_BATCH_SIZE)
            self._scratch = bytearray(scratch_bytes)
            self.reader = reader_type(
                store, *first, self._scratch, additional_targets=others
            )
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._scratch))
            if store.register_buffer(ptr, len(self._scratch)) != 0:
                raise RuntimeError("failed to register GatherPageReader scratch")
            self._scratch_ptr = ptr
            # K and V share rank ownership. Every replica keeps its own read binding.
            config.kv_reshard_writer = self.components[0].writer is not None
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

    def _page_slots(self, keys, indices):
        for component in self.components:
            component.validate_binding()
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
        return [slot // self._slot_unit for slot in slots]

    def prepare_read(self, keys, host_indices):
        """Prepare v1 vectors in component order: all K pages, then all V pages."""
        slots = self._page_slots(keys, host_indices)
        key_strs, ptrs, sizes = [], [], []
        for c in self.components:
            key_strs.extend(c.object_keys(keys))
            for ptr, size in c.read_buffers(slots):
                ptrs.append(ptr)
                sizes.append(size)
        return key_strs, ptrs, sizes

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        """Read the component batches from prepare_read and return page-major results."""
        pages, remainder = divmod(len(key_strs), len(self.components))
        if (
            remainder
            or len(buffer_ptrs) != len(key_strs)
            or len(buffer_sizes) != len(key_strs)
        ):
            raise ValueError("invalid Gather read batch dimensions")
        # prepare_read lays out one contiguous batch per component.
        groups = []
        for i, component in enumerate(self.components):
            batch = slice(i * pages, (i + 1) * pages)
            codes = self.reader.batch_get_into(
                key_strs[batch],
                buffer_ptrs[batch],
                buffer_sizes[batch],
                target_manifest=component.placement,
            )
            if len(codes) != pages:
                raise ValueError("unexpected Gather read result length")
            groups.append(codes)
        if len(groups) == 1:
            return groups[0]
        # Match the physical result order expected by _batch_postprocess.
        return [code for page in zip(*groups, strict=True) for code in page]

    def batch_set_v1(self, keys, host_indices):
        slots = self._page_slots(keys, host_indices)
        page_meta = None
        if self._needs_batch_write_meta:
            page_meta = self.pool.get_page_buffer_meta(host_indices)
        per_component = [
            c.upload(c.object_keys(keys), slots, page_meta) for c in self.components
        ]
        return [
            all(code == 0 for code in page) for page in zip(*per_component, strict=True)
        ]

    def close(self):
        if close_reader := getattr(getattr(self, "reader", None), "close", None):
            close_reader()
        for c in self.components:
            c.close()
        if self._scratch_ptr is not None:
            if self._store.unregister_buffer(self._scratch_ptr) != 0:
                raise RuntimeError("failed to unregister GatherPageReader scratch")
            self._scratch_ptr = None
