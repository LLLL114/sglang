"""Host-pool byte geometry and operation-context tests for the Store adapter."""

import ctypes
from types import MethodType, SimpleNamespace

import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

pytest.importorskip("mooncake.reshard.kv_cache")

from sglang.srt.mem_cache.storage.mooncake_store.reshard import (
    MooncakeKVReshardAdapter,
    make_placement,
)


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.parts = {}
        self.queries = self.manifest_reads = self.metadata_puts = self.range_reads = 0
        self.short_key = None

    def put(self, key, value, config):
        self.metadata_puts += 1
        self.values.setdefault(key, value)
        return 0

    def get(self, key):
        self.manifest_reads += 1
        return self.values.get(key, b"")

    def batch_put_parts_from(self, keys, index, count, manifest, pointers, sizes):
        result = []
        for key, ptrs, lengths in zip(keys, pointers, sizes):
            existing = self.parts.get(key)
            if existing is not None and all(v is not None for v in existing[1]):
                result.append(0)
                continue
            if existing is None:
                existing = self.parts[key] = (manifest, [None] * count)
            if existing[0] != manifest or len(existing[1]) != count:
                result.append(-1)
                continue
            if existing[1][index] is None:
                existing[1][index] = b"".join(
                    ctypes.string_at(p, n) for p, n in zip(ptrs, lengths)
                )
            result.append(0)
        return result

    def batch_query_parts(self, keys):
        self.queries += 1
        result = []
        for key in keys:
            entry = self.parts.get(key)
            result.append(
                (0, entry[0], len(entry[1]))
                if entry and all(v is not None for v in entry[1])
                else (-1, "", 0)
            )
        return result

    def prepare_get_into_ranges_template(self, dst, src, sizes):
        return (dst, src, sizes)

    def prepare_get_parts_snapshot(self, keys, manifests, counts):
        result = {}
        for key, ref, count in zip(keys, manifests, counts):
            entry = self.parts.get(key)
            if (
                entry
                and entry[0] == ref
                and len(entry[1]) == count
                and all(v is not None for v in entry[1])
            ):
                result.update(
                    {key + f"\x1fp{i}": value for i, value in enumerate(entry[1])}
                )
        return result

    def get_into_ranges_from_template(
        self, snapshot, templates, buffers, indices, keys, deltas, **kwargs
    ):
        self.range_reads += 1
        result = []
        for (dst, src, sizes), i, key, delta in zip(templates, indices, keys, deltas):
            data = snapshot.get(key)
            ok = data is not None and key != self.short_key
            if ok:
                for d, s, n in zip(dst, src, sizes):
                    assert s + n <= len(data)
                    ctypes.memmove(buffers[i] + delta + d, data[s : s + n], n)
            result.append(ok)
        return result


def metadata(tp, pp, fmt="PLHD", heads=4):
    return dict(
        model_name="test-model",
        model_revision="weights-1",
        semantic="rope-1",
        total_kv_heads=heads,
        num_layers=12,
        tp_size=tp,
        pp_size=pp,
        parts=[
            dict(
                tp=t,
                pp=p,
                layers=list(range(p * (12 // pp), (p + 1) * (12 // pp))),
                heads=max(1, heads // tp),
                head_dim=4,
                dtype="float16",
                page_size=2,
                format=fmt,
            )
            for p in range(pp)
            for t in range(tp)
        ],
    )


def adapter(native, layout, tp, pp, rank=0, heads=4, revision="weights-1"):
    fmt = {
        "page_first": "PLHD",
        "layer_first": "LPHD",
        "page_first_direct": "LPHD",
        "page_head": "HPLD",
    }[layout]
    m = metadata(tp, pp, fmt, heads)
    m["model_revision"] = revision
    part = m["parts"][rank]
    layers, nheads = len(part["layers"]), part["heads"]
    dims = {
        "page_first": (8, layers, nheads, 4),
        "layer_first": (layers, 8, nheads, 4),
        "page_first_direct": (4, layers, 2, nheads, 4),
        "page_head": (4, nheads, 2, layers, 4),
    }[layout]
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    tensor = torch.zeros((2, *dims), dtype=torch.float16)
    pool = SimpleNamespace(
        layout=layout,
        page_size=2,
        size=8,
        kv_buffer=tensor,
        k_buffer=tensor[0],
        v_buffer=tensor[1],
        layer_num=layers,
        head_num=nheads,
        head_dim=4,
        dtype=torch.float16,
    )
    pool.get_page_buffer_meta = MethodType(
        MHATokenToKVPoolHost.get_page_buffer_meta, pool
    )
    config = SimpleNamespace(
        kv_reshard_metadata=m, tp_rank=part["tp"], pp_rank=part["pp"], extra_config={}
    )
    return MooncakeKVReshardAdapter(native, pool, config)


def upload(a, keys, indices, *, grouped=False):
    # Exercise the real HiCache write path with a memory-backed native Store.
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

    backend = MooncakeStore.__new__(MooncakeStore)
    backend.kv_reshard = a
    backend.mem_pool_host = a.pool
    backend.config_prefix = None
    backend.store = a.store.backend.store
    backend.is_mla_backend = backend.should_split_heads = False
    backend._use_group_semantics = grouped
    backend.enable_storage_metrics = False
    return backend.batch_set_v1(keys, indices)


def logical_tensor(pool, tensor, slot, layer):
    if pool.layout == "page_first":
        return tensor[slot, layer]
    if pool.layout == "layer_first":
        return tensor[layer, slot]
    if pool.layout == "page_first_direct":
        return tensor[slot // 2, layer, slot % 2]
    return tensor[slot // 2, :, slot % 2, layer]


def fill_or_check(a, slots, *, check=False):
    part = a.placement.part(a.participant)
    for component, tensor in enumerate((a.pool.k_buffer, a.pool.v_buffer)):
        for token, slot in enumerate(slots):
            for local_layer, global_layer in enumerate(part.layer_ids):
                expected = torch.arange(part.head_count * 4).view(part.head_count, 4)
                expected = (
                    expected
                    + part.head_start * 4
                    + global_layer * 32
                    + token * 512
                    + component * 2048
                ).to(torch.float16)
                value = logical_tensor(a.pool, tensor, slot, local_layer)
                if check:
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    value.copy_(expected)


@pytest.mark.parametrize(
    "layout", ["page_first", "layer_first", "page_first_direct", "page_head"]
)
@pytest.mark.parametrize("source,target", [((1, 3), (2, 2)), ((2, 2), (1, 3))])
def test_tp_pp_reshard_with_noncontiguous_host_pages(layout, source, target):
    native = MemoryStore()
    keys = ["hash0", "hash1"]
    for rank in range(source[0] * source[1]):
        a = adapter(native, layout, *source, rank)
        fill_or_check(a, [4, 5, 0, 1])
        assert upload(a, keys, torch.tensor([4, 5, 0, 1])) == [True, True]
    readers = [
        adapter(native, layout, *target, rank) for rank in range(target[0] * target[1])
    ]
    before = native.queries
    context = readers[0].discover(keys, "prefetch")
    assert native.queries == before + 1
    assert len(context.page_keys) == 2
    for reader in readers:
        # Each actual controller batch rebinds token offsets relative to its own start.
        assert reader.load(keys[:1], torch.tensor([2, 3]), context, 0, None) == [True]
        assert reader.load(keys[1:], torch.tensor([6, 7]), context, 1, None) == [True]
        fill_or_check(reader, [2, 3, 6, 7], check=True)


def test_context_key_validation_cancel_and_eviction():
    native = MemoryStore()
    a = adapter(native, "page_first", 1, 1)
    fill_or_check(a, [0, 1])
    assert upload(a, ["a"], torch.tensor([0, 1])) == [True]
    c = a.discover(["a"], "first")
    assert a.load(["a"], torch.tensor([2, 3]), c, 0, lambda: True) == [False]
    with pytest.raises(ValueError, match="context"):
        a.load(["b"], torch.tensor([2, 3]), c, 0, None)
    with pytest.raises(ValueError, match="aligned"):
        a.binding(torch.tensor([1, 2]), "bad")
    native.parts.pop(a._page_keys(["a"])[0])
    assert a.discover(["a"], "second").page_keys == ()
    assert a.load(["a"], torch.tensor([2, 3]), c, 0, None) == [False]


def test_replicated_gqa_heads_and_incomplete_pp_ownership():
    placement = make_placement(metadata(4, 3, heads=2))
    assert placement.parts[1].replica_ordinal == 1
    broken = metadata(1, 3)
    broken["parts"].pop()
    with pytest.raises(ValueError):
        make_placement(broken)


def test_same_hash_under_different_model_revisions_is_isolated():
    native = MemoryStore()
    a = adapter(native, "page_first", 1, 1)
    upload(a, ["same-hash"], torch.tensor([0, 1]))
    b = adapter(native, "page_first", 1, 1, revision="weights-2")
    assert b.discover(["same-hash"], "new-weights").page_keys == ()


def test_unified_controller_preserves_prefetch_context_and_hit_accounting():
    from unittest.mock import Mock

    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )

    controller = HybridCacheController.__new__(HybridCacheController)
    context = SimpleNamespace(page_keys=("hash",))
    controller.storage_backend = SimpleNamespace(
        supports_prefetch_context=True, prepare_prefetch=Mock(return_value=context)
    )
    controller.tp_rank = controller.pp_rank = 0
    controller.page_size = 2
    controller.prefetch_hits_sync_groups = []
    controller.get_hash_str = lambda *a, **kw: ["hash"]
    op = SimpleNamespace(
        token_ids=[1, 2],
        last_hash=None,
        prefix_keys=None,
        is_terminated=lambda: False,
        pool_storage_result=Mock(),
    )
    assert controller._storage_hit_query(op) == (["hash"], 2)
    assert op.storage_read_context is context
    op.pool_storage_result.update_kv_hit_pages.assert_called_once_with(1)


@pytest.mark.parametrize("enabled", [False, True])
def test_controller_passes_context_and_offsets_to_each_io_batch(enabled):
    from queue import Queue

    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE

    controller = HiCacheController.__new__(HiCacheController)
    controller.storage_backend = SimpleNamespace(supports_prefetch_context=enabled)
    controller.page_size = 2
    controller.prefetch_sync_queue = Queue()
    calls = []

    def transfer(operation, keys, indices, extra, sidecars):
        calls.append(extra)
        return len(keys)

    controller._page_transfer_kv_batch = transfer
    context = object()
    n = STORAGE_BATCH_SIZE + 1
    op = SimpleNamespace(
        prefix_keys=None,
        hash_value=[str(i) for i in range(n)],
        host_indices=torch.arange(n * 2),
        pool_transfers=None,
        is_terminated=lambda: False,
        storage_read_context=context,
        request_id="request",
    )
    assert controller._page_transfer(op) == n
    if enabled:
        assert [c.page_offset for c in calls] == [0, STORAGE_BATCH_SIZE]
        assert all(
            c.prefetch_context is context and c.cancelled is op.is_terminated
            for c in calls
        )
    else:
        assert all(
            c.prefetch_context is None and c.cancelled is None and c.page_offset == 0
            for c in calls
        )


@pytest.mark.parametrize(
    "layout", ["page_first", "layer_first", "page_first_direct", "page_head"]
)
def test_upload_uses_original_buffers_without_lowering(layout, monkeypatch):
    from mooncake.reshard.kv_cache._store import multipart as execution
    from mooncake.reshard.kv_cache._store import page_reader

    native = MemoryStore()
    a = adapter(native, layout, 1, 1)
    fill_or_check(a, [4, 5, 0, 1])
    keys = ["a", "b"]
    # Retain the old generic uploader as an independent byte-order reference.
    assert a.store.upload(
        a.upload_plan,
        a._page_keys(keys),
        a.binding(torch.tensor([4, 5, 0, 1]), "reference"),
        operation_id="reference",
    ) == (True, True)
    payload_keys = a._page_keys(keys)
    expected = {key: native.parts.pop(key) for key in payload_keys}
    queries = native.queries

    def forbidden(*args, **kwargs):
        raise AssertionError("upload must not plan or lower")

    monkeypatch.setattr(execution, "lower_store_ranges", forbidden)
    monkeypatch.setattr(page_reader, "lower_store_ranges", forbidden)
    assert upload(a, keys, torch.tensor([4, 5, 0, 1])) == [True, True]
    assert {key: native.parts[key] for key in payload_keys} == expected
    assert native.queries == queries
    context = a.discover(keys, "read")
    # First read compiles; this test deliberately stops before that stage.
    assert context.page_keys == a._page_keys(keys)


@pytest.mark.parametrize(
    "layout", ["page_first", "layer_first", "page_first_direct", "page_head"]
)
def test_same_layout_part_reads_cache_templates_and_rebind_pages(layout, monkeypatch):
    from mooncake.reshard.kv_cache._store import page_reader

    native = MemoryStore()
    a = adapter(native, layout, 1, 1)
    fill_or_check(a, [0, 1, 2, 3])
    assert upload(a, ["a", "b"], torch.tensor([0, 1, 2, 3])) == [True, True]
    context = a.discover(["a", "b"], "first")
    assert a.load(["a", "b"], torch.tensor([4, 5, 6, 7]), context, 0, None) == [
        True,
        True,
    ]
    fill_or_check(a, [4, 5, 6, 7], check=True)
    assert native.range_reads > 0
    reads, queries = native.manifest_reads, native.queries

    def forbidden(*args, **kwargs):
        raise AssertionError("cached layout must not plan or lower again")

    monkeypatch.setattr(page_reader, "plan_kv_cache_store_load", forbidden)
    monkeypatch.setattr(page_reader, "lower_store_ranges", forbidden)
    # Different page count, key ordering, operation ID and physical slots.
    second = a.discover(["b"], "second")
    assert a.load(["b"], torch.tensor([0, 1]), second, 0, None) == [True]
    for tensor in (a.pool.k_buffer, a.pool.v_buffer):
        for layer in range(a.pool.layer_num):
            for i in range(2):
                torch.testing.assert_close(
                    logical_tensor(a.pool, tensor, i, layer),
                    logical_tensor(a.pool, tensor, 6 + i, layer),
                )
    assert native.manifest_reads == reads and native.queries == queries + 1


def test_mixed_source_layouts_are_cached_separately_and_eviction_is_fresh():
    native = MemoryStore()
    for topology, key in [((1, 3), "a"), ((2, 2), "b")]:
        for rank in range(topology[0] * topology[1]):
            a = adapter(native, "page_first", *topology, rank)
            fill_or_check(a, [0, 1])
            assert upload(a, [key], torch.tensor([0, 1])) == [True]
    target = adapter(native, "page_first", 1, 1)
    context = target.discover(["a", "b"], "mixed")
    assert len(set(context.layout_ids)) == 2
    assert target.load(["a", "b"], torch.tensor([0, 1, 4, 5]), context, 0, None) == [
        True,
        True,
    ]
    assert len(target.reader._templates) == 2
    fill_or_check(target, [0, 1], check=True)
    fill_or_check(target, [4, 5], check=True)
    native.parts[context.page_keys[0]][1][0] = None
    assert target.discover(["a", "b"], "evicted").page_keys == ()
    assert target.load(["a", "b"], torch.tensor([0, 1, 4, 5]), context, 0, None) == [
        False,
        False,
    ]


def test_page_reader_rejects_overlapping_out_of_bounds_and_changed_pool():
    native = MemoryStore()
    a = adapter(native, "page_first", 1, 1)
    upload(a, ["a", "b"], torch.tensor([0, 1, 2, 3]))
    context = a.discover(["a", "b"], "bounds")
    with pytest.raises(ValueError, match="overlap"):
        a.load(["a", "b"], torch.tensor([0, 1, 0, 1]), context, 0, None)
    with pytest.raises(ValueError, match="outside"):
        a.load(["a", "b"], torch.tensor([0, 1, 8, 9]), context, 0, None)
    offsets = [
        {key: 0 for key in a.reader.regions},
        {key: region.nbytes for key, region in a.reader.regions.items()},
    ]
    with pytest.raises(ValueError, match="bounds"):
        a.reader.load(context, offsets)
    a.pool.k_buffer = a.pool.k_buffer.clone()
    with pytest.raises(ValueError, match="pool changed"):
        a.load(["a", "b"], torch.tensor([0, 1, 2, 3]), context, 0, None)


def test_replicated_writer_ranks_are_explicitly_unsupported():
    with pytest.raises(ValueError, match="distinct KV shard"):
        adapter(MemoryStore(), "page_head", 4, 3, heads=2)


def test_compact_context_roundtrip_checks_local_keys_and_domain():
    import pickle

    native = MemoryStore()
    keys = ["a", "b"]
    for rank in range(3):
        writer = adapter(native, "page_head", 1, 3, rank)
        fill_or_check(writer, [0, 1, 2, 3])
        assert upload(writer, keys, torch.arange(4)) == [True, True]
    reader = adapter(native, "page_first", 2, 2)
    context = reader.discover(keys, "compact")
    packed = reader.pack_prefetch_context(context)
    assert reader.unpack_prefetch_context(packed, keys) == context
    assert len(packed) < len(pickle.dumps(context, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(ValueError, match="local page keys"):
        reader.unpack_prefetch_context(packed, ["a", "different"])
    with pytest.raises(ValueError, match="exceeds local"):
        reader.unpack_prefetch_context(packed, ["a"])
    wrong_model = adapter(native, "page_first", 2, 2, revision="weights-2")
    with pytest.raises(ValueError, match="model domain"):
        wrong_model.unpack_prefetch_context(packed, keys)
    fields = list(pickle.loads(packed))
    fields[0] = 999
    with pytest.raises(ValueError, match="version"):
        reader.unpack_prefetch_context(pickle.dumps(tuple(fields)), keys)


def test_legacy_backend_selection_is_rejected():
    config = SimpleNamespace(extra_config={"kv_reshard": {"multipart": False}})
    with pytest.raises(ValueError, match="legacy KV reshard was removed"):
        MooncakeKVReshardAdapter(None, None, config)


def test_prefetched_host_prefix_does_not_trigger_write_through_again():
    from unittest.mock import Mock

    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

    cache = HiRadixCache.__new__(HiRadixCache)
    cache.page_size = 2
    cache.cache_controller = SimpleNamespace(write_policy="write_through")
    cache.write_through_threshold = 1
    cache.write_backup = Mock()
    cache._update_host_leaf_status = Mock()
    cache._update_leaf_status = Mock()
    cache.kv_events = SimpleNamespace(record_store=Mock())
    root = TreeNode()
    root.key = RadixKey([])
    key = RadixKey([1, 2])
    assert cache._insert_helper_host(root, key, torch.tensor([10, 11]), ["hash"]) == 0
    node = root.children[key.child_key(2)]
    assert node.backuped
    node.value = torch.tensor([20, 21])  # GPU slots after load-back.
    cache._inc_hit_count(node)
    cache.write_backup.assert_not_called()

    fresh = TreeNode()
    fresh.value = torch.tensor([30, 31])
    assert not fresh.backuped
    cache._inc_hit_count(fresh)
    cache.write_backup.assert_called_once_with(fresh)


def test_only_global_rank_zero_publishes_the_shared_manifest():
    native = MemoryStore()
    ranks = [adapter(native, "page_first", 2, 3, rank) for rank in range(6)]
    assert native.metadata_puts == 1
    assert len({rank.store.manifest.manifest_key for rank in ranks}) == 1
    assert [rank._part_index for rank in ranks] == list(range(6))
