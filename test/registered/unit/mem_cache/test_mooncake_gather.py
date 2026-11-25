"""CPU tests for the C++ Store rank-gather HiCache integration."""

import ctypes
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.reshard_adapter import (
    ReshardManager,
    collect_reshard_metadata,
    get_reshard_reader_type,
    make_placement,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def pool(layers=3, heads=2, start=0, layout="page_first"):
    obj = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    obj.layout = layout
    obj.page_size = 4
    obj.size = 16
    obj.layer_num = obj.target_layer_num = layers
    obj.head_num = heads
    obj.head_dim = obj.v_head_dim = 8
    obj.start_layer = start
    obj.end_layer = start + layers
    obj.dtype = torch.float16
    pages = obj.size // obj.page_size
    dims = {
        "page_first": (obj.size, layers, heads, 8),
        "layer_first": (layers, obj.size, heads, 8),
        "page_first_direct": (pages, layers, obj.page_size, heads, 8),
        "page_head": (pages, heads, obj.page_size, layers, 8),
    }[layout]
    obj.kv_buffer = torch.zeros((2, *dims), dtype=torch.float16)
    return obj


def config():
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=2,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="cpu-test",
        extra_config={"kv_reshard": {"backend": "cpp", "model_revision": "revision"}},
    )


def collect(ctrl, cfg):
    model = SimpleNamespace(
        is_multimodal=False,
        num_hidden_layers=5,
        get_total_num_kv_heads=lambda: 4,
        hf_config=SimpleNamespace(to_dict=lambda: {"layers": 5}),
    )

    def gather(output, parts, group):
        if group == "tp":
            for t in range(2):
                output[t] = [dict(parts[0], tp=t, writer_instance=f"tp-{t}")]
        else:
            for pp, layers in enumerate(([0, 1, 2], [3, 4])):
                output[pp] = [
                    dict(
                        row,
                        pp=pp,
                        layers=layers,
                        writer_instance=f"pp-{pp}-tp-{row['tp']}",
                    )
                    for row in parts
                ]

    with (
        patch(
            "sglang.srt.runtime_context.get_server_args",
            return_value=SimpleNamespace(dp_size=1, enable_lora=False),
        ),
        patch("sglang.srt.runtime_context.process_model_config", return_value=model),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather_object", side_effect=gather),
    ):
        return collect_reshard_metadata(ctrl, cfg)


@unittest.skipUnless(
    get_reshard_reader_type() is not None,
    "requires the optional Mooncake reshard SDK with GatherPageReader",
)
class TestGatherStore(unittest.TestCase):
    def test_collects_actual_unequal_layers_and_shared_writer_cohort(self):
        host = pool()
        ctrl = SimpleNamespace(
            storage_host_pool=host, mem_pool_host=host, tp_group="tp", pp_group="pp"
        )
        cfg = config()
        meta = collect(ctrl, cfg)
        self.assertEqual(
            [p["layers"] for p in meta["parts"]], [[0, 1, 2], [0, 1, 2], [3, 4], [3, 4]]
        )
        self.assertEqual(len(meta["writer_id"]), 64)
        placement = make_placement(meta, "key")
        self.assertEqual(
            [
                (p.rank.pp, p.rank.tp, p.head_start, p.head_count)
                for p in placement.parts
            ],
            [(0, 0, 0, 2), (0, 1, 2, 2), (1, 0, 0, 2), (1, 1, 2, 2)],
        )

    def test_rejects_unsupported_hybrid_and_missing_revision(self):
        host = pool()
        ctrl = SimpleNamespace(
            storage_host_pool=host,
            mem_pool_host=SimpleNamespace(entries=[1, 2]),
            tp_group="tp",
            pp_group="pp",
        )
        with self.assertRaises(ValueError):
            collect(ctrl, config())
        ctrl.mem_pool_host = host
        cfg = config()
        cfg.extra_config["kv_reshard"].pop("model_revision")
        with self.assertRaises(ValueError):
            collect(ctrl, cfg)

    def adapter(self, layout="page_first", settings=None, host=None):
        if host is None:
            host = pool(layout=layout)
        cfg = config()
        cfg.extra_config["kv_reshard"].update(settings or {})
        cfg.kv_reshard_metadata = collect(
            SimpleNamespace(
                storage_host_pool=host, mem_pool_host=host, tp_group="tp", pp_group="pp"
            ),
            cfg,
        )
        self.writer_id = cfg.kv_reshard_metadata["writer_id"]
        raw = Mock()
        raw.register_buffer.return_value = raw.unregister_buffer.return_value = 0
        adapter = ReshardManager(raw, host, cfg)
        self.addCleanup(adapter.close)
        return adapter, host, raw

    def test_reader_supplies_registration_size_with_legacy_fallback(self):
        from mooncake.reshard.kv_cache import GATHER_HEADER_BYTES

        from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE

        reader = get_reshard_reader_type()
        with patch.object(
            reader, "scratch_bytes", return_value=8192, create=True
        ) as size:
            manager, _, store = self.adapter()
            size.assert_called_once_with(STORAGE_BATCH_SIZE)
            self.assertEqual(len(manager._scratch), 8192)
            store.register_buffer.assert_any_call(manager._scratch_ptr, 8192)
        with patch.object(reader, "scratch_bytes", None, create=True):
            manager, _, store = self.adapter()
            expected = STORAGE_BATCH_SIZE * (GATHER_HEADER_BYTES + 32)
            self.assertEqual(len(manager._scratch), expected)
            store.register_buffer.assert_any_call(manager._scratch_ptr, expected)

    def test_rank_gather_publishes_one_object_in_rank_order(self):
        self.check_rank_gather_publishes_one_object(False)

    def test_peer_accepts_all_hits_from_the_missing_bitmap(self):
        from sglang.srt.mem_cache.storage.mooncake_store.rank_gather import (
            RankGatherWriter,
        )

        manager, _, store = self.adapter()
        manifest = manager.components[0].writer.manifest
        writer = RankGatherWriter(store, manifest, 1, self.writer_id, 2)
        peer, coordinator = socket.socketpair()
        with peer, coordinator:
            writer._peers[0] = peer
            coordinator.sendall(bytes(2))
            coordinator.shutdown(socket.SHUT_WR)
            spans = [(0, writer.part_sizes[1])]
            self.assertEqual(writer.upload_slots(["a", "b"], [0, 1], spans, 1), [0, 0])

    def test_writer_rejects_negative_spans_before_starting_a_batch(self):
        manager, _, _ = self.adapter()
        writer = manager.components[0].writer
        size = writer.part_sizes[writer.rank]
        with patch.object(writer, "_upload") as start_batch:
            with self.assertRaisesRegex(ValueError, "payload size"):
                writer.upload(["page"], [[0, 0]], [[size + 1, -1]])
        start_batch.assert_not_called()

    def test_peer_sends_adjacent_source_spans_before_releasing_them(self):
        from sglang.srt.mem_cache.storage.mooncake_store import rank_gather

        manager, _, store = self.adapter()
        writer = rank_gather.RankGatherWriter(
            store, manager.components[0].writer.manifest, 1, self.writer_id, 3
        )

        def rows(selected):
            self.assertEqual(selected, [0, 2])
            source = ctypes.create_string_buffer(b"abXYcde")
            base = ctypes.addressof(source)
            try:
                yield [(base, 1), (base + 1, 1), (base + 4, 1)]
                yield [(base + 5, 2)]
            finally:
                ctypes.memset(base, 0, len(source))

        peer, coordinator = socket.socketpair()
        with (
            peer,
            coordinator,
            patch.object(rank_gather, "_view", wraps=rank_gather._view) as views,
        ):
            writer._peers[0] = peer
            coordinator.sendall(bytes([1, 0, 1, 1, 1, 1]))
            results = writer._send_pages(b"S", ["first", "hit", "last"], rows)
            self.assertEqual(results, [0, 0, 0])
            self.assertEqual(rank_gather._read_exact(coordinator, 6), b"Sabcde")
            self.assertEqual([call.args[1] for call in views.call_args_list], [2, 3])

    def test_failed_first_buffer_registration_releases_peers_without_publishing(self):
        from sglang.srt.mem_cache.storage.mooncake_store.rank_gather import (
            RankGatherWriter,
        )

        manager, _, store = self.adapter()
        manifest = manager.components[0].writer.manifest
        sizes = manifest.layout.part_sizes
        writers = [
            RankGatherWriter(store, manifest, rank, self.writer_id, 1)
            for rank in range(len(sizes))
        ]
        buffers = [ctypes.create_string_buffer(n) for n in sizes]
        store.batch_is_exist.return_value = [0]
        store.register_buffer.return_value = -1
        try:
            with ThreadPoolExecutor(len(writers)) as executor:
                tasks = [
                    executor.submit(
                        writer.upload, ["missing"], [[ctypes.addressof(buf)]], [[n]]
                    )
                    for writer, buf, n in zip(writers, buffers, sizes, strict=True)
                ]
                with self.assertRaisesRegex(RuntimeError, "failed to register"):
                    tasks[0].result(timeout=10)
                for peer in tasks[1:]:
                    with self.assertRaises(ConnectionError):
                        peer.result(timeout=10)
            store.batch_put_from.assert_not_called()
        finally:
            for writer in writers:
                writer.close()

    def test_idempotent_put_only_acknowledges_readable_pages_and_can_retry(self):
        from sglang.srt.mem_cache.storage.mooncake_store.rank_gather import (
            RankGatherWriter,
        )

        manager, _, store = self.adapter()
        manifest = manager.components[0].writer.manifest
        sizes = manifest.layout.part_sizes
        writers = [
            RankGatherWriter(store, manifest, rank, self.writer_id, 2)
            for rank in range(len(sizes))
        ]
        buffers = [ctypes.create_string_buffer(n) for n in sizes]
        # Both puts report success, but an independent writer still owns the
        # incomplete second key. Only that key should be sent on the retry.
        store.batch_is_exist.side_effect = [[0, 0], [1, 0], [1, 0], [1], [1, 1]]
        store.batch_put_from.side_effect = [[0, 0], [0]]
        keys = ["complete", "in-progress"]
        try:
            with ThreadPoolExecutor(len(writers)) as pool:
                for expected in ([0, -1], [0, 0], [0, 0]):
                    tasks = [
                        pool.submit(
                            w.upload, keys, [[ctypes.addressof(buf)]] * 2, [[n]] * 2
                        )
                        for w, buf, n in zip(writers, buffers, sizes, strict=True)
                    ]
                    self.assertEqual(
                        [t.result(timeout=10) for t in tasks], [expected] * len(writers)
                    )
            self.assertEqual(
                [call.args[0] for call in store.batch_put_from.call_args_list],
                [keys, [keys[1]]],
            )
            self.assertEqual(store.batch_is_exist.call_count, 5)
        finally:
            for writer in writers:
                writer.close()

    def test_rank_gather_publishes_extended_header_before_all_rank_payloads(self):
        self.check_rank_gather_publishes_one_object(True)

    def check_rank_gather_publishes_one_object(self, extended):
        from dataclasses import replace

        from mooncake.reshard.kv_cache import GATHER_HEADER_BYTES

        from sglang.srt.mem_cache.storage.mooncake_store.rank_gather import (
            RankGatherWriter,
        )

        adapter, _, _ = self.adapter(settings={"storage_mode": "rank_gather"})
        root = adapter.components[0].writer
        root = RankGatherWriter(root.native, root.manifest, 0, self.writer_id, 2)
        if extended:
            manifest = replace(
                root.manifest, model_revision="long-revision-" + "x" * 9000
            )
            root = RankGatherWriter(root.native, manifest, 0, self.writer_id, 2)
            self.assertGreater(len(root.header), 2 * GATHER_HEADER_BYTES)
        else:
            self.assertEqual(len(root.header), GATHER_HEADER_BYTES)
        written = {}

        def put(keys, pointers, sizes):
            written.update(
                (key, ctypes.string_at(ptr, size))
                for key, ptr, size in zip(keys, pointers, sizes)
            )
            return [0] * len(keys)

        root.native.batch_put_from.side_effect = put
        peers = [
            RankGatherWriter(root.native, root.manifest, rank, self.writer_id, 2)
            for rank in range(1, len(root.part_sizes))
        ]
        writers = [root, *peers]
        contents = [
            bytes([rank + 1]) * size for rank, size in enumerate(root.part_sizes)
        ]
        buffers = [ctypes.create_string_buffer(data, len(data)) for data in contents]
        try:
            before = root.native.register_buffer.call_count
            with (
                patch.object(ctypes, "memmove", wraps=ctypes.memmove) as copy_bytes,
                ThreadPoolExecutor(len(writers)) as executor,
            ):
                for iteration, mask in enumerate(([1], [0], [1], [0, 0], [0])):
                    keys = ["page", "next"][: len(mask)]
                    root.native.batch_is_exist.side_effect = (
                        [mask, [1] * mask.count(0)] if 0 in mask else [mask]
                    )
                    pending = [
                        executor.submit(
                            w.upload,
                            keys,
                            [[ctypes.addressof(buf)]] * len(keys),
                            [[len(buf)]] * len(keys),
                        )
                        for w, buf in zip(writers, buffers)
                    ]
                    self.assertEqual(
                        [f.result() for f in pending], [[0] * len(keys)] * len(writers)
                    )
                    self.assertEqual(
                        root.native.register_buffer.call_count - before,
                        int(iteration > 0),
                    )
                self.assertEqual(
                    sum(
                        call.args[1] is root.header
                        for call in copy_bytes.call_args_list
                    ),
                    2,
                )
            self.assertEqual(root.native.batch_put_from.call_count, 3)
            self.assertEqual(list(written), ["page", "next"])
            self.assertEqual(written["next"], written["page"])
            self.assertEqual(written["page"][: len(root.header)], root.header)
            self.assertEqual(written["page"][len(root.header) :], b"".join(contents))
        finally:
            for writer in writers:
                writer.close()

    def test_page_keys_are_topology_free_and_pool_replacement_is_rejected(self):
        adapter, host, _ = self.adapter()
        tagged = "tenant_model_hash"
        k, v = adapter.components
        self.assertEqual(
            adapter.object_keys([tagged]),
            [
                f"{tagged}_m{k.target.model_domain}_rsrg4_k",
                f"{tagged}_m{v.target.model_domain}_rsrg4_v",
            ],
        )
        host.kv_buffer = host.kv_buffer.clone()
        with self.assertRaisesRegex(ValueError, "pool changed"):
            adapter.prepare_read(["hash"], torch.arange(4))

    def test_page_template_and_translations_address_nonzero_slots(self):
        from mooncake.reshard.kv_cache import KVCacheComponent

        for layout in ("page_first", "layer_first", "page_first_direct", "page_head"):
            with self.subTest(layout=layout):
                adapter, host, _ = self.adapter(layout)
                # Two non-adjacent pages, in reverse slot order.
                slots = (8, 0)
                indices = torch.cat(
                    [torch.arange(s, s + host.page_size) for s in slots]
                )
                keys = ("a", "b")
                local_slots = adapter._page_slots(keys, indices)
                pointers, lengths = host.get_page_buffer_meta(indices)
                for plane, c in enumerate(adapter.components):
                    translated = [
                        (address + slot * c._slot_stride, length)
                        for slot in local_slots
                        for address, length in c._write_spans
                    ]
                    self.assertEqual(
                        translated, list(zip(pointers[plane::2], lengths[plane::2]))
                    )
                    buffers = c.read_buffers(local_slots)
                    binding = c._page_binding()
                    regions = {r.region_id: r for r in binding.regions}
                    for slot, (ptr, capacity) in zip(slots, buffers):
                        delta = {r.region_id: ptr - r.address for r in binding.regions}
                        self.assertGreater(capacity, 0)
                        for span in binding.ranges:
                            tensor = (
                                host.k_buffer
                                if span.component is KVCacheComponent.KEY
                                else host.v_buffer
                            )
                            for t in range(host.page_size):
                                for h in range(span.head_start, span.head_end):
                                    layer = span.global_layer_id
                                    idx = {
                                        "page_first": (slot + t, layer, h),
                                        "layer_first": (layer, slot + t, h),
                                        "page_first_direct": (
                                            slot // host.page_size,
                                            layer,
                                            t,
                                            h,
                                        ),
                                        "page_head": (
                                            slot // host.page_size,
                                            h,
                                            t,
                                            layer,
                                        ),
                                    }[layout]
                                    address = span.address(
                                        regions[span.region_id], t, h
                                    )
                                    self.assertEqual(
                                        address + delta[span.region_id],
                                        tensor[idx].data_ptr(),
                                    )

    def test_shares_slot_calculation_and_reuses_bound_writer_spans(self):
        manager, host, _ = self.adapter()
        keys = ["a", "b"]
        indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
        with (
            patch.object(manager, "_page_slots", wraps=manager._page_slots) as slots,
            patch.object(
                host, "get_page_buffer_meta", wraps=host.get_page_buffer_meta
            ) as meta,
            patch.object(
                type(manager.components[0].writer), "upload_slots", return_value=[0, 0]
            ) as upload,
        ):
            manager.prepare_read(keys, indices)
            self.assertEqual(slots.call_count, 1)
            self.assertEqual(manager.batch_set_v1(keys, indices), [True, True])
            self.assertEqual(slots.call_count, 2)
            meta.assert_not_called()
            self.assertEqual(upload.call_count, 2)

    def test_custom_source_mapping_rejects_other_component_buffer(self):
        host = pool()
        indices = torch.arange(host.page_size)
        pointers, sizes = host.get_page_buffer_meta(indices)
        with patch.object(
            host, "get_page_buffer_meta", return_value=(pointers[::-1], sizes)
        ):
            adapter, _, _ = self.adapter(host=host)
            with patch.object(
                type(adapter.components[0].writer), "upload", return_value=[0]
            ) as upload:
                with self.assertRaisesRegex(ValueError, "source page spans"):
                    adapter.batch_set_v1(["page"], indices)
            upload.assert_not_called()

    def test_custom_source_mapping_is_queried_for_the_actual_batch(self):
        host = pool()
        original = host.get_page_buffer_meta
        indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
        remapped = indices.reshape(-1, host.page_size).flip(0).flatten()
        pointers, sizes = original(remapped)
        with patch.object(
            host, "get_page_buffer_meta", return_value=(pointers, sizes)
        ) as lookup:
            adapter, _, _ = self.adapter(host=host)
            with patch.object(
                type(adapter.components[0].writer), "upload", return_value=[0, 0]
            ) as upload:
                self.assertEqual(
                    adapter.batch_set_v1(["a", "b"], indices), [True, True]
                )
            lookup.assert_called_once_with(indices)
            for plane, call in enumerate(upload.call_args_list):
                self.assertEqual(call.args[1], [[p] for p in pointers[plane::2]])
                self.assertEqual(call.args[2], [[n] for n in sizes[plane::2]])

    def test_sdk_round_trip_through_adapter(self):
        # One rank uses the real upload, discovery, planning and load code.
        # Only ordinary Store object I/O is replaced with in-process buffers.
        cfg = config()
        cfg.tp_size = cfg.pp_size = 1
        host = pool(layers=5, heads=4)
        cfg.kv_reshard_metadata = collect(
            SimpleNamespace(
                storage_host_pool=host, mem_pool_host=host, tp_group=None, pp_group=None
            ),
            cfg,
        )
        objects = {}
        raw = Mock()
        raw.register_buffer.return_value = raw.unregister_buffer.return_value = 0
        raw.batch_is_exist.side_effect = lambda keys: [int(k in objects) for k in keys]

        def put(keys, ptrs, sizes):
            objects.update(
                (k, ctypes.string_at(p, n)) for k, p, n in zip(keys, ptrs, sizes)
            )
            return [0] * len(keys)

        def get(snapshot, pointers, keys, destinations, sources, sizes):
            result = []
            for ptr, names, dsts, srcs, lengths in zip(
                pointers, keys, destinations, sources, sizes
            ):
                group = []
                for key, doffs, soffs, lens in zip(names, dsts, srcs, lengths):
                    counts = []
                    for dst, src, n in zip(doffs, soffs, lens):
                        data = snapshot[key][src : src + n]
                        ctypes.memmove(ptr + dst, data, len(data))
                        counts.append(len(data))
                    group.append(counts)
                result.append(group)
            return result

        raw.batch_put_from.side_effect = put
        raw.prepare_get_into_ranges_snapshot.side_effect = lambda keys: {
            k: objects.get(k, b"") for k in keys
        }
        raw.get_into_ranges_from_snapshot.side_effect = get
        adapter = ReshardManager(raw, host, cfg)
        self.addCleanup(adapter.close)
        for n, tensor in enumerate((host.k_buffer, host.v_buffer), start=1):
            tensor[:4].copy_(
                n * 100 + torch.arange(tensor[:4].numel()).reshape_as(tensor[:4])
            )
        keys = ["hit", "miss"]
        self.assertEqual(adapter.batch_set_v1(keys[:1], torch.arange(4)), [True])
        from contextlib import nullcontext

        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        backend = MooncakeStore.__new__(MooncakeStore)
        backend.kv_reshard = adapter
        backend.is_mla_backend = cfg.is_mla_model
        backend.should_split_heads = cfg.should_split_heads
        backend.config_prefix = None
        backend._batch_exist = raw.batch_is_exist
        backend.request_context = lambda **kwargs: nullcontext()
        backend._request_context_from_extra_info = lambda info: {}
        backend.enable_storage_metrics = False
        backend.mem_pool_host = host
        # The very same reader handles two homogeneous physical batches.
        self.assertTrue(backend.batch_set_v1(["second"], torch.arange(4, 8))[0])
        with patch.object(
            adapter.reader, "batch_get_into", wraps=adapter.reader.batch_get_into
        ) as read:
            self.assertEqual(
                backend.batch_get_v1(["hit", "second"], torch.arange(8, 16)),
                [True, True],
            )
        k0, v0, k1, v1 = adapter.object_keys(["hit", "second"])
        self.assertEqual(
            [tuple(c.args[0]) for c in read.call_args_list], [(k0, k1), (v0, v1)]
        )
        self.assertEqual(
            [c.kwargs["target_manifest"] for c in read.call_args_list],
            [c.placement for c in adapter.components],
        )
        self.assertEqual(backend.batch_get_v1(keys, torch.arange(8, 16)), [True, False])
        for t in (host.k_buffer, host.v_buffer):
            torch.testing.assert_close(t[8:12], t[:4], rtol=0, atol=0)
        k, v = adapter.object_keys(keys[:1])
        self.assertEqual(backend.batch_exists(keys), 1)
        # Even if the first K is missing, the V reader receives its full batch.
        first_k = objects.pop(k0)
        with patch.object(
            adapter.reader, "batch_get_into", wraps=adapter.reader.batch_get_into
        ) as read:
            self.assertEqual(
                backend.batch_get_v1(["hit", "second"], torch.arange(8, 16)),
                [False, False],
            )
        self.assertEqual(
            [tuple(call.args[0]) for call in read.call_args_list], [(k0, k1), (v0, v1)]
        )
        objects[k0] = first_k
        second_v = objects.pop(v1)
        self.assertEqual(
            backend.batch_get_v1(["hit", "second"], torch.arange(8, 16)), [True, False]
        )
        objects[v1] = second_v
        saved = objects[k]
        del objects[v]
        self.assertEqual(backend.batch_exists(keys), 0)
        self.assertEqual(backend.batch_get_v1(keys[:1], torch.arange(8, 12)), [False])
        self.assertEqual(backend.batch_set_v1(keys[:1], torch.arange(4)), [True])
        self.assertEqual(backend.batch_exists(keys), 1)
        self.assertEqual(backend.batch_get_v1(keys, torch.arange(8, 16)), [True, False])
        self.assertEqual(objects[k], saved)
        self.assertEqual(set(objects), {k, v, k1, v1})
        self.assertEqual(backend.batch_get_v1([], torch.arange(0)), [])


class TestReshardIntegration(unittest.TestCase):
    def test_reader_discovery_without_supported_sdk(self):
        from sglang.srt.mem_cache.storage.mooncake_store import reshard_adapter

        with patch.object(
            reshard_adapter,
            "import_module",
            side_effect=ModuleNotFoundError(name="mooncake.reshard"),
        ):
            self.assertIsNone(get_reshard_reader_type())
        with patch.object(
            reshard_adapter,
            "import_module",
            return_value=SimpleNamespace(GatherPageReader=object),
        ):
            self.assertIsNone(get_reshard_reader_type())
        with patch.object(
            reshard_adapter,
            "import_module",
            side_effect=ModuleNotFoundError(name="broken_dependency"),
        ):
            with self.assertRaises(ModuleNotFoundError):
                get_reshard_reader_type()


if __name__ == "__main__":
    unittest.main()
