# SPDX-License-Identifier: Apache-2.0
"""Endpoint ordering retains batch order and bounded scheduling metadata."""

import pytest
from mooncake.reshard.transfer_engine.contracts import TransferBatch, TransferDirection

from sglang.srt.weight_cache.inplace.endpoint_order import BalancedEndpointExecutor


def _gloo_worker(rank, path, results):
    from datetime import timedelta

    import torch.distributed as dist

    class Engine:
        def batch_transfer_sync_read(self, endpoint, targets, sources, sizes):
            return 0

    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=20),
    )
    try:
        endpoints = tuple(f"worker-{i}" for i in range(8))
        executor = BalancedEndpointExecutor(Engine())
        executor.configure(
            rank=rank,
            target_attention_tp=1,
            local_endpoint=endpoints[rank],
            endpoints=endpoints,
            phase_barrier=dist.barrier,
        )
        batches = (
            [] if rank == 7 else [TransferBatch(endpoints[rank ^ 1], (1,), (2,), (2,))]
        )
        with executor.submission() as submission:
            submission.execute_batches(
                batches, TransferDirection.READ, max_inflight_batches=1
            )
        profile = executor.profiles[-1]
        results.put(
            (
                rank,
                profile["phase_barrier_count"],
                profile["completed"],
                tuple(p["completed"] for p in profile["phases"]),
            )
        )
    finally:
        dist.destroy_process_group()


def test_real_gloo_empty_phases_have_symmetric_barriers(tmp_path):
    import multiprocessing

    import torch.distributed as dist

    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    path = str(tmp_path / "gloo-init")
    processes = [
        context.Process(target=_gloo_worker, args=(rank, path, results))
        for rank in range(8)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        records = [results.get(timeout=5) for _ in processes]
        assert {r[0] for r in records} == set(range(8))
        assert all(
            count == 8 and completed and all(phases)
            for _, count, completed, phases in records
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


class TestEndpointOrder:
    def test_endpoint_group_keeps_original_batch_order(self):
        class Engine:
            def __init__(self):
                self.calls = []

            def batch_transfer_sync_read(self, endpoint, targets, sources, sizes):
                self.calls.append((endpoint, sources[0]))
                return 0

        engine = Engine()
        executor = BalancedEndpointExecutor(engine)
        executor.configure(
            rank=1,
            target_attention_tp=1,
            local_endpoint="b",
            endpoints=["a", "b", "c", "d", "e", "f", "g", "h"],
            phase_barrier=lambda: None,
        )
        batches = [
            TransferBatch(e, (i,), (i + 10,), (2,))
            for e, i in [("a", 1), ("b", 2), ("a", 3)]
        ]
        with executor.submission() as submission:
            submission.execute_batches(
                batches, TransferDirection.READ, max_inflight_batches=1
            )
        assert engine.calls == [("b", 2), ("a", 1), ("a", 3)]
        profile = executor.profiles[-1]
        assert profile["actual_batch_indices"] == [1, 0, 2]
        assert profile["planned_batch_indices"] == [1, 0, 2]
        assert profile["actual_batch_operations"] == [1, 1, 1]
        assert profile["actual_batch_nbytes"] == [2, 2, 2]

    def test_caps_reject_before_native(self):
        class Engine:
            def batch_transfer_sync_read(self, *args):
                raise AssertionError("native must not start")

        executor = BalancedEndpointExecutor(Engine())
        executor.configure(
            rank=0,
            target_attention_tp=1,
            local_endpoint="a",
            endpoints=["a", "b", "c", "d", "e", "f", "g", "h"],
            phase_barrier=lambda: None,
        )
        TransferBatch("a", (1,), (2,), (2,))
        with executor.submission() as submission:
            with pytest.raises(ValueError):
                submission.execute_batches(
                    (TransferBatch("a", (1,), (2,), (2,)) for _ in range(129)),
                    TransferDirection.READ,
                    max_inflight_batches=1,
                )
            oversized = TransferBatch("a", (1,) * 131073, (2,) * 131073, (2,) * 131073)
            with pytest.raises(ValueError):
                submission.execute_batches(
                    [oversized], TransferDirection.READ, max_inflight_batches=1
                )
