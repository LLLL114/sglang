"""Real TP/PP CPU collectives for compact, large, cancelled and failed frames."""

import datetime
import pickle
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _frame_worker(rank, path):
    from sglang.srt.managers.cache_controller import HiCacheController

    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        groups = [
            dist.new_group(ranks=ranks, backend="gloo")
            for ranks in ([0, 1], [2, 3], [0, 2], [1, 3])
        ]
        controller = object.__new__(HiCacheController)
        controller.tp_rank, controller.pp_rank = rank % 2, rank // 2
        controller.page_size = 16
        controller.prefetch_hits_sync_groups = [groups[rank // 2], groups[2 + rank % 2]]

        class Backend:
            mode = "small"

            def prepare_prefetch(self, keys, operation_id):
                if self.mode == "missing":
                    return None
                return SimpleNamespace(
                    page_keys=tuple(keys),
                    padding=b"x" * (12000 if self.mode == "large" else 0),
                )

            def pack_prefetch_context(self, context):
                if self.mode == "encode_failure":
                    raise ValueError("injected encode failure")
                return pickle.dumps(context)

            def unpack_prefetch_context(self, payload, keys):
                context = pickle.loads(payload)
                if context.page_keys != tuple(keys):
                    raise ValueError("mismatched request keys")
                return context

        backend = Backend()
        controller.storage_backend = backend
        original = dist.broadcast
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        dist.broadcast = counted
        for mode in [
            "small",
            "large",
            "missing",
            "encode_failure",
            "mismatch",
            "cancelled_peer",
            "cancelled_root",
            "small",
        ]:
            backend.mode = mode
            calls.clear()
            keys = ["a", "b", "c"]
            if mode == "mismatch" and rank == 3:
                keys[-1] = "different"
            cancelled = (mode == "cancelled_peer" and rank == 3) or (
                mode == "cancelled_root" and rank == 0
            )
            operation = SimpleNamespace(is_terminated=lambda: cancelled)
            _, count = controller._storage_context_query(operation, keys)
            assert len(calls) == (4 if mode == "large" else 2), (mode, len(calls))
            # This is the existing post-discovery consensus; a local decode or
            # cancellation failure must veto reads on every rank.
            total = torch.tensor(count)
            for group in controller.prefetch_hits_sync_groups:
                dist.all_reduce(total, op=dist.ReduceOp.MIN, group=group)
            assert total.item() == (48 if mode in ("small", "large") else 0), mode
            if mode in ("small", "large"):
                assert operation.storage_read_context.page_keys == ("a", "b", "c")
    finally:
        dist.destroy_process_group()


def test_compact_frames_preserve_tp_pp_consensus(tmp_path):
    mp.spawn(_frame_worker, args=(str(tmp_path / "group"),), nprocs=4, join=True)
