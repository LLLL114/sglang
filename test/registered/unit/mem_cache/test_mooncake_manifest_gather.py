"""Every TP/PP rank participates and receives the same full layout metadata."""

import datetime
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _gather_worker(rank, path):
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.storage.mooncake_store.reshard import (
        collect_reshard_metadata,
    )

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
        tp, pp = rank % 2, rank // 2
        pool = object.__new__(MHATokenToKVPoolHost)
        pool.layout, pool.dtype, pool.page_size = "page_first", torch.float16, 16
        pool.head_num, pool.layer_num, pool.head_dim = 2, 2, 64
        pool.device_pool = SimpleNamespace(
            start_layer=pp * 2,
            end_layer=(pp + 1) * 2,
            head_dim=64,
            v_head_dim=64,
            layer_num=2,
        )
        controller = SimpleNamespace(
            storage_host_pool=pool, tp_group=groups[pp], pp_group=groups[2 + tp]
        )
        config = SimpleNamespace(
            extra_config={"kv_reshard": {"model_revision": "weights-1"}},
            is_mla_model=False,
            attn_cp_size=1,
            tp_lcm_size=None,
            tp_rank=tp,
            pp_rank=pp,
            tp_size=2,
            pp_size=2,
            model_name="model",
        )
        model = SimpleNamespace(
            num_hidden_layers=4,
            is_multimodal=False,
            get_total_num_kv_heads=lambda: 4,
            hf_config=SimpleNamespace(to_dict=lambda: {"rope_theta": 10000}),
        )
        with (
            patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=SimpleNamespace(dp_size=1),
            ),
            patch(
                "sglang.srt.runtime_context.process_model_config", return_value=model
            ),
        ):
            metadata = collect_reshard_metadata(controller, config)
        assert [(p["pp"], p["tp"]) for p in metadata["parts"]] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        assert [p["layers"] for p in metadata["parts"]] == [
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
        ]
        gathered = [None] * 4
        dist.all_gather_object(gathered, metadata)
        assert all(value == metadata for value in gathered)
    finally:
        dist.destroy_process_group()


def test_two_stage_gather_builds_the_full_manifest_on_every_rank(tmp_path):
    mp.spawn(
        _gather_worker, args=(str(tmp_path / "manifest-group"),), nprocs=4, join=True
    )
