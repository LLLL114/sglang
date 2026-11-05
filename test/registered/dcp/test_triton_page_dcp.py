# SPDX-License-Identifier: Apache-2.0
"""Real page pools, Triton attention and NCCL, compared with dense FP32 attention."""

import socket
import traceback
import unittest
from types import SimpleNamespace as NS

import torch
import torch.multiprocessing as mp

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.layers.attention.triton_backend import (
    ForwardMetadata,
    TritonAttnBackend,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.page_interleave import PageShardSpec
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveMHATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

PAGE = 64
DIM = 64
KV_HEADS = 2
LOCAL_Q_HEADS = 2


def _reference(q, k, v, scale):
    repeats = q.shape[1] // k.shape[1]
    keys = k.float().repeat_interleave(repeats, dim=1)
    values = v.float().repeat_interleave(repeats, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), keys) * scale
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), values)


def _make_case(rank, group, lengths, gather_prefix=False):
    device = f"cuda:{rank}"
    spec = PageShardSpec(
        rank, group.world_size, PAGE, 4096, 4096, gather_prefix=gather_prefix
    )
    pool = PageInterleaveMHATokenToKVPool(
        32768,
        PAGE,
        torch.bfloat16,
        KV_HEADS,
        DIM,
        1,
        device,
        False,
        start_layer=0,
        end_layer=0,
        enable_alt_stream=False,
        shard_spec=spec,
        shard_group=group,
    )
    if not gather_prefix:
        assert not hasattr(pool, "kv_gather_comm")
        assert not hasattr(pool, "_slots")
    allocator = PageInterleavePoolAllocator(
        32768, PAGE, group.world_size, torch.bfloat16, device, pool, False, spec
    )
    # Preserve owner classes but force physically fragmented request chains.
    for owner, pages in enumerate(allocator.class_free_pages):
        allocator.class_free_pages[owner] = torch.cat((pages[::2], pages[1::2]))
    lengths_cpu = torch.tensor(lengths, dtype=torch.int64)
    bases = [i % group.world_size for i in range(len(lengths))]
    locs = allocator.alloc_extend(
        torch.zeros_like(lengths_cpu, device=device),
        torch.zeros_like(lengths_cpu),
        lengths_cpu.to(device),
        lengths_cpu,
        torch.full((len(lengths),), -1, dtype=torch.int64, device=device),
        sum(lengths),
        rotation_bases=bases,
    )
    table = torch.zeros(
        (len(lengths), max(lengths) + PAGE), device=device, dtype=torch.int64
    )
    offset = 0
    for row, length in enumerate(lengths):
        table[row, :length] = locs[offset : offset + length]
        offset += length
    req_ids = torch.arange(len(lengths), dtype=torch.int64, device=device)
    torch.manual_seed(42)
    k = torch.randn(sum(lengths), KV_HEADS, DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    layer = RadixAttention(LOCAL_Q_HEADS, DIM, DIM**-0.5, KV_HEADS, 0)
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend._page_dcp = True
    backend.dcp_size = group.world_size
    backend.dcp_rank = rank
    backend.device = device
    backend.req_to_token = table
    backend.token_to_kv_pool = pool
    backend.decode_attention_fwd = decode_attention_fwd
    backend.extend_attention_fwd = extend_attention_fwd
    backend.is_draft_runner = False
    backend.use_mla = False
    backend.max_kv_splits = 4
    pool.k_buffer[0].fill_(-123)
    pool.v_buffer[0].fill_(-123)
    if gather_prefix:
        # Exercise the actual PCP write/scratch path, then reuse its persistent
        # bytes for DCP without packing or translating the cache tensor.
        pool.begin_shard_extend(table, req_ids, [0] * len(lengths), lengths)
    else:
        pool.begin_local_forward(locs)
    batch = NS(out_cache_loc=locs, out_cache_loc_is_physical=False)
    backend._set_kv_buffer(batch, layer, KVWriteLoc(locs), k, v)
    if gather_prefix:
        pool.end_shard_extend()
    return backend, layer, req_ids, lengths_cpu.to(device), k, v, bases


def _metadata(backend, req_ids, lengths, query_lengths=None):
    batch = len(req_ids)
    indptr, indices, local_lens = backend._dcp_kv_indices(
        req_ids,
        lengths,
        torch.zeros(batch + 1, dtype=torch.int32, device=backend.device),
    )
    heads = LOCAL_Q_HEADS * backend.dcp_size
    qo_indptr = None
    if query_lengths is not None:
        qo_indptr = (
            torch.tensor([0] + query_lengths, dtype=torch.int32, device=backend.device)
            .cumsum(0)
            .int()
        )
    backend.forward_metadata = ForwardMetadata(
        attn_logits=torch.empty(
            batch, heads, 4, DIM, device=backend.device, dtype=torch.float32
        ),
        attn_lse=torch.empty(
            batch, heads, 4, device=backend.device, dtype=torch.float32
        ),
        max_extend_len=max(query_lengths) if query_lengths else 1,
        num_kv_splits=torch.full((batch,), 4, dtype=torch.int32, device=backend.device),
        kv_indptr=indptr,
        kv_indices=indices,
        qo_indptr=qo_indptr,
        custom_mask=None,
        mask_indptr=None,
        window_kv_indptr=None,
        window_kv_indices=None,
        window_num_kv_splits=None,
        window_kv_offsets=None,
    )
    backend._page_dcp_empty_rows = local_lens == 0
    return indptr, indices, local_lens


def _decode_case(rank, group, lengths, gather_prefix=False):
    backend, layer, req_ids, lens, k, v, bases = _make_case(
        rank, group, lengths, gather_prefix
    )
    indptr, indices, local_lens = _metadata(backend, req_ids, lens)
    torch.manual_seed(84)
    q = torch.randn(
        len(lengths),
        LOCAL_Q_HEADS * group.world_size,
        DIM,
        device=backend.device,
        dtype=torch.bfloat16,
    )
    head_start = rank * LOCAL_Q_HEADS
    expected = []
    expected_local_lse = []
    offset = 0
    for request, length in enumerate(lengths):
        owned_positions = [
            p
            for p in range(length)
            if (bases[request] + p // PAGE) % group.world_size == rank
        ]
        assert local_lens[request].item() == len(owned_positions)
        local_rows = indices[indptr[request] : indptr[request + 1]]
        # Byte equality checks the actual writes and CSR reads together,
        # independently of the implementation's logical-to-local formula.
        torch.testing.assert_close(
            backend.token_to_kv_pool.get_key_buffer(0)[local_rows],
            k[offset : offset + length][owned_positions],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            backend.token_to_kv_pool.get_value_buffer(0)[local_rows],
            v[offset : offset + length][owned_positions],
            rtol=0,
            atol=0,
        )
        full = _reference(
            q[request : request + 1],
            k[offset : offset + length],
            v[offset : offset + length],
            layer.scaling,
        )
        expected.append(full[:, head_start : head_start + LOCAL_Q_HEADS])
        local_keys = (
            k[offset : offset + length][owned_positions]
            .float()
            .repeat_interleave(LOCAL_Q_HEADS * group.world_size // KV_HEADS, dim=1)
        )
        scores = (
            torch.einsum("hd,khd->hk", q[request].float(), local_keys) * layer.scaling
        )
        expected_local_lse.append(scores.logsumexp(-1))
        offset += length
    actual = backend.forward_decode(
        q[:, head_start : head_start + LOCAL_Q_HEADS].contiguous(),
        None,
        None,
        layer,
        NS(),
        save_kv_cache=False,
    ).view(len(lengths), LOCAL_Q_HEADS, DIM)
    torch.testing.assert_close(
        actual.float(), torch.cat(expected), atol=2e-2, rtol=2e-2
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        backend.forward_metadata.attn_lse.logsumexp(-1),
        torch.stack(expected_local_lse),
        atol=5e-3,
        rtol=1e-3,
    )
    return backend


def _extend_case(rank, group, prefixes, chunks):
    lengths = [prefix + chunk for prefix, chunk in zip(prefixes, chunks)]
    backend, layer, req_ids, _, k, v, _ = _make_case(rank, group, lengths)
    _metadata(backend, req_ids, torch.tensor(prefixes, device=backend.device), chunks)
    torch.manual_seed(123)
    q = torch.randn(
        sum(chunks),
        LOCAL_Q_HEADS * group.world_size,
        DIM,
        device=backend.device,
        dtype=torch.bfloat16,
    )
    head_start = rank * LOCAL_Q_HEADS
    expected, current_k, current_v = [], [], []
    kv_offset = q_offset = 0
    for prefix, chunk, length in zip(prefixes, chunks, lengths):
        current_k.append(k[kv_offset + prefix : kv_offset + length])
        current_v.append(v[kv_offset + prefix : kv_offset + length])
        for j in range(chunk):
            full = _reference(
                q[q_offset + j : q_offset + j + 1],
                k[kv_offset : kv_offset + prefix + j + 1],
                v[kv_offset : kv_offset + prefix + j + 1],
                layer.scaling,
            )
            expected.append(full[:, head_start : head_start + LOCAL_Q_HEADS])
        kv_offset += length
        q_offset += chunk
    actual = backend._forward_extend_dcp(
        q=q[:, head_start : head_start + LOCAL_Q_HEADS].contiguous(),
        k=torch.cat(current_k),
        v=torch.cat(current_v),
        layer=layer,
        forward_batch=NS(extend_prefix_lens_cpu=prefixes),
        causal=True,
        logits_soft_cap=0.0,
        sinks=None,
    ).view(sum(chunks), LOCAL_Q_HEADS, DIM)
    torch.testing.assert_close(
        actual.float(), torch.cat(expected), atol=2e-2, rtol=2e-2
    )
    assert torch.isfinite(actual).all()
    return backend


def _worker(rank, world, port):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
        timeout=90,
    )
    publish_build_topology(tp_size=world, dcp_size=world, world_rank=rank)
    initialize_model_parallel()
    try:
        group = get_parallel().dcp_group
        cases = []
        cases.append(
            _decode_case(
                rank, group, [1, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 8192]
            )
        )
        torch.distributed.barrier(group=group.cpu_group)
        print(f"rank={rank} fragmented decode passed", flush=True)
        cases.append(_decode_case(rank, group, [64, 128], gather_prefix=True))
        torch.distributed.barrier(group=group.cpu_group)
        print(f"rank={rank} PCP pool reuse passed", flush=True)
        cases.append(_extend_case(rank, group, [64], [3]))
        torch.distributed.barrier(group=group.cpu_group)
        print(f"rank={rank} empty-rank prefix passed", flush=True)
        cases.append(_extend_case(rank, group, [64, 0, 128, 1], [3, 2, 5, 1]))
        torch.distributed.barrier(group=group.cpu_group)
        print(f"rank={rank} ragged prefixes passed", flush=True)
        cases.append(_extend_case(rank, group, [0, 0], [3, 5]))
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group.cpu_group)
    except Exception:
        traceback.print_exc()
        raise
    destroy_model_parallel()
    destroy_distributed_environment()


class TestTritonPageDcp(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "requires two CUDA GPUs",
    )
    def test_real_page_pool_and_attention(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
