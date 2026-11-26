"""Allocation and view contracts for the actual input-buffer pool leaves."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def leaves(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "python/sglang/srt/model_executor"
    utils = types.ModuleType("sglang.srt.utils")
    utils.is_npu = lambda: False
    context = types.ModuleType("sglang.srt.runtime_context")
    context.get_parallel = lambda: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, context.__name__, context)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    pool = load("sglang.srt.model_executor.input_buffers", "input_buffers.py")
    registry = load("_tested_graph_registry", "cuda_graph_buffer_registry.py")
    return pool, registry


def test_public_registry_hit_does_not_allocate(leaves):
    pool, module = leaves

    def register(shape, **kwargs):
        registry = module.CudaGraphBufferRegistry(
            device="cpu", max_bs=1, max_num_tokens=410000, share_pool=True
        )
        return registry.register_slot(
            module.GraphSlot(
                name="mrope_positions",
                shape_fn=lambda bs, n: shape,
                dtype=torch.int64,
                **kwargs,
            )
        ).buffer

    first = register((3, 410000))
    first.fill_(7)
    with patch.object(torch, "zeros", wraps=torch.zeros) as zeros:
        second = register((410000, 3))
    assert zeros.call_count == 0
    assert second.data_ptr() == first.data_ptr()
    assert second.stride() == (3, 1)
    assert torch.all(second == 7)
    assert len(pool._forward_input_buffer_pool) == 1


def test_pool_key_and_legacy_noncontiguous_storage(leaves):
    pool, _ = leaves
    original = torch.arange(12).reshape(3, 4).t()
    pool.share_input_buffer("x", original)
    view = pool.get_shared_input_buffer("x", (2, 6), dtype=torch.int64, device="cpu:0")
    assert view.data_ptr() == original.data_ptr()
    assert view.stride() == (6, 1)
    assert torch.equal(view, torch.arange(12).reshape(2, 6))
    for name, shape, dtype in (
        ("y", (12,), torch.int64),
        ("x", (13,), torch.int64),
        ("x", (12,), torch.float32),
    ):
        other = pool.get_shared_input_buffer(name, shape, dtype=dtype, device="cpu")
        assert other.data_ptr() != original.data_ptr()
        assert torch.count_nonzero(other) == 0
    empty = pool.get_shared_input_buffer(
        "empty", (0, 3), dtype=torch.int64, device="cpu"
    )
    assert empty.shape == (0, 3)
    scalar = pool.get_shared_input_buffer(
        "scalar", (), dtype=torch.float32, device="cpu"
    )
    assert scalar.shape == ()
    assert scalar.item() == 0
    with pytest.raises(RuntimeError):
        pool.get_shared_input_buffer("bad", (-1, 3), dtype=torch.int64, device="cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA device normalization"
)
def test_implicit_cuda_device_matches_indexed_device(leaves):
    pool, _ = leaves
    original = pool.get_shared_input_buffer(
        "cuda", (4,), dtype=torch.int64, device="cuda"
    )
    with patch.object(torch, "zeros", wraps=torch.zeros) as zeros:
        reused = pool.get_shared_input_buffer(
            "cuda",
            (2, 2),
            dtype=torch.int64,
            device=f"cuda:{torch.cuda.current_device()}",
        )
    assert zeros.call_count == 0
    assert original.data_ptr() == reused.data_ptr()


def test_registry_bind_disabled_nonpool_and_sentinel(leaves):
    pool, module = leaves

    def registry(shared):
        return module.CudaGraphBufferRegistry(
            device="cpu", max_bs=2, max_num_tokens=4, share_pool=shared
        )

    def slot(name, **kwargs):
        return module.GraphSlot(
            name=name, shape_fn=lambda bs, n: (n,), dtype=torch.int64, **kwargs
        )

    bound = torch.full((4,), 9)
    with patch.object(torch, "zeros", wraps=torch.zeros) as zeros:
        assert registry(True).register_slot(slot("bound"), bind=bound).buffer is bound
        assert (
            registry(True).register_slot(slot("disabled", enabled=False)).buffer is None
        )
        assert zeros.call_count == 0
        a = registry(False).register_slot(slot("private")).buffer
        b = registry(False).register_slot(slot("private")).buffer
        assert zeros.call_count == 2
        assert a.data_ptr() != b.data_ptr()
    first = (
        registry(True)
        .register_slot(
            slot(
                "sentinel",
                padding_policy=module.PaddingPolicy.FILL_SENTINEL,
                pad_value=-1,
            )
        )
        .buffer
    )
    first.fill_(5)
    second = (
        registry(True)
        .register_slot(
            slot(
                "sentinel", padding_policy=module.PaddingPolicy.FILL_ONCE, pad_value=-2
            )
        )
        .buffer
    )
    assert first.data_ptr() == second.data_ptr()
    assert torch.all(first == -2)
    assert len(pool._forward_input_buffer_pool) == 1
