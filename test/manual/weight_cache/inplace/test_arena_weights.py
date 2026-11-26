# SPDX-License-Identifier: Apache-2.0
"""CPU checks for bounded chunk planning and allocator release interpretation."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch as real_torch

from sglang.srt.weight_cache.inplace.arena_weights import (
    CAP_BYTES,
    active_blocks,
    chunks,
    compact_source,
)


class TestArenaTest:
    def test_cap_dtype_and_contiguity(self):
        def tensor(size, dtype="bf16", contiguous=True):
            return NS(
                numel=lambda: size // 2,
                element_size=lambda: 2,
                dtype=dtype,
                is_contiguous=lambda: contiguous,
            )

        parameters = {str(i): tensor(600 << 20) for i in range(3)}
        assert [size for (names, size) in chunks(parameters, parameters)] == [
            600 << 20
        ] * 3
        with pytest.raises(ValueError):
            chunks({"x": tensor(CAP_BYTES + 2)}, ["x"])
        with pytest.raises(ValueError):
            chunks({"x": tensor(4, contiguous=False)}, ["x"])
        assert len(chunks({"a": tensor(4), "b": tensor(4, "fp32")}, ["a", "b"])) == 2

    def test_only_active_blocks_hold_old_storage(self):
        snapshot = [
            dict(
                address=100,
                blocks=[
                    dict(size=20, state="inactive"),
                    dict(size=30, state="active_allocated"),
                    dict(size=10, state="active_awaiting_free"),
                ],
            )
        ]
        assert active_blocks(snapshot, [100, 119, 120, 149, 150, 159, 160]) == [
            120,
            149,
            150,
            159,
        ]


class TestFlushTest:
    def test_multiple_chunks_capture_once(self, monkeypatch):
        from sglang.srt.weight_cache.inplace import arena_weights

        torch, model, parameters, events = self.fixture()
        for i in range(2, 7):
            name = f"extra_{i}"
            model._parameters[name] = type(parameters["a"])(
                (600 << 20) // 2, 1000 * (i + 1)
            )
            parameters[name] = model._parameters[name]
        capture = arena_weights.capture_tensor_aliases
        calls = []

        def counted_capture(model, registered):
            calls.append(set(registered))
            return capture(model, registered)

        monkeypatch.setattr(arena_weights, "capture_tensor_aliases", counted_capture)
        compact_source(torch, model, parameters, parameters.keys(), {}, [])
        assert calls == [set(parameters)]

    def fixture(self, retained=False):
        events = []

        class Tensor:
            __slots__ = (
                "count",
                "address",
                "dtype",
                "device",
                "shape",
                "requires_grad",
                "__dict__",
                "__weakref__",
            )

            def __init__(self, count, address):
                self.count, self.address = count, address
                self.dtype, self.device, self.shape, self.requires_grad = (
                    "bf16",
                    "cuda",
                    (count,),
                    False,
                )

            def numel(self):
                return self.count

            def element_size(self):
                return 2

            def is_contiguous(self):
                return True

            def data_ptr(self):
                return self.address

            def untyped_storage(self):
                return NS(data_ptr=self.data_ptr, nbytes=lambda: self.count * 2)

            def storage_offset(self):
                return 0

            def narrow(self, dim, offset, count):
                return Tensor(count, self.address + offset * 2)

            def view(self, shape):
                self.shape = shape
                return self

            def copy_(self, other):
                events.append("copy")

        class Cuda:
            def synchronize(self):
                events.append("sync")

            def memory_allocated(self):
                return 1200 << 20

            def memory_reserved(self):
                return 1300 << 20

            def max_memory_allocated(self):
                return 1800 << 20

            def memory_snapshot(self):
                events.append("snapshot")
                return (
                    [
                        dict(
                            address=1000,
                            blocks=[dict(size=100, state="active_allocated")],
                        )
                    ]
                    if retained
                    else []
                )

            def empty_cache(self):
                events.append("flush")

        cursor = [100000]

        def empty(count, **kwargs):
            events.append("allocate")
            cursor[0] += 100000
            return Tensor(count, cursor[0])

        def parameter(view, requires_grad):
            view.requires_grad = requires_grad
            return view

        torch = NS(
            cuda=Cuda(), empty=empty, no_grad=nullcontext, nn=NS(Parameter=parameter)
        )
        model = real_torch.nn.Module()
        model._parameters = {
            n: Tensor((600 << 20) // 2, 1000 + i * 1000)
            for i, n in enumerate(("a", "b"))
        }
        return torch, model, dict(model._parameters), events

    def test_each_verified_chunk_flushes_before_next_allocation(self):
        torch, model, parameters, events = self.fixture()
        profile = {}
        owners = []
        compact_source(torch, model, parameters, parameters.keys(), profile, owners)
        rows = profile["source_release_chunks"]
        assert len(rows) == 2
        assert all((r["flush_performed"] and r["release_verified"] for r in rows))
        assert all(
            (
                r["flush_seconds"] >= 0 and r["reserved_after_flush"] == 1300 << 20
                for r in rows
            )
        )
        assert [e for e in events if e in ("allocate", "flush")] == [
            "allocate",
            "flush",
            "allocate",
            "flush",
            "flush",
        ]
        assert all(
            (r["reserved_after_flush"] <= r["reserved_before_flush"] for r in rows)
        )
        assert profile["source_final_empty_cache_seconds"] >= 0
        assert (
            profile["source_empty_cache_seconds"]
            == sum((r["flush_seconds"] for r in rows))
            + profile["source_final_empty_cache_seconds"]
        )

    def test_retained_old_block_refuses_before_any_flush(self):
        torch, model, parameters, events = self.fixture(retained=True)
        profile = {}
        with pytest.raises(RuntimeError):
            compact_source(torch, model, parameters, parameters.keys(), profile, [])
        assert "flush" not in events
        assert not (profile["source_release_chunks"][0]["flush_performed"])
        assert profile["source_empty_cache_seconds"] == 0
