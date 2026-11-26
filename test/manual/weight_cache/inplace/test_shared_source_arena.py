# SPDX-License-Identifier: Apache-2.0
"""Repeated epochs must release complete physical source storage units."""

import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weight_cache.inplace import arena_weights, binding


class CpuCuda:
    def __init__(self):
        self.blocks = []
        self.events = []

    def track(self, tensor):
        storage = tensor.untyped_storage()
        self.blocks.append((storage.data_ptr(), storage.nbytes(), storage._weak_ref()))

    def memory_snapshot(self):
        self.events.append("snapshot")
        return [
            dict(address=address, blocks=[dict(size=size, state="active_allocated")])
            for address, size, reference in self.blocks
            if not torch.UntypedStorage._expired(reference)
        ]

    def synchronize(self):
        pass

    def current_device(self):
        return 0

    def memory_allocated(self):
        return sum(
            size
            for _, size, reference in self.blocks
            if not torch.UntypedStorage._expired(reference)
        )

    memory_reserved = memory_allocated
    max_memory_allocated = memory_allocated

    def empty_cache(self):
        self.events.append("flush")

    def close(self):
        for _, _, reference in self.blocks:
            torch.UntypedStorage._free_weak_ref(reference)
        self.blocks.clear()


@pytest.fixture
def cpu():
    cuda = CpuCuda()

    def empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        result = torch.empty(*args, **kwargs)
        cuda.track(result)
        cuda.events.append("allocate")
        return result

    result = SimpleNamespace(cuda=cuda, empty=empty, nn=torch.nn, no_grad=torch.no_grad)
    try:
        yield result
    finally:
        cuda.close()


def _previous_target(cpu):
    model = torch.nn.Module()
    model.changed_a = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    model.retained = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + 20)
    model.changed_z = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + 40)
    # Exercise the exact previous-epoch allocation/install path, not a manual
    # allocator snapshot that pretends each parameter owns a separate block.
    allocated = arena_weights.allocate_target(cpu, model, {}, set(), {})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            allocated[name].copy_(parameter)
    binding._install_target(torch, model, allocated, model)
    return model


@pytest.mark.parametrize("with_consumers", [False, True])
def test_capture_once_consumes_only_completed_release_unit(
    cpu, monkeypatch, with_consumers
):
    @dataclass(frozen=True)
    class Runtime:
        weight: torch.Tensor

    monkeypatch.setattr(arena_weights, "CAP_BYTES", 32)
    model = torch.nn.Module()
    references = []
    expected = {}
    for i in range(7):
        name = f"weight_{i}"
        parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32) + i)
        model.register_parameter(name, parameter)
        if with_consumers:
            setattr(model, f"runtime_{i}", Runtime(parameter))
            object.__setattr__(model, f"direct_{i}", parameter)
        cpu.cuda.track(parameter)
        references.append(weakref.ref(parameter))
        expected[name] = parameter.detach().clone()
    del parameter
    parameters = dict(model.named_parameters())
    capture = arena_weights.capture_tensor_aliases
    calls = []

    def counted_capture(model, registered):
        calls.append(set(registered))
        return capture(model, registered)

    monkeypatch.setattr(arena_weights, "capture_tensor_aliases", counted_capture)
    snapshot = cpu.cuda.memory_snapshot
    completed = []

    def checked_snapshot():
        index = len(completed)
        assert all(ref() is None for ref in references[: index + 1])
        assert all(ref() is not None for ref in references[index + 1 :])
        completed.append(index)
        return snapshot()

    monkeypatch.setattr(cpu.cuda, "memory_snapshot", checked_snapshot)
    profile = {}
    arena_weights.compact_source(cpu, model, parameters, parameters.keys(), profile, [])
    assert calls == [set(parameters)]
    assert len(completed) == 7
    assert all(
        row["gc_fallback_seconds"] == 0 for row in profile["source_release_chunks"]
    )
    for i, (name, parameter) in enumerate(parameters.items()):
        if with_consumers:
            assert getattr(model, f"runtime_{i}").weight is parameter
            assert getattr(model, f"direct_{i}") is parameter
        assert torch.equal(parameter, expected[name])


@pytest.mark.parametrize("has_parameters", [False, True])
def test_no_changed_names_skips_capture(cpu, monkeypatch, has_parameters):
    model = torch.nn.Module()
    if has_parameters:
        model.weight = torch.nn.Parameter(torch.ones(8))
    parameters = dict(model.named_parameters())

    def unexpected_capture(*args):
        pytest.fail("no moved storage requires no alias scan")

    monkeypatch.setattr(arena_weights, "capture_tensor_aliases", unexpected_capture)
    profile = {}
    arena_weights.compact_source(cpu, model, parameters, set(), profile, [])
    assert profile["source_release_chunks"] == []
    assert profile["source_release_verified"]
    assert cpu.cuda.events == ["flush"]


def test_previous_target_mixed_storage_is_released_and_alias_backing_separated(
    cpu, monkeypatch
):
    from sglang.srt.weight_cache.weight_layout import WeightLayoutBuilder

    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    model = _previous_target(cpu)
    parameters = dict(model.named_parameters())
    assert len({p.untyped_storage().data_ptr() for p in parameters.values()}) == 1
    expected = {n: p.detach().clone() for n, p in parameters.items()}
    owners, profile = [], {}
    builder = WeightLayoutBuilder(model)
    for name, p in parameters.items():
        builder.region(p, name, global_shape=p.shape)
    template, tensors = builder.build().as_meta(model)
    del builder, p
    cpu.cuda.events.clear()
    arena_weights.compact_source(
        cpu, model, parameters, {"changed_a", "changed_z"}, profile, owners
    )
    for name, parameter in parameters.items():
        assert torch.equal(parameter, expected[name])
    assert (
        parameters["retained"].untyped_storage().data_ptr()
        != parameters["changed_a"].untyped_storage().data_ptr()
    )
    assert parameters["retained"].untyped_storage().nbytes() == 16
    assert profile["source_release_verified"]
    assert all(
        row["old_active_allocation_count"] == 0
        for row in profile["source_release_chunks"]
    )
    assert all(
        row["old_parameter_objects_alive"] == 0
        for row in profile["source_release_chunks"]
    )
    assert profile["source_arena_max_bytes"] <= 64
    assert profile["source_release_unit_max_bytes"] == 64
    assert profile["source_arena_payload_bytes"] == 64
    assert profile["source_changed_payload_bytes"] == 48
    assert profile["source_relocated_alias_payload_bytes"] == 16
    assert profile["source_arena_count"] == 2
    template.rebind(model, tensors).validate_bindings(model)
    target = torch.nn.Module()
    for name, p in model.named_parameters():
        target.register_parameter(name, torch.nn.Parameter(torch.empty_like(p)))
    allocated = arena_weights.allocate_target(cpu, target, parameters, {"retained"}, {})
    binding._install_target(torch, target, allocated, model)
    assert target.retained.untyped_storage().nbytes() == 16
    assert (
        target.retained.untyped_storage().data_ptr()
        != target.changed_a.untyped_storage().data_ptr()
    )


def test_hidden_storage_view_keeps_original_fail_stop_release_gate(cpu, monkeypatch):
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    model = _previous_target(cpu)
    parameters = dict(model.named_parameters())
    hidden = parameters["retained"].detach()
    cpu.cuda.events.clear()
    profile, owners = {}, []
    with pytest.raises(
        RuntimeError, match="old changed-source allocation remains active"
    ):
        arena_weights.compact_source(
            cpu, model, parameters, {"changed_a", "changed_z"}, profile, owners
        )
    assert not profile["source_release_chunks"][0]["release_verified"]
    assert not profile["source_release_chunks"][0]["flush_performed"]
    assert torch.equal(hidden, model.retained)


def test_all_changed_shared_storage_is_not_split_by_name_order(cpu, monkeypatch):
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    model = torch.nn.Module()
    storage = torch.arange(16, dtype=torch.float32)
    model.a = torch.nn.Parameter(storage[:8])
    model.z = torch.nn.Parameter(storage[8:])
    model.m = torch.nn.Parameter(torch.arange(8, dtype=torch.float32) + 100)
    cpu.cuda.track(storage)
    cpu.cuda.track(model.m)
    del storage
    parameters = dict(model.named_parameters())
    assert arena_weights.chunks(parameters, parameters) == [
        (["a", "m"], 64),
        (["z"], 32),
    ]
    assert arena_weights.source_chunks(parameters, parameters) == [
        (["a", "z"], 64),
        (["m"], 32),
    ]
    expected = {n: p.detach().clone() for n, p in parameters.items()}
    profile = {}
    cpu.cuda.events.clear()
    arena_weights.compact_source(cpu, model, parameters, parameters.keys(), profile, [])
    assert profile["source_release_verified"]
    assert profile["source_relocated_alias_payload_bytes"] == 0
    for name, p in parameters.items():
        assert torch.equal(p, expected[name])
    allocations = [i for i, e in enumerate(cpu.cuda.events) if e == "allocate"]
    assert cpu.cuda.events.index("flush") < allocations[1]
    assert len(profile["source_release_chunks"]) == 2


def test_registered_group_over_cap_is_rejected_before_mutation(cpu, monkeypatch):
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    model = _previous_target(cpu)
    parameters = dict(model.named_parameters())
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 48)
    previous = {n: id(p) for n, p in parameters.items()}
    cpu.cuda.events.clear()
    with pytest.raises(ValueError, match="release unit exceeds"):
        arena_weights.compact_source(cpu, model, parameters, {"changed_a"}, {}, [])
    assert {n: id(p) for n, p in parameters.items()} == previous
    assert "allocate" not in cpu.cuda.events


def test_offset_members_and_unused_capacity_are_supported(cpu, monkeypatch):
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 32)
    model = torch.nn.Module()
    storage = torch.arange(32, dtype=torch.float32)
    model.a = torch.nn.Parameter(storage[4:8])
    model.b = torch.nn.Parameter(storage[16:20])
    cpu.cuda.track(storage)
    del storage
    parameters = dict(model.named_parameters())
    assert model.a.untyped_storage().nbytes() > arena_weights.CAP_BYTES
    assert arena_weights.source_chunks(parameters, parameters) == [(["a", "b"], 32)]
    expected = {n: p.detach().clone() for n, p in parameters.items()}
    profile = {}
    arena_weights.compact_source(cpu, model, parameters, parameters.keys(), profile, [])
    assert profile["source_release_verified"]
    assert profile["source_arena_payload_bytes"] == 32
    for name, p in parameters.items():
        assert torch.equal(p, expected[name])


def test_registered_overlap_is_rejected_before_mutation(cpu, monkeypatch):
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    model = torch.nn.Module()
    storage = torch.arange(16, dtype=torch.float32)
    model.a = torch.nn.Parameter(storage[:8])
    model.b = torch.nn.Parameter(storage[4:12])
    cpu.cuda.track(storage)
    del storage
    cpu.cuda.events.clear()
    with pytest.raises(ValueError, match="overlapping source storage"):
        arena_weights.compact_source(
            cpu, model, dict(model.named_parameters()), {"a"}, {}, []
        )
    assert "allocate" not in cpu.cuda.events


def test_frozen_dataclass_alias_consumers_are_rebound_for_both_parts(cpu, monkeypatch):
    @dataclass(frozen=True)
    class Nested:
        weight: torch.Tensor

    @dataclass(frozen=True)
    class Runtime:
        first: torch.Tensor
        retained: torch.Tensor
        nested: Nested

    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    model = _previous_target(cpu)
    model.runtime = Runtime(model.changed_a, model.retained, Nested(model.changed_z))
    parameters = dict(model.named_parameters())
    profile = {}
    arena_weights.compact_source(
        cpu, model, parameters, {"changed_a", "changed_z"}, profile, []
    )
    assert model.runtime.first is model.changed_a
    assert model.runtime.retained is model.retained
    assert model.runtime.nested.weight is model.changed_z
    assert profile["source_release_verified"]
    assert profile["source_release_chunks"][0]["old_parameter_objects_alive"] == 0


def test_fresh_independent_storages_keep_original_chunks(cpu, monkeypatch):
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    model = torch.nn.Module()
    for name in ("a", "m", "z"):
        model.register_parameter(name, torch.nn.Parameter(torch.ones(8)))
    parameters = dict(model.named_parameters())
    for p in parameters.values():
        cpu.cuda.track(p)
    del p
    assert arena_weights.source_chunks(parameters, parameters) == arena_weights.chunks(
        parameters, parameters
    )
    profile = {}
    arena_weights.compact_source(cpu, model, parameters, parameters.keys(), profile, [])
    assert profile["source_arena_count"] == 2
    assert profile["source_arena_payload_bytes"] == 96
    assert profile["source_relocated_alias_payload_bytes"] == 0


def test_source_grouping_rejects_multiple_devices_before_mutation():
    values = {"a": torch.empty(1), "b": torch.empty(1, device="meta")}
    with pytest.raises(ValueError, match="single source device"):
        arena_weights.source_chunks(values, {"a"})


@pytest.mark.parametrize("kind", ["empty", "meta"])
def test_source_grouping_refuses_null_storage_identity(kind):
    value = torch.empty(0) if kind == "empty" else torch.empty(1, device="meta")
    with pytest.raises(ValueError, match="nonempty materialized"):
        arena_weights.source_chunks({"weight": value}, {"weight"})


def test_public_bind_rejects_oversize_storage_unit_before_compaction(cpu, monkeypatch):
    from sglang.srt.weight_cache.weight_layout import WeightLayoutBuilder

    monkeypatch.setattr(arena_weights, "CAP_BYTES", 64)
    source = _previous_target(cpu)
    target = torch.nn.Module()
    target.changed_a = torch.nn.Parameter(torch.empty(4, 2, device="meta"))
    target.changed_z = torch.nn.Parameter(torch.empty(4, device="meta"))
    target.retained = torch.nn.Parameter(torch.empty(4, device="meta"))
    source.config = target.config = SimpleNamespace(num_experts=8)

    def plan(model, *, meta=False):
        builder = WeightLayoutBuilder(model, allow_meta=meta)
        for name, p in model.named_parameters():
            builder.region(p, name, global_shape=p.shape)
        return builder.build()

    args = SimpleNamespace(tp_size=8, ep_size=1, dp_size=1)
    monkeypatch.setattr(binding, "_normalize_args", lambda value: value)
    monkeypatch.setattr(binding, "_validate_models", lambda *args: None)
    monkeypatch.setattr(binding, "_build_plan", plan)
    monkeypatch.setattr(binding, "_PARKED", [])
    monkeypatch.setenv("MC_INTRANODE_NVLINK", "1")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: None)
    monkeypatch.setattr(arena_weights, "CAP_BYTES", 48)
    previous = {n: id(p) for n, p in source.named_parameters()}
    cpu.cuda.events.clear()
    with pytest.raises(ValueError, match="release unit exceeds"):
        binding.bind_weights(source, target, args, args, 0, group=object())
    assert {n: id(p) for n, p in source.named_parameters()} == previous
    assert not binding._PARKED
    assert "allocate" not in cpu.cuda.events
