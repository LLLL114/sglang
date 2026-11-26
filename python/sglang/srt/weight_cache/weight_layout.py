# SPDX-License-Identifier: Apache-2.0
"""Explicit, bounded descriptions of immutable runtime weight storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any

import torch

from .weight_runtime_manifest import (
    LogicalWeightTensor,
    WeightManifestError,
    WeightPartition,
    _storage_key,
)


@dataclass(frozen=True)
class WeightRegion:
    tensor_id: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    parameter: torch.Tensor
    parameter_names: tuple[str, ...]
    byte_offset: int
    partition: WeightPartition
    expert_id: int | None = None
    layout_fingerprint: str = "sglang:row-major"

    @property
    def nbytes(self) -> int:
        return prod(self.local_shape) * self.parameter.element_size()


@dataclass(frozen=True)
class WeightInitialization:
    parameter: torch.Tensor
    parameter_names: tuple[str, ...]
    byte_offset: int
    nbytes: int
    value: bool | int | float


def registered_tensors(model):
    return dict(
        (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
    )


def _layout_storage_key(tensor):
    # Meta tensors have no allocation and all report the same null data pointer.
    return ("meta", id(tensor)) if tensor.is_meta else _storage_key(tensor)


@dataclass(frozen=True)
class WeightLayoutPlan:
    tensors: tuple[LogicalWeightTensor, ...]
    views: tuple[WeightRegion, ...]
    initializations: tuple[WeightInitialization, ...]
    local_buffers: frozenset[str]
    derived_state: tuple[str, ...]
    bindings: dict[str, tuple]

    def _with_tensors(self, tensors):
        return replace(
            self,
            views=tuple(
                replace(v, parameter=tensors[v.parameter_names[0]]) for v in self.views
            ),
            initializations=tuple(
                replace(v, parameter=tensors[v.parameter_names[0]])
                for v in self.initializations
            ),
            bindings={n: _layout_storage_key(tensors[n]) for n in self.bindings},
        )

    def as_meta(self, model):
        """Detach a stopped epoch's geometry without retaining tensor attributes."""
        self.validate_bindings(model)
        tensors, by_storage = {}, {}
        for name, tensor in registered_tensors(model).items():
            key = _layout_storage_key(tensor)
            if key not in by_storage:
                by_storage[key] = torch.empty_strided(
                    tensor.shape, tensor.stride(), dtype=tensor.dtype, device="meta"
                )
            tensors[name] = by_storage[key]
        return self._with_tensors(tensors), tensors

    def rebind(self, model, previous_tensors):
        """Rebind same-epoch meta geometry through all physical builder checks."""
        builder = WeightLayoutBuilder(model)
        if set(builder.tensors) != set(previous_tensors):
            raise WeightManifestError("registered tensor names changed before rebind")
        previous_names = {}
        for name, tensor in previous_tensors.items():
            if not tensor.is_meta:
                raise WeightManifestError(
                    "layout rebind requires detached meta tensors"
                )
            previous_names.setdefault(id(tensor), []).append(name)
        for name, current in builder.tensors.items():
            previous = previous_tensors[name]
            aliases = tuple(sorted(previous_names[id(previous)]))
            if (
                tuple(current.shape) != tuple(previous.shape)
                or tuple(current.stride()) != tuple(previous.stride())
                or current.dtype != previous.dtype
                or tuple(sorted(builder._names[_layout_storage_key(current)]))
                != aliases
            ):
                raise WeightManifestError(f"layout geometry or aliases changed: {name}")
            if name not in self.local_buffers:
                builder._parameter_names(current)
        for region in (*self.views, *self.initializations):
            previous = previous_tensors[region.parameter_names[0]]
            if region.parameter is not previous or region.parameter_names != tuple(
                sorted(previous_names[id(previous)])
            ):
                raise WeightManifestError(
                    "layout template references unexpected storage"
                )
        rebound = self._with_tensors(builder.tensors)
        for region in (*rebound.views, *rebound.initializations):
            builder._check_range(region)
        builder._views = list(rebound.views)
        builder._initializations = list(rebound.initializations)
        builder._declarations = {t.tensor_id: t for t in self.tensors}
        for name in self.local_buffers:
            builder.local_buffer(name)
        builder.derived(*self.derived_state)
        result = builder.build()
        result.validate_derived_state(model)
        return result

    def validate_bindings(self, model) -> None:
        tensors = registered_tensors(model)
        signatures = {}

        def signature(tensor):
            identity = id(tensor)
            if identity not in signatures:
                signatures[identity] = _layout_storage_key(tensor)
            return signatures[identity]

        if set(tensors) != self.bindings.keys() | self.local_buffers:
            raise WeightManifestError(
                "registered tensor names changed after layout construction"
            )
        for name, expected in self.bindings.items():
            if signature(tensors[name]) != expected:
                raise WeightManifestError(
                    f"weight storage changed after layout construction: {name}"
                )
        for region in (*self.views, *self.initializations):
            if signature(region.parameter) != self.bindings[region.parameter_names[0]]:
                raise WeightManifestError(
                    f"declared tensor storage changed: {region.parameter_names[0]}"
                )

    def initialize_local_regions(self) -> None:
        with torch.no_grad():
            for region in self.initializations:
                itemsize = region.parameter.element_size()
                region.parameter.view(-1).narrow(
                    0, region.byte_offset // itemsize, region.nbytes // itemsize
                ).fill_(region.value)

    def validate_derived_state(self, model) -> None:
        for name in self.derived_state:
            module_name, _, attr = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            if not isinstance(
                getattr(module, attr, None), (torch.Tensor, bool, int, float)
            ):
                raise WeightManifestError(
                    f"derived weight state was not rebuilt: {name}"
                )


class WeightLayoutBuilder:
    """Account for every parameter/buffer without inspecting tensor values.

    Each declared region is a logical box stored as one contiguous byte range.
    Helpers emit a fixed number of boxes per fusion segment or expert; arbitrary
    tensor transforms and per-element fragmentation are deliberately unsupported.
    """

    def __init__(self, model: Any, *, allow_meta: bool = False):
        self.model = model
        self.allow_meta = allow_meta
        self.tensors = registered_tensors(model)
        self._names = {}
        for name, tensor in self.tensors.items():
            self._names.setdefault(_layout_storage_key(tensor), []).append(name)
        self._views = []
        self._initializations = []
        self._local_buffers = set()
        self._derived_state = []
        self._view_index = {}
        self._initialization_index = set()
        self._declarations = {}

    def _parameter_names(self, tensor):
        names = self._names.get(_layout_storage_key(tensor))
        if names is None:
            raise WeightManifestError(
                "layout region does not reference registered storage"
            )
        if (
            tensor.is_sparse
            or not tensor.is_contiguous()
            or (tensor.is_meta and not self.allow_meta)
        ):
            raise WeightManifestError(f"unsupported runtime storage: {names[0]}")
        return tuple(sorted(names))

    def region(
        self,
        tensor,
        tensor_id,
        *,
        global_shape,
        global_offset=None,
        local_shape=None,
        byte_offset=0,
        expert_id=None,
        partition=WeightPartition(),
        layout_fingerprint="sglang:row-major",
    ):
        global_shape = tuple(global_shape)
        local_shape = tuple(local_shape if local_shape is not None else tensor.shape)
        global_offset = tuple(
            global_offset if global_offset is not None else (0,) * len(global_shape)
        )
        if (
            not tensor_id
            or not layout_fingerprint
            or not (len(global_shape) == len(global_offset) == len(local_shape))
            or any(
                g <= 0 or n <= 0 or o < 0 or o + n > g
                for g, o, n in zip(global_shape, global_offset, local_shape)
            )
        ):
            raise WeightManifestError(f"invalid logical region: {tensor_id}")
        names = self._parameter_names(tensor)
        self._declare(
            tensor,
            tensor_id,
            global_shape=global_shape,
            partition=partition,
            expert_id=expert_id,
            layout_fingerprint=layout_fingerprint,
        )
        region = WeightRegion(
            tensor_id,
            global_shape,
            global_offset,
            local_shape,
            tensor,
            names,
            byte_offset,
            partition,
            expert_id,
            layout_fingerprint,
        )
        self._check_range(region)
        # The same exact storage can be exposed by multiple names (tied heads).
        key = (names, byte_offset, region.nbytes)
        for previous in self._view_index.get(key, ()):
            if (
                previous.tensor_id,
                previous.global_shape,
                previous.global_offset,
                previous.local_shape,
                previous.layout_fingerprint,
                previous.expert_id,
            ) == (
                tensor_id,
                global_shape,
                global_offset,
                local_shape,
                layout_fingerprint,
                expert_id,
            ):
                return
        self._views.append(region)
        self._view_index.setdefault(key, []).append(region)

    def declare(
        self,
        tensor,
        tensor_id,
        *,
        global_shape,
        partition=WeightPartition(),
        expert_id=None,
        layout_fingerprint="sglang:row-major",
    ):
        """Keep ownership even when this rank has only vocabulary padding."""
        self._parameter_names(tensor)
        self._declare(
            tensor,
            tensor_id,
            global_shape=global_shape,
            partition=partition,
            expert_id=expert_id,
            layout_fingerprint=layout_fingerprint,
        )

    def _declare(
        self,
        tensor,
        tensor_id,
        *,
        global_shape,
        partition=WeightPartition(),
        expert_id=None,
        layout_fingerprint="sglang:row-major",
    ):
        """Declare already validated storage without repeating its lookup."""
        shape = tuple(global_shape) or (1,)
        if (
            not tensor_id
            or any(n <= 0 for n in shape)
            or (partition.dim is not None and partition.dim >= len(shape))
            or (expert_id is not None and partition.kind != "expert")
        ):
            raise WeightManifestError(f"invalid tensor declaration: {tensor_id}")
        declaration = LogicalWeightTensor(
            tensor_id=tensor_id,
            global_shape=shape,
            dtype=str(tensor.dtype).removeprefix("torch."),
            itemsize=tensor.element_size(),
            partition=partition,
            layout_fingerprint=layout_fingerprint,
            expert_id=expert_id,
        )
        if self._declarations.setdefault(tensor_id, declaration) != declaration:
            raise WeightManifestError(f"conflicting tensor declaration: {tensor_id}")

    def initialize(self, tensor, *, byte_offset, nbytes, value=0):
        if nbytes == 0:
            return
        region = WeightInitialization(
            tensor, self._parameter_names(tensor), byte_offset, nbytes, value
        )
        self._check_range(region)
        key = (region.parameter_names, byte_offset, nbytes, value)
        if key not in self._initialization_index:
            self._initializations.append(region)
            self._initialization_index.add(key)

    @staticmethod
    def _check_range(region):
        itemsize = region.parameter.element_size()
        if (
            region.byte_offset < 0
            or region.nbytes <= 0
            or region.byte_offset % itemsize
            or region.nbytes % itemsize
            or region.byte_offset + region.nbytes > region.parameter.numel() * itemsize
        ):
            raise WeightManifestError(
                f"layout byte range is out of bounds: {region.parameter_names[0]}"
            )

    def local_buffer(self, name):
        try:
            buffer = self.model.get_buffer(name)
        except AttributeError:
            buffer = None
        if buffer is None:
            raise WeightManifestError(
                f"local initialization must name a buffer: {name}"
            )
        self._local_buffers.add(name)

    def derived(self, *names):
        for name in names:
            if name in self.tensors:
                raise WeightManifestError(
                    f"derived state must not replace registered storage: {name}"
                )
            self._derived_state.append(name)

    def build(self):
        # Exact aliases were grouped above. Different overlapping views cannot
        # independently own the same bytes, even if both declarations are full.
        intervals = sorted(
            (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                names[0],
            )
            for names in self._names.values()
            if (tensor := self.tensors[names[0]]).numel() and not tensor.is_meta
        )
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise WeightManifestError(
                    f"overlapping registered tensor storage: {previous[2]}, {current[2]}"
                )
        grouped, signatures = {}, {}
        for region in (*self._views, *self._initializations):
            identity = id(region.parameter)
            if identity not in signatures:
                signatures[identity] = _layout_storage_key(region.parameter)
            grouped.setdefault(signatures[identity], []).append(region)
        for key, names in self._names.items():
            local = set(names) & self._local_buffers
            if local:
                if local != set(names) or key in grouped:
                    raise WeightManifestError(
                        f"conflicting local buffer declaration: {names[0]}"
                    )
                continue
            tensor = self.tensors[names[0]]
            cursor = 0
            for region in sorted(grouped.get(key, ()), key=lambda r: r.byte_offset):
                if region.byte_offset != cursor:
                    raise WeightManifestError(
                        f"layout regions overlap or leave a gap: {names[0]}"
                    )
                cursor += region.nbytes
            if cursor != tensor.numel() * tensor.element_size():
                raise WeightManifestError(
                    f"layout does not cover complete tensor: {names[0]}"
                )
        plan = WeightLayoutPlan(
            tuple(self._declarations[name] for name in sorted(self._declarations)),
            tuple(self._views),
            tuple(self._initializations),
            frozenset(self._local_buffers),
            tuple(self._derived_state),
            {
                n: _layout_storage_key(t)
                for n, t in self.tensors.items()
                if n not in self._local_buffers
            },
        )
        plan.validate_bindings(self.model)
        return plan
