# SPDX-License-Identifier: Apache-2.0
"""Private grouped wire manifests derived from validated canonical views."""

from collections import defaultdict
from dataclasses import replace as replace_plan

from msgspec.structs import replace

from ..weight_layout import WeightLayoutBuilder
from ..weight_runtime_manifest import (
    LogicalWeightTensor,
    RuntimeWeightFragment,
    WeightManifestError,
)


def alias_names(source_plan, target_plan):
    def signatures(plan):
        grouped = defaultdict(list)
        for view in plan.views:
            grouped[view.parameter_names[0]].append(
                (
                    view.tensor_id,
                    view.global_shape,
                    view.global_offset,
                    view.local_shape,
                    view.byte_offset,
                    view.layout_fingerprint,
                    str(view.parameter.dtype),
                    tuple(view.parameter.shape),
                )
            )
        for region in plan.initializations:
            grouped[region.parameter_names[0]].append(
                ("initialize", region.byte_offset, region.nbytes, region.value)
            )
        return {name: tuple(values) for name, values in grouped.items()}

    source, target = signatures(source_plan), signatures(target_plan)
    return {name for name in target if source.get(name) == target[name]}


def derive_wire_manifest(manifest, plan, args, peer_args, config, *, exclude=()):
    declarations, fragments = {}, []

    def emit(view, name, shape, offset, local, byte_offset, partition, expert=None):
        tensor = view.parameter
        declaration = LogicalWeightTensor(
            tensor_id=name,
            global_shape=shape,
            dtype=str(tensor.dtype).removeprefix("torch."),
            itemsize=tensor.element_size(),
            partition=partition,
            expert_id=expert,
            layout_fingerprint="sglang:inplace-grouped-block-v1",
        )
        if declarations.setdefault(name, declaration) != declaration:
            raise WeightManifestError("inconsistent grouped wire declaration")
        if view.parameter_names[0] in exclude:
            return
        storage = tensor.untyped_storage()
        address = tensor.data_ptr() + byte_offset
        size = tensor.element_size()
        for extent in local:
            size *= extent
        if address + size > storage.data_ptr() + storage.nbytes():
            raise WeightManifestError("grouped wire exceeds registered storage")
        fragments.append(
            RuntimeWeightFragment(
                tensor_id=name,
                global_offset=offset,
                local_shape=local,
                address=address,
                nbytes=size,
                storage_address=storage.data_ptr(),
                storage_nbytes=storage.nbytes(),
                storage_offset_bytes=address - storage.data_ptr(),
            )
        )

    _visit_wire_regions(plan, args, peer_args, config, emit)
    return replace(
        manifest, tensors=tuple(declarations.values()), fragments=tuple(fragments)
    )


def build_wire_plan(model, plan, args, peer_args, config):
    """Lower validated canonical views before retaining same-epoch meta geometry."""
    builder = WeightLayoutBuilder(
        model,
        allow_meta=any(
            v.parameter.is_meta for v in (*plan.views, *plan.initializations)
        ),
    )

    def emit(view, name, shape, offset, local, byte_offset, partition, expert=None):
        builder.region(
            view.parameter,
            name,
            global_shape=shape,
            global_offset=offset,
            local_shape=local,
            byte_offset=byte_offset,
            partition=partition,
            expert_id=expert,
            layout_fingerprint="sglang:inplace-grouped-block-v1",
        )

    _visit_wire_regions(plan, args, peer_args, config, emit)
    for region in plan.initializations:
        builder.initialize(
            region.parameter,
            byte_offset=region.byte_offset,
            nbytes=region.nbytes,
            value=region.value,
        )
    for name in plan.local_buffers:
        builder.local_buffer(name)
    builder.derived(*plan.derived_state)
    result = builder.build()
    represented = {v.tensor_id for v in plan.views}
    padding = tuple(
        replace(t, layout_fingerprint="sglang:inplace-grouped-block-v1")
        for t in plan.tensors
        if t.tensor_id not in represented
    )
    declarations = {t.tensor_id: t for t in result.tensors}
    for tensor in padding:
        if declarations.setdefault(tensor.tensor_id, tensor) != tensor:
            raise WeightManifestError("conflicting padding-only wire declaration")
    return replace_plan(
        result, tensors=tuple(declarations[name] for name in sorted(declarations))
    )


def exclude_parameters(manifest, plan, names):
    """Exclude only fragments owned by the named, already validated parameters."""
    excluded = {
        (
            v.tensor_id,
            v.global_offset,
            v.local_shape,
            v.parameter.data_ptr() + v.byte_offset,
        )
        for v in plan.views
        if v.parameter_names[0] in names
    }
    return replace(
        manifest,
        fragments=tuple(
            f
            for f in manifest.fragments
            if (f.tensor_id, f.global_offset, f.local_shape, f.address) not in excluded
        ),
    )


def _visit_wire_regions(plan, args, peer_args, config, emit):
    groups = defaultdict(list)
    for view in plan.views:
        groups[view.parameter_names[0]].append(view)
    for parameter_name, views in groups.items():
        view = views[0]
        if parameter_name.endswith(("experts.w13_weight", "experts.w2_weight")):
            ordered = sorted(views, key=lambda v: v.byte_offset)
            experts = sorted({v.expert_id for v in ordered})
            if experts != list(range(experts[0], experts[0] + len(experts))):
                raise WeightManifestError("grouped wire requires consecutive experts")
            intermediate = config.moe_intermediate_size
            hidden = config.hidden_size
            local_i = intermediate // (args.tp_size // args.ep_size)
            start = view.partition.rank * local_i
            local_e = len(experts)
            is_gate = parameter_name.endswith("w13_weight")
            expected_count = 2 * local_e if is_gate else local_e
            if len(ordered) != expected_count:
                raise WeightManifestError("incomplete canonical expert inventory")
            for index, region in enumerate(ordered):
                expert = experts[index // 2] if is_gate else experts[index]
                suffix = (
                    ("gate_proj.weight", "up_proj.weight")[index % 2]
                    if is_gate
                    else "down_proj.weight"
                )
                shape = (local_i, hidden) if is_gate else (hidden, local_i)
                offset = (start, 0) if is_gate else (0, start)
                if (
                    region.expert_id != expert
                    or not region.tensor_id.endswith(suffix)
                    or region.local_shape != shape
                    or region.global_offset != offset
                    or region.byte_offset
                    != index * local_i * hidden * region.parameter.element_size()
                ):
                    raise WeightManifestError("unsupported canonical expert byte order")
            partition = replace(view.partition, dim=2)
            name = parameter_name + ".grouped"
            if is_gate:
                emit(
                    view,
                    name,
                    (config.num_experts, 2, intermediate, hidden),
                    (experts[0], 0, start, 0),
                    (local_e, 2, local_i, hidden),
                    0,
                    partition,
                )
            else:
                blocks = max(1, args.ep_size // peer_args.ep_size)
                if local_i % blocks:
                    raise WeightManifestError("expert width must divide wire blocks")
                chunk = local_i // blocks
                for block in range(blocks):
                    emit(
                        view,
                        name,
                        (config.num_experts, hidden, intermediate),
                        (experts[0], 0, start + block * chunk),
                        (local_e, hidden, chunk),
                        block
                        * local_e
                        * hidden
                        * chunk
                        * view.parameter.element_size(),
                        partition,
                    )
        else:
            for view in views:
                blocks = (
                    max(1, args.dp_size // peer_args.dp_size)
                    if "self_attn.o_proj." in view.tensor_id
                    else 1
                )
                rows, width = view.local_shape if blocks > 1 else (0, 0)
                if blocks > 1 and width % blocks:
                    raise WeightManifestError("attention width must divide wire blocks")
                for block in range(blocks):
                    local = (rows, width // blocks) if blocks > 1 else view.local_shape
                    offset = (
                        (
                            view.global_offset[0],
                            view.global_offset[1] + block * width // blocks,
                        )
                        if blocks > 1
                        else view.global_offset
                    )
                    byte_offset = view.byte_offset + (
                        block * rows * width // blocks * view.parameter.element_size()
                        if blocks > 1
                        else 0
                    )
                    emit(
                        view,
                        view.tensor_id,
                        view.global_shape,
                        offset,
                        local,
                        byte_offset,
                        view.partition,
                        view.expert_id,
                    )


def close_inventory(sources, targets):
    needed = {f.tensor_id for manifest in targets for f in manifest.fragments}
    if not needed:
        return (), ()

    def retain(manifest):
        return replace(
            manifest,
            tensors=tuple(t for t in manifest.tensors if t.tensor_id in needed),
            fragments=tuple(f for f in manifest.fragments if f.tensor_id in needed),
        )

    return tuple(map(retain, sources)), tuple(map(retain, targets))
