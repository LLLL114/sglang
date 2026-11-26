# SPDX-License-Identifier: Apache-2.0
"""Native loader mappings must agree with canonical expert declarations."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("triton", reason="native SGLang layer imports require Triton")

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    _compute_logical_to_all_physical_map,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_resources
from sglang.srt.weight_cache.model_adapters.layers import LayerLayoutBuilder
from sglang.srt.weight_cache.weight_runtime_manifest import WeightManifestError


def _metadata(mapping, ep_size):
    inverse = _compute_logical_to_all_physical_map(mapping, 4, ep_size, None)
    with patch(
        "sglang.srt.eplb.expert_location.get_exec",
        return_value=SimpleNamespace(moe=SimpleNamespace(ep_dispatch_algorithm=None)),
    ):
        return ExpertLocationMetadata._init_raw(
            ep_size, mapping, inverse, moe_ep_rank=0
        )


def _module(ep_size, ep_rank, tp_size, tp_rank):
    module = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(module)
    with patch(
        "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
        return_value=MoeRunnerBackend.TRITON,
    ):
        method = UnquantizedFusedMoEMethod()
    method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
    module.quant_method, module.quant_config = method, None
    module.uses_expert_location = True
    module.num_fused_shared_experts, module._has_fused_shared = 0, False
    module._num_global_routed = 4
    module._num_local_routed = module.num_local_experts = 4 // ep_size
    module._expert_storage_rank = module.moe_ep_rank = ep_rank
    module.moe_ep_size = ep_size
    module.moe_tp_rank, module.moe_tp_size = tp_rank, tp_size
    module.hidden_size = 4
    module.layer_id = 1
    module.intermediate_size_per_partition = 8 // tp_size
    module.use_flashinfer_trtllm_moe = module.use_presharded_weights = False
    module.moe_runner_config = SimpleNamespace(is_gated=True)
    module.w13_weight = torch.nn.Parameter(
        torch.zeros(4 // ep_size, 16 // tp_size, 4, dtype=torch.bfloat16)
    )
    module.w2_weight = torch.nn.Parameter(
        torch.zeros(4 // ep_size, 4, 8 // tp_size, dtype=torch.bfloat16)
    )
    return module


@pytest.mark.parametrize(
    "ep_size,tp_size,metadata_ep", [(1, 4, 4), (2, 2, 1), (4, 1, 2)]
)
def test_identity_metadata_matches_native_loader_across_ep_contexts(
    monkeypatch, ep_size, tp_size, metadata_ep
):
    mapping = torch.arange(4).repeat(2, 1)
    metadata = _metadata(mapping, metadata_ep)
    monkeypatch.setattr(get_resources(), "expert_location_metadata", metadata)
    for ep_rank in range(ep_size):
        for tp_rank in range(tp_size):
            module = _module(ep_size, ep_rank, tp_size, tp_rank)
            checkpoint = {}
            with (
                torch.no_grad(),
                patch(
                    "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                    return_value=SimpleNamespace(
                        is_flashinfer_trtllm_routed=lambda: False
                    ),
                ),
            ):
                for expert in range(4):
                    for component, (part, shard, shape) in enumerate(
                        (
                            ("gate_proj", "w1", (8, 4)),
                            ("up_proj", "w3", (8, 4)),
                            ("down_proj", "w2", (4, 8)),
                        )
                    ):
                        values = (
                            torch.arange(32).reshape(shape)
                            + expert * 32
                            + component * 16
                        ).to(torch.bfloat16)
                        checkpoint[f"experts.{expert}.{part}.weight"] = values
                        parameter = (
                            module.w2_weight if shard == "w2" else module.w13_weight
                        )
                        module.weight_loader(parameter, values, "weight", shard, expert)
                expected = {n: p.detach().clone() for n, p in module.named_parameters()}
                # A target meta construction context must not turn the CPU identity into meta.
                with torch.device("meta"):
                    builder = LayerLayoutBuilder(module)
                    builder.fused_moe(module, "experts")
                    plan = builder.build()
                for parameter in module.parameters():
                    parameter.fill_(-1)
                for view in plan.views:
                    slices = tuple(
                        slice(o, o + n)
                        for o, n in zip(view.global_offset, view.local_shape)
                    )
                    values = checkpoint[view.tensor_id][slices].contiguous().flatten()
                    view.parameter.flatten().narrow(
                        0, view.byte_offset // 2, values.numel()
                    ).copy_(values)
                for name, parameter in module.named_parameters():
                    assert torch.equal(parameter, expected[name])


@pytest.mark.parametrize(
    "kind", ["swapped-remote-slots", "redundant", "missing-layer", "negative-layer"]
)
def test_nonidentity_native_metadata_rejected_without_mutation(monkeypatch, kind):
    mapping = torch.arange(4).repeat(2, 1)
    if kind == "swapped-remote-slots":
        mapping[1, 2:] = torch.tensor([3, 2])
    elif kind == "redundant":
        mapping = torch.cat((mapping, torch.zeros(2, 1, dtype=mapping.dtype)), dim=1)
    metadata = _metadata(mapping, 1)
    monkeypatch.setattr(get_resources(), "expert_location_metadata", metadata)
    module = _module(4, 0, 1, 0)
    if kind == "missing-layer":
        module.layer_id = 2
    elif kind == "negative-layer":
        module.layer_id = -1
    before = {
        name: (parameter.data_ptr(), parameter.detach().clone())
        for name, parameter in module.named_parameters()
    }
    with pytest.raises(WeightManifestError, match="unsupported expert mapping"):
        LayerLayoutBuilder(module).fused_moe(module, "experts")
    for name, parameter in module.named_parameters():
        assert before[name][0] == parameter.data_ptr()
        assert torch.equal(before[name][1], parameter)
