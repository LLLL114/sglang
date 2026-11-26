# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3 MoE layouts using the shared layer declarations."""

import torch

from ..weight_runtime_manifest import WeightManifestError
from .qwen3 import Qwen3WeightAdapter


class Qwen3MoeWeightAdapter(Qwen3WeightAdapter):
    def validate(self, model, context):
        super().validate(model, context)
        from sglang.srt.runtime_context import get_exec

        if get_exec().moe.enable_eplb or get_exec().moe.ep_num_redundant_experts:
            raise WeightManifestError("Qwen3 MoE reshard does not support EPLB")
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise WeightManifestError("Qwen3 MoE reshard requires BF16 weights")

    def _mlp(self, builder, module, name):
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
        from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        if type(module) is Qwen3MoeSparseMoeBlock:
            builder.replicated_linear(module.gate, name + ".gate")
            builder.fused_moe(module.experts, name + ".experts")
        elif type(module) is Qwen2MoeMLP:
            builder.mlp(module, name)
        else:
            raise WeightManifestError(f"unsupported Qwen3 MoE MLP: {name}")
