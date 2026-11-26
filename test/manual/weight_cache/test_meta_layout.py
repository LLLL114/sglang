# SPDX-License-Identifier: Apache-2.0
"""Target logical layouts must not allocate or publish meta storage."""

import pytest
import torch

from sglang.srt.weight_cache.weight_layout import WeightLayoutBuilder
from sglang.srt.weight_cache.weight_runtime_manifest import (
    ImmutableWeightRuntimeManifestBuilder,
    WeightManifestError,
    WeightParallelTopology,
)


def test_meta_layout_keeps_distinct_storage_and_refuses_physical_export():
    model = torch.nn.Linear(4, 4, device="meta", dtype=torch.bfloat16)
    builder = WeightLayoutBuilder(model, allow_meta=True)
    builder.region(model.weight, "weight", global_shape=(4, 4))
    builder.region(model.bias, "bias", global_shape=(4,))
    plan = builder.build()
    assert len(plan.bindings) == 2
    with pytest.raises(WeightManifestError, match="unsupported runtime devices"):
        ImmutableWeightRuntimeManifestBuilder(
            model=model, layout_plan=plan, topology=WeightParallelTopology()
        ).build(
            model_id="m", revision="r", instance_id="i", worker_id="w", endpoint="e"
        )
    with pytest.raises(WeightManifestError, match="unsupported runtime storage"):
        WeightLayoutBuilder(model).region(model.weight, "weight", global_shape=(4, 4))
