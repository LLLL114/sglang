# SPDX-License-Identifier: Apache-2.0
"""Registered replacement must preserve frozen runtime tensor consumers."""

from dataclasses import dataclass

import torch

from sglang.srt.weight_cache.tensor_aliases import (
    capture_tensor_aliases,
    rebind_tensor_aliases,
)


def test_direct_frozen_consumer_uses_replacement():
    @dataclass(frozen=True)
    class Runtime:
        weight: torch.Tensor

    model = torch.nn.Linear(4, 4, bias=False)
    model.runtime = Runtime(model.weight)
    aliases = capture_tensor_aliases(model, dict(model.named_parameters()))
    replacement = torch.nn.Parameter(torch.ones(4, 4))
    model.weight = replacement
    rebind_tensor_aliases(aliases, {"weight": replacement})
    assert model.runtime.weight is replacement
    assert torch.equal(model.runtime.weight @ torch.ones(4), torch.full((4,), 4.0))
