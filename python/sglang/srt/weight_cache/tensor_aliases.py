# SPDX-License-Identifier: Apache-2.0
"""Preserve direct tensor consumers when registered storage is replaced."""

import torch
from torch import nn


def capture_tensor_aliases(model, registered):
    from dataclasses import fields, is_dataclass

    names = {id(tensor): name for name, tensor in registered.items()}
    aliases, visited = [], set()

    def visit(obj):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        attributes = (
            vars(obj).items()
            if isinstance(obj, nn.Module)
            else ((field.name, getattr(obj, field.name)) for field in fields(obj))
        )
        for key, value in attributes:
            if isinstance(value, torch.Tensor) and id(value) in names:
                aliases.append((obj, key, names[id(value)], value))
            elif not isinstance(value, type) and is_dataclass(value):
                visit(value)

    for module in model.modules():
        visit(module)
    # Break the recursive closure cycle so consumed aliases release immediately.
    visit = None
    return aliases


def rebind_tensor_aliases(aliases, replacements):
    for obj, key, name, previous in aliases:
        if name in replacements and getattr(obj, key) is previous:
            # Runtime configs may be frozen dataclasses holding registered tensors.
            object.__setattr__(obj, key, replacements[name])
