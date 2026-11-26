# SPDX-License-Identifier: Apache-2.0
"""Run CPU leaf tests without frontend or GPU-kernel package initializers."""

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "python"))
for name in ("sglang", "sglang.srt.distributed", "sglang.srt.utils"):
    module = types.ModuleType(name)
    module.__path__ = [str(root / "python" / name.replace(".", "/"))]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
