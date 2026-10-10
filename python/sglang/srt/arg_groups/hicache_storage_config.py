# SPDX-License-Identifier: Apache-2.0
"""Read the same HiCache backend configuration at argument and attach time."""

from __future__ import annotations

import json
from pathlib import Path


def load_storage_backend_extra_config(value: str | None) -> dict:
    if not value:
        return {}
    if not value.startswith("@"):
        result = json.loads(value)
    else:
        path = Path(value[1:])
        ext = path.suffix.lower()
        with path.open("rb" if ext == ".toml" else "r") as stream:
            if ext == ".json":
                result = json.load(stream)
            elif ext == ".toml":
                try:
                    import tomllib
                except ImportError:  # Python 3.10
                    import tomli as tomllib
                result = tomllib.load(stream)
            elif ext in (".yaml", ".yml"):
                import yaml

                result = yaml.safe_load(stream)
            else:
                raise ValueError(
                    f"Unsupported config file {path} (config format: {ext})"
                )
    if not isinstance(result, dict):
        raise ValueError("HiCache storage backend extra config must be a mapping")
    return result
