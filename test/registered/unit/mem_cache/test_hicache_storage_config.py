"""Argument-time backend detection agrees with both runtime config consumers."""

import json

import pytest

from sglang.srt.arg_groups.hicache_storage_config import (
    load_storage_backend_extra_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("file_config", [False, True])
def test_backend_detection_and_runtime_options_agree(tmp_path, file_config):
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )

    expected = {"kv_reshard": {"model_revision": "revision"}, "protocol": "rdma"}
    config = {
        **expected,
        "prefetch_threshold": 192,
        "prefetch_timeout_base": 2,
        "prefetch_timeout_per_ki_token": 0.5,
        "hicache_storage_pass_prefix_keys": True,
    }
    value = json.dumps(config)
    if file_config:
        path = tmp_path / "storage.json"
        path.write_text(value)
        value = "@" + str(path)
    assert load_storage_backend_extra_config(value) == config
    extra, threshold, timeout, prefix = (
        HiRadixCache._parse_storage_backend_extra_config(None, value)
    )
    assert extra == expected and threshold == 192 and prefix is True
    assert timeout.base == 2 and timeout.per_ki_token == 0.5
    assert HybridCacheController.parse_storage_backend_extra_config(value) == (
        expected,
        192,
        2.0,
        0.5,
        True,
    )
    # Runtime extraction must not mutate subsequent argument-time inspection.
    assert load_storage_backend_extra_config(value) == config


def test_invalid_file_type_cannot_silently_enable_reshard(tmp_path):
    path = tmp_path / "storage.txt"
    path.write_text('{"kv_reshard":{"model_revision":"r"}}')
    with pytest.raises(ValueError, match="Unsupported config file"):
        load_storage_backend_extra_config("@" + str(path))
