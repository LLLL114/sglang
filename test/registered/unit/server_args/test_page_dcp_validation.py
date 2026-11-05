# SPDX-License-Identifier: Apache-2.0
"""The page-DCP switch must refuse combinations outside its validated route."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import kv_shard_hook
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return NS(
        device="cuda",
        model_impl="auto",
        nnodes=1,
        pp_size=1,
        dp_size=1,
        attn_dp_size=1,
        attn_cp_size=1,
        tp_size=2,
        dcp_size=2,
        kv_cache_dtype="auto",
        page_size=64,
        chunked_prefill_size=2048,
        disaggregation_mode="null",
        dcp_comm_backend="ag_rs",
        cuda_graph_config=NS(
            prefill=NS(backend=Backend.DISABLED), decode=NS(backend=Backend.DISABLED)
        ),
        disable_overlap_schedule=True,
        enable_prefill_cp=False,
        cp_kv_layout=None,
        enable_kv_cache_sharding=False,
        enable_hierarchical_cache=False,
        enable_unified_memory=False,
        enable_hisparse=False,
        enable_lmcache=False,
        enable_flexkv=False,
        enable_unified_cache_external_linker=False,
        enable_lora=False,
        enable_mixed_chunk=False,
        enable_two_batch_overlap=False,
        enable_pd_role_switch=False,
        enable_streaming_session=False,
        enable_session_radix_cache=False,
        speculative_algorithm=None,
        dllm_algorithm=None,
        quantization=None,
        radix_cache_backend=None,
    )


class TestPageDcpValidation(CustomTestCase):
    def _validate(
        self,
        cfg,
        dtype=torch.bfloat16,
        arch="Qwen3ForCausalLM",
        backends=("triton", "triton"),
    ):
        model = NS(dtype=dtype, hf_config=NS(architectures=[arch]))
        with (
            patch.object(kv_shard_hook, "model_config_of", return_value=model),
            patch.object(kv_shard_hook, "resolved_view", return_value=cfg),
            patch.object(kv_shard_hook, "attention_backends_of", return_value=backends),
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
        ):
            kv_shard_hook.validate_page_dcp(None, cfg)

    def test_accepts_eager_qwen3(self):
        self._validate(_config())

    def test_rejects_unsupported_compositions(self):
        cases = [
            ("dcp_size", 1),
            ("tp_size", 3),
            ("nnodes", 2),
            ("pp_size", 2),
            ("model_impl", "transformers"),
            ("attn_dp_size", 2),
            ("attn_cp_size", 2),
            ("page_size", 128),
            ("chunked_prefill_size", 65),
            ("disable_overlap_schedule", False),
            ("enable_prefill_cp", True),
            ("enable_hierarchical_cache", True),
            ("enable_unified_memory", True),
            ("enable_mixed_chunk", True),
            ("enable_pd_role_switch", True),
            ("speculative_algorithm", "EAGLE"),
            ("disaggregation_mode", "decode"),
            ("kv_cache_dtype", "fp8_e4m3"),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                cfg = _config()
                setattr(cfg, name, value)
                with self.assertRaises(ValueError):
                    self._validate(cfg)

    def test_rejects_graphs_and_other_models_or_backends(self):
        cfg = _config()
        cfg.cuda_graph_config.decode.backend = Backend.FULL
        with self.assertRaises(ValueError):
            self._validate(cfg)
        for kwargs in (
            dict(dtype=torch.float16),
            dict(arch="GlmMoeDsaForCausalLM"),
            dict(backends=("fa3", "triton")),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._validate(_config(), **kwargs)


if __name__ == "__main__":
    unittest.main()
