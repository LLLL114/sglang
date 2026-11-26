"""Multimem capability guards must agree before allocating symmetric payloads."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as ag
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMultimemCapabilityGuard(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            device_group=object(),
            cpu_group=object(),
            all_gather_object=Mock(return_value=[False, False]),
        )
        self.stack.enter_context(
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=self.group,
            )
        )
        self.stack.enter_context(
            patch.object(ag, "get_parallel", return_value=SimpleNamespace(nnodes=1))
        )
        self.stack.enter_context(
            patch.object(ag, "is_gfx95_supported", return_value=False)
        )
        self.cuda = self.stack.enter_context(
            patch.object(ag, "is_cuda", return_value=True)
        )
        self.stack.enter_context(
            patch("torch.cuda.is_current_stream_capturing", return_value=False)
        )
        self.support = self.stack.enter_context(
            patch(
                "sglang.srt.distributed.device_communicators.multicast_all_gather.has_multicast_support",
                return_value=False,
            )
        )
        self.create = self.stack.enter_context(
            patch.object(ag, "create_state", return_value=self.state(multicast_ptr=0))
        )
        self.x = torch.ones((1, 8), dtype=torch.bfloat16)
        self.expected = torch.cat([self.x, self.x], dim=-1)
        self.fallback = self.stack.enter_context(
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_gather",
                return_value=self.expected,
            )
        )
        self.inner = self.stack.enter_context(
            patch.object(ag, "all_gather_inner", return_value=self.expected)
        )

    def state(self, multicast_ptr=1):
        return SimpleNamespace(
            symm_mem_hdl=SimpleNamespace(multicast_ptr=multicast_ptr),
            max_token_num=128,
            world_size=2,
            hidden_dim=16,
        )

    def test_unsupported_device_falls_back_without_payload_and_caches_result(self):
        gather = ag.MultimemAllGatherer(128)
        for _ in range(2):
            self.assertIs(gather(self.x), self.expected)
        self.create.assert_not_called()
        self.support.assert_called_once_with(self.x.device)
        self.group.all_gather_object.assert_called_once_with(False)
        self.inner.assert_not_called()
        self.assertEqual(self.fallback.call_count, 2)

    def test_mixed_capabilities_fall_back_collectively(self):
        self.support.return_value = True
        self.group.all_gather_object.return_value = [True, False]
        self.assertIs(ag.MultimemAllGatherer(128)(self.x), self.expected)
        self.group.all_gather_object.assert_called_once_with(True)
        self.create.assert_not_called()

    def test_driver_error_still_votes_false(self):
        self.support.side_effect = RuntimeError("driver query failed")
        self.assertIs(ag.MultimemAllGatherer(128)(self.x), self.expected)
        self.group.all_gather_object.assert_called_once_with(False)
        self.create.assert_not_called()

    def test_supported_mapping_keeps_original_multimem_path(self):
        self.support.return_value = True
        self.group.all_gather_object.return_value = [True, True]
        self.create.return_value = self.state()
        gather = ag.MultimemAllGatherer(128)
        self.assertIs(gather(self.x), self.expected)
        self.assertIs(gather(self.x), self.expected)
        self.create.assert_called_once_with(
            group=self.group.device_group,
            rank_in_group=0,
            max_tokens=128,
            hidden_size=16,
        )
        self.assertEqual(self.inner.call_count, 2)
        self.fallback.assert_not_called()

    def test_supported_device_without_mapping_keeps_original_fallback(self):
        self.support.return_value = True
        self.group.all_gather_object.return_value = [True, True]
        self.create.return_value = self.state(multicast_ptr=0)
        self.assertIs(ag.MultimemAllGatherer(128)(self.x), self.expected)
        self.create.assert_called_once()
        self.inner.assert_not_called()
        self.fallback.assert_called_once_with(self.x, dim=-1)

    def test_single_rank_skips_capability_query_and_vote(self):
        self.group.world_size = 1
        self.assertIs(ag.MultimemAllGatherer(128)(self.x), self.expected)
        self.support.assert_not_called()
        self.group.all_gather_object.assert_not_called()
        self.create.assert_not_called()

    def test_capture_defers_query_and_allocation(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            gather = ag.MultimemAllGatherer(128)
            self.assertIs(gather(self.x), self.expected)
        self.assertIs(gather._state, gather._UNINIT)
        self.support.assert_not_called()
        self.group.all_gather_object.assert_not_called()
        self.create.assert_not_called()

    def test_invalid_input_skips_query_and_allocation(self):
        for x in (self.x.float(), self.x.flatten(), self.x[:, :7]):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assertIs(ag.MultimemAllGatherer(128)(x), self.expected)
        self.support.assert_not_called()
        self.group.all_gather_object.assert_not_called()
        self.create.assert_not_called()

    def test_non_cuda_keeps_original_build_path(self):
        self.cuda.return_value = False
        self.create.return_value = self.state(multicast_ptr=0)
        self.assertIs(ag.MultimemAllGatherer(128)(self.x), self.expected)
        self.support.assert_not_called()
        self.group.all_gather_object.assert_not_called()
        self.create.assert_called_once()

    def test_disabled_gatherer_keeps_original_fallback(self):
        self.assertIs(ag.MultimemAllGatherer(128, enabled=False)(self.x), self.expected)
        self.support.assert_not_called()
        self.group.all_gather_object.assert_not_called()
        self.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
