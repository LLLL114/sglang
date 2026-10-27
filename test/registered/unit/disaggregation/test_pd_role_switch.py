import concurrent.futures
import signal
import threading
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import torch
import torch.distributed as dist

from sglang.srt import runtime_context as rc  # noqa: E402
from sglang.srt.disaggregation import role_switch, role_switch_pp
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers import scheduler_pp_mixin as pp
from sglang.srt.managers.io_struct import (  # noqa: E402
    PdRoleSwitchReqInput,
    PdRoleSwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


class TestHandlePdRoleSwitch(unittest.TestCase):
    """Cover the control-plane contract of Scheduler.handle_pd_role_switch.

    Only the role-flip *decision* logic is exercised here (no GPU): the heavy
    teardown/rebuild is mocked, so this asserts the guard branches and the
    orchestration order without standing up a model.
    """

    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def _scheduler(self, mode, *, enable=True, idle=True):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = mode
        sa = ServerArgs(
            model_path="dummy",
            disaggregation_mode=mode.value,
            enable_pd_role_switch=enable,
        )
        rc.get_context().set_server_args(sa)
        s.server_args = sa
        s.is_fully_idle = MagicMock(return_value=idle)
        teardown_patcher = patch.object(role_switch, "teardown_disaggregation")
        s.teardown_disaggregation = teardown_patcher.start()
        self.addCleanup(teardown_patcher.stop)
        s.init_disaggregation = MagicMock()
        s._sync_disaggregation_mode_to_subcomponents = MagicMock()
        s._event_loop_should_restart = False
        s._pd_role_switch_in_progress = False
        s._pd_role_switch_unhealthy = False
        s.tp_worker = MagicMock()
        return s

    def test_preflight_checks_graph_headroom_without_releasing_resources(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.device = "cuda"
        cases = (
            ("missing", None, [], 2.0, False, "is required"),
            ("negative", -1.0, [], 2.0, False, "non-negative"),
            ("insufficient", 1.0, [], 0.5, False, "insufficient"),
            (
                "probe failure",
                1.0,
                [],
                RuntimeError("probe failed"),
                False,
                "failed to check",
            ),
            ("sufficient", 1.0, [], 1.0, True, "ready"),
            ("already captured", None, [1], 0.0, True, "ready"),
        )
        for name, budget, captured_bs, memory, success, message in cases:
            with self.subTest(name=name):
                s.tp_worker.get_decode_cuda_graph_bs.return_value = captured_bs
                with patch.object(
                    role_switch, "get_available_gpu_memory"
                ) as memory_probe:
                    if isinstance(memory, Exception):
                        memory_probe.side_effect = memory
                    else:
                        memory_probe.return_value = memory
                    out = Scheduler.handle_pd_role_switch(
                        s,
                        PdRoleSwitchReqInput(
                            new_role="decode",
                            prepare_only=True,
                            decode_cuda_graph_memory_gb=budget,
                        ),
                    )
                    self.assertEqual(out.success, success)
                    self.assertTrue(out.safe_to_restore)
                    self.assertIn(message, out.message)
                    if not captured_bs and budget is not None and budget >= 0:
                        memory_probe.assert_called_once()
                    else:
                        memory_probe.assert_not_called()
                s.tp_worker.ensure_decode_cuda_graphs.assert_not_called()
                s.teardown_disaggregation.assert_not_called()
                s.init_disaggregation.assert_not_called()
                self.assertEqual(rc.get_disagg().disaggregation_mode, "prefill")
                self.assertFalse(s._event_loop_should_restart)

    def test_rejected_when_flag_disabled(self):
        s = self._scheduler(DisaggregationMode.PREFILL, enable=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertIsInstance(out, PdRoleSwitchReqOutput)
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("enable-pd-role-switch", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_on_invalid_role(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="both"))
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("invalid new_role", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_not_in_pd_mode(self):
        s = self._scheduler(DisaggregationMode.NULL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("not running in PD", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_same_role_is_noop(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.message, "already in target role")
        s.teardown_disaggregation.assert_not_called()
        s.init_disaggregation.assert_not_called()
        self.assertFalse(s._event_loop_should_restart)

    def test_rejected_when_not_idle(self):
        s = self._scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("not idle", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_decode_graph_headroom_is_missing(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.tp_worker.get_decode_cuda_graph_bs.return_value = []
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("decode_cuda_graph_memory_gb is required", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_decode_graph_headroom_is_insufficient(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.device = "cuda"
        s.tp_worker.get_decode_cuda_graph_bs.return_value = []
        with patch.object(role_switch, "get_available_gpu_memory", return_value=0.5):
            out = Scheduler.handle_pd_role_switch(
                s,
                PdRoleSwitchReqInput(
                    new_role="decode",
                    decode_cuda_graph_memory_gb=1.0,
                ),
            )

        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("insufficient decode CUDA graph headroom", out.message)
        s.teardown_disaggregation.assert_not_called()
        self.assertEqual(rc.get_disagg().disaggregation_mode, "prefill")

    def test_decode_graph_headroom_allows_flip(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.device = "cuda"
        s.tp_worker.get_decode_cuda_graph_bs.return_value = []
        with patch.object(role_switch, "get_available_gpu_memory", return_value=1.0):
            out = Scheduler.handle_pd_role_switch(
                s,
                PdRoleSwitchReqInput(
                    new_role="decode",
                    decode_cuda_graph_memory_gb=1.0,
                ),
            )

        self.assertTrue(out.success)
        s.teardown_disaggregation.assert_called_once_with(s)

    def test_successful_flip_orchestration(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertTrue(out.success)
        self.assertEqual(out.old_role, "prefill")
        self.assertEqual(out.new_role, "decode")
        # Orchestration: drain -> teardown -> flip config bag -> rebuild -> signal.
        s.teardown_disaggregation.assert_called_once_with(s)
        self.assertEqual(rc.get_disagg().disaggregation_mode, "decode")
        # The pristine startup record is never mutated.
        self.assertEqual(s.server_args.disaggregation_mode, "prefill")
        s.init_disaggregation.assert_called_once()
        s._sync_disaggregation_mode_to_subcomponents.assert_called_once()
        self.assertTrue(s._event_loop_should_restart)
        # Flip to decode ensures decode CUDA graphs exist (idempotent capture).
        s.tp_worker.ensure_decode_cuda_graphs.assert_called_once()
        # The in-progress guard is released after a successful flip.
        self.assertFalse(s._pd_role_switch_in_progress)

    def test_flip_to_prefill_skips_decode_graph_capture(self):
        s = self._scheduler(DisaggregationMode.DECODE)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.new_role, "prefill")
        s.init_disaggregation.assert_called_once()
        # Flipping to prefill must not capture decode graphs.
        s.tp_worker.ensure_decode_cuda_graphs.assert_not_called()
        self.assertTrue(s._event_loop_should_restart)

    def test_rejected_when_switch_in_progress(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_in_progress = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertFalse(out.safe_to_restore)
        self.assertIn("in progress", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_unhealthy(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_unhealthy = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertFalse(out.safe_to_restore)
        self.assertIn("unhealthy", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rebuild_failure_marks_unhealthy_and_notifies(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        # Rebuild of the new role fails after the old role was torn down.
        s.init_disaggregation = MagicMock(side_effect=RuntimeError("boom"))

        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        # Fail loud (notify), mark unhealthy, no in-place rollback attempt.
        self.assertFalse(out.success)
        self.assertFalse(out.safe_to_restore)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown + rebuild attempted exactly once (no rollback).
        self.assertEqual(s.teardown_disaggregation.call_count, 1)
        self.assertEqual(s.init_disaggregation.call_count, 1)
        s._sync_disaggregation_mode_to_subcomponents.assert_not_called()
        # A subsequent switch is rejected because the instance is unhealthy.
        out2 = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertFalse(out2.success)
        self.assertIn("unhealthy", out2.message)

    def test_teardown_failure_marks_unhealthy(self):
        """Teardown, the role flip and rebuild are one atomic step: a failure
        during teardown (not only rebuild) must also mark the instance unhealthy
        and must not proceed to rebuild."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.teardown_disaggregation.side_effect = RuntimeError("boom")

        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertFalse(out.success)
        self.assertFalse(out.safe_to_restore)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown raised, so rebuild is never attempted.
        self.assertEqual(s.teardown_disaggregation.call_count, 1)
        s.init_disaggregation.assert_not_called()
        s._sync_disaggregation_mode_to_subcomponents.assert_not_called()


class TestPdRoleSwitchReqSerialization(unittest.TestCase):
    """Guard the wire contract of the /pd_role_switch req/resp structs.

    These caught real breakages when upstream moved BaseReq to msgspec: the
    request must accept an optional decode_cuda_graph_bs body field, and the
    response must be encodable for the HTTP layer (msgspec_to_builtins).
    """

    def test_req_accepts_optional_decode_cuda_graph_bs(self):
        req = PdRoleSwitchReqInput(
            new_role="decode",
            decode_cuda_graph_bs=[1, 2, 4],
            decode_cuda_graph_memory_gb=1.25,
        )
        self.assertEqual(req.new_role, "decode")
        self.assertEqual(req.decode_cuda_graph_bs, [1, 2, 4])
        self.assertEqual(req.decode_cuda_graph_memory_gb, 1.25)
        # Field is optional and defaults to None.
        default_req = PdRoleSwitchReqInput(new_role="prefill")
        self.assertIsNone(default_req.decode_cuda_graph_bs)
        self.assertIsNone(default_req.decode_cuda_graph_memory_gb)

    def test_resp_is_json_encodable(self):
        from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

        out = PdRoleSwitchReqOutput(
            success=True, message="ok", old_role="prefill", new_role="decode"
        )
        d = msgspec_to_builtins(out)
        self.assertEqual(d["success"], True)
        self.assertEqual(d["old_role"], "prefill")
        self.assertEqual(d["new_role"], "decode")
        self.assertEqual(d["message"], "ok")
        self.assertEqual(d["safe_to_restore"], False)


class TestPdRoleSwitchStartupValidation(unittest.TestCase):
    """--enable-pd-role-switch only rebuilds the small role-specific disagg
    structures on a flip; the per-role buffers of EP / MoE
    all-to-all are sized at startup and not rebuilt, so
    a flip with those on would silently deadlock. The PD arg hook must reject
    the combination up-front instead of failing at flip time."""

    def _sa(self, **kw):
        base = dict(
            disaggregation_transfer_backend="mori",
            disaggregation_mode="prefill",
            disaggregation_decode_host_receive_threshold=1.0,
            enable_pd_role_switch=True,
            attn_dp_size=1,
            ep_join_mode=None,
            ep_size=1,
            moe_a2a_backend="none",
            pp_size=1,
            dp_size=1,
            dcp_size=1,
            speculative_algorithm=None,
        )
        base.update(kw)
        # Keep the production defaults for fields read by the shared PD hook.
        return ServerArgs(model_path="dummy", **base)

    def _run(self, sa):
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_pd_disaggregation,
        )

        handle_pd_disaggregation(sa)

    def test_pure_tp_role_switch_accepted(self):
        # No raise for the validated pure-TP configuration.
        self._run(self._sa())

    def test_dp_attention_role_switch_accepted(self):
        self._run(self._sa(tp_size=2, attn_dp_size=2))

    def test_reject_dp_attention_local_control_broadcast(self):
        with self.assertRaisesRegex(ValueError, "local control broadcast"):
            self._run(
                self._sa(
                    tp_size=2,
                    attn_dp_size=2,
                    enable_dp_attention_local_control_broadcast=True,
                )
            )

    def test_reject_expert_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(ep_size=8))
        self.assertIn("expert parallelism", str(ctx.exception))

    def test_reject_moe_a2a(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(moe_a2a_backend="mori"))
        self.assertIn("MoE all-to-all", str(ctx.exception))

    def test_pipeline_mooncake_role_switch_accepted(self):
        for role in ("prefill", "decode"):
            self._run(
                self._sa(
                    pp_size=2,
                    disaggregation_transfer_backend="mooncake",
                    disaggregation_mode=role,
                )
            )

    def test_reject_unsupported_pipeline_combinations(self):
        for role in ("prefill", "decode"):
            self._check_unsupported_pipeline_combinations(role)

    def _check_unsupported_pipeline_combinations(self, role):
        for extra in (
            {"dp_size": 2},
            {"tp_size": 2, "attn_dp_size": 2},
            {"tp_size": 2, "attn_cp_size": 2},
            {"disaggregation_transfer_backend": "mori"},
            {"ep_size": 2},
            {"speculative_algorithm": "EAGLE"},
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaises(ValueError),
            ):
                args = {
                    "pp_size": 2,
                    "disaggregation_mode": role,
                    "disaggregation_transfer_backend": "mooncake",
                    **extra,
                }
                self._run(self._sa(**args))

    def test_data_parallel_role_switch_accepted(self):
        self._run(self._sa(dp_size=2))

    def test_reject_decode_context_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(dcp_size=2))
        self.assertIn("decode context parallelism", str(ctx.exception))

    def test_reject_speculative_decoding(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(speculative_algorithm="EAGLE"))
        self.assertIn("speculative decoding", str(ctx.exception))

    def test_no_role_switch_is_unaffected(self):
        # The same unsupported feature is fine when role switch is off.
        self._run(self._sa(enable_pd_role_switch=False, moe_a2a_backend="mori"))

    def test_decode_pp1_requires_supported_role_switch_features(self):
        for extra in (
            {"speculative_algorithm": "EAGLE"},
            {"ep_size": 2},
            {"dcp_size": 2},
            {"moe_a2a_backend": "mori"},
            {
                "tp_size": 2,
                "attn_dp_size": 2,
                "enable_dp_attention_local_control_broadcast": True,
            },
        ):
            for enabled in (False, True):
                with self.subTest(extra=extra, enabled=enabled):
                    args = self._sa(
                        disaggregation_mode="decode",
                        disaggregation_transfer_backend="mooncake",
                        pp_size=1,
                        enable_pd_role_switch=enabled,
                        **extra,
                    )
                    if enabled:
                        with self.assertRaises(ValueError):
                            self._run(args)
                    else:
                        self._run(args)


# --- teardown: transfer-worker thread-leak fix + prefix-cache release (radix ON) ---
import threading  # noqa: E402
import time  # noqa: E402

import zmq  # noqa: E402

try:
    from sglang.srt.disaggregation.common.utils import (  # noqa: E402,F811
        FastQueue as _FQ,
    )
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager  # noqa: E402

    _HAS_MOONCAKE = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MOONCAKE = False

try:
    from sglang.srt.disaggregation.role_switch import (  # noqa: E402
        _release_prefix_cache_for_role_switch,
        teardown_disaggregation,
    )

    _HAS_ROLE_SWITCH = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_ROLE_SWITCH = False


@unittest.skipUnless(_HAS_MOONCAKE, "mooncake not importable in this environment")
class TestMooncakeTeardownNoThreadLeak(unittest.TestCase):
    """teardown() must stop+join the transfer workers it started, so a P->D->P
    flip loop does not leak transfer threads per cycle."""

    def test_teardown_joins_transfer_workers(self):
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._stopped = False
        m.enable_trace = False
        m._worker_threads = []
        m.transfer_queues = [_FQ() for _ in range(3)]
        m.executors = [concurrent.futures.ThreadPoolExecutor(1) for _ in range(3)]
        m.server_socket = MagicMock()
        m._zmq_ctx = MagicMock()
        m._socket_lock = threading.Lock()
        m._socket_cache = {}
        m._monitor_cache = {}
        m.engine = MagicMock()
        # Built from KVArgs' own annotations: teardown walks several ptr/len
        # pairs, and hardcoding them here goes stale every time one is added.
        from sglang.srt.disaggregation.base.conn import KVArgs

        m.kv_args = SimpleNamespace(**{name: [] for name in KVArgs.__annotations__})
        for i, (q, ex) in enumerate(zip(m.transfer_queues, m.executors)):
            t = threading.Thread(
                target=m.transfer_worker, args=(q, ex, None, i), daemon=True
            )
            t.start()
            m._worker_threads.append(t)
        started = list(m._worker_threads)
        time.sleep(0.05)  # let workers park in FastQueue.get()
        for t in started:
            self.assertTrue(t.is_alive())

        MooncakeKVManager.teardown(m)

        for t in started:
            self.assertFalse(t.is_alive(), "transfer worker survived teardown (leak)")
        self.assertEqual(m._worker_threads, [])
        self.assertEqual(m.transfer_queues, [])
        self.assertEqual(m.executors, [])

    def test_pp_teardown_keeps_peer_registration_for_fixed_decode(self):
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m._reuse_role_switch_registration = True
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._worker_threads = []
        m.server_socket = MagicMock()
        m._zmq_ctx = MagicMock()
        m._socket_lock = threading.Lock()
        m._socket_cache = {}
        m._monitor_cache = {}
        m.engine = SimpleNamespace()
        registration = object()
        m.decode_kv_args_table = {"fixed-decode": registration}
        MooncakeKVManager.teardown(m)
        self.assertEqual(
            m.engine._pp_pd_peer_registrations, {"fixed-decode": registration}
        )
        m.decode_kv_args_table.clear()
        self.assertIs(m.engine._pp_pd_peer_registrations["fixed-decode"], registration)
        # A subsequent Decode teardown must keep the saved Prefill peer table.
        m.disaggregation_mode = DisaggregationMode.DECODE
        MooncakeKVManager.teardown(m)
        self.assertIs(m.engine._pp_pd_peer_registrations["fixed-decode"], registration)


@unittest.skipUnless(_HAS_MOONCAKE, "mooncake not importable in this environment")
class TestMooncakeBootstrapThreadRobustness(unittest.TestCase):
    """The prefill bootstrap loop moved from a blocking recv_multipart() to a
    500ms poll + _stopped check (so teardown, i.e. a runtime role switch, can
    stop it). That loop runs on every mooncake PD instance, so pin the
    contract with real ZMQ traffic driven through the ABORT -> ABORT_ACK
    path: no message loss while idle or bursting, and prompt exit once
    _stopped is set. Unlike mori, the loop has no try/except around recv: a
    recv error terminates the thread (see test_recv_error_kills_thread).
    """

    class _FlakySocket(zmq.Socket):
        """Real PULL socket whose next recv can be forced to fail once,
        emulating a transient ZMQ error between poll() and recv()."""

        fail_next_recv = False

        def recv_multipart(self, *args, **kwargs):
            if type(self).fail_next_recv:
                type(self).fail_next_recv = False
                raise RuntimeError("transient recv failure")
            return super().recv_multipart(*args, **kwargs)

    def setUp(self):
        self._FlakySocket.fail_next_recv = False
        self._ctx = zmq.Context()
        sock = self._FlakySocket(self._ctx, zmq.PULL)
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m._stopped = False
        m._worker_threads = []
        m.server_socket = sock
        # The receive path is gated on this flag: role switch must be on for
        # the poll-with-timeout loop these tests exercise.
        m.server_args = SimpleNamespace(enable_pd_role_switch=True)
        # Read by the receive loop; off keeps these tests on the plain ACK path.
        m.enable_deferred_decode_kv_release = False
        # ABORT for an unknown room takes the "ignoring" branch and still
        # ACKs, giving a side-effect-free probe of the receive loop.
        m.request_status = {}
        m._socket_send_locks = {}

        def _connect(endpoint, is_ipv6=False):
            m._socket_send_locks.setdefault(endpoint, threading.Lock())
            return m._connect.return_value

        m._connect = MagicMock(side_effect=_connect)
        self.m = m
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(f"tcp://127.0.0.1:{port}")

    def tearDown(self):
        self.m._stopped = True
        for t in self.m._worker_threads:
            t.join(timeout=3.0)
        self._push.close(linger=0)
        self.m.server_socket.close(linger=0)
        self._ctx.destroy(linger=0)

    def _start(self):
        MooncakeKVManager.start_prefill_thread(self.m)
        (thread,) = self.m._worker_threads
        return thread

    def _send_abort(self, room):
        self._push.send_multipart(
            [b"ABORT", str(room).encode("ascii"), b"127.0.0.1", b"9999"]
        )

    def _wait_acks(self, n, timeout=10.0):
        send = self.m._connect.return_value.send_multipart
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if send.call_count >= n:
                return
            time.sleep(0.02)
        self.fail(f"expected {n} ABORT_ACKs, got {send.call_count}")

    def test_messages_processed_across_idle_poll_timeouts(self):
        self._start()
        self._send_abort(1)
        self._wait_acks(1)
        # Idle past a full poll timeout, then traffic must still flow: the
        # empty-poll -> continue path must not disturb the socket.
        time.sleep(0.8)
        self._send_abort(2)
        self._wait_acks(2)

    def test_prefill_drops_late_decode_frames_and_keeps_receiving(self):
        thread = self._start()
        for message in (
            [b"19", b"4", b"0"],  # Old Decode transfer status.
            [b"AUX_DATA", b"19", b"0", b"0", bytes(4), b""],
            [b"ABORT_ACK", b"19"],
            [b"None", b"127.0.0.1", b"1234", b"session"],
            [b"invalid-room", b"127.0.0.1", b"1234", b"session", b"", b"0", b"", b"1"],
            [
                b"None",
                b"127.0.0.1",
                b"1234",
                b"session",
                b"x",
                b"",
                b"",
                b"0",
                b"1",
                b"64",
            ],
        ):
            self._push.send_multipart(message)
        # The same socket must still process the next valid message.
        self._send_abort(20)
        self._wait_acks(1)
        self.assertTrue(thread.is_alive())
        self.assertEqual(self.m.request_status, {})

    def test_decode_drops_late_prefill_frames_and_keeps_receiving(self):
        self.m.apply_prefill_status = Mock()
        with patch.object(
            self.m, "_start_heartbeat_checker_thread", return_value=Mock()
        ):
            self.m.start_decode_thread()
        thread = self.m._worker_threads[0]
        for message in (
            [b"None", b"127.0.0.1", b"1234", b"session"] + [b""] * 16,
            [b"ABORT", b"19", b"127.0.0.1", b"1234"],
            [b"19", b"127.0.0.1", b"1234", b"session", b"", b"0", b"", b"1", b"0", b""],
        ):
            self._push.send_multipart(message)
        self._push.send_multipart([b"20", b"4", b"0"])
        deadline = time.monotonic() + 3
        while not self.m.apply_prefill_status.called and time.monotonic() < deadline:
            time.sleep(0.01)
        self.m.apply_prefill_status.assert_called_once_with(
            bootstrap_room=20, status=4, prefill_rank=0, failure_reason=None
        )
        self.assertTrue(thread.is_alive())

    def test_no_message_loss_under_burst(self):
        self._start()
        n = 200
        for i in range(n):
            self._send_abort(i)
        # Two-step poll+recv must consume every queued message exactly once.
        self._wait_acks(n)

    def test_recv_error_kills_thread(self):
        # No try/except guards recv() in the mooncake loop (unlike mori): a
        # recv error terminates the thread and the loop stops processing.
        # Pin that contract so adding error handling stays a deliberate,
        # reviewed change rather than a silent behavior shift.
        thread = self._start()
        self._FlakySocket.fail_next_recv = True
        self._send_abort(3)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "bootstrap thread survived recv error")
        self.assertFalse(self._FlakySocket.fail_next_recv)  # fault consumed

    def test_cached_peer_reconnects_after_port_rebind(self):
        from sglang.srt.utils.network import get_zmq_socket_on_host

        self._start()
        self._send_abort(1)
        self._wait_acks(1)
        endpoint = self.m.server_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        port = int(endpoint.rsplit(":", 1)[1])
        self.m._stopped = True
        for thread in self.m._worker_threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.m.server_socket.close(linger=0)
        rebound_port, self.m.server_socket = get_zmq_socket_on_host(
            self._ctx, zmq.PULL, host="127.0.0.1", port=port
        )
        self.assertEqual(rebound_port, port)
        self.m._worker_threads = []
        self.m._stopped = False
        self._start()
        # The remote PUSH socket remains unchanged across the local rebuild.
        self._send_abort(2)
        self._wait_acks(2)

    def test_exits_promptly_when_stopped_while_idle(self):
        thread = self._start()
        self.m._stopped = True
        # Poll timeout is 500ms, so the flag must be observed within ~1 cycle
        # (this is what keeps teardown / role switch from hanging).
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "bootstrap thread leaked past stop")


def _radix_scheduler(disable_radix_cache):
    s = MagicMock()
    s.disable_radix_cache = disable_radix_cache
    tree = MagicMock()
    del tree.clear_storage_backend  # plain RadixCache has none
    s.tree_cache = tree
    s.req_to_token_pool = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock()
    return s


@unittest.skipUnless(_HAS_ROLE_SWITCH, "role_switch not importable in this env")
class TestReleasePrefixCacheOnRoleSwitch(unittest.TestCase):
    """The flip may run with radix cache ENABLED: teardown resets the tree cache
    + KV pools when radix is on, and is a no-op on the radix-disabled path."""

    def test_noop_when_radix_disabled(self):
        s = _radix_scheduler(disable_radix_cache=True)
        _release_prefix_cache_for_role_switch(s)
        s.tree_cache.reset.assert_not_called()
        s.token_to_kv_pool_allocator.clear.assert_not_called()

    def test_releases_when_radix_enabled(self):
        s = _radix_scheduler(disable_radix_cache=False)
        _release_prefix_cache_for_role_switch(s)
        s.tree_cache.reset.assert_called_once_with()
        s.req_to_token_pool.clear.assert_called_once_with()
        s.token_to_kv_pool_allocator.clear.assert_called_once_with()

    def test_teardown_invokes_release(self):
        s = _radix_scheduler(disable_radix_cache=False)
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.disagg_prefill_bootstrap_queue = None  # no queue -> skip km.teardown()
        teardown_disaggregation(s)
        self.assertIsNone(s.disagg_metadata_buffers)
        s.tree_cache.reset.assert_called_once_with()


class TestDPRoleSwitch(unittest.IsolatedAsyncioTestCase):
    async def test_all_workers_must_be_ready_before_switching(self):
        for dp_size, attn_dp_size in ((2, 1), (1, 2)):
            with self.subTest(dp_size=dp_size, attn_dp_size=attn_dp_size):
                await self._check_preflight(dp_size, attn_dp_size)

    async def test_insufficient_headroom_prevents_switching_all_workers(self):
        for dp_size, attn_dp_size in ((2, 1), (1, 2)):
            with self.subTest(dp_size=dp_size, attn_dp_size=attn_dp_size):
                await self._check_preflight(
                    dp_size, attn_dp_size, insufficient_memory=True
                )

    async def _check_preflight(self, dp_size, attn_dp_size, insufficient_memory=False):
        from sglang.srt.managers import tokenizer_control_mixin as control

        fixture = TestHandlePdRoleSwitch()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.addCleanup(fixture.doCleanups)
        schedulers = [fixture._scheduler(DisaggregationMode.PREFILL) for _ in range(2)]
        available_memory = [2.0, 2.0]
        for scheduler in schedulers:
            scheduler.device = "cuda"
            scheduler.tp_worker.get_decode_cuda_graph_bs.return_value = (
                [] if insufficient_memory else [1]
            )
            scheduler.init_disaggregation.side_effect = lambda s=scheduler: setattr(
                s,
                "disaggregation_mode",
                DisaggregationMode(rc.get_disagg().disaggregation_mode),
            )

        async def fanout(req):
            results = []
            for index, scheduler in enumerate(schedulers):
                rc.get_context().set_server_args(scheduler.server_args)
                rc.get_context().override(
                    "test.role", disaggregation_mode=scheduler.disaggregation_mode.value
                )
                with patch.object(
                    role_switch,
                    "get_available_gpu_memory",
                    return_value=available_memory[index],
                ):
                    results.append(role_switch.handle_pd_role_switch(scheduler, req))
            if req.prepare_only:
                for scheduler in schedulers:
                    scheduler.init_disaggregation.assert_not_called()
            return results

        manager = SimpleNamespace(
            server_args=SimpleNamespace(enable_pd_role_switch=True),
            auto_create_handle_loop=MagicMock(),
            pd_role_switch_communicator=AsyncMock(side_effect=fanout),
            record_config_updates=MagicMock(),
            disaggregation_mode=DisaggregationMode.PREFILL,
        )
        req = PdRoleSwitchReqInput(
            new_role="decode",
            decode_cuda_graph_memory_gb=1.0 if insufficient_memory else None,
        )
        with patch.object(
            control,
            "get_parallel",
            return_value=SimpleNamespace(
                dp_size=dp_size,
                attn_dp_size=attn_dp_size,
                num_dp_ranks=dp_size * attn_dp_size,
            ),
        ):
            if insufficient_memory:
                available_memory[1] = 0.5
            else:
                schedulers[1].is_fully_idle.return_value = False
            out = await control.TokenizerControlMixin.pd_role_switch(manager, req)
            self.assertFalse(out.success)
            self.assertTrue(out.safe_to_restore)
            self.assertIn(
                "insufficient" if insufficient_memory else "not idle", out.message
            )
            manager.pd_role_switch_communicator.assert_awaited_once()
            self.assertTrue(
                manager.pd_role_switch_communicator.call_args.args[0].prepare_only
            )
            manager.record_config_updates.assert_not_called()
            for scheduler in schedulers:
                scheduler.init_disaggregation.assert_not_called()
                scheduler.teardown_disaggregation.assert_not_called()
                self.assertEqual(
                    scheduler.disaggregation_mode, DisaggregationMode.PREFILL
                )

            available_memory[1] = 2.0
            schedulers[1].is_fully_idle.return_value = True
            out = await control.TokenizerControlMixin.pd_role_switch(manager, req)
            self.assertTrue(out.success)
            self.assertEqual(manager.disaggregation_mode, DisaggregationMode.DECODE)
            for scheduler in schedulers:
                scheduler.init_disaggregation.assert_called_once()
                self.assertEqual(
                    scheduler.disaggregation_mode, DisaggregationMode.DECODE
                )


class TestStartupDecodeGraphReuse(unittest.TestCase):
    def test_switch_reuses_startup_graph_or_captures_when_missing(self):
        from sglang.srt.managers.tp_worker import TpModelWorker
        from sglang.srt.model_executor import model_runner as mr

        for graph in (SimpleNamespace(capture_bs=[1, 2, 4]), None):
            with self.subTest(captured_at_startup=graph is not None):
                rc.reset_context()
                try:
                    rc.get_context().set_server_args(ServerArgs(model_path="dummy"))
                    runner = mr.ModelRunner.__new__(mr.ModelRunner)
                    runner.model = MagicMock()
                    runner.decode_cuda_graph_captured = False
                    runner.decode_cuda_graph_capture_bs = []
                    runner.init_decode_cuda_graph = MagicMock()
                    capture = SimpleNamespace(
                        eager_runner=MagicMock(),
                        prefill=SimpleNamespace(runner=None),
                        decode=SimpleNamespace(runner=graph),
                        memory_usage=None,
                        time_usage=None,
                    )
                    with (
                        patch.object(mr, "capture_cuda_graphs", return_value=capture),
                        patch.object(mr, "prebuild_deepep_v2_buffer"),
                    ):
                        runner.init_cuda_graphs()
                    worker = SimpleNamespace(model_runner=runner)
                    self.assertEqual(
                        TpModelWorker.get_decode_cuda_graph_bs(worker),
                        [1, 2, 4] if graph is not None else [],
                    )
                    runner.ensure_decode_cuda_graphs()
                    if graph is not None:
                        runner.init_decode_cuda_graph.assert_not_called()
                        self.assertIs(runner.decode_cuda_graph_runner, graph)
                    else:
                        runner.init_decode_cuda_graph.assert_called_once_with()
                finally:
                    rc.reset_context()


class Done(Exception):
    pass


class EmptyBatch:
    def __init__(self, **kwargs):
        self.batch_is_full = False

    def is_empty(self):
        return True


class TestPPPreflightAndResources(unittest.TestCase):
    def test_cached_prefix_transfer_waits_for_pp_forward(self):
        from sglang.srt.disaggregation import prefill

        for pp_size, overlap in ((1, False), (1, True), (2, False)):
            with self.subTest(pp=pp_size, overlap=overlap):
                kv = {"value": "computed" if pp_size == 1 and not overlap else "stale"}
                event = Mock()
                event.synchronize.side_effect = lambda: kv.update(value="computed")
                sent = []
                req = SimpleNamespace(
                    pending_bootstrap=False,
                    prefix_len=8,
                    host_hit_length=0,
                    start_send_idx=0,
                    disagg_kv_sender=SimpleNamespace(),
                )

                def send_chunk(request, **kwargs):
                    wait_event = getattr(
                        request.disagg_kv_sender, "_early_send_wait_event", None
                    )
                    if wait_event is not None:
                        wait_event.synchronize()
                    sent.append(kv["value"])

                s = SimpleNamespace(
                    enable_staging=False,
                    enable_overlap=overlap,
                    token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
                    forward_stream=object(),
                    send_kv_chunk=send_chunk,
                )
                with (
                    patch.object(
                        prefill.envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX,
                        "get",
                        return_value=True,
                    ),
                    patch.object(
                        prefill,
                        "get_parallel",
                        return_value=SimpleNamespace(pp_size=pp_size),
                    ),
                    patch.object(prefill.torch.cuda, "Event", return_value=event),
                ):
                    prefill.SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk(
                        s, req
                    )
                self.assertEqual(sent, ["computed"])
                if pp_size > 1 or overlap:
                    event.record.assert_called_once_with(s.forward_stream)

    def test_pp_chunk_send_waits_for_kv_writes(self):
        # Model a forward that has enqueued its KV writes on another stream.
        # Sending a chunk before its event completes would copy stale KV bytes.
        kv = {"value": "stale"}
        event = SimpleNamespace(synchronize=lambda: kv.update(value="computed"))
        sent = []

        def send_chunk(**kwargs):
            sent.append(kv["value"])
            raise Done()

        s = SimpleNamespace(
            pp_loop_size=1,
            running_mbs=[EmptyBatch()],
            last_mbs=[None],
            init_pp_loop_state=lambda: None,
            ingest_requests=lambda: [],
            _pp_role_switch=None,
            pp_group=SimpleNamespace(is_last_rank=True),
            process_bootstrapped_queue=lambda: None,
            _pp_pd_get_prefill_transferred_ids=lambda: [],
            _pp_commit_comm_work=lambda work: None,
            chunked_req=object(),
            launch_event=event,
            process_prefill_chunk=send_chunk,
        )
        with patch.object(pp, "get_parallel", return_value=SimpleNamespace(pp_size=2)):
            with self.assertRaises(Done):
                pp.SchedulerPPMixin.event_loop_pp_disagg_prefill(s)
        self.assertEqual(sent, ["computed"])

    def test_restart_drops_old_batch_aliases_before_ready(self):
        old = object()
        s = SimpleNamespace(
            running_batch=old,
            last_batch=old,
            cur_batch_for_debug=old,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            device="cpu",
        )

        def ready():
            self.assertIs(s.running_batch, s.running_mbs[0])
            self.assertIsNone(s.last_batch)
            self.assertIsNone(s.cur_batch_for_debug)

        s._pp_role_switch = SimpleNamespace(on_loop_started=ready)
        with (
            patch.object(
                pp,
                "get_parallel",
                return_value=SimpleNamespace(pp_size=2, pp_async_batch_depth=1),
            ),
            patch.object(pp, "ScheduleBatch", EmptyBatch),
            patch("torch.cuda.is_available", return_value=False),
        ):
            pp.SchedulerPPMixin.init_pp_loop_state(s)

    def test_teardown_accepts_backend_without_pp_consensus_store(self):
        manager = SimpleNamespace(teardown=Mock())
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=manager),
            disable_radix_cache=True,
        )
        role_switch_pp.role_switch.teardown_disaggregation(scheduler)
        manager.teardown.assert_called_once()
        self.assertIsNone(scheduler.disagg_prefill_bootstrap_queue)

    def test_aggregated_pp_keeps_legacy_rejection_path(self):
        scheduler = SimpleNamespace(disaggregation_mode=DisaggregationMode.NULL)
        dispatch = Mock(return_value="done")
        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(pp_size=2),
            ),
            patch.object(role_switch_pp, "PPRoleSwitch") as constructor,
        ):
            result = role_switch_pp.role_switch.run_event_loop_supervisor(
                scheduler, dispatch
            )
        self.assertEqual(result, "done")
        constructor.assert_not_called()
        self.assertIsNone(scheduler._pp_role_switch)

    def test_fatal_switch_reaches_parent_shutdown(self):
        from sglang.srt.managers import scheduler as module

        parent = Mock()
        s = SimpleNamespace(
            server_args=SimpleNamespace(enable_pd_role_switch=True),
            disaggregation_mode=DisaggregationMode.PREFILL,
            get_init_info=lambda: {},
            run_event_loop=lambda: module.dispatch_event_loop(s),
            metrics_reporter=Mock(),
            async_d2h_worker=None,
            gracefully_exit=False,
        )
        with (
            patch.object(module, "load_plugins"),
            patch.object(module, "publish"),
            patch.object(module, "spawn_world_rank", return_value=0),
            patch.object(module, "configure_scheduler_process"),
            patch.object(
                module.psutil,
                "Process",
                return_value=SimpleNamespace(parent=lambda: parent),
            ),
            patch.object(
                module,
                "get_observability",
                return_value=SimpleNamespace(enable_trace=False),
            ),
            patch.object(module, "Scheduler", return_value=s),
            patch.object(
                module,
                "_dispatch_event_loop_once",
                side_effect=RuntimeError("injected phase failure"),
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(pp_size=2),
            ),
            patch.object(role_switch_pp, "PPRoleSwitch") as controller,
            patch.object(
                module.envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION,
                "get",
                return_value=False,
            ),
        ):
            module.run_scheduler_process(s.server_args, None, 0, 0, 0, None, Mock())
        controller.return_value.close.assert_called_once()
        parent.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_idle_switch_arms_its_own_watchdog(self):
        world, markers = object(), object()
        parallel = SimpleNamespace(world_group=SimpleNamespace(ranks=[0, 1]), tp_size=1)
        with (
            patch.object(role_switch_pp, "get_parallel", return_value=parallel),
            patch.object(
                role_switch_pp,
                "get_device",
                return_value=SimpleNamespace(watchdog_timeout=7),
            ),
            patch.object(
                dist, "new_group", side_effect=[world, markers]
            ) as create_group,
            patch.object(dist, "get_rank", return_value=0),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "destroy_process_group"),
            patch.object(role_switch_pp, "WatchdogRaw") as watchdog,
        ):
            controller = role_switch_pp.PPRoleSwitch(SimpleNamespace())
            callbacks = watchdog.call_args.kwargs
            self.assertFalse(callbacks["is_active"]())
            self.assertEqual(callbacks["get_counter"](), 0)
            controller.enqueue(PdRoleSwitchReqInput(new_role="decode"))
            self.assertTrue(callbacks["is_active"]())
            self.assertEqual(callbacks["get_counter"](), 1)
            self.assertEqual(callbacks["watchdog_timeout"], 7)
            self.assertIn("prepare", callbacks["dump_info"]())
            for call in create_group.call_args_list:
                self.assertEqual(call.kwargs["timeout"], timedelta(seconds=30))
            controller.close()
            self.assertFalse(callbacks["is_active"]())

    def test_every_rank_can_reject(self):
        controller = role_switch_pp.PPRoleSwitch.__new__(role_switch_pp.PPRoleSwitch)
        for rank in range(4):
            for message in (
                "instance is not idle",
                "insufficient decode CUDA graph headroom",
            ):
                values = [
                    dict(
                        old_role="prefill", new_role="decode", graphs=[], rejection=None
                    )
                    for _ in range(4)
                ]
                values[rank]["rejection"] = (message, True)
                result = controller._merge_preflight(values)
                self.assertFalse(result.success)
                self.assertTrue(result.safe_to_restore)
                self.assertIn(f"rank {rank}", result.message)
                self.assertIn(message, result.message)

    def test_consensus_store_closes_threads_and_is_idempotent(self):
        info = None

        def broadcast(value, src):
            nonlocal info
            if value is not None:
                info = value
            return info

        for _ in range(3):
            leader = PPConsensusStore(2, 0, SimpleNamespace(broadcast_object=broadcast))
            peer = PPConsensusStore(2, 1, SimpleNamespace(broadcast_object=broadcast))
            try:
                peer[7] = 1
                deadline = time.monotonic() + 3
                while leader.collect(7) != [None, 1] and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(leader.collect(7), [None, 1])
            finally:
                peer.close()
                peer.close()
                leader.close()
                leader.close()
            self.assertFalse(peer._worker_thread.is_alive())
            self.assertFalse(leader._worker_thread.is_alive())
            self.assertTrue(peer._zmq_ctx.closed)
            self.assertTrue(leader._zmq_ctx.closed)


class TestPPInputOrdering(unittest.TestCase):
    def test_invalid_slot_does_not_call_payload_equality(self):
        class Payload:
            def __eq__(self, other):
                raise AssertionError("Protocol detection called payload equality")

        parallel = SimpleNamespace(
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_tp_size=1,
            pp_rank=1,
            pp_size=2,
            tp_size=1,
        )
        scheduler = SimpleNamespace(world_group=SimpleNamespace(cpu_group=object()))
        for data in ([None], [], [Payload()], [torch.tensor([1, 2])], (None,)):
            with (
                patch.object(pp, "get_parallel", return_value=parallel),
                patch.object(pp, "point_to_point_pyobj", return_value=data),
                patch.object(pp, "attn_cp_tp_broadcast_pyobj", side_effect=lambda x: x),
            ):
                actual = pp.SchedulerPPMixin._pp_recv_pyobj_from_prev_stage(scheduler)
            if isinstance(data, list) and len(data) == 1 and data[0] is None:
                self.assertIsNone(actual)
            else:
                self.assertIs(actual, data)

    def test_later_ingress_waits_for_role_switch(self):
        from sglang.srt.managers import scheduler as module

        before, after = object(), object()
        request = PdRoleSwitchReqInput(new_role="decode")
        controller = SimpleNamespace(request=None)
        s = SimpleNamespace(
            _deferred_input_requests=[],
            _pp_role_switch=controller,
            _poll_timeout_aborts=lambda: [],
            request_receiver=SimpleNamespace(
                recv_requests=Mock(side_effect=[[before, request, after], [], []])
            ),
            process_input_requests=Mock(),
            metrics_reporter=SimpleNamespace(record_scheduler_active=Mock()),
        )
        parallel = SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
        with patch.object(module, "get_parallel", return_value=parallel):
            self.assertEqual(module.Scheduler.ingest_requests(s), [before, request])
            self.assertEqual(s._deferred_input_requests, [after])
            controller.request = request
            self.assertEqual(module.Scheduler.ingest_requests(s), [])
            self.assertEqual(s._deferred_input_requests, [after])
            controller.request = None
            self.assertEqual(module.Scheduler.ingest_requests(s), [after])
            self.assertEqual(s._deferred_input_requests, [])


class TestPPRegisteredBuffers(unittest.TestCase):
    def manager(self, engine, reuse=True):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.engine = engine
        manager._reuse_role_switch_registration = reuse
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.decode_kv_args_table = {}
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100],
            kv_data_lens=[64],
            aux_data_ptrs=[200],
            aux_data_lens=[32],
            state_data_ptrs=[[300]],
            state_data_lens=[[16]],
        )
        manager.connection_pool = {"old": 1}
        manager.connection_lock = threading.Lock()
        return manager

    def test_pp_reuses_all_regions_but_legacy_still_deregisters(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                engine = SimpleNamespace(
                    batch_register=Mock(return_value=0),
                    batch_deregister=Mock(return_value=0),
                )
                first = self.manager(engine, reuse)
                first.register_buffer_to_engine()
                first.deregister_buffer_to_engine()
                self.assertEqual(first.connection_pool, {})
                second = self.manager(engine, reuse)
                second.register_buffer_to_engine()
                self.assertEqual(engine.batch_register.call_count, 1 if reuse else 2)
                self.assertEqual(engine.batch_deregister.call_count, 0 if reuse else 1)

    def test_changed_regions_and_failed_registration_are_fatal(self):
        engine = SimpleNamespace(
            batch_register=Mock(return_value=0), batch_deregister=Mock()
        )
        self.manager(engine).register_buffer_to_engine()
        changed = self.manager(engine)
        changed.kv_args.aux_data_ptrs = [201]
        with self.assertRaisesRegex(RuntimeError, "changed registered"):
            changed.register_buffer_to_engine()
        engine.batch_register.assert_called_once()
        failed = SimpleNamespace(batch_register=Mock(return_value=-1))
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            self.manager(failed).register_buffer_to_engine()
        self.assertFalse(hasattr(failed, "_pp_pd_registered_regions"))

    def test_unsettled_send_or_live_worker_prevents_reuse(self):
        engine = SimpleNamespace(
            batch_register=Mock(return_value=0), batch_deregister=Mock()
        )
        poisoned = self.manager(engine)
        poisoned._deferred_ack_poisoned_rooms = {1}
        with self.assertRaisesRegex(RuntimeError, "unsettled"):
            poisoned.teardown()
        live = self.manager(engine)
        live.transfer_queues = []
        live.executors = []
        thread = SimpleNamespace(join=Mock(), is_alive=lambda: True)
        live._worker_threads = [thread]
        with self.assertRaisesRegex(RuntimeError, "did not stop"):
            live.teardown()
        engine.batch_deregister.assert_not_called()

    def test_pp_cleanup_errors_propagate(self):
        for failing in (
            "queue",
            "socket",
            "monitor",
            "server",
            "context",
            "deregister",
        ):
            with self.subTest(failing=failing):
                manager = self.manager(SimpleNamespace())
                manager.transfer_queues = [Mock()]
                manager.executors = []
                manager._worker_threads = []
                manager._socket_lock = threading.Lock()
                manager._socket_cache = {"peer": Mock()}
                manager._monitor_cache = {"peer": Mock()}
                manager.server_socket = Mock()
                manager._zmq_ctx = Mock()
                manager.deregister_buffer_to_engine = Mock()
                actions = {
                    "queue": manager.transfer_queues[0].put,
                    "socket": manager._socket_cache["peer"].close,
                    "monitor": manager._monitor_cache["peer"].close,
                    "server": manager.server_socket.close,
                    "context": manager._zmq_ctx.destroy,
                    "deregister": manager.deregister_buffer_to_engine,
                }
                actions[failing].side_effect = RuntimeError("injected cleanup failure")
                with self.assertRaisesRegex(RuntimeError, "injected cleanup failure"):
                    manager.teardown()

    def test_metadata_reuses_addresses_and_resets_contents(self):
        from sglang.srt.managers import scheduler as module

        for pp_size in (1, 2):
            s = SimpleNamespace(
                _pp_role_metadata_buffers=None,
                max_running_requests=6,
                req_to_token_pool=SimpleNamespace(size=8),
                model_config=SimpleNamespace(hf_config=SimpleNamespace()),
                server_args=SimpleNamespace(sampling_mask_max_tokens=8),
                token_to_kv_pool_allocator=SimpleNamespace(
                    get_kvcache=lambda: SimpleNamespace(
                        maybe_get_custom_mem_pool=lambda: None
                    )
                ),
                device_module=SimpleNamespace(synchronize=Mock()),
            )
            with (
                patch.object(
                    module,
                    "get_parallel",
                    return_value=SimpleNamespace(pp_size=pp_size),
                ),
                patch.object(
                    module,
                    "get_disagg",
                    return_value=SimpleNamespace(
                        enable_pd_role_switch=True,
                        disaggregation_enable_kv_checksum=True,
                    ),
                ),
                patch.object(module, "is_minimax_sparse", return_value=False),
            ):
                first = module.Scheduler._get_disaggregation_metadata_buffers(
                    s, 12, 16, torch.float32, 4
                )
                self.assertEqual(first.output_ids.shape[0], 16 if pp_size > 1 else 12)
                pointers = first.get_buf_infos()[0]
                for value in vars(first).values():
                    if isinstance(value, torch.Tensor):
                        value.fill_(7)
                req = SimpleNamespace(metadata_buffer_index=0)
                first.set_kv_checksum(req, digest=11, signature=13)
                self.assertEqual(first.get_kv_checksum(0), (11, 13))
                second = module.Scheduler._get_disaggregation_metadata_buffers(
                    s, 16, 16, torch.float32, 4
                )
                if pp_size > 1:
                    self.assertIs(first, second)
                    self.assertEqual(pointers, second.get_buf_infos()[0])
                    s.device_module.synchronize.assert_called_once()
                    with self.assertRaisesRegex(RuntimeError, "capacity changed"):
                        module.Scheduler._get_disaggregation_metadata_buffers(
                            s, 17, 16, torch.float32, 4
                        )
                else:
                    self.assertIsNot(first, second)
                    s.device_module.synchronize.assert_not_called()
                self.assertEqual(second.get_kv_checksum(0), (0, 0))
                for name, value in vars(second).items():
                    if isinstance(value, torch.Tensor):
                        expected = -1 if name == "output_dsa_topk_indices" else 0
                        self.assertTrue(torch.all(value == expected), name)


class TestPPRoleSwitchController(unittest.TestCase):
    def test_pp_launch_orders_scheduler_writes_after_forward_reads(self):
        from contextlib import contextmanager

        events = []
        on_forward_stream = False

        @contextmanager
        def forward_context():
            nonlocal on_forward_stream
            on_forward_stream = True
            try:
                yield
            finally:
                on_forward_stream = False

        result = SimpleNamespace(
            can_run_cuda_graph=True,
            spec_verify_out_cache_loc=None,
            fpm_timing=None,
        )

        def forward(*args):
            events.append("forward_reads")
            return result

        def fence():
            self.assertFalse(on_forward_stream)
            events.append("scheduler_waits_for_reads")

        event = Mock()
        scheduler = SimpleNamespace(
            forward_stream_ctx=forward_context(),
            _pp_wait_forward_dependencies=Mock(),
            run_batch=forward,
            device_module=SimpleNamespace(Event=lambda: event, current_stream=Mock()),
            pp_group=SimpleNamespace(is_last_rank=False),
            _apply_war_barrier=Mock(side_effect=fence),
        )
        metadata = [None]
        with (
            patch.object(pp, "set_time_batch"),
            patch.object(pp, "_pp_snapshot_forward_batch", return_value=object()),
        ):
            actual, ready = pp.SchedulerPPMixin._pp_launch_batch(
                scheduler, 0, SimpleNamespace(reqs=[]), None, metadata, []
            )
        events.append("next_scheduler_write")
        self.assertEqual(
            events,
            ["forward_reads", "scheduler_waits_for_reads", "next_scheduler_write"],
        )
        scheduler._apply_war_barrier.assert_called_once_with()
        self.assertIs(actual, result)
        self.assertIs(ready, event)

    def controller(self):
        controller = role_switch_pp.PPRoleSwitch.__new__(role_switch_pp.PPRoleSwitch)
        controller.scheduler = SimpleNamespace(
            send_control_output=Mock(),
            _pd_role_switch_in_progress=True,
            _pd_role_switch_unhealthy=False,
        )
        controller.rank = 0
        controller.size = 2
        controller.sequence = 1
        controller.phase = "prepare"
        controller.request = PdRoleSwitchReqInput(new_role="decode")
        controller.vote = concurrent.futures.Future()
        controller.result = None
        controller.marker_group = object()
        controller.executor = Mock()
        return controller

    def test_prepare_vote_must_finish_before_injecting_a_marker(self):
        controller = self.controller()
        requests = []
        parallel = SimpleNamespace(pp_rank=0, tp_group=SimpleNamespace(ranks=[0]))
        ready = dict(old_role="prefill", new_role="decode", rejection=None, graphs=[])
        with (
            patch.object(role_switch_pp, "get_parallel", return_value=parallel),
            patch.object(dist, "broadcast_object_list"),
        ):
            self.assertFalse(controller.poll_inputs(requests))
            self.assertEqual(requests, [])
            controller.vote.set_result([ready, ready])
            self.assertFalse(controller.poll_inputs(requests))
        self.assertEqual(requests, [role_switch_pp.PPRoleSwitchMarker(1, "drain")])
        self.assertTrue(controller.draining)
        controller.scheduler.send_control_output.assert_not_called()

    def test_prepare_only_noop_and_rejection_finish_without_draining(self):
        for case in ("prepare_only", "same_role", "rejection"):
            with self.subTest(case=case):
                controller = self.controller()
                controller.request.prepare_only = case == "prepare_only"
                target = "prefill" if case == "same_role" else "decode"
                controller.request.new_role = target
                ready = dict(
                    old_role="prefill", new_role=target, rejection=None, graphs=[]
                )
                peer = (
                    dict(ready, rejection=("busy", True))
                    if case == "rejection"
                    else ready
                )
                controller.vote.set_result([ready, peer])
                with (
                    patch.object(
                        role_switch_pp,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            pp_rank=0, tp_group=SimpleNamespace(ranks=[0])
                        ),
                    ),
                    patch.object(dist, "broadcast_object_list"),
                ):
                    requests = []
                    self.assertFalse(controller.poll_inputs(requests))
                self.assertEqual(requests[0].phase, "finish")
                result = controller.scheduler.send_control_output.call_args.args[0]
                self.assertEqual(result.success, case != "rejection")
                self.assertIsNone(controller.request)
                self.assertFalse(controller.scheduler._pd_role_switch_in_progress)

    def test_empty_valid_slots_and_tails_still_block_drain(self):
        controller = self.controller()
        controller.phase = "drain"
        controller.vote = None
        controller.after_iteration(([None, []],), (None,))
        controller.after_iteration(([None, None],), ([],))
        controller.executor.submit.assert_not_called()
        controller.after_iteration(([None, None],), (None,))
        controller.executor.submit.assert_called_once_with(
            controller._agree, "drained", None
        )

    def test_remote_destructive_phase_failure_is_fatal(self):
        for phase in ("teardown", "rebuild", "decode_graphs"):
            with self.subTest(phase=phase):
                controller = self.controller()
                controller._agree = Mock(return_value=[None, "peer failed"])
                action = Mock()
                with self.assertRaisesRegex(RuntimeError, phase):
                    controller._run_phase(phase, action)
                action.assert_called_once_with()
                self.assertTrue(controller.scheduler._pd_role_switch_unhealthy)
                controller.scheduler.send_control_output.assert_not_called()

    def test_restarted_loop_is_acknowledged_before_reply(self):
        controller = self.controller()
        controller.phase = "restarting"
        controller.result = PdRoleSwitchReqOutput(
            success=True, message="ok", old_role="prefill", new_role="decode"
        )
        events = []
        controller._agree = lambda *args: events.append("restarted")
        controller.scheduler.send_control_output.side_effect = lambda *args: (
            events.append("response")
        )
        controller.on_loop_started()
        self.assertEqual(events, ["restarted", "response"])
        self.assertIsNone(controller.request)

    def test_control_sequence_mismatch_is_rejected(self):
        controller = self.controller()
        controller.group = object()
        for mismatched in ((2, "prepare", None), (1, "drained", None)):
            with self.subTest(peer=mismatched):

                def gather(values, local, **kwargs):
                    values[:] = [local, mismatched]

                with patch.object(dist, "all_gather_object", side_effect=gather):
                    with self.assertRaisesRegex(RuntimeError, "sequence mismatch"):
                        controller._agree("prepare", None)


class TestPPKVMapping(unittest.TestCase):
    def test_plain_mha_pp_to_pp1_maps_all_k_and_v_layers(self):
        # Uneven PP shards must not be mistaken for an appended draft model.
        for pp_size in (3, 4, 5, 6, 7):
            for draft_layers in (0, 1):
                dst = list(range(1000, 1064 + 2 * draft_layers))
                received = []
                for rank in range(pp_size):
                    start = 32 * rank // pp_size
                    end = 32 * (rank + 1) // pp_size
                    count = end - start
                    src = list(range(2000, 2000 + count * 2))
                    manager = SimpleNamespace(
                        pp_size=pp_size,
                        kv_args=SimpleNamespace(prefill_start_layer=start),
                    )
                    manager.kv_args.prefill_num_hidden_layers = 32
                    with self.subTest(pp_size=pp_size, rank=rank, draft=draft_layers):
                        k, v, dst_k, dst_v, layers = (
                            CommonKVManager.get_mha_kv_ptrs_with_pp(manager, src, dst)
                        )
                        self.assertEqual(
                            (k, v, layers), (src[:count], src[count:], count)
                        )
                        self.assertEqual(dst_k, dst[start:end])
                        self.assertEqual(dst_v, dst[32 + start : 32 + end])
                        received.extend(dst_k + dst_v)
                self.assertEqual(sorted(received), dst[:64])


if __name__ == "__main__":
    unittest.main()
