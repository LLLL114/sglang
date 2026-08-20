import argparse
import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt import runtime_context as rc  # noqa: E402
from sglang.srt.disaggregation import role_switch  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    PdRoleSwitchReqInput,
    PdRoleSwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestPdRoleSwitchServerArg(unittest.TestCase):
    def test_cli_flag_parses(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        off = parser.parse_args(["--model-path", "dummy"])
        self.assertFalse(off.enable_pd_role_switch)

        on = parser.parse_args(["--model-path", "dummy", "--enable-pd-role-switch"])
        self.assertTrue(on.enable_pd_role_switch)


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
        switch_patcher = patch.object(role_switch, "switch_tree_cache")
        s.switch_tree_cache = switch_patcher.start()
        self.addCleanup(switch_patcher.stop)
        s.init_disaggregation = MagicMock()
        s._sync_disaggregation_mode_to_subcomponents = MagicMock()
        s._event_loop_should_restart = False
        s._pd_role_switch_in_progress = False
        s._pd_role_switch_unhealthy = False
        s.tp_worker = MagicMock()
        return s

    def test_rejected_when_flag_disabled(self):
        s = self._scheduler(DisaggregationMode.PREFILL, enable=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertIsInstance(out, PdRoleSwitchReqOutput)
        self.assertFalse(out.success)
        self.assertIn("enable-pd-role-switch", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_on_invalid_role(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="both"))
        self.assertFalse(out.success)
        self.assertIn("invalid new_role", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_not_in_pd_mode(self):
        s = self._scheduler(DisaggregationMode.NULL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
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
        self.assertIn("not idle", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_successful_flip_orchestration(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertTrue(out.success)
        self.assertEqual(out.old_role, "prefill")
        self.assertEqual(out.new_role, "decode")
        # Orchestration: drain -> teardown -> flip config bag -> tree cache
        # reset/rebuild -> rebuild queues -> signal.
        s.teardown_disaggregation.assert_called_once_with(s)
        self.assertEqual(rc.get_disagg().disaggregation_mode, "decode")
        # The pristine startup record is never mutated.
        self.assertEqual(s.server_args.disaggregation_mode, "prefill")
        # The tree cache switch runs after the mode flip, before the queues.
        s.switch_tree_cache.assert_called_once_with(s, "prefill", "decode")
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
        self.assertIn("in progress", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_unhealthy(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_unhealthy = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
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
        req = PdRoleSwitchReqInput(new_role="decode", decode_cuda_graph_bs=[1, 2, 4])
        self.assertEqual(req.new_role, "decode")
        self.assertEqual(req.decode_cuda_graph_bs, [1, 2, 4])
        # Field is optional and defaults to None.
        self.assertIsNone(PdRoleSwitchReqInput(new_role="prefill").decode_cuda_graph_bs)

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


class TestPdRoleSwitchStartupValidation(unittest.TestCase):
    """--enable-pd-role-switch only rebuilds the small role-specific disagg
    structures on a flip; the per-role buffers of DP attention / EP / MoE
    all-to-all / pipeline parallelism are sized at startup and not rebuilt, so
    a flip with those on would silently deadlock. The PD arg hook must reject
    the combination up-front instead of failing at flip time."""

    def _sa(self, **kw):
        base = dict(
            disaggregation_transfer_backend="mori",
            disaggregation_mode="prefill",
            enable_pd_role_switch=True,
            enable_dp_attention=False,
            ep_size=1,
            moe_a2a_backend="none",
            pp_size=1,
            dp_size=1,
            dcp_size=1,
            speculative_algorithm=None,
            disable_radix_cache=False,
            disaggregation_decode_enable_radix_cache=False,
            disaggregation_decode_extra_slots=None,
            max_running_requests=None,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def _run(self, sa):
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_pd_disaggregation,
        )

        handle_pd_disaggregation(sa)

    def test_pure_tp_role_switch_accepted(self):
        # No raise for the validated pure-TP configuration.
        self._run(self._sa())

    def test_reject_dp_attention(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(enable_dp_attention=True))
        self.assertIn("DP attention", str(ctx.exception))

    def test_reject_expert_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(ep_size=8))
        self.assertIn("expert parallelism", str(ctx.exception))

    def test_reject_moe_a2a(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(moe_a2a_backend="mori"))
        self.assertIn("MoE all-to-all", str(ctx.exception))

    def test_reject_pipeline_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(pp_size=2))
        self.assertIn("pipeline parallelism", str(ctx.exception))

    def test_reject_data_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(dp_size=2))
        self.assertIn("data parallelism", str(ctx.exception))

    def test_reject_speculative_decoding(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(speculative_algorithm="EAGLE"))
        self.assertIn("speculative decoding", str(ctx.exception))

    def test_no_role_switch_is_unaffected(self):
        # The same unsupported feature is fine when role switch is off.
        self._run(self._sa(enable_pd_role_switch=False, moe_a2a_backend="mori"))

    def test_per_role_radix_intent_recorded_on_prefill_start(self):
        sa = self._sa()
        self._run(sa)
        self.assertEqual(
            sa._pd_role_disable_radix_cache, {"prefill": False, "decode": True}
        )
        # The started role's flag is untouched for a prefill node.
        self.assertFalse(sa.disable_radix_cache)

    def test_per_role_radix_intent_survives_decode_forcing(self):
        # A decode-started node has disable_radix_cache force-overwritten; the
        # stash must still carry the pre-forcing prefill intent.
        sa = self._sa(disaggregation_mode="decode")
        self._run(sa)
        self.assertTrue(sa.disable_radix_cache)  # forced for decode
        self.assertEqual(
            sa._pd_role_disable_radix_cache, {"prefill": False, "decode": True}
        )

    def test_per_role_radix_intent_honors_decode_radix_flag(self):
        sa = self._sa(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            enable_hisparse=False,
        )
        self._run(sa)
        self.assertEqual(
            sa._pd_role_disable_radix_cache, {"prefill": False, "decode": False}
        )

    def test_no_intent_recorded_without_role_switch(self):
        sa = self._sa(enable_pd_role_switch=False)
        self._run(sa)
        self.assertFalse(hasattr(sa, "_pd_role_disable_radix_cache"))


# --- teardown: transfer-worker thread-leak fix + prefix-cache release (radix ON) ---
import threading  # noqa: E402
import time  # noqa: E402

import zmq  # noqa: E402

try:
    from sglang.srt.disaggregation.common.utils import FastQueue  # noqa: E402
    from sglang.srt.disaggregation.mori.conn import MoriKVManager  # noqa: E402

    _HAS_MORI = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MORI = False

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


@unittest.skipUnless(_HAS_MORI, "mori not importable in this environment")
class TestMoriTeardownNoThreadLeak(unittest.TestCase):
    """teardown() must stop+join the transfer workers it started, so a P->D->P
    flip loop does not leak _num_shards transfer threads per cycle."""

    def test_teardown_joins_transfer_workers(self):
        m = MoriKVManager.__new__(MoriKVManager)
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._stopped = False
        m._worker_threads = []
        m._transfer_queues = [FastQueue() for _ in range(3)]
        m.server_socket = MagicMock()
        m._zmq_ctx = MagicMock()
        m.engine = MagicMock()
        m.kv_mem_descs = m.aux_mem_descs = m.state_mem_descs = []
        for q in m._transfer_queues:
            t = threading.Thread(target=m._transfer_worker, args=(q,), daemon=True)
            t.start()
            m._worker_threads.append(t)
        started = list(m._worker_threads)
        time.sleep(0.05)  # let workers park in FastQueue.get()
        for t in started:
            self.assertTrue(t.is_alive())

        MoriKVManager.teardown(m)

        for t in started:
            self.assertFalse(t.is_alive(), "transfer worker survived teardown (leak)")
        self.assertEqual(m._worker_threads, [])
        self.assertEqual(m._transfer_queues, [])


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
        m.kv_args = SimpleNamespace(
            kv_data_ptrs=[], aux_data_ptrs=[], state_data_ptrs=[]
        )
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
    + KV pools when radix is on, and is a no-op on the historical chunk-cache path."""

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

    def test_teardown_leaves_prefix_cache_to_switch(self):
        # The cache release moved out of teardown into switch_tree_cache,
        # which runs after the config-bag mode flip (recipe re-resolution
        # must see the new role).
        s = _radix_scheduler(disable_radix_cache=False)
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.disagg_prefill_bootstrap_queue = None  # no queue -> skip km.teardown()
        teardown_disaggregation(s)
        self.assertIsNone(s.disagg_metadata_buffers)
        s.tree_cache.reset.assert_not_called()


@unittest.skipUnless(_HAS_ROLE_SWITCH, "role_switch not importable in this env")
class TestSwitchTreeCacheDispatch(unittest.TestCase):
    """switch_tree_cache: equal recipes take the historical reset path;
    differing recipes destroy the old cache and rebuild one for the new role."""

    def _dispatch(self, recipes):
        s = _radix_scheduler(disable_radix_cache=False)
        with patch.object(
            role_switch, "_tree_cache_recipe_for_role", side_effect=recipes
        ), patch.object(
            role_switch, "_release_prefix_cache_for_role_switch"
        ) as release, patch.object(
            role_switch, "_destroy_tree_cache_for_role_switch"
        ) as destroy, patch.object(
            role_switch, "_rebuild_tree_cache_for_role_switch"
        ) as rebuild:
            role_switch.switch_tree_cache(s, "prefill", "decode")
        return s, release, destroy, rebuild

    def test_same_recipe_takes_reset_path(self):
        s, release, destroy, rebuild = self._dispatch(
            [(False, "cpu_tensor"), (False, "cpu_tensor")]
        )
        release.assert_called_once_with(s)
        destroy.assert_not_called()
        rebuild.assert_not_called()

    def test_recipe_change_destroys_and_rebuilds(self):
        s, release, destroy, rebuild = self._dispatch(
            [(False, "cpu_tensor"), (True, "host_pool")]
        )
        release.assert_not_called()
        destroy.assert_called_once_with(s)
        rebuild.assert_called_once_with(s, "decode")


@unittest.skipUnless(_HAS_ROLE_SWITCH, "role_switch not importable in this env")
class TestRoleTreeCacheRecipe(unittest.TestCase):
    """The recipe (disable_radix_cache, retraction_backup) drives the rebuild
    decision; both legs must be computed per role, not read from the started
    role's resolved flags."""

    def _s(self, stash, *, started="prefill", resolved_disable=False):
        s = MagicMock()
        s.server_args = SimpleNamespace(
            disaggregation_mode=started,
            disable_radix_cache=resolved_disable,
            disaggregation_decode_retraction_backup=None,
        )
        if stash is not None:
            s.server_args._pd_role_disable_radix_cache = stash
        return s

    def test_stash_maps_roles(self):
        s = self._s({"prefill": False, "decode": True})
        self.assertFalse(role_switch._role_disable_radix_cache(s, "prefill"))
        self.assertTrue(role_switch._role_disable_radix_cache(s, "decode"))

    def test_later_model_forcing_applies_to_both_roles(self):
        # Started as prefill with radix-ON intent, but a model-specific pass
        # after the PD hook forced radix off: that forcing is role-independent
        # and must survive every flip.
        s = self._s({"prefill": False, "decode": True}, resolved_disable=True)
        self.assertTrue(role_switch._role_disable_radix_cache(s, "prefill"))
        self.assertTrue(role_switch._role_disable_radix_cache(s, "decode"))

    def test_explicit_retraction_backup_is_role_independent(self):
        s = self._s({"prefill": False, "decode": True})
        s.server_args.disaggregation_decode_retraction_backup = "cpu_tensor"
        self.assertEqual(
            role_switch._role_retraction_backup(s, "decode"), "cpu_tensor"
        )

    def test_auto_retraction_backup_resolves_per_role(self):
        s = self._s({"prefill": False, "decode": True})
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.compute_auto_decode_retraction_backup",
            side_effect=lambda *, tp_worker, mode: (
                "host_pool" if mode == "decode" else "cpu_tensor"
            ),
        ):
            self.assertEqual(
                role_switch._tree_cache_recipe_for_role(s, "prefill"),
                (False, "cpu_tensor"),
            )
            self.assertEqual(
                role_switch._tree_cache_recipe_for_role(s, "decode"),
                (True, "host_pool"),
            )


@unittest.skipUnless(_HAS_ROLE_SWITCH, "role_switch not importable in this env")
class TestDestroyTreeCacheOnRoleSwitch(unittest.TestCase):
    """A recipe-changing flip must fully release the old cache: L3 cleared and
    detached (storage daemon threads stopped), pinned host pools destroyed,
    shared device-pool hooks unhooked, and the atexit auto-detach unregistered
    so the dead cache is collectable."""

    def _scheduler(self):
        s = MagicMock()
        tree = MagicMock()
        s.tree_cache = tree
        s.req_to_token_pool = MagicMock()
        s.token_to_kv_pool_allocator = MagicMock()
        s.tp_worker = MagicMock()
        return s, tree

    def test_full_release_of_hicache_variant(self):
        s, tree = self._scheduler()
        with patch("atexit.unregister") as unreg:
            role_switch._destroy_tree_cache_for_role_switch(s)
        tree.clear_storage_backend.assert_called_once_with()
        tree.detach_storage_backend.assert_called_once_with()
        tree.reset.assert_called_once_with()
        tree.release_host_resources.assert_called_once_with()
        kv = s.token_to_kv_pool_allocator.get_kvcache.return_value
        kv.register_layer_transfer_counter.assert_called_once_with(None)
        s.tp_worker.register_hicache_layer_transfer_counter.assert_called_once_with(
            None
        )
        unreg.assert_called_once_with(tree.shutdown)
        s.req_to_token_pool.clear.assert_called_once_with()
        s.token_to_kv_pool_allocator.clear.assert_called_once_with()

    def test_plain_radix_cache_release(self):
        s, tree = self._scheduler()
        del tree.clear_storage_backend  # plain RadixCache has none of these
        del tree.detach_storage_backend
        del tree.shutdown
        role_switch._destroy_tree_cache_for_role_switch(s)
        tree.reset.assert_called_once_with()
        tree.release_host_resources.assert_called_once_with()
        s.token_to_kv_pool_allocator.clear.assert_called_once_with()

    def test_storage_failure_does_not_abort_destruction(self):
        s, tree = self._scheduler()
        tree.clear_storage_backend.side_effect = RuntimeError("backend gone")
        tree.detach_storage_backend.side_effect = RuntimeError("backend gone")
        role_switch._destroy_tree_cache_for_role_switch(s)
        tree.reset.assert_called_once_with()
        tree.release_host_resources.assert_called_once_with()


@unittest.skipUnless(_HAS_ROLE_SWITCH, "role_switch not importable in this env")
class TestRebuildTreeCacheOnRoleSwitch(unittest.TestCase):
    """The rebuild must land the new role's cache config on the bags, rebuild
    through the regular kv_cache_builder path, and rebind every holder."""

    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def _scheduler(self, sa):
        rc.get_context().set_server_args(sa)
        s = MagicMock()
        s.server_args = sa
        s._kv_cache_build_kwargs = {}
        return s

    def test_rebuild_to_decode_re_resolves_config(self):
        sa = ServerArgs(model_path="dummy", disaggregation_mode="prefill")
        sa._pd_role_disable_radix_cache = {"prefill": False, "decode": True}
        s = self._scheduler(sa)
        new_cache = MagicMock()
        result = SimpleNamespace(disable_radix_cache=True, tree_cache=new_cache)
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.build_kv_cache",
            return_value=result,
        ) as build:
            role_switch._rebuild_tree_cache_for_role_switch(s, "decode")
        # Role-derived config landed on the bags: radix off, retraction backend
        # and hicache ratio cleared for re-resolution against the decode role.
        self.assertTrue(rc.get_memory().disable_radix_cache)
        self.assertIsNone(rc.get_disagg().disaggregation_decode_retraction_backup)
        self.assertIsNone(rc.get_memory().hicache_ratio)
        build.assert_called_once_with(for_pd_role_switch=True)
        self.assertIs(s.disable_radix_cache, True)
        s._rebind_tree_cache.assert_called_once_with(new_cache)

    def test_rebuild_to_prefill_restores_static_hicache_ratio(self):
        # A decode-started node kept the ratio unset for the retraction
        # resolver; flipping to prefill must restore the static default.
        sa = ServerArgs(model_path="dummy", disaggregation_mode="decode")
        sa._pd_role_disable_radix_cache = {"prefill": False, "decode": True}
        s = self._scheduler(sa)
        result = SimpleNamespace(disable_radix_cache=False, tree_cache=MagicMock())
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.build_kv_cache",
            return_value=result,
        ):
            role_switch._rebuild_tree_cache_for_role_switch(s, "prefill")
        self.assertFalse(rc.get_memory().disable_radix_cache)
        self.assertEqual(rc.get_memory().hicache_ratio, 2.0)

    def test_explicit_hicache_ratio_is_kept(self):
        sa = ServerArgs(
            model_path="dummy", disaggregation_mode="prefill", hicache_ratio=3.0
        )
        sa._pd_role_disable_radix_cache = {"prefill": False, "decode": True}
        s = self._scheduler(sa)
        result = SimpleNamespace(disable_radix_cache=True, tree_cache=MagicMock())
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.build_kv_cache",
            return_value=result,
        ):
            role_switch._rebuild_tree_cache_for_role_switch(s, "decode")
        self.assertEqual(rc.get_memory().hicache_ratio, 3.0)


if __name__ == "__main__":
    unittest.main()
