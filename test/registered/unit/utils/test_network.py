"""Unit tests for srt/utils/network.py: ZMQ peer allowlist and source pinning."""

import unittest
from unittest.mock import Mock, patch

import zmq

from sglang.srt.environ import envs
from sglang.srt.utils import network
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestZmqSourceIpEndpoint(CustomTestCase):
    def test_rewrites_only_matching_tcp_ip_endpoints(self):
        """`SGLANG_ZMQ_SOURCE_IP` is spliced in with libzmq's `src:0;dst` form;
        hostnames, ipc endpoints and the other address family pass unchanged."""
        cases = [
            ("10.0.0.5", "tcp://10.0.0.1:5555", "tcp://10.0.0.5:0;10.0.0.1:5555"),
            (
                "2001:db8::2",
                "tcp://[2001:db8::1]:5555",
                "tcp://[2001:db8::2]:0;[2001:db8::1]:5555",
            ),
            ("2001:db8::2", "tcp://127.0.0.1:5555", "tcp://127.0.0.1:5555"),
            ("10.0.0.5", "tcp://somehost:5555", "tcp://somehost:5555"),
            ("10.0.0.5", "ipc:///tmp/sock", "ipc:///tmp/sock"),
            (
                "10.0.0.5",
                "tcp://10.0.0.9:0;10.0.0.1:5555",
                "tcp://10.0.0.9:0;10.0.0.1:5555",
            ),
            (None, "tcp://10.0.0.1:5555", "tcp://10.0.0.1:5555"),
        ]
        for source, endpoint, expected in cases:
            with self.subTest(source=source, endpoint=endpoint):
                with envs.SGLANG_ZMQ_SOURCE_IP.override(source):
                    if source is None:
                        envs.SGLANG_ZMQ_SOURCE_IP.clear()
                    self.assertEqual(network._with_source_ip(endpoint), expected)


class TestZmqAllowlistAdmitsPinnedSource(CustomTestCase):
    """A multi-homed client whose kernel picks a source address outside the
    peer allowlist is rejected and its PUSH blocks; pinning the source to the
    allowlisted address must get the message through."""

    def _delivered(self, source: str | None) -> bool:
        server_ctx, client_ctx = zmq.Context(), zmq.Context()
        try:
            with envs.SGLANG_ZMQ_ALLOWED_PEERS.override("127.0.0.2"):
                port, server = network.get_zmq_socket(server_ctx, zmq.PULL)
                with envs.SGLANG_ZMQ_SOURCE_IP.override(source):
                    if source is None:
                        envs.SGLANG_ZMQ_SOURCE_IP.clear()
                    client = network.get_zmq_socket(
                        client_ctx, zmq.PUSH, f"tcp://127.0.0.1:{port}", bind=False
                    )
            client.send(b"ping")
            return server.poll(2000) != 0
        finally:
            with network._zmq_authenticators_lock:
                auth = network._zmq_authenticators.pop(server_ctx.underlying, None)
            if auth is not None:
                auth.stop()
            server_ctx.destroy(linger=0)
            client_ctx.destroy(linger=0)

    def test_unpinned_connect_is_rejected(self):
        self.assertFalse(self._delivered(None))

    def test_pinned_connect_is_admitted(self):
        self.assertTrue(self._delivered("127.0.0.2"))


class TestFixedZmqEndpoint(CustomTestCase):
    def test_rebinds_same_port_after_previous_listener_releases_it(self):
        previous, current = zmq.Context(), zmq.Context()
        try:
            port, _ = network.get_zmq_socket_on_host(previous, zmq.PULL)
            with patch.object(
                network.time, "sleep", side_effect=lambda _: previous.destroy(linger=0)
            ) as retry:
                rebound, receiver = network.get_zmq_socket_on_host(
                    current, zmq.PULL, port=port
                )
            self.assertEqual(rebound, port)
            self.assertEqual(retry.call_count, 1)
            sender = current.socket(zmq.PUSH)
            sender.connect(f"tcp://127.0.0.1:{port}")
            sender.send(b"rebound")
            self.assertTrue(receiver.poll(2000))
            self.assertEqual(receiver.recv(), b"rebound")
        finally:
            previous.destroy(linger=0)
            current.destroy(linger=0)

    def test_busy_port_fails_with_bounded_retries_and_endpoint(self):
        previous, current = zmq.Context(), zmq.Context()
        try:
            port, _ = network.get_zmq_socket_on_host(previous, zmq.PULL)
            with patch.object(network.time, "sleep") as retry:
                with self.assertRaises(zmq.ZMQError) as error:
                    network.get_zmq_socket_on_host(current, zmq.PULL, port=port)
            self.assertEqual(error.exception.errno, zmq.EADDRINUSE)
            self.assertIn(f"127.0.0.1:{port}", str(error.exception))
            self.assertIn("3 bind attempts", str(error.exception))
            self.assertEqual(retry.call_count, 2)
        finally:
            previous.destroy(linger=0)
            current.destroy(linger=0)

    def test_other_bind_errors_are_not_retried(self):
        context = Mock()
        socket = context.socket.return_value
        socket.bind.side_effect = zmq.ZMQError(zmq.EINVAL)
        with (
            patch.object(network, "config_socket"),
            patch.object(network.time, "sleep") as retry,
        ):
            with self.assertRaises(zmq.ZMQError) as error:
                network.get_zmq_socket_on_host(context, zmq.PULL, port=12345)
        self.assertEqual(error.exception.errno, zmq.EINVAL)
        socket.bind.assert_called_once()
        socket.close.assert_called_once_with(linger=0)
        retry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
