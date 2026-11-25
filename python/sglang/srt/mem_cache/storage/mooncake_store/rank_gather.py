"""Single-node HiCache gather using ordinary Mooncake Store objects."""

import ctypes
import hashlib
import json
import socket
import tempfile
import threading
import time
from itertools import accumulate
from pathlib import Path

import torch
from mooncake.reshard.kv_cache import encode_gather_object_header

_TIMEOUT = 60


def _view(pointer, size):
    return memoryview((ctypes.c_ubyte * size).from_address(pointer)).cast("B")


def _recv(peer, buffer):
    while buffer:
        received = peer.recv_into(buffer)
        if not received:
            raise ConnectionError("rank gather peer disconnected")
        buffer = buffer[received:]


def _read_exact(peer, size):
    buffer = bytearray(size)
    _recv(peer, memoryview(buffer))
    return bytes(buffer)


class RankGatherWriter:
    """Store each page as a source manifest followed by TP/PP rank payloads."""

    def __init__(self, native, manifest, rank, cohort, max_pages):
        self.native = native
        self.manifest = manifest
        self.rank = rank
        self.header = encode_gather_object_header(manifest)
        self.part_sizes = manifest.layout.part_sizes
        self.part_offsets = tuple(accumulate(self.part_sizes, initial=len(self.header)))
        self.page_bytes = self.part_offsets[-1]
        self.max_pages = max_pages
        self._pages = None
        self._socket_dir = (
            Path(tempfile.gettempdir()) / f"sglang-rank-gather-{cohort[:24]}"
        )
        self._peers = {}
        self._listener = None
        self._failed = False
        self._write_lock = threading.Lock()

    def _allocate(self, size):
        buffer = torch.empty(size, dtype=torch.uint8, device="cpu")
        if self.native.register_buffer(buffer.data_ptr(), size) != 0:
            raise RuntimeError("failed to register rank gather buffer")
        return buffer

    def _open(self):
        path = str(self._socket_dir / "gather.sock")
        if self.rank == 0:
            self._pages = self._allocate(self.max_pages * self.page_bytes)
            if len(self.part_sizes) == 1:
                return
            self._socket_dir.mkdir(mode=0o700)
            self._listener = socket.socket(socket.AF_UNIX)
            self._listener.settimeout(_TIMEOUT)
            self._listener.bind(path)
            self._listener.listen(len(self.part_sizes) - 1)
            for _ in self.part_sizes[1:]:
                peer, _ = self._listener.accept()
                peer.settimeout(_TIMEOUT)
                try:
                    rank = int.from_bytes(_read_exact(peer, 4), "big")
                    if rank <= 0 or rank >= len(self.part_sizes) or rank in self._peers:
                        raise ValueError("unexpected rank gather participant")
                    self._peers[rank] = peer
                except Exception:
                    peer.close()
                    raise
        else:
            peer = socket.socket(socket.AF_UNIX)
            peer.settimeout(_TIMEOUT)
            self._peers[0] = peer
            deadline = time.monotonic() + _TIMEOUT
            while True:
                try:
                    peer.connect(path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "rank gather coordinator did not start"
                        ) from None
                    time.sleep(0.01)
            peer.sendall(self.rank.to_bytes(4, "big"))

    def _close_peers(self):
        for peer in self._peers.values():
            peer.close()
        self._peers.clear()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
            (self._socket_dir / "gather.sock").unlink(missing_ok=True)
            self._socket_dir.rmdir()

    def close(self):
        self._close_peers()
        if self._pages is not None:
            if self.native.unregister_buffer(self._pages.data_ptr()) != 0:
                raise RuntimeError("failed to unregister rank gather buffer")
            self._pages = None

    def upload(self, keys, pointers, sizes):
        if not keys:
            return []
        if (
            len(keys) > self.max_pages
            or len(keys) != len(pointers)
            or len(keys) != len(sizes)
        ):
            raise ValueError("rank gather batch shape differs")
        if any(
            len(ptr) != len(size) or sum(size) != self.part_sizes[self.rank]
            for ptr, size in zip(pointers, sizes)
        ):
            raise ValueError("rank gather payload size differs from the manifest")

        # HiCache schedules the backup stream in lockstep. Check the complete
        # batch before allowing peers to send payloads or publishing any key.
        signature = hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode()
        ).digest()
        with self._write_lock:
            if self._failed:
                raise RuntimeError("rank gather cohort needs to be reattached")
            try:
                if not self._peers and self._pages is None:
                    self._open()
                if self.rank:
                    return self._send_pages(signature, keys, pointers, sizes)
                return self._publish_pages(signature, keys, pointers, sizes)
            except Exception:
                self._failed = True
                self._close_peers()
                raise

    def _send_pages(self, signature, keys, pointers, sizes):
        peer = self._peers[0]
        peer.sendall(signature)
        missing = _read_exact(peer, len(keys))
        for put, ptr, size in zip(missing, pointers, sizes):
            if put:
                for address, length in zip(ptr, size):
                    peer.sendall(_view(address, length))
        return [0 if ok else -1 for ok in _read_exact(peer, len(keys))]

    def _publish_pages(self, signature, keys, pointers, sizes):
        for peer in self._peers.values():
            if _read_exact(peer, len(signature)) != signature:
                raise ValueError("rank gather backup batches are out of order")
        exists = self.native.batch_is_exist(list(keys))
        if len(exists) != len(keys):
            raise ValueError("ordinary object existence result shape differs")
        if any(value not in (0, 1) for value in exists):
            raise RuntimeError("rank gather page lookup failed")
        missing = bytes(value == 0 for value in exists)
        for peer in self._peers.values():
            peer.sendall(missing)
        selected = [i for i, put in enumerate(missing) if put]
        base = self._pages.data_ptr()
        pages = [base + i * self.page_bytes for i in range(len(selected))]
        for i, dst in zip(selected, pages):
            ctypes.memmove(dst, self.header, len(self.header))
            dst += self.part_offsets[0]
            for address, length in zip(pointers[i], sizes[i]):
                ctypes.memmove(dst, address, length)
                dst += length
        for rank, peer in sorted(self._peers.items()):
            for dst in pages:
                _recv(
                    peer,
                    _view(dst + self.part_offsets[rank], self.part_sizes[rank]),
                )
        result = [0] * len(keys)
        if selected:
            values = self.native.batch_put_from(
                [keys[i] for i in selected],
                pages,
                [self.page_bytes] * len(pages),
            )
            if len(values) != len(selected):
                raise ValueError("ordinary object write result shape differs")
            for i, code in zip(selected, values):
                result[i] = code
        reply = bytes(code == 0 for code in result)
        for peer in self._peers.values():
            peer.sendall(reply)
        if not all(reply):
            # Retain the staging allocation and stop reusing it if
            # the transport did not confirm the writes.
            self._failed = True
            self._close_peers()
        return result
