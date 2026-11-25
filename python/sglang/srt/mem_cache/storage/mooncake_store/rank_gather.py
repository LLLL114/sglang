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
        if not 0 <= rank < len(self.part_sizes):
            raise ValueError("invalid gather writer rank")
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
        if len(keys) != len(pointers) or len(keys) != len(sizes):
            raise ValueError("rank gather batch shape differs")
        if any(
            len(ptr) != len(size)
            or sum(size) != self.part_sizes[self.rank]
            or (len(size) > 1 and min(size) < 0)
            for ptr, size in zip(pointers, sizes)
        ):
            raise ValueError("rank gather payload size differs from the manifest")

        def source_rows(selected):
            for i in selected:
                yield zip(pointers[i], sizes[i], strict=True)

        return self._upload(keys, source_rows)

    def upload_slots(self, keys, slots, spans, slot_stride):
        """Translate a bound source page only for keys the coordinator requests."""
        if len(slots) != len(keys):
            raise ValueError("rank gather batch shape differs")

        if sum(n for _, n in spans) != self.part_sizes[self.rank] or any(
            n < 0 for _, n in spans
        ):
            raise ValueError("rank gather payload size differs from the manifest")

        def source_rows(selected):
            for i in selected:
                delta = slots[i] * slot_stride
                yield [(address + delta, length) for address, length in spans]

        return self._upload(keys, source_rows)

    def _upload(self, keys, rows):
        if not keys:
            return []
        if len(keys) > self.max_pages:
            raise ValueError("rank gather batch shape differs")
        # HiCache schedules the backup stream in lockstep. Check the complete
        # batch before allowing peers to send payloads or publishing any key.
        # Reject peers expecting the old two-message all-hit reply up front.
        signature = hashlib.sha256(
            b"gather-hit-mask\0" + json.dumps(keys, separators=(",", ":")).encode()
        ).digest()
        with self._write_lock:
            if self._failed:
                raise RuntimeError("rank gather cohort needs to be reattached")
            try:
                if len(self.part_sizes) > 1 and not self._peers:
                    self._open()
                if self.rank:
                    return self._send_pages(signature, keys, rows)
                return self._publish_pages(signature, keys, rows)
            except Exception:
                self._failed = True
                self._close_peers()
                raise

    def _send_pages(self, signature, keys, rows):
        peer = self._peers[0]
        peer.sendall(signature)
        missing = _read_exact(peer, len(keys))
        selected = [i for i, put in enumerate(missing) if put]
        if not selected:
            return [0] * len(keys)
        source = iter(rows(selected))
        run_start = run_size = 0
        for _ in selected:
            for address, length in next(source):
                if not length:
                    continue
                if run_size and run_start + run_size != address:
                    peer.sendall(_view(run_start, run_size))
                    run_size = 0
                if not run_size:
                    run_start = address
                run_size += length
        if run_size:
            peer.sendall(_view(run_start, run_size))
        return [0 if ok else -1 for ok in _read_exact(peer, len(keys))]

    def _readable(self, keys):
        exists = self.native.batch_is_exist(list(keys))
        if len(exists) != len(keys):
            raise ValueError("ordinary object existence result shape differs")
        if any(value not in (0, 1) for value in exists):
            raise RuntimeError("rank gather page lookup failed")
        return exists

    def _publish_pages(self, signature, keys, rows):
        for peer in self._peers.values():
            if _read_exact(peer, len(signature)) != signature:
                raise ValueError("rank gather backup batches are out of order")
        exists = self._readable(keys)
        missing = bytes(value == 0 for value in exists)
        # An all-zero mask confirms every key is already readable.
        for peer in self._peers.values():
            peer.sendall(missing)
        selected = [i for i, put in enumerate(missing) if put]
        if not selected:
            return [0] * len(keys)
        if self._pages is None:
            self._pages = self._allocate(self.max_pages * self.page_bytes)
            self._initialized_headers = 0
        base = self._pages.data_ptr()
        pages = [base + i * self.page_bytes for i in range(len(selected))]
        # Payloads start after the immutable header in every reusable slot.
        for dst in pages[self._initialized_headers :]:
            ctypes.memmove(dst, self.header, len(self.header))
        self._initialized_headers = max(self._initialized_headers, len(pages))
        for dst, row in zip(pages, rows(selected), strict=True):
            dst += self.part_offsets[0]
            for address, length in row:
                ctypes.memmove(dst, address, length)
                dst += length
        for rank, peer in sorted(self._peers.items()):
            for dst in pages:
                _recv(
                    peer,
                    _view(dst + self.part_offsets[rank], self.part_sizes[rank]),
                )
        result = [0] * len(keys)
        values = self.native.batch_put_from(
            [keys[i] for i in selected],
            pages,
            [self.page_bytes] * len(pages),
        )
        if len(values) != len(selected):
            raise ValueError("ordinary object write result shape differs")
        transfer_failed = any(code != 0 for code in values)
        successful = []
        for i, code in zip(selected, values, strict=True):
            result[i] = code
            if code == 0:
                successful.append(i)
        # A competing, unfinished put can also yield an idempotent success.
        # Confirm readability before acknowledging a page to the other ranks.
        if successful:
            visible = self._readable([keys[i] for i in successful])
            for i, present in zip(successful, visible, strict=True):
                if not present:
                    result[i] = -1
        reply = bytes(code == 0 for code in result)
        for peer in self._peers.values():
            peer.sendall(reply)
        if transfer_failed:
            # Retain the staging allocation and stop reusing it if
            # the transport did not confirm the writes.
            self._failed = True
            self._close_peers()
        return result
