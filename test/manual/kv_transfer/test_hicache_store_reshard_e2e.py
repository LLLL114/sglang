"""Manual 7-GPU HiCache e2e: TP2/PP2 <-> TP1/PP3 through an owned Store.

Run with --model-path, --build-dir, and --output-dir. The model must be a
text-only MHA/GQA model supported by both topologies. Uses real model weights,
HTTP generation, L2 GPU transfers, native Store, and independent server exits.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

# Match production initialization: SGLang/Torch before the native Store module.
from sglang.srt.mem_cache.utils import get_hash_str


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def wait_server(process, port, log, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited: {log}\n{log.read_text()[-10000:]}")
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"server startup timed out: {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "requests": {},
    }
    children, provider = [], None

    def launch(command, name, env=None):
        logfile = output / f"{name}.log"
        with logfile.open("w") as f:
            p = subprocess.Popen(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        children.append(p)
        result.setdefault("processes", []).append(
            {"name": name, "pid": p.pid, "command": command}
        )
        (output / "progress.json").write_text(json.dumps(result, indent=2))
        return p, logfile

    def generate(port, tokens, label, storage=False):
        request = {
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
        }
        start = time.monotonic()
        response = requests.post(
            f"http://127.0.0.1:{port}/generate", json=request, timeout=180
        )
        response.raise_for_status()
        body = response.json()
        result["requests"][label] = {
            "elapsed": time.monotonic() - start,
            "response": body,
        }
        (output / "progress.json").write_text(json.dumps(result, indent=2))
        details = body["meta_info"].get("cached_tokens_details", {})
        print(
            label,
            json.dumps(
                {
                    "text": body.get("text"),
                    "cached_tokens": body["meta_info"].get("cached_tokens"),
                    "details": details,
                }
            ),
            flush=True,
        )
        if storage:
            assert details.get("storage", 0) >= len(tokens) // 16 * 16, body
        return body

    try:
        from mooncake.reshard.kv_cache import (
            kv_cache_store_catalog_from_json,
            kv_cache_store_manifest_from_json,
        )
        from mooncake.store import MooncakeDistributedStore
        from transformers import AutoTokenizer

        rpc, http, metrics = free_port(), free_port(), free_port()
        master, _ = launch(
            [
                str(Path(args.build_dir) / "mooncake-store/src/mooncake_master"),
                f"--port={rpc}",
                "--rpc_address=127.0.0.1",
                "--rpc_thread_num=4",
                "--enable_metric_reporting=false",
                f"--metrics_port={metrics}",
                "--enable_http_metadata_server=true",
                "--http_metadata_server_host=127.0.0.1",
                f"--http_metadata_server_port={http}",
            ],
            "master",
        )
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", http), timeout=0.2):
                    break
            except OSError:
                assert master.poll() is None
                time.sleep(0.1)
        provider = MooncakeDistributedStore()
        assert (
            provider.setup(
                f"127.0.0.1:{free_port()}",
                f"http://127.0.0.1:{http}/metadata",
                2 << 30,
                32 << 20,
                "tcp",
                "",
                f"127.0.0.1:{rpc}",
            )
            == 0
        )
        tag = "phase-c-" + uuid.uuid4().hex
        served_name = "kv-reshard-qwen"
        config = {
            "local_hostname": "127.0.0.1",
            "metadata_server": f"http://127.0.0.1:{http}/metadata",
            "global_segment_size": 0,
            "protocol": "tcp",
            "device_name": "",
            "master_server_address": f"127.0.0.1:{rpc}",
            "check_server": False,
            "standalone_storage": False,
            "extra_backend_tag": tag,
            "prefetch_threshold": 16,
            "kv_reshard": {"model_revision": args.model_revision},
        }
        config_path = output / "storage.json"
        config_path.write_text(json.dumps(config))
        result["storage"] = config

        def server(name, tp, pp, gpus):
            port = free_port()
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpus,
                HF_HUB_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
            )
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                args.model_path,
                "--served-model-name",
                served_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tp-size",
                str(tp),
                "--pp-size",
                str(pp),
                "--dtype",
                "float16",
                "--mem-fraction-static",
                "0.5",
                "--max-total-tokens",
                "4096",
                "--context-length",
                "4096",
                "--max-running-requests",
                "4",
                "--chunked-prefill-size",
                "512",
                "--attention-backend",
                "triton",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-custom-all-reduce",
                "--disable-overlap-schedule",
                "--enable-hierarchical-cache",
                "--page-size",
                "16",
                "--hicache-size",
                "1",
                "--hicache-write-policy",
                "write_through",
                "--hicache-mem-layout",
                "page_first",
                "--hicache-io-backend",
                "kernel",
                "--hicache-storage-backend",
                "mooncake",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-storage-backend-extra-config",
                "@" + str(config_path),
            ]
            p, log = launch(command, name, env)
            return p, port, log

        a, ap, alog = server("tp2pp2", 2, 2, "0,1,2,3")
        b, bp, blog = server("tp1pp3", 1, 3, "4,5,6")
        wait_server(a, ap, alog)
        wait_server(b, bp, blog)
        print("BOTH_SERVERS_READY", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True
        )
        ta = tokenizer.encode(
            "Alpha records: " + "The river flows through the green valley. " * 200
        )[:1025]
        tb = tokenizer.encode(
            "Beta records: " + "Stars shine over the quiet mountain. " * 200
        )[:769]
        ca = generate(ap, ta, "cold_tp2pp2")
        cb = generate(bp, tb, "cold_tp1pp3")

        import re

        catalog_keys = set(
            re.findall(
                r"catalog=(kv/[0-9a-f]+/layouts)", alog.read_text() + blog.read_text()
            )
        )
        assert len(catalog_keys) == 1, catalog_keys
        catalog_key = catalog_keys.pop()
        status, data, _ = provider.read_metadata_for_update(catalog_key)
        assert status == 1
        catalog = kv_cache_store_catalog_from_json(data.decode())
        assert len(catalog.entries) == 2, catalog
        manifests = [
            kv_cache_store_manifest_from_json(
                provider.get(entry.manifest_key(catalog.model_domain)).decode()
            )
            for entry in catalog.entries
        ]
        result["catalog"] = json.loads(data)

        def wait_uploaded(tokens, tp):
            manifest = next(m for m in manifests if len(m.layout.shards) == tp)
            hashes = get_hash_str(tokens[: len(tokens) // 16 * 16], None, page_size=16)
            keys = [
                key
                for h in hashes
                for key in manifest.object_keys(
                    f"{tag}_{served_name}_{h}_m{manifest.model_domain}"
                )
            ]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if all(value == 1 for value in provider.batch_is_exist(keys)):
                    return
                time.sleep(0.2)
            raise AssertionError(f"incomplete upload for {len(keys)} keys")

        wait_uploaded(ta, 4)
        wait_uploaded(tb, 3)
        wa = generate(bp, ta, "tp2pp2_to_tp1pp3", storage=True)
        wb = generate(ap, tb, "tp1pp3_to_tp2pp2", storage=True)
        assert wa["text"] == ca["text"], (wa, ca)
        assert wb["text"] == cb["text"], (wb, cb)
        stop(a)
        stop(b)
        b, bp, blog = server("tp1pp3_after_source_exit", 1, 3, "4,5,6")
        wait_server(b, bp, blog)
        after_a = generate(bp, ta, "tp2pp2_source_exited", storage=True)
        assert after_a["text"] == ca["text"]
        stop(b)
        a, ap, alog = server("tp2pp2_after_source_exit", 2, 2, "0,1,2,3")
        wait_server(a, ap, alog)
        after_b = generate(ap, tb, "tp1pp3_source_exited", storage=True)
        assert after_b["text"] == cb["text"]
        result["status"] = "passed"
        print("E2E_PASSED", flush=True)
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        for process in reversed(children):
            # Keep the Store Master until providers have closed.
            if process is not children[0]:
                stop(process)
        if provider is not None:
            provider.close()
        if children:
            stop(children[0])
        (output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
