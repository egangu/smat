"""Minimal SGLang process lifecycle used by the TRACE eval-TA scheduler."""

from __future__ import annotations

import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path


_active: set[subprocess.Popen[object]] = set()
_active_lock = threading.Lock()


def _server_ports(preferred: int) -> tuple[int, int]:
    nccl = preferred + 20_000
    if not 0 < preferred <= 65_535:
        raise ValueError("--port-base must be between 1 and 65535")
    port_range = Path("/proc/sys/net/ipv4/ip_local_port_range")
    low, high = (
        map(int, port_range.read_text().split())
        if port_range.exists()
        else (32768, 65535)
    )
    if nccl <= 65_535 and not any(low <= value <= high for value in (preferred, nccl)):
        return preferred, nccl
    # Health/client connections may otherwise occupy a server's port before it binds.
    slots = (low - 1024) // 2
    for _ in range(32 if slots > 0 else 0):
        port = 1024 + 2 * secrets.randbelow(slots)
        try:
            with socket.socket() as http, socket.socket() as collective:
                http.bind(("127.0.0.1", port))
                collective.bind(("0.0.0.0", port + 1))
            return port, port + 1
        except OSError:
            continue
    raise RuntimeError(
        "No available HTTP/NCCL port pair below the ephemeral port range"
    )


def _track(process: subprocess.Popen[object]) -> None:
    with _active_lock:
        _active.add(process)


def _forget(process: subprocess.Popen[object]) -> None:
    with _active_lock:
        _active.discard(process)


def stop(process: subprocess.Popen[object]) -> None:
    # A failed launch parent can leave live SGLang children in its session.
    # The owned process group must be cleaned even after that parent exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        _forget(process)
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        _forget(process)


def stop_all() -> None:
    """Stop every server started by this process, including on TERM/HUP."""

    with _active_lock:
        processes = tuple(_active)
    for process in processes:
        stop(process)


def start(
    model: Path,
    port: int,
    concurrency: int,
    log_path: Path,
    environment: dict[str, str],
) -> tuple[subprocess.Popen[object], str]:
    """Start a server and verify its served checkpoint, not health alone."""

    for attempt in range(2):
        instance_id = f"smat-{uuid.uuid4().hex}"
        candidate, nccl_port = _server_ports(port + attempt * 1_000)
        log = log_path if attempt == 0 else log_path.with_stem(f"{log_path.stem}-retry")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    str(model),
                    "--served-model-name",
                    instance_id,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(candidate),
                    "--nccl-port",
                    str(nccl_port),
                    "--dtype",
                    "bfloat16",
                    "--mem-fraction-static",
                    environment.get("SMAT_EVAL_MEM_FRACTION", "0.30"),
                    "--max-running-requests",
                    str(concurrency),
                    "--max-total-tokens",
                    "262144",
                    "--cuda-graph-max-bs",
                    str(concurrency),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        _track(process)
        endpoint = f"http://127.0.0.1:{candidate}"
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"SGLang server exited with code {process.returncode}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"{endpoint}/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            with urllib.request.urlopen(
                                f"{endpoint}/get_model_info", timeout=2
                            ) as info:
                                served = json.loads(info.read()).get("model_path")
                            if (
                                not isinstance(served, str)
                                or Path(served).resolve() != model.resolve()
                            ):
                                raise RuntimeError(
                                    f"SGLang endpoint identity mismatch at {endpoint}: "
                                    f"expected {model}, served {served}",
                                )
                            with urllib.request.urlopen(
                                f"{endpoint}/v1/models", timeout=2
                            ) as info:
                                identities = json.loads(info.read()).get("data", [])
                            if not any(
                                item.get("id") == instance_id for item in identities
                            ):
                                raise RuntimeError(
                                    f"SGLang server instance identity mismatch at {endpoint}"
                                )
                            if process.poll() is not None:
                                raise RuntimeError(
                                    f"SGLang server exited with code {process.returncode}"
                                )
                            log.with_suffix(".identity.json").write_text(
                                json.dumps(
                                    {
                                        "endpoint": endpoint,
                                        "pid": process.pid,
                                        "requested_model": str(model),
                                        "served_model": served,
                                        "verified_instance_id": instance_id,
                                        "verified_at_unix": time.time(),
                                    },
                                    indent=2,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            return process, endpoint
                except (OSError, http.client.BadStatusLine):
                    time.sleep(1)
            raise TimeoutError(
                f"SGLang server did not become ready within 600s: {endpoint}"
            )
        except BaseException as error:
            stop(process)
            message = log.read_text(encoding="utf-8", errors="replace")
            lower_message = message.lower()
            if (
                attempt == 0
                and isinstance(error, RuntimeError)
                and (
                    "address already in use" in lower_message
                    or "eaddrinuse" in lower_message
                )
            ):
                continue
            raise
    raise RuntimeError("unreachable")
