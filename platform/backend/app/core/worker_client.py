"""JSON-RPC client for the canonical platform/worker/daemon.py socket."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any

HISTORICAL_SOCKET = Path("/tmp/learningos_worker.sock")
DEFAULT_TIMEOUT_SEC = 2.0
MAX_RESPONSE_BYTES = 1_000_000


def resolve_worker_socket(explicit: str | Path | None = None) -> Path:
    """LEARNINGOS_WORKER_SOCKET, else $LEARNINGOS_HOME/run/worker.sock, else /tmp.

    Keep this algorithm in lockstep with platform/worker/daemon.py.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env_socket = os.environ.get("LEARNINGOS_WORKER_SOCKET")
    if env_socket:
        return Path(env_socket).expanduser()
    env_home = os.environ.get("LEARNINGOS_HOME")
    if env_home:
        home = Path(env_home).expanduser().resolve()
        return home / "run" / "worker.sock"
    return HISTORICAL_SOCKET


def _unavailable(socket_path: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"socket": socket_path}
    if details:
        payload.update(details)
    return {
        "error": {
            "code": "WORKER_UNAVAILABLE",
            "message": message,
            "details": payload,
        }
    }


class WorkerClient:
    def __init__(self, socket_path: str | Path | None = None, timeout: float = DEFAULT_TIMEOUT_SEC) -> None:
        self.socket_path = str(resolve_worker_socket(socket_path))
        self.timeout = timeout

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request = {
            "jsonrpc": "2.0",
            "id": f"req_{uuid.uuid4().hex}",
            "method": method,
            "params": params or {},
        }
        path = self.socket_path
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(path)
            sock.sendall((json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            total = 0
            response: dict[str, Any] | None = None
            while total < MAX_RESPONSE_BYTES:
                try:
                    data = sock.recv(65536)
                except socket.timeout:
                    if method == "shutdown":
                        return {"status": "SHUTDOWN"}
                    return _unavailable(path, "Timed out waiting for the worker", details={"method": method})
                if not data:
                    break
                chunks.append(data)
                total += len(data)
                try:
                    parsed = json.loads(b"".join(chunks).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(parsed, dict):
                    response = parsed
                    break
            if response is None:
                if method == "shutdown":
                    return {"status": "SHUTDOWN"}
                return _unavailable(path, "No response from worker", details={"method": method})
        except FileNotFoundError:
            return _unavailable(path, f"Worker socket not found at {path}")
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as exc:
            return _unavailable(path, f"Worker daemon is not available: {exc}")
        except OSError as exc:
            return _unavailable(path, f"Worker communication error: {exc}")
        except TimeoutError:
            return _unavailable(path, "Timed out contacting the worker")
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

        if "error" in response and "result" not in response:
            err = response["error"]
            if isinstance(err, dict):
                return {"error": err}
            return {"error": {"code": "INTERNAL", "message": str(err), "details": {}}}
        result = response.get("result", {})
        return result if isinstance(result, dict) else {"result": result}

    def health(self) -> bool:
        response = self._rpc("health")
        if not isinstance(response, dict) or "error" in response:
            return False
        return response.get("alive") is True

    def execute(self, code: str, parameters: dict | None = None) -> dict:
        params: dict[str, Any] = {
            "code": code,
            "parameters": parameters or {},
            "limits": {"timeout_sec": 30, "memory_mb": 2048},
        }
        if isinstance(parameters, dict) and "echo" in parameters:
            params["echo"] = parameters["echo"]
        return self._rpc("execute_task", params)

    def cancel(self, job_id: str) -> dict:
        return self._rpc("cancel", {"job_id": job_id})

    def shutdown(self) -> dict:
        return self._rpc("shutdown", {})
