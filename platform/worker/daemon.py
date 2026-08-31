#!/usr/bin/env python3
"""Canonical LearningOS G3 execution worker (JSON-RPC 2.0 over a Unix socket).

G3 implements start/health/cancel/shutdown and a job boundary only. User code is
never exec'd or eval'd; the WP400 sandbox is the execution surface.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

HISTORICAL_SOCKET = Path("/tmp/learningos_worker.sock")
MAX_REQUEST_BYTES = 1_000_000
G3_UNSUPPORTED_REASON = "G3 job boundary — execution sandbox is WP400"


def resolve_socket_path(explicit: str | None = None) -> Path:
    """LEARNINGOS_WORKER_SOCKET, else $LEARNINGOS_HOME/run/worker.sock, else /tmp."""
    if explicit:
        return Path(explicit).expanduser()
    env_socket = os.environ.get("LEARNINGOS_WORKER_SOCKET")
    if env_socket:
        return Path(env_socket).expanduser()
    env_home = os.environ.get("LEARNINGOS_HOME")
    if env_home:
        home = Path(env_home).expanduser().resolve()
        return home / "run" / "worker.sock"
    return HISTORICAL_SOCKET


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _read_request(conn: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < MAX_REQUEST_BYTES:
        try:
            data = conn.recv(65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
        total += len(data)
        blob = b"".join(chunks)
        try:
            json.loads(blob.decode("utf-8"))
            return blob
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return b"".join(chunks)


class WorkerDaemon:
    def __init__(self, socket_path: Path | None = None) -> None:
        self.socket_path = socket_path or resolve_socket_path()
        self._server: socket.socket | None = None
        self._stopping = False
        self._jobs: dict[str, str] = {}
        self.pid = os.getpid()

    def _install_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum: int, _frame: object) -> None:
        self._stopping = True
        server = self._server
        if server is not None:
            try:
                server.close()
            except OSError:
                pass

    def _prepare_socket(self) -> socket.socket:
        path = self.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.settimeout(0.2)
                probe.connect(str(path))
                probe.close()
            except OSError:
                path.unlink(missing_ok=True)
            else:
                raise SystemExit(f"Worker already listening on {path}")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(path))
        except OSError as exc:
            server.close()
            raise SystemExit(f"Unable to bind worker socket {path}: {exc}") from exc
        server.listen(5)
        return server

    def _cleanup(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        try:
            self.socket_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _health(self) -> dict[str, Any]:
        return {"alive": True, "pid": self.pid}

    def _execute_task(self, params: dict[str, Any]) -> dict[str, Any]:
        # Job boundary only: never exec/eval/compile/import user payloads (WP400).
        job_id = str(params.get("job_id") or f"job_{uuid.uuid4().hex}")
        echo = params.get("echo")
        nested = params.get("parameters")
        if echo is None and isinstance(nested, dict):
            echo = nested.get("echo")
        if echo is not None:
            self._jobs[job_id] = "ACCEPTED"
            return {"status": "ACCEPTED", "job_id": job_id, "echo": echo}
        self._jobs[job_id] = "UNSUPPORTED"
        return {
            "status": "UNSUPPORTED",
            "job_id": job_id,
            "reason": G3_UNSUPPORTED_REASON,
        }

    def _cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        job_id = str(params.get("job_id") or "")
        if not job_id:
            return {"status": "NOT_FOUND", "job_id": job_id}
        if job_id in self._jobs:
            self._jobs[job_id] = "CANCELLED"
            return {"status": "CANCELLED", "job_id": job_id}
        return {"status": "NOT_FOUND", "job_id": job_id}

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method in {"health", "ping"}:
            return self._health()
        if method == "execute_task":
            return self._execute_task(params)
        if method == "cancel":
            return self._cancel(params)
        if method == "shutdown":
            self._stopping = True
            return {"status": "SHUTDOWN", "pid": self.pid}
        raise KeyError(method)

    def _handle(self, conn: socket.socket) -> None:
        raw = _read_request(conn)
        if not raw:
            return
        try:
            req = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            conn.sendall(
                _json_dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                )
            )
            return
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0":
            conn.sendall(
                _json_dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req.get("id") if isinstance(req, dict) else None,
                        "error": {"code": -32600, "message": "Invalid Request"},
                    }
                )
            )
            return
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") if isinstance(req.get("params"), dict) else {}
        try:
            if not isinstance(method, str):
                raise KeyError("method")
            result = self._dispatch(method, params)
            response: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "result": result}
        except KeyError:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
        conn.sendall(_json_dumps(response))
        if method == "shutdown":
            self._stopping = True

    def serve_forever(self) -> int:
        self._install_signals()
        self._server = self._prepare_socket()
        print(f"Worker daemon listening on {self.socket_path}", flush=True)
        try:
            while not self._stopping:
                server = self._server
                if server is None:
                    break
                try:
                    conn, _addr = server.accept()
                except OSError:
                    if self._stopping:
                        break
                    continue
                try:
                    self._handle(conn)
                except OSError:
                    pass
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        finally:
            self._cleanup()
        return 0


def main() -> int:
    try:
        return WorkerDaemon().serve_forever()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1


if __name__ == "__main__":
    raise SystemExit(main())
