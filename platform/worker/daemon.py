#!/usr/bin/env python3
"""Canonical LearningOS execution worker (JSON-RPC 2.0 over a Unix socket).

Learner code is executed only inside an isolated subprocess (WP400 sandbox).
This daemon never calls exec/eval/compile on user payloads and never imports
sqlite3 — a worker crash must not corrupt the learner database.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

HISTORICAL_SOCKET = Path("/tmp/learningos_worker.sock")
MAX_REQUEST_BYTES = 1_000_000
G3_UNSUPPORTED_REASON = "G3 job boundary — execution sandbox is WP400"

_WORKER_DIR = Path(__file__).resolve().parent
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))

from sandbox import (  # noqa: E402
    SandboxViolation,
    coerce_memory_mb,
    coerce_timeout_sec,
    kill_process_group,
    run_job,
    sanitize_job_id,
)


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


class _Job:
    def __init__(self, job_id: str, status: str) -> None:
        self.job_id = job_id
        self.status = status
        self.pgid: int | None = None
        self.cancel_event = threading.Event()


class WorkerDaemon:
    def __init__(self, socket_path: Path | None = None) -> None:
        self.socket_path = socket_path or resolve_socket_path()
        self._server: socket.socket | None = None
        self._stopping = False
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
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
        server.listen(16)
        server.settimeout(0.5)
        return server

    def _kill_all_jobs(self) -> None:
        with self._lock:
            pgids = [job.pgid for job in self._jobs.values() if job.pgid is not None]
            for job in self._jobs.values():
                job.cancel_event.set()
        for pgid in pgids:
            kill_process_group(None, pgid)

    def _cleanup(self) -> None:
        self._kill_all_jobs()
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

    def _register_job(self, job_id: str, status: str) -> _Job:
        rec = _Job(job_id, status)
        with self._lock:
            self._jobs[job_id] = rec
        return rec

    def _set_pgid(self, job_id: str, pgid: int) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.pgid = pgid

    def _set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.status = status

    def _execute_task(self, params: dict[str, Any]) -> dict[str, Any]:
        echo = params.get("echo")
        nested = params.get("parameters") if isinstance(params.get("parameters"), dict) else {}
        if echo is None and isinstance(nested, dict):
            echo = nested.get("echo")
        code = params.get("code") if isinstance(params.get("code"), str) else ""
        raw_id = str(params.get("job_id") or "")
        try:
            job_id = sanitize_job_id(raw_id or None)
        except SandboxViolation:
            job_id = uuid.uuid4().hex
            if code.strip():
                rec = self._register_job(job_id, "DENIED")
                rec.status = "DENIED"
                return {
                    "status": "DENIED",
                    "job_id": job_id,
                    "reason": "invalid job_id",
                    "exit_code": 1,
                }

        if echo is not None and not code.strip():
            self._register_job(job_id, "ACCEPTED")
            return {"status": "ACCEPTED", "job_id": job_id, "echo": echo}
        if not code.strip():
            self._register_job(job_id, "UNSUPPORTED")
            return {
                "status": "UNSUPPORTED",
                "job_id": job_id,
                "reason": G3_UNSUPPORTED_REASON,
            }

        limits = params.get("limits") if isinstance(params.get("limits"), dict) else {}
        if not limits and isinstance(nested.get("limits"), dict):
            limits = nested["limits"]
        timeout_sec = coerce_timeout_sec(limits.get("timeout_sec"))
        memory_mb = coerce_memory_mb(limits.get("memory_mb"))
        rec = self._register_job(job_id, "RUNNING")

        def _pgid(pgid: int) -> None:
            self._set_pgid(job_id, pgid)

        try:
            result = run_job(
                {
                    "job_id": job_id,
                    "code": code,
                    "parameters": nested,
                    "timeout_sec": timeout_sec,
                    "memory_mb": memory_mb,
                    "limits": limits,
                    "data_home": os.environ.get("LEARNINGOS_HOME"),
                },
                cancel_event=rec.cancel_event,
                pgid_callback=_pgid,
            )
        except SandboxViolation as exc:
            result = {
                "status": "DENIED",
                "job_id": job_id,
                "reason": str(exc),
                "exit_code": 1,
            }
        except Exception as exc:
            result = {
                "status": "CRASHED",
                "job_id": job_id,
                "reason": f"sandbox error: {exc}",
                "exit_code": -1,
            }
        status = str(result.get("status") or "FAILED")
        self._set_status(job_id, status)
        with self._lock:
            if rec.pgid is not None:
                rec.pgid = None
        result["job_id"] = job_id
        return result

    def _cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        job_id = str(params.get("job_id") or "")
        if not job_id:
            return {"status": "NOT_FOUND", "job_id": job_id}
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return {"status": "NOT_FOUND", "job_id": job_id}
            pgid = rec.pgid
            rec.cancel_event.set()
            rec.status = "CANCELLED"
        if pgid is not None:
            kill_process_group(None, pgid)
        return {"status": "CANCELLED", "job_id": job_id}

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

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            self._handle(conn)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

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
                except (TimeoutError, socket.timeout):
                    continue
                except OSError:
                    if self._stopping:
                        break
                    continue
                worker = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True)
                worker.start()
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
