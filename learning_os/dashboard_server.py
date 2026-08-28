from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .app import AppService
from .mission_context import MissionContextAssembler


MAX_REQUEST_BYTES = 1_000_000


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def make_handler(service: AppService, html_path: Path, m01_path: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: object) -> None:
            self._send(status, "application/json; charset=utf-8", _json_bytes(payload))

        def _read_json(self) -> dict[str, object]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length") from exc
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body is too large")
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Request body must be valid UTF-8 JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            return payload

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            mission = query.get("mission", [None])[0]
            try:
                if parsed.path == "/":
                    body = html_path.read_bytes()
                    self._send(200, "text/html; charset=utf-8", body)
                    return
                if parsed.path == "/m01":
                    body = m01_path.read_bytes()
                    self._send(200, "text/html; charset=utf-8", body)
                    return
                if parsed.path == "/api/dashboard":
                    self._send_json(200, service.snapshot(mission))
                    return
                if parsed.path == "/api/m01":
                    self._send_json(200, service.m01_view())
                    return
                if parsed.path == "/api/step":
                    self._send_json(200, service.loop.step(mission))
                    return
                if parsed.path == "/api/context":
                    current = mission or (service.loop.runner.status().get("mission") or {}).get("id")
                    if not current:
                        self._send_json(400, {"error": "No active mission"})
                        return
                    payload = MissionContextAssembler(service.root, service.loop.gates).build(current)
                    self._send_json(200, payload)
                    return
                if parsed.path == "/healthz":
                    self._send_json(200, {"status": "ok", "surface": "learningos-app", "version": "v2-m01-reference"})
                    return
                self._send_json(404, {"error": "Not found"})
            except (ValueError, KeyError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/start":
                    self._send_json(200, service.start(payload.get("mission_id")))
                    return
                if parsed.path == "/api/evidence":
                    self._send_json(201, service.record_evidence(payload))
                    return
                if parsed.path == "/api/gate":
                    self._send_json(200, service.run_gate(payload.get("mission_id")))
                    return
                if parsed.path == "/api/player/complete":
                    self._send_json(200, service.complete_player_step(payload))
                    return
                if parsed.path == "/api/player/reset":
                    self._send_json(200, service.reset_player(payload.get("mission_id")))
                    return
                if parsed.path == "/api/tutor":
                    self._send_json(200, service.ask_tutor(payload))
                    return
                if parsed.path == "/api/lab/run":
                    self._send_json(200, service.run_lab(payload))
                    return
                if parsed.path == "/api/m01/stage":
                    self._send_json(200, service.m01_save_stage(payload))
                    return
                if parsed.path == "/api/m01/prediction":
                    self._send_json(200, service.m01_prediction(payload))
                    return
                if parsed.path == "/api/m01/experiment/run":
                    self._send_json(200, service.m01_run_experiment(payload))
                    return
                if parsed.path == "/api/m01/reflection":
                    self._send_json(200, service.m01_reflection(payload))
                    return
                if parsed.path == "/api/retention/complete":
                    self._send_json(200, service.complete_retention(payload.get("event_id"), payload.get("passed")))
                    return
                if parsed.path == "/api/sidequest/open":
                    self._send_json(201, service.open_side_quest(payload))
                    return
                if parsed.path == "/api/sidequest/close":
                    self._send_json(200, service.close_side_quest(payload))
                    return
                self._send_json(404, {"error": "Not found"})
            except (ValueError, KeyError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def serve_app(root: str | Path = ".", host: str = "127.0.0.1", port: int = 8765) -> None:
    root_path = Path(root).resolve()
    html_path = root_path / "web" / "dashboard.html"
    m01_path = root_path / "web" / "m01.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Learning OS app HTML not found: {html_path}")
    if not m01_path.exists():
        raise FileNotFoundError(f"M01 guided workspace HTML not found: {m01_path}")
    service = AppService(root_path)
    server = ThreadingHTTPServer((host, port), make_handler(service, html_path, m01_path))
    print(f"Learning OS app: http://{host}:{port}")
    print(f"M01 guided workspace: http://{host}:{port}/m01")
    print("Local-first learner surface. State is stored under tracking/. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve_dashboard(root: str | Path = ".", host: str = "127.0.0.1", port: int = 8765) -> None:
    """Backward-compatible alias for the original dashboard command."""
    serve_app(root, host, port)
