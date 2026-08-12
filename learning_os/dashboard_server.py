from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .dashboard import DashboardService
from .mission_context import MissionContextAssembler


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def make_handler(service: DashboardService, html_path: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            mission = query.get("mission", [None])[0]
            try:
                if parsed.path == "/":
                    body = html_path.read_bytes()
                    self._send(200, "text/html; charset=utf-8", body)
                    return
                if parsed.path == "/api/dashboard":
                    self._send(200, "application/json; charset=utf-8", _json_bytes(service.snapshot(mission)))
                    return
                if parsed.path == "/api/step":
                    self._send(200, "application/json; charset=utf-8", _json_bytes(service.loop.step(mission)))
                    return
                if parsed.path == "/api/context":
                    current = mission or (service.loop.runner.status().get("mission") or {}).get("id")
                    if not current:
                        self._send(400, "application/json; charset=utf-8", _json_bytes({"error": "No active mission"}))
                        return
                    payload = MissionContextAssembler(service.root, service.loop.gates).build(current)
                    self._send(200, "application/json; charset=utf-8", _json_bytes(payload))
                    return
                if parsed.path == "/healthz":
                    self._send(200, "application/json; charset=utf-8", b'{"status":"ok"}')
                    return
                self._send(404, "application/json; charset=utf-8", _json_bytes({"error": "Not found"}))
            except Exception as exc:  # dashboard boundary: return a readable local error
                self._send(500, "application/json; charset=utf-8", _json_bytes({"error": str(exc)}))

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def serve_dashboard(root: str | Path = ".", host: str = "127.0.0.1", port: int = 8765) -> None:
    root_path = Path(root).resolve()
    html_path = root_path / "web" / "dashboard.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Dashboard HTML not found: {html_path}")
    service = DashboardService(root_path)
    server = ThreadingHTTPServer((host, port), make_handler(service, html_path))
    print(f"Learning OS dashboard: http://{host}:{port}")
    print("Read-only runtime surface. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
