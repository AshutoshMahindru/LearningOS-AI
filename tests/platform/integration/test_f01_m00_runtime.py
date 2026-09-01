"""WP400 F01/M00 lifecycle: predict → execute WP-137 → evidence → gate → next-action."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
DAEMON_PATH = REPO_ROOT / "platform" / "worker" / "daemon.py"
F01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "f01"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_EXPERIMENT = "stage_02_experiment"
STAGE_TRANSFER = "stage_03_transfer"
STAGE_GATE = "stage_04_gate"

PREDICT_BODY = {
    "hypothesis": "scale=2 doubles each value",
    "expected_values": {"series": [2, 4, 6, 8], "scale": 2},
}
EXECUTE_BODY = {
    "code": (
        "series = [value * parameters['scale'] for value in parameters['series']]\n"
        "print(series)\n"
        "{'type': 'metric', 'title': 'scaled', 'payload': {'series': series, 'scale': parameters['scale']}}"
    ),
    "parameters": {"scale": 2, "series": [1, 2, 3, 4]},
}
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _start_worker(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(DAEMON_PATH)],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_worker(proc: subprocess.Popen[bytes] | None, sig: signal.Signals = signal.SIGTERM) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(sig)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _outside_repo(path: Path) -> None:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    with pytest.raises(ValueError):
        resolved.relative_to(repo)


@pytest.fixture
def f01_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g4-f01-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g4-f01-{uuid.uuid4().hex}.sock")
    sock.unlink(missing_ok=True)
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-never-leak")
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(home)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        yield {"home": home, "sock": sock, "env": env}
    finally:
        sock.unlink(missing_ok=True)
        shutil.rmtree(home, ignore_errors=True)


def _bootstrap(client):
    boot = client.post("/api/v1/auth/bootstrap")
    assert boot.status_code == 200, boot.text
    token = boot.json()["token"]
    assert boot.json()["token_type"] == "bearer"
    return {"Authorization": f"Bearer {token}"}


def test_platform_has_no_f01_special_case_routes_or_conditionals():
    platform = REPO_ROOT / "platform"
    skip_dirs = {"node_modules", "dist", "__pycache__", ".git"}
    eq_f01 = re.compile(r"""==\s*["']F01["']""")
    route_f01 = re.compile(r"/missions/F01")
    hits: list[str] = []
    for path in platform.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if "fixtures" in path.parts and "f01" in path.parts:
            continue
        if path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".css", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        if eq_f01.search(text) or route_f01.search(text):
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_frontend_does_not_embed_provider_secrets():
    frontend = REPO_ROOT / "platform" / "frontend" / "src"
    skip_dirs = {"node_modules", "dist", "__pycache__"}
    secret_re = re.compile(r"OPENAI_API_KEY|ANTHROPIC_API_KEY|AZURE_OPENAI|sk-[A-Za-z0-9]{12,}")
    hits: list[str] = []
    for path in frontend.rglob("*"):
        if not path.is_file() or any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        if secret_re.search(text):
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_f01_m00_runtime_roundtrip(f01_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = f01_env["home"]
    sock: Path = f01_env["sock"]
    env: dict[str, str] = f01_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(F01_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g4.fixture.f01"
            assert loaded_body.get("version") == "4.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M00" in mission_ids
            assert "F01" not in mission_ids

            mission = client.get("/api/v1/missions/M00", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M00"
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == [STAGE_ORIENT, STAGE_EXPERIMENT, STAGE_TRANSFER, STAGE_GATE]

            learner = client.post(
                "/api/v1/learners",
                json={"username": "f01-learner", "display_name": "F01 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M00", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M00"
            assert session_body["status"] == "ACTIVE"
            assert session_body["current_stage_id"] == STAGE_ORIENT
            session_id = session_body["session_id"]

            entered = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/enter",
                headers=headers,
            )
            assert entered.status_code == 200, entered.text
            assert entered.json()["current_stage_id"] == STAGE_ORIENT
            assert entered.json()["status"] == "ACTIVE"

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            got = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert got.status_code == 200, got.text
            assert got.json()["current_stage_id"] == STAGE_ORIENT
            assert got.json()["status"] == "ACTIVE"
            assert got.json().get("current_stage", {}).get("id") == STAGE_ORIENT

            again = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/enter",
                headers=headers,
            )
            assert again.status_code == 200, again.text
            assert again.json()["resumed"] is True

            submitted = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/submit",
                json={"explanation": "framed the generic journey", "artifacts": [{"artifact_type": "markdown"}]},
                headers=headers,
            )
            assert submitted.status_code == 200, submitted.text
            assert submitted.json()["current_stage_id"] == STAGE_EXPERIMENT

            lab_enter = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/enter",
                headers=headers,
            )
            assert lab_enter.status_code == 200, lab_enter.text
            assert lab_enter.json()["stage_type"] == "experiment"

            blocked = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXECUTE_BODY,
                headers=headers,
            )
            assert blocked.status_code == 409, blocked.text
            assert blocked.json()["error"]["code"] == "CONFLICT"
            assert blocked.json()["error"]["details"].get("reason") == "PREDICTION_REQUIRED"

            predicted = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/predict",
                json=PREDICT_BODY,
                headers=headers,
            )
            assert predicted.status_code == 200, predicted.text
            assert predicted.json()["is_sealed"] is True
            assert predicted.json()["prediction_hash"] != "dummy_hash"
            assert len(predicted.json()["prediction_hash"]) == 64

            executed = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXECUTE_BODY,
                headers=headers,
            )
            assert executed.status_code == 200, executed.text
            exec_body = executed.json()
            assert exec_body["status"] == "SUCCESS", exec_body
            assert exec_body["status"] != "UNSUPPORTED"
            assert exec_body["code_hash"] != "dummy_hash"
            structured = exec_body.get("structured_result") or {}
            for key in ("execution_id", "status", "exit_code", "duration_ms", "blocks"):
                assert key in structured, key
            assert structured["status"] in WP137_STATUSES
            assert structured["status"] == "SUCCESS"
            assert structured["execution_id"] == exec_body["execution_id"]
            blocks = structured.get("blocks") or exec_body.get("blocks") or []
            assert isinstance(blocks, list) and blocks
            assert blocks[0]["type"] == "metric"
            assert blocks[0]["payload"]["series"] == [2, 4, 6, 8]
            diagnostics = structured.get("diagnostics") or exec_body.get("diagnostics") or {}
            assert "2" in str(diagnostics.get("stdout") or "")
            assert "dummy_hash" not in json.dumps(exec_body)
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)

            lab_submit = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/submit",
                json={
                    "explanation": "observed scaled series [2, 4, 6, 8]",
                    "artifacts": [{"artifact_type": "metric"}],
                },
                headers=headers,
            )
            assert lab_submit.status_code == 200, lab_submit.text
            assert lab_submit.json()["current_stage_id"] == STAGE_TRANSFER
            assert lab_submit.json()["payload_hash"] != "dummy_hash"

            transfer_enter = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_TRANSFER}/enter",
                headers=headers,
            )
            assert transfer_enter.status_code == 200, transfer_enter.text
            transfer_submit = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_TRANSFER}/submit",
                json={
                    "explanation": "fresh-case transform without assistance",
                    "artifacts": [{"artifact_type": "metric"}],
                },
                headers=headers,
            )
            assert transfer_submit.status_code == 200, transfer_submit.text
            assert transfer_submit.json()["current_stage_id"] == STAGE_GATE

            gate_enter = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_GATE}/enter",
                headers=headers,
            )
            assert gate_enter.status_code == 200, gate_enter.text
            gate_submit = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_GATE}/submit",
                json={"explanation": "ready for gate evaluation"},
                headers=headers,
            )
            assert gate_submit.status_code == 200, gate_submit.text

            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = gate_body.get("competency_increments") or []
            assert {item["competency_id"] for item in increments} == {
                "comp.generic.orientation",
                "comp.generic.experiment",
                "comp.generic.transfer",
            }
            assert all("comp.sys." not in json.dumps(item) for item in increments)

            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            evidence_body = evidence.json()
            claims = evidence_body.get("evidence") or []
            assert claims
            assert "dummy_hash" not in json.dumps(evidence_body)
            provenance_hashes = []
            for claim in claims:
                provenance = claim.get("provenance") or claim
                for key in ("artifact_hash", "runner_hash", "curriculum_sha"):
                    digest = provenance.get(key) or claim.get(key)
                    if digest:
                        provenance_hashes.append(digest)
                        assert digest != "dummy_hash"
                        assert len(str(digest)) == 64
            assert provenance_hashes

            today = resumed.get(f"/api/v1/learners/{learner_id}/next-action", headers=headers)
            assert today.status_code == 200, today.text
            action_body = today.json()
            assert action_body["action"] == "IDLE"
            assert action_body["reason"] == "ALL_MISSIONS_COMPLETE"
            competencies = action_body.get("competencies") or []
            assert {item["competency_id"] for item in competencies} == {
                "comp.generic.orientation",
                "comp.generic.experiment",
                "comp.generic.transfer",
            }
            assert "M01" not in json.dumps(action_body)

            final = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert final.status_code == 200, final.text
            assert final.json()["status"] == "COMPLETED"
            assert "dummy_hash" not in json.dumps(final.json())
            assert "openai" not in sys.modules

            tutor = resumed.post(
                "/api/v1/tutor/chat",
                json={
                    "session_id": session_id,
                    "stage_id": STAGE_ORIENT,
                    "role": "learner",
                    "prompt": "hello",
                },
                headers=headers,
            )
            assert tutor.status_code == 501, tutor.text
            assert tutor.json()["error"]["code"] == "TUTOR_NOT_AVAILABLE"

        completed = subprocess.run(
            [sys.executable, str(STATE_GUARD), "--repo", str(REPO_ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert not (REPO_ROOT / ".learningos").exists()
        assert not (REPO_ROOT / "learningos.db").exists()
    finally:
        _stop_worker(worker)


def test_live_worker_client_denies_io_open_db_write(f01_env):
    """Independent-review probe: import io; io.open(db, 'w') on live WorkerClient."""
    from app.core.worker_client import WorkerClient

    home: Path = f01_env["home"]
    sock: Path = f01_env["sock"]
    env: dict[str, str] = f01_env["env"]
    db_path = home / "learningos.db"
    db_path.write_bytes(b"untouched")
    repo_marker = REPO_ROOT / f".wp400-pwn-{uuid.uuid4().hex}.txt"
    limits = {"limits": {"timeout_sec": 5, "memory_mb": 256}}
    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        client = WorkerClient(sock)
        assert _wait_until(client.health)
        allowed = client.execute("print('sandbox-ok')", limits)
        assert allowed.get("status") == "SUCCESS", allowed
        assert allowed.get("status") != "UNSUPPORTED"
        probe = client.execute(
            f"import io\nio.open({str(db_path)!r}, 'w').write('corrupt-via-io')",
            limits,
        )
        assert probe.get("status") in {"DENIED", "FAILED"}, probe
        assert db_path.read_bytes() == b"untouched"
        tree = client.execute(
            f"import io\nio.open({str(repo_marker)!r}, 'w').write('pwned')",
            limits,
        )
        assert tree.get("status") in {"DENIED", "FAILED"}, tree
        assert not repo_marker.exists()
        client.shutdown()
    finally:
        repo_marker.unlink(missing_ok=True)
        _stop_worker(worker)
