"""G5 M04 on the generic runtime: predict → execute WP-137 quality table → gate."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
DAEMON_PATH = REPO_ROOT / "platform" / "worker" / "daemon.py"
M04_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M04"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(M04_PACKAGE) not in sys.path:
    sys.path.insert(0, str(M04_PACKAGE))

from g5.reference.M04.experiment import QUALITY_EXECUTE_SOURCE, orders_csv_text  # noqa: E402

STAGE_ORIENT = "stage_01_orientation"
STAGE_EXPERIMENT = "stage_02_experiment"
STAGE_FAILURE = "stage_03_controlled_failure"
STAGE_TRANSFER = "stage_04_transfer"
STAGE_GATE = "stage_05_gate"

PREDICT_BODY = {
    "hypothesis": "36 raw rows with one exact duplicate; conflicts and outliers stay visible",
    "expected_values": {"raw_rows": 36, "exact_duplicate_rows": 1},
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
def m04_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g5-m04-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g5-m04-{uuid.uuid4().hex}.sock")
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


def test_m04_runtime_quality_roundtrip(m04_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = m04_env["home"]
    sock: Path = m04_env["sock"]
    env: dict[str, str] = m04_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M04_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M04"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M04" in mission_ids

            mission = client.get("/api/v1/missions/M04", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M04"
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == [
                STAGE_ORIENT,
                STAGE_EXPERIMENT,
                STAGE_FAILURE,
                STAGE_TRANSFER,
                STAGE_GATE,
            ]

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m04-learner", "display_name": "M04 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M04", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M04"
            assert session_body["status"] == "ACTIVE"
            assert session_body["current_stage_id"] == STAGE_ORIENT
            session_id = session_body["session_id"]

            entered = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/enter",
                headers=headers,
            )
            assert entered.status_code == 200, entered.text

            submitted = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/submit",
                json={
                    "explanation": "framed lossless load, defect classes and V01 quality evidence",
                    "artifacts": [{"artifact_type": "markdown"}],
                },
                headers=headers,
            )
            assert submitted.status_code == 200, submitted.text
            assert submitted.json()["current_stage_id"] == STAGE_EXPERIMENT

            lab_enter = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/enter",
                headers=headers,
            )
            assert lab_enter.status_code == 200, lab_enter.text
            assert lab_enter.json()["stage_type"] == "experiment"

            blocked = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json={
                    "code": QUALITY_EXECUTE_SOURCE,
                    "parameters": {"csv_text": orders_csv_text()},
                },
                headers=headers,
            )
            assert blocked.status_code == 409, blocked.text
            assert blocked.json()["error"]["code"] == "CONFLICT"
            assert blocked.json()["error"]["details"].get("reason") == "PREDICTION_REQUIRED"

            predicted = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/predict",
                json=PREDICT_BODY,
                headers=headers,
            )
            assert predicted.status_code == 200, predicted.text
            assert predicted.json()["is_sealed"] is True
            assert predicted.json()["prediction_hash"] != "dummy_hash"
            assert len(predicted.json()["prediction_hash"]) == 64

            executed = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json={
                    "code": QUALITY_EXECUTE_SOURCE,
                    "parameters": {"csv_text": orders_csv_text()},
                },
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
            blocks = structured.get("blocks") or exec_body.get("blocks") or []
            assert isinstance(blocks, list) and len(blocks) >= 2
            types = [block["type"] for block in blocks]
            assert "table" in types
            assert "metric" in types
            table = next(block for block in blocks if block["type"] == "table")
            metric = next(block for block in blocks if block["type"] == "metric")
            accounting = {row[0]: row[1] for row in table["payload"]["rows"]}
            assert accounting["raw_rows"] == 36
            assert accounting["exact_duplicate_rows"] == 1
            assert accounting["rows_after_exact_dedupe"] == 35
            assert metric["payload"]["raw_rows"] == 36
            assert metric["payload"]["exact_duplicate_rows"] == 1
            diagnostics = structured.get("diagnostics") or exec_body.get("diagnostics") or {}
            assert "36" in str(diagnostics.get("stdout") or "")
            assert "dummy_hash" not in json.dumps(exec_body)
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)

            lab_submit = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/submit",
                json={
                    "explanation": "observed 36 raw rows, 1 exact duplicate logged, 35 retained",
                    "artifacts": [{"artifact_type": "table"}],
                },
                headers=headers,
            )
            assert lab_submit.status_code == 200, lab_submit.text
            assert lab_submit.json()["current_stage_id"] == STAGE_FAILURE
            assert lab_submit.json()["payload_hash"] != "dummy_hash"

            fail_enter = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_FAILURE}/enter",
                headers=headers,
            )
            assert fail_enter.status_code == 200, fail_enter.text
            fail_submit = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_FAILURE}/submit",
                json={
                    "explanation": "dropna is the first irreversible loss; repair logs exact duplicates only",
                    "artifacts": [{"artifact_type": "metric"}],
                },
                headers=headers,
            )
            assert fail_submit.status_code == 200, fail_submit.text
            assert fail_submit.json()["current_stage_id"] == STAGE_TRANSFER

            transfer_enter = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_TRANSFER}/enter",
                headers=headers,
            )
            assert transfer_enter.status_code == 200, transfer_enter.text
            transfer_submit = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_TRANSFER}/submit",
                json={
                    "explanation": "fresh inventory: exact duplicate logged, SKU conflict preserved, bulk receipt retained",
                    "artifacts": [{"artifact_type": "table"}],
                },
                headers=headers,
            )
            assert transfer_submit.status_code == 200, transfer_submit.text
            assert transfer_submit.json()["current_stage_id"] == STAGE_GATE

            gate_enter = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_GATE}/enter",
                headers=headers,
            )
            assert gate_enter.status_code == 200, gate_enter.text
            gate_submit = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_GATE}/submit",
                json={"explanation": "quality evidence ready for gate evaluation"},
                headers=headers,
            )
            assert gate_submit.status_code == 200, gate_submit.text

            gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = gate_body.get("competency_increments") or []
            assert {item["competency_id"] for item in increments} == {
                "comp.data.schema_normalization",
                "comp.data.missingness_decisions",
                "comp.data.outlier_judgment",
            }

            evidence = client.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            claims = evidence.json().get("evidence") or []
            assert claims
            assert "dummy_hash" not in json.dumps(evidence.json())

            today = client.get(f"/api/v1/learners/{learner_id}/next-action", headers=headers)
            assert today.status_code == 200, today.text
            action_body = today.json()
            assert action_body["action"] == "IDLE"
            assert action_body["reason"] == "ALL_MISSIONS_COMPLETE"

            final = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert final.status_code == 200, final.text
            assert final.json()["status"] == "COMPLETED"

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
