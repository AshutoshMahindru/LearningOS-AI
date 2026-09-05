"""G6 M11 on the frozen generic runtime: predict → WP-137 path trace → gate."""

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
M11_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M11"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
REAL_HOME_DB = Path.home() / ".learningos" / "learningos.db"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_experiment():
    import importlib.util

    path = M11_PACKAGE / "g5" / "reference" / "M11" / "experiment.py"
    spec = importlib.util.spec_from_file_location("m11_fixture_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATH_EXECUTE_SOURCE = _load_experiment().PATH_EXECUTE_SOURCE

STAGE_ORIENT = "stage_01_orientation"
STAGE_TRACE = "stage_02_trace_map"
STAGE_CODE = "stage_03_code_reading"
STAGE_EXPERIMENT = "stage_04_experiment"
STAGE_FAILURE = "stage_05_controlled_failure"
STAGE_TRANSFER = "stage_06_transfer"
STAGE_ADR = "stage_07_adr"
STAGE_FLAGSHIP = "stage_08_flagship"
STAGE_GATE = "stage_09_gate"
STAGE_SEQUENCE = [
    STAGE_ORIENT,
    STAGE_TRACE,
    STAGE_CODE,
    STAGE_EXPERIMENT,
    STAGE_FAILURE,
    STAGE_TRANSFER,
    STAGE_ADR,
    STAGE_FLAGSHIP,
    STAGE_GATE,
]
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}
PREDICT_BODY = {
    "hypothesis": "practice_accuracy 0.70 follows attendance_pct > 86.50 to class 1",
    "expected_values": {"predicted_class": 1},
}
EXECUTE_BODY = {
    "code": PATH_EXECUTE_SOURCE,
    "parameters": {
        "query": {"study_hours_week": 6, "practice_accuracy": 0.70, "attendance_pct": 91}
    },
}


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


def _home_mtime() -> int | None:
    return REAL_HOME_DB.stat().st_mtime_ns if REAL_HOME_DB.exists() else None


def _assert_home_untouched(before: int | None) -> None:
    if before is None:
        assert not REAL_HOME_DB.exists()
    else:
        assert REAL_HOME_DB.stat().st_mtime_ns == before
    assert not (REPO_ROOT / ".learningos").exists()
    assert not (REPO_ROOT / "learningos.db").exists()


@pytest.fixture
def m11_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-m11-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m11-{uuid.uuid4().hex}.sock")
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
    return {"Authorization": f"Bearer {token}"}


def _enter(client, headers, session_id: str, stage_id: str):
    response = client.post(
        f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response


def _submit(client, headers, session_id: str, stage_id: str, explanation: str, artifacts: list[dict]):
    response = client.post(
        f"/api/v1/sessions/{session_id}/stages/{stage_id}/submit",
        json={"explanation": explanation, "artifacts": artifacts},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response


def test_m11_runtime_path_trace_roundtrip(m11_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m11_env["home"]
    sock: Path = m11_env["sock"]
    env: dict[str, str] = m11_env["env"]
    worker: subprocess.Popen[bytes] | None = None
    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M11_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M11"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert "M11" in [item.get("id") for item in missions.json().get("missions", [])]
            spec = client.get("/api/v1/missions/M11", headers=headers).json()
            assert spec["id"] == "M11"
            assert [stage["id"] for stage in spec["stages"]] == STAGE_SEQUENCE

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m11-learner", "display_name": "M11 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]
            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M11", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            assert session.json()["current_stage_id"] == STAGE_ORIENT
            session_id = session.json()["session_id"]
            _enter(client, headers, session_id, STAGE_ORIENT)

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            got = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert got.json()["status"] == "ACTIVE"
            submitted = _submit(
                resumed,
                headers,
                session_id,
                STAGE_ORIENT,
                "framed shallow-tree interrogation and V03 baseline",
                [{"artifact_type": "markdown"}],
            )
            assert submitted.json()["current_stage_id"] == STAGE_TRACE

            early_gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert early_gate.status_code == 200, early_gate.text
            early_body = early_gate.json()
            assert early_body["status"] == "REPAIR_REQUIRED"
            assert early_body["reason"] == "GATE_CRITERIA_UNMET"
            plan = early_body["repair_plan"]
            assert all(item.get("action") == "targeted_repair" for item in plan["drills"])
            nodes = set(plan["failed_knowledge_nodes"])
            assert "kn.m11.decision_path" in nodes
            assert "kn.m11.overfitting_diagnosis" in nodes
            assert "kn.m11.no_ai_transfer" in nodes
            stages = set(plan["target_stage_ids"])
            assert STAGE_EXPERIMENT in stages
            assert STAGE_FAILURE in stages
            assert STAGE_TRANSFER in stages

            for stage_id, explanation, artifacts in (
                (STAGE_TRACE, "mapped thresholds, left <=, leaves and class counts", [{"artifact_type": "diagram"}]),
                (STAGE_CODE, "tree_ arrays and decision_path match a manual export_text trace", [{"artifact_type": "trace"}]),
            ):
                _enter(resumed, headers, session_id, stage_id)
                submitted = _submit(resumed, headers, session_id, stage_id, explanation, artifacts)
            assert submitted.json()["current_stage_id"] == STAGE_EXPERIMENT

            lab_enter = _enter(resumed, headers, session_id, STAGE_EXPERIMENT)
            assert lab_enter.json()["stage_type"] == "experiment"
            blocked = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXECUTE_BODY,
                headers=headers,
            )
            assert blocked.status_code == 409, blocked.text
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
            structured = exec_body.get("structured_result") or {}
            for key in ("execution_id", "status", "exit_code", "duration_ms", "blocks"):
                assert key in structured, key
            assert structured["status"] in WP137_STATUSES
            blocks = structured.get("blocks") or exec_body.get("blocks") or []
            assert blocks and blocks[0]["type"] == "trace"
            payload = blocks[0]["payload"]
            assert payload["predicted_class"] == 1
            assert payload["path"][0] == "practice_accuracy <= 0.73"
            assert payload["causal_claim_licensed"] is False
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)

            lab_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_EXPERIMENT,
                "observed attendance branch to class 1; leaf is a training distribution",
                [{"artifact_type": "trace"}],
            )
            assert lab_submit.json()["current_stage_id"] == STAGE_FAILURE

            _enter(resumed, headers, session_id, STAGE_FAILURE)
            _submit(
                resumed,
                headers,
                session_id,
                STAGE_FAILURE,
                "unconstrained tree overfits; importance is not causation",
                [{"artifact_type": "metric"}],
            )
            _enter(resumed, headers, session_id, STAGE_TRANSFER)
            _submit(
                resumed,
                headers,
                session_id,
                STAGE_TRANSFER,
                "textual tree traces and non-causal rewrite without assistance",
                [{"artifact_type": "artifact"}],
            )
            _enter(resumed, headers, session_id, STAGE_ADR)
            _submit(
                resumed,
                headers,
                session_id,
                STAGE_ADR,
                "retain constrained max_depth using held-out gap, not train accuracy",
                [{"artifact_type": "markdown"}],
            )
            _enter(resumed, headers, session_id, STAGE_FLAGSHIP)
            _submit(
                resumed,
                headers,
                session_id,
                STAGE_FLAGSHIP,
                "V03 keeps path diagnostics and non-causal importance caveats",
                [{"artifact_type": "markdown"}],
            )
            _enter(resumed, headers, session_id, STAGE_GATE)
            _submit(resumed, headers, session_id, STAGE_GATE, "ready for gate evaluation", [])

            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = {item["competency_id"] for item in (gate_body.get("competency_increments") or [])}
            assert "comp.tree.path_reasoning" in increments
            assert "comp.tree.unassisted_transfer" in increments

            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            assert evidence.json().get("evidence")
            assert "dummy_hash" not in json.dumps(evidence.json())
            today = resumed.get(f"/api/v1/learners/{learner_id}/next-action", headers=headers)
            assert today.json()["action"] == "IDLE"
            assert today.json()["reason"] == "ALL_MISSIONS_COMPLETE"
            final = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert final.json()["status"] == "COMPLETED"
            assert "openai" not in sys.modules

        completed = subprocess.run(
            [sys.executable, str(STATE_GUARD), "--repo", str(REPO_ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        _assert_home_untouched(before)
    finally:
        _stop_worker(worker)
