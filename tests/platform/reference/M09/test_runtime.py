"""G6 M09 on the frozen generic runtime: predict seal, no-AI, evidence, gate, repair."""

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
PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M09"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
REAL_HOME_DB = Path.home() / ".learningos" / "learningos.db"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

PACKAGE_ID = "g6.reference.M09"
MISSION_ID = "M09"
EXPERIMENT_STAGE = "stage_03_experiment"
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}
WP137_BLOCK_TYPES = {
    "table",
    "chart",
    "trace",
    "state_diff",
    "diagram",
    "markdown",
    "metric",
    "artifact",
}

PREDICT_BODY = {"hypothesis": "threshold 0.50 reports high accuracy with low recall; 0.30 raises recall at FP cost", "expected_values": {"accuracy_at_default": 0.85}}

EXECUTE_CODE = """
rows = [
    ['threshold', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall'],
    [0.50, 2, 15, 1, 6, 0.708, 0.667, 0.25],
    [0.30, 5, 12, 4, 3, 0.708, 0.556, 0.625],
]
payload = {'default_accuracy': 0.708, 'default_recall': 0.25, 'moved_recall': 0.625, 'majority_accuracy': 0.667}
print(payload)
[
    {'type': 'table', 'title': 'thresholds', 'payload': {'columns': rows[0], 'rows': rows[1:]}},
    {'type': 'metric', 'title': 'threshold-policy', 'payload': payload},
]
"""

EXECUTE_BODY = {"code": EXECUTE_CODE, "parameters": {}}
EXPECTED_BLOCK = "table"
REPAIR_NODE = "kn.m09.probability_not_label"
REPAIR_STAGE = "stage_02_trace_map"


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


def _home_mtime() -> int | None:
    return REAL_HOME_DB.stat().st_mtime_ns if REAL_HOME_DB.exists() else None


def _assert_home_untouched(before: int | None) -> None:
    if before is None:
        assert not REAL_HOME_DB.exists()
    else:
        assert REAL_HOME_DB.stat().st_mtime_ns == before
    assert not (REPO_ROOT / ".learningos").exists()
    assert not (REPO_ROOT / "learningos.db").exists()


def _outside_repo(path: Path) -> None:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    with pytest.raises(ValueError):
        resolved.relative_to(repo)


@pytest.fixture
def env_home(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    home = Path(tempfile.mkdtemp(prefix="los-g6-m09-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m09-{uuid.uuid4().hex}.sock")
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


def _load_and_start(client, headers):
    loaded = client.post(
        "/api/v1/curriculum/packages/load",
        json={"package_dir": str(PACKAGE)},
        headers=headers,
    )
    assert loaded.status_code == 200, loaded.text
    body = loaded.json()
    assert body.get("id") == PACKAGE_ID
    assert body.get("version") == "6.0.0"
    learner = client.post(
        "/api/v1/learners",
        json={"username": "m09-learner", "display_name": "M09 Learner"},
        headers=headers,
    )
    assert learner.status_code == 200, learner.text
    learner_id = learner.json()["learner_id"]
    session = client.post(
        "/api/v1/sessions",
        json={"mission_id": MISSION_ID, "learner_id": learner_id},
        headers=headers,
    )
    assert session.status_code == 200, session.text
    session_body = session.json()
    assert session_body["mission_id"] == MISSION_ID
    assert session_body["status"] == "ACTIVE"
    assert session_body["current_stage_id"] == "stage_01_orientation"
    return learner_id, session_body["session_id"]


def _artifacts_for_stage(spec: dict, stage: dict) -> list[dict]:
    stage_id = stage["id"]
    contract = spec.get("gate_contract") if isinstance(spec.get("gate_contract"), dict) else {}
    required = contract.get("required_evidence") if isinstance(contract.get("required_evidence"), list) else []
    for item in required:
        if isinstance(item, dict) and item.get("stage_id") == stage_id and item.get("artifact_type"):
            return [{"artifact_type": item["artifact_type"]}]
    rubric = stage.get("validation_rubric") if isinstance(stage.get("validation_rubric"), dict) else {}
    required_type = rubric.get("required_evidence_type") or rubric.get("artifact_type")
    if isinstance(required_type, str) and required_type:
        return [{"artifact_type": required_type}]
    return [{"artifact_type": "markdown"}]


def test_load_package_and_create_session_without_writing_home(env_home):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = env_home["home"]
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        headers = _bootstrap(client)
        learner_id, session_id = _load_and_start(client, headers)
        missions = client.get("/api/v1/missions", headers=headers)
        assert missions.status_code == 200, missions.text
        mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
        assert MISSION_ID in mission_ids
        assert PACKAGE_ID not in mission_ids
        mission = client.get(f"/api/v1/missions/{MISSION_ID}", headers=headers)
        assert mission.status_code == 200, mission.text
        spec = mission.json()
        assert spec["id"] == MISSION_ID
        types = [stage["type"] for stage in spec["stages"]]
        assert "experiment" in types
        assert "controlled_failure" in types
        assert "transfer_assessment" in types
        got = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        assert got.status_code == 200, got.text
        assert got.json()["current_stage_id"] == "stage_01_orientation"
        assert learner_id
    assert home.exists()
    assert str(home).startswith("/tmp/")
    _outside_repo(home)
    _assert_home_untouched(before)


def test_gate_targeted_repair_when_evidence_missing(env_home):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        headers = _bootstrap(client)
        _learner_id, session_id = _load_and_start(client, headers)
        gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
        assert gate.status_code == 200, gate.text
        body = gate.json()
        assert body["status"] == "REPAIR_REQUIRED"
        assert body["reason"] == "GATE_CRITERIA_UNMET"
        plan = body["repair_plan"]
        assert plan["drills"]
        assert all(item.get("action") == "targeted_repair" for item in plan["drills"])
        nodes = set(plan["failed_knowledge_nodes"])
        assert REPAIR_NODE in nodes
        stages = set(plan["target_stage_ids"])
        assert REPAIR_STAGE in stages
        assert EXPERIMENT_STAGE in stages or any("experiment" in sid for sid in stages)
    _assert_home_untouched(before)


def test_prediction_seal_structured_output_and_full_gate(env_home):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = env_home["home"]
    sock: Path = env_home["sock"]
    env: dict[str, str] = env_home["env"]
    worker: subprocess.Popen[bytes] | None = None
    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            learner_id, session_id = _load_and_start(client, headers)
            spec_resp = client.get(f"/api/v1/missions/{MISSION_ID}", headers=headers)
            assert spec_resp.status_code == 200, spec_resp.text
            spec = spec_resp.json()
            stages = spec["stages"]
            predicted_once = False
            submitted = None
            for stage in stages:
                stage_id = stage["id"]
                entered = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
                    headers=headers,
                )
                assert entered.status_code == 200, entered.text
                if stage.get("type") == "experiment":
                    if not predicted_once:
                        blocked = client.post(
                            f"/api/v1/sessions/{session_id}/stages/{stage_id}/execute",
                            json=EXECUTE_BODY,
                            headers=headers,
                        )
                        assert blocked.status_code == 409, blocked.text
                        assert blocked.json()["error"]["code"] == "CONFLICT"
                        assert blocked.json()["error"]["details"].get("reason") == "PREDICTION_REQUIRED"
                        predicted_once = True
                    predicted = client.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/predict",
                        json=PREDICT_BODY,
                        headers=headers,
                    )
                    assert predicted.status_code == 200, predicted.text
                    assert predicted.json()["is_sealed"] is True
                    assert predicted.json()["prediction_hash"] != "dummy_hash"
                    assert len(predicted.json()["prediction_hash"]) == 64
                    executed = client.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/execute",
                        json=EXECUTE_BODY,
                        headers=headers,
                    )
                    assert executed.status_code == 200, executed.text
                    exec_body = executed.json()
                    assert exec_body["status"] == "SUCCESS", exec_body
                    assert exec_body["status"] != "UNSUPPORTED"
                    structured = exec_body.get("structured_result") or {}
                    for key in ("execution_id", "status", "exit_code", "duration_ms", "blocks"):
                        assert key in structured, key
                    assert structured["status"] in WP137_STATUSES
                    assert structured["status"] == "SUCCESS"
                    blocks = structured.get("blocks") or exec_body.get("blocks") or []
                    assert isinstance(blocks, list) and blocks
                    types = [block["type"] for block in blocks]
                    assert EXPECTED_BLOCK in types
                    assert set(types) <= WP137_BLOCK_TYPES
                    assert "sk-test-should-never-leak" not in json.dumps(exec_body)
                    assert "dummy_hash" not in json.dumps(exec_body)
                submitted = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{stage_id}/submit",
                    json={
                        "explanation": f"completed {stage_id} on generic runtime",
                        "artifacts": _artifacts_for_stage(spec, stage),
                    },
                    headers=headers,
                )
                assert submitted.status_code == 200, submitted.text
                assert submitted.json()["payload_hash"] != "dummy_hash"

            assert submitted is not None
            gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED", gate_body
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = {item["competency_id"] for item in (gate_body.get("competency_increments") or [])}
            assert increments
            assert all(str(item).startswith("comp.") for item in increments)

            evidence = client.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            claims = evidence.json().get("evidence") or []
            assert claims
            assert "dummy_hash" not in json.dumps(evidence.json())

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
        assert str(home) not in str(REPO_ROOT)
        _assert_home_untouched(before)
    finally:
        _stop_worker(worker)
