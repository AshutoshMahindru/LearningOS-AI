"""G5 M39 on the frozen generic runtime: predict → WP-137 execute → gate."""

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
M39_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M39"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
REAL_HOME_DB = Path.home() / ".learningos" / "learningos.db"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_CODE = "stage_02_code_reading"
STAGE_EXPERIMENT = "stage_03_experiment"
STAGE_FAILURE = "stage_04_controlled_failure"
STAGE_TRANSFER = "stage_05_transfer"
STAGE_ADR = "stage_06_adr"
STAGE_FLAGSHIP = "stage_07_flagship"
STAGE_GATE = "stage_08_gate"

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

PREDICT_BODY = {'hypothesis': 'SKU-7 routes catalog_purchase, posts 42.0, excludes stale memory, and stays non-degraded; circuit stays closed', 'expected_values': {'route': 'catalog_purchase', 'degraded': False, 'stale_excluded': True, 'posted_amount': 42.0}}

EXECUTE_CODE = """
mode = str(parameters.get("mode") or "happy")
stale_price = float(parameters.get("stale_price") or 99.0)
catalog_price = float(parameters.get("catalog_price") or 42.0)
max_attempts = int(parameters.get("max_attempts") or 3)
circuit_threshold = int(parameters.get("circuit_threshold") or 2)
if mode == "stale_trust":
    route, degraded, circuit_open, stale_excluded, posted, attempts, terminal = (
        "catalog_purchase", False, False, False, stale_price, 1, "complete"
    )
elif mode == "primary_fail":
    route, degraded, circuit_open, stale_excluded, posted, attempts, terminal = (
        "catalog_purchase", True, False, True, None, 2, "degraded"
    )
elif mode == "oscillate":
    route, degraded, circuit_open, stale_excluded, posted, attempts, terminal = (
        "catalog_purchase", True, False, True, None, max_attempts + 3, "loop"
    )
else:
    route, degraded, circuit_open, stale_excluded, posted, attempts, terminal = (
        "catalog_purchase", False, False, True, catalog_price, 1, "complete"
    )
if mode == "oscillate_repair":
    circuit_open, attempts, terminal, degraded = True, circuit_threshold, "circuit_open", True
    posted = None
payload = {
    "route": route,
    "degraded": degraded,
    "circuit_open": circuit_open,
    "stale_excluded": stale_excluded,
    "posted_amount": posted,
    "attempts": attempts,
    "terminal": terminal,
}
print(payload)
[
    {"type": "metric", "title": "robust-run", "payload": payload},
    {
        "type": "table",
        "title": "memory",
        "payload": {
            "columns": ["id", "action"],
            "rows": [["mem-catalog", "retrieved"], ["mem-stale", "excluded" if stale_excluded else "trusted"]],
        },
    },
]
"""

EXECUTE_BODY = {"code": EXECUTE_CODE, "parameters": {'mode': 'happy', 'catalog_price': 42.0, 'stale_price': 99.0}}

SUBMIT_PLAN = [
    (STAGE_ORIENT, 'framed working versus persistent memory; routes plus no-match; bounded fallback', [{"artifact_type": "markdown"}]),
    (STAGE_CODE, 'retrieve_memory, select_route, run_fallback_ladder; expired ids excluded', [{"artifact_type": "trace"}]),
    (STAGE_EXPERIMENT, 'observed catalog_purchase post 42.0 after a sealed prediction; stale excluded', [{"artifact_type": 'metric'}]),
    (STAGE_FAILURE, 'stale 99.0 posted as complete; repair_run used M38 lookup 42.0; circuit bound restored', [{"artifact_type": "table"}]),
    (STAGE_TRANSFER, 'classified memory/routes/fallbacks without inventing an eval harness', [{"artifact_type": "artifact"}]),
    (STAGE_ADR, 'provenance-bearing memory, explicit routes, bounded ladder, labeled degraded', [{"artifact_type": "markdown"}]),
    (STAGE_FLAGSHIP, 'M40 receives declared memory, route, fallback, and traces; not an eval harness', [{"artifact_type": "markdown"}]),
    (STAGE_GATE, 'ready for gate evaluation', []),
]


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
def m39_env(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    home = Path(tempfile.mkdtemp(prefix="los-g6-m39-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m39-{uuid.uuid4().hex}.sock")
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


def test_sample_result_matches_frozen_wp137() -> None:
    from app.execution.result_schema import validate_structured_result

    sample = {
        "execution_id": "exec-m39-sample",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [{'type': 'metric', 'title': 'robust-run', 'payload': {'route': 'catalog_purchase', 'degraded': False, 'stale_excluded': True, 'posted_amount': 42.0}}, {'type': 'table', 'title': 'memory', 'payload': {'columns': ['id', 'action'], 'rows': [['mem-catalog', 'retrieved'], ['mem-stale', 'excluded']]}}],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert set(types) <= WP137_BLOCK_TYPES
    assert 'metric' in types


def test_m39_runtime_roundtrip(m39_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m39_env["home"]
    sock: Path = m39_env["sock"]
    env: dict[str, str] = m39_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M39_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M39"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M39" in mission_ids
            assert "g5.reference.M39" not in mission_ids

            mission = client.get("/api/v1/missions/M39", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M39"
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == [
                STAGE_ORIENT,
                STAGE_CODE,
                STAGE_EXPERIMENT,
                STAGE_FAILURE,
                STAGE_TRANSFER,
                STAGE_ADR,
                STAGE_FLAGSHIP,
                STAGE_GATE,
            ]

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m39-learner", "display_name": "M39 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M39", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M39"
            assert session_body["status"] == "ACTIVE"
            assert session_body["current_stage_id"] == STAGE_ORIENT
            session_id = session_body["session_id"]

            entered = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/enter",
                headers=headers,
            )
            assert entered.status_code == 200, entered.text

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            got = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert got.status_code == 200, got.text
            assert got.json()["current_stage_id"] == STAGE_ORIENT
            assert got.json()["status"] == "ACTIVE"

            early_gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert early_gate.status_code == 200, early_gate.text
            early_body = early_gate.json()
            assert early_body["status"] == "REPAIR_REQUIRED"
            assert early_body["reason"] == "GATE_CRITERIA_UNMET"
            plan = early_body["repair_plan"]
            assert plan["drills"]
            assert all(item.get("action") == "targeted_repair" for item in plan["drills"])
            nodes = set(plan["failed_knowledge_nodes"])
            assert 'kn.m39.memory' in nodes
            stages = set(plan["target_stage_ids"])
            assert STAGE_EXPERIMENT in stages
            assert STAGE_FAILURE in stages
            assert STAGE_TRANSFER in stages

            submitted = None
            for stage_id, explanation, artifacts in SUBMIT_PLAN:
                if stage_id != STAGE_EXPERIMENT:
                    entered = resumed.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
                        headers=headers,
                    )
                    assert entered.status_code == 200, entered.text
                    submitted = resumed.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/submit",
                        json={"explanation": explanation, "artifacts": artifacts},
                        headers=headers,
                    )
                    assert submitted.status_code == 200, submitted.text
                    continue

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
                structured = exec_body.get("structured_result") or {}
                for key in ("execution_id", "status", "exit_code", "duration_ms", "blocks"):
                    assert key in structured, key
                assert structured["status"] in WP137_STATUSES
                assert structured["status"] == "SUCCESS"
                blocks = structured.get("blocks") or exec_body.get("blocks") or []
                assert isinstance(blocks, list) and blocks
                types = [block["type"] for block in blocks]
                assert 'metric' in types
                assert set(types) <= WP137_BLOCK_TYPES
                metric = next(block for block in blocks if block["type"] == "metric")["payload"]
                assert metric['route'] == 'catalog_purchase'
                assert metric['degraded'] == False
                assert metric['stale_excluded'] == True
                assert metric['posted_amount'] == 42.0
                assert "dummy_hash" not in json.dumps(exec_body)
                assert "sk-test-should-never-leak" not in json.dumps(exec_body)

                from app.execution.result_schema import validate_structured_result

                validate_structured_result(structured)

                submitted = resumed.post(
                    f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/submit",
                    json={"explanation": explanation, "artifacts": artifacts},
                    headers=headers,
                )
                assert submitted.status_code == 200, submitted.text

            assert submitted is not None
            assert submitted.json()["current_stage_id"] == STAGE_GATE

            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = {item["competency_id"] for item in (gate_body.get("competency_increments") or [])}
            assert increments == {'comp.robust.memory', 'comp.robust.routing', 'comp.robust.fallback', 'comp.robust.unassisted_transfer'}

            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            claims = evidence.json().get("evidence") or []
            assert claims
            assert "dummy_hash" not in json.dumps(evidence.json())

            today = resumed.get(f"/api/v1/learners/{learner_id}/next-action", headers=headers)
            assert today.status_code == 200, today.text
            action_body = today.json()
            assert action_body["action"] == "IDLE"
            assert action_body["reason"] == "ALL_MISSIONS_COMPLETE"

            final = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert final.status_code == 200, final.text
            assert final.json()["status"] == "COMPLETED"
            assert "openai" not in sys.modules

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
