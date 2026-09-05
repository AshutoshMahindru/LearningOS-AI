"""G5 M05 on the frozen generic runtime: predict → WP-137 chart execute → gate."""

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
M05_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M05"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_EXPERIMENT = "stage_02_experiment"
STAGE_FAILURE = "stage_03_controlled_failure"
STAGE_TRANSFER = "stage_04_transfer"
STAGE_GATE = "stage_05_gate"

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

PREDICT_BODY = {
    "hypothesis": "bulk/vectorized work is faster than Python-level loops at n=4000; totals agree within tolerance; timing is hardware-dependent",
    "expected_values": {"faster": "bulk", "n": 4000, "correctness_before_speed": True},
}

EXECUTE_CODE = """
import time
n = int(parameters.get('n') or 4000)
qty = [2, 1, 0, 3]
prices = [10.0, 4.0, 1.5, 2.5]
expected = 0.0
for q, p in zip(qty, prices):
    expected += q * p
t0 = time.perf_counter()
loop_totals = []
for _ in range(n):
    total = 0.0
    for q, p in zip(qty, prices):
        total += q * p
    loop_totals.append(total)
loop_s = time.perf_counter() - t0
t1 = time.perf_counter()
line = [q * p for q, p in zip(qty, prices)]
bulk_total = sum(line)
bulk_totals = [bulk_total for _ in range(n)]
bulk_s = time.perf_counter() - t1
speedup = (loop_s / bulk_s) if bulk_s else 0.0
print({'loop_seconds': loop_s, 'bulk_seconds': bulk_s, 'n': n})
[
    {
        'type': 'chart',
        'title': 'loop vs bulk timing',
        'payload': {
            'chart_type': 'bar',
            'series': [
                {'name': 'loop', 'x': [0], 'y': [loop_s]},
                {'name': 'bulk', 'x': [1], 'y': [bulk_s]},
            ],
        },
    },
    {
        'type': 'metric',
        'title': 'timing',
        'payload': {
            'loop_seconds': loop_s,
            'bulk_seconds': bulk_s,
            'observed_speedup': speedup,
            'n': n,
            'correct': loop_totals[0] == expected and bulk_totals[0] == expected,
        },
    },
    {
        'type': 'table',
        'title': 'benchmark',
        'payload': {
            'columns': ['method', 'seconds', 'n'],
            'rows': [['loop', loop_s, n], ['bulk', bulk_s, n]],
        },
    },
]
"""

EXECUTE_BODY = {
    "code": EXECUTE_CODE,
    "parameters": {"n": 4000},
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


@pytest.fixture
def m05_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g5-m05-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g5-m05-{uuid.uuid4().hex}.sock")
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


def test_sample_timing_result_matches_frozen_wp137() -> None:
    from app.execution.result_schema import validate_structured_result

    sample = {
        "execution_id": "exec-m05-timing",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [
            {
                "type": "chart",
                "title": "loop vs bulk timing",
                "payload": {
                    "chart_type": "bar",
                    "series": [
                        {"name": "loop", "x": [0], "y": [0.042]},
                        {"name": "bulk", "x": [1], "y": [0.003]},
                    ],
                },
            },
            {
                "type": "metric",
                "title": "timing",
                "payload": {
                    "loop_seconds": 0.042,
                    "bulk_seconds": 0.003,
                    "observed_speedup": 14.0,
                    "n": 4000,
                    "correct": True,
                },
            },
            {
                "type": "table",
                "title": "benchmark",
                "payload": {
                    "columns": ["method", "seconds", "n"],
                    "rows": [["loop", 0.042, 4000], ["bulk", 0.003, 4000]],
                },
            },
        ],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert types == ["chart", "metric", "table"]
    assert set(types) <= WP137_BLOCK_TYPES


def test_m05_runtime_roundtrip_emits_wp137_chart(m05_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = m05_env["home"]
    sock: Path = m05_env["sock"]
    env: dict[str, str] = m05_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M05_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M05"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M05" in mission_ids

            mission = client.get("/api/v1/missions/M05", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M05"
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == [
                STAGE_ORIENT,
                STAGE_EXPERIMENT,
                STAGE_FAILURE,
                STAGE_TRANSFER,
                STAGE_GATE,
            ]
            experiment = next(stage for stage in spec["stages"] if stage["id"] == STAGE_EXPERIMENT)
            assert experiment["type"] == "experiment"
            assert experiment["validation_rubric"]["required_evidence_type"] == "chart"

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m05-learner", "display_name": "M05 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M05", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M05"
            assert session_body["status"] == "ACTIVE"
            assert session_body["current_stage_id"] == STAGE_ORIENT
            session_id = session_body["session_id"]

            entered = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/enter",
                headers=headers,
            )
            assert entered.status_code == 200, entered.text
            assert entered.json()["current_stage_id"] == STAGE_ORIENT

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            got = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert got.status_code == 200, got.text
            assert got.json()["current_stage_id"] == STAGE_ORIENT
            assert got.json()["status"] == "ACTIVE"

            submitted = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/submit",
                json={
                    "explanation": "framed the timing whole; gate needs chart, table, metric",
                    "artifacts": [{"artifact_type": "markdown"}],
                },
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
            structured = exec_body.get("structured_result") or {}
            for key in ("execution_id", "status", "exit_code", "duration_ms", "blocks"):
                assert key in structured, key
            assert structured["status"] in WP137_STATUSES
            assert structured["status"] == "SUCCESS"
            blocks = structured.get("blocks") or exec_body.get("blocks") or []
            assert isinstance(blocks, list) and blocks
            types = [block["type"] for block in blocks]
            assert "chart" in types
            assert set(types) <= WP137_BLOCK_TYPES
            chart = next(block for block in blocks if block["type"] == "chart")
            payload = chart["payload"]
            assert payload.get("chart_type") == "bar"
            series = payload.get("series") or []
            assert {item["name"] for item in series} == {"loop", "bulk"}
            assert all(isinstance(item["y"][0], (int, float)) for item in series)
            metric = next(block for block in blocks if block["type"] == "metric")
            assert metric["payload"]["n"] == 4000
            assert metric["payload"]["correct"] is True
            table = next(block for block in blocks if block["type"] == "table")
            assert table["payload"]["columns"] == ["method", "seconds", "n"]
            diagnostics = structured.get("diagnostics") or exec_body.get("diagnostics") or {}
            assert "loop_seconds" in str(diagnostics.get("stdout") or "")
            assert "dummy_hash" not in json.dumps(exec_body)
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)

            from app.execution.result_schema import validate_structured_result

            validate_structured_result(structured)

            lab_submit = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/submit",
                json={
                    "explanation": "observed chart of loop vs bulk durations; speed is not proof of correctness",
                    "artifacts": [{"artifact_type": "chart"}],
                },
                headers=headers,
            )
            assert lab_submit.status_code == 200, lab_submit.text
            assert lab_submit.json()["current_stage_id"] == STAGE_FAILURE

            fail_enter = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_FAILURE}/enter",
                headers=headers,
            )
            assert fail_enter.status_code == 200, fail_enter.text
            fail_submit = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_FAILURE}/submit",
                json={
                    "explanation": "setup inside one timing is unfair; wrong axis preserves products not orders",
                    "artifacts": [{"artifact_type": "table"}],
                },
                headers=headers,
            )
            assert fail_submit.status_code == 200, fail_submit.text
            assert fail_submit.json()["current_stage_id"] == STAGE_TRANSFER

            transfer_enter = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_TRANSFER}/enter",
                headers=headers,
            )
            assert transfer_enter.status_code == 200, transfer_enter.text
            transfer_submit = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_TRANSFER}/submit",
                json={
                    "explanation": "fresh stores×products shapes and broadcasts without assistance",
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
                "comp.array.performance_reasoning",
                "comp.array.timing_prediction",
                "comp.array.benchmark_fairness",
                "comp.array.vectorization_transfer",
            }

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
        assert not (REPO_ROOT / ".learningos").exists()
        assert not (REPO_ROOT / "learningos.db").exists()
    finally:
        _stop_worker(worker)
