"""G6 M42 on the frozen generic runtime: predict → WP-137 execute → gate."""

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
M42_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M42"
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

PREDICT_BODY = {'hypothesis': 'under-budget query uses full capability; latency above budget degrades to cached fallback; this is a WP-136 package not a one-off app', 'expected_values': {'full_degraded': False, 'over_budget_degraded': True, 'one_off_app': False}}

EXECUTE_CODE = 'components = ["HybridRetriever", "LLMReasoningEngine", "ToolExecutor", "StatefulMemoryStore", "SystemEvalHarness"]\nmax_latency_ms = float(parameters.get("max_latency_ms") or 2000.0)\nquery = str(parameters.get("query") or "reconcile BIN-12 occupancy")\n\ndef run(latency_ms):\n    degraded = latency_ms > max_latency_ms\n    if degraded:\n        output = "[DEGRADED_CACHED_FALLBACK] cached answer"\n        eval_score = 0.75\n        cost = 0.001\n        memory_depth = 0\n        mode = "FALLBACK_CACHED"\n    else:\n        output = "[FULL_CAPABILITY_RESPONSE] retrieval + tool calling"\n        eval_score = 0.95\n        cost = 0.005\n        memory_depth = 2\n        mode = "FULL_CAPABILITY"\n    return {\n        "status": "SUCCESS",\n        "output_kind": "degraded" if degraded else "full",\n        "latency_ms": latency_ms,\n        "cost_usd": cost,\n        "eval_score": eval_score,\n        "degradation_applied": degraded,\n        "degradation_mode": mode,\n        "memory_depth": memory_depth,\n        "active_components": len(components),\n        "eval_version": "m40.eval.v1",\n        "one_off_app": False,\n        "wp136_package": True,\n    }\n\nfull = run(float(parameters.get("full_latency_ms") or 120.0))\nslow = run(float(parameters.get("slow_latency_ms") or 2500.0))\npayload = {\n    "query": query,\n    "components": components,\n    "full": full,\n    "slow": slow,\n    "one_off_app": False,\n    "wp136_package": True,\n}\nprint(payload)\n[\n    {"type": "metric", "title": "capstone_run", "payload": payload},\n    {\n        "type": "table",\n        "title": "path_comparison",\n        "payload": {\n            "columns": ["path", "mode", "eval_score", "degraded"],\n            "rows": [\n                ["full", full["degradation_mode"], full["eval_score"], full["degradation_applied"]],\n                ["slow", slow["degradation_mode"], slow["eval_score"], slow["degradation_applied"]],\n            ],\n        },\n    },\n    {\n        "type": "trace",\n        "title": "orchestrator_steps",\n        "payload": {\n            "steps": ["retrieve", "reason", "tool", "memory", "evaluate"],\n            "degraded_slow": True,\n        },\n    },\n]'

EXECUTE_BODY = {"code": EXECUTE_CODE, "parameters": {'query': 'reconcile BIN-12 occupancy', 'max_latency_ms': 2000, 'full_latency_ms': 120, 'slow_latency_ms': 2500}}

SUBMIT_PLAN = [
    (STAGE_ORIENT, "framed WP-136 capstone package not a one-off app", [{"artifact_type": "markdown"}]),
    (STAGE_CODE, "traced orchestrator budget check and defense evaluation", [{"artifact_type": "trace"}]),
    (STAGE_EXPERIMENT, "full capability under budget; cached fallback when slow", [{"artifact_type": "metric"}]),
    (STAGE_FAILURE, "eval gate rejects hidden-critical even if average is high", [{"artifact_type": "table"}]),
    (STAGE_TRANSFER, "defended caching versus freshness without assistance", [{"artifact_type": "artifact"}]),
    (STAGE_ADR, "chose generic-runtime package; rejected a one-off app", [{"artifact_type": "markdown"}]),
    (STAGE_FLAGSHIP, "V12 tagged on the generic Mission Player", [{"artifact_type": "markdown"}]),
    (STAGE_GATE, "ready for gate evaluation", []),
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
def m42_env(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    home = Path(tempfile.mkdtemp(prefix="los-g6-m42-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m42-{uuid.uuid4().hex}.sock")
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
        "execution_id": "exec-m42-sample",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [
            {
                "type": "metric",
                "title": "capstone_run",
                "payload": {"one_off_app": False, "wp136_package": True},
            },
            {
                "type": "table",
                "title": "path_comparison",
                "payload": {
                    "columns": ["path", "mode"],
                    "rows": [["full", "FULL_CAPABILITY"], ["slow", "FALLBACK_CACHED"]],
                },
            },
            {
                "type": "trace",
                "title": "orchestrator_steps",
                "payload": {"steps": ["retrieve", "reason", "tool", "memory", "evaluate"]},
            },
        ],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert types == ['metric', 'table', 'trace']
    assert set(types) <= WP137_BLOCK_TYPES


def test_m42_runtime_roundtrip(m42_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m42_env["home"]
    sock: Path = m42_env["sock"]
    env: dict[str, str] = m42_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M42_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g6.reference.M42"
            assert loaded_body.get("version") == "6.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M42" in mission_ids
            assert "g6.reference.M42" not in mission_ids

            mission = client.get("/api/v1/missions/M42", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M42"
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
                json={"username": "m42-learner", "display_name": "M42 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M42", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M42"
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
            assert 'kn.m42.m40_eval_reuse' in nodes
            stages = set(plan["target_stage_ids"])
            assert 'stage_03_experiment' in stages

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
                for required in ['metric', 'table', 'trace']:
                    assert required in types
                assert set(types) <= WP137_BLOCK_TYPES
                metric = next(block for block in blocks if block["type"] == "metric")["payload"]
                assert metric['one_off_app'] == False
                assert metric['wp136_package'] == True
                assert metric['full']['degradation_applied'] == False
                assert metric['full']['eval_score'] == 0.95
                assert metric['slow']['degradation_applied'] == True
                assert metric['slow']['eval_score'] == 0.75
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
            assert increments == {'comp.cap.orchestration', 'comp.cap.unassisted_defense', 'comp.cap.degradation_defense', 'comp.cap.evaluation_reuse'}

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
