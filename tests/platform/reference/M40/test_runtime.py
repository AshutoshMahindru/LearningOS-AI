"""G6 M40 on the frozen generic runtime: predict → WP-137 execute → gate."""

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
M40_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M40"
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

PREDICT_BODY = {'hypothesis': 'aggregate-only 0.80 can pass while unsupported_citation on rag-grounded-reset fails the canonical slice gate', 'expected_values': {'eval_version': 'm40.eval.v1', 'n': 12, 'aggregate_only_passed': True, 'canonical_gate_passed': False}}

EXECUTE_CODE = 'EVAL_VERSION = "m40.eval.v1"\ncases = [\n    {"id": "rag-grounded-reset", "family": "rag", "grader": "citation_support", "severity": "critical", "ok": True},\n    {"id": "rag-abstain-ceo", "family": "rag", "grader": "citation_support", "severity": "critical", "ok": True},\n    {"id": "rag-holdout-email", "family": "rag", "grader": "citation_support", "severity": "major", "ok": True},\n    {"id": "rag-ticket-recovered", "family": "rag", "grader": "citation_support", "severity": "major", "ok": True},\n    {"id": "agent-purchase-complete", "family": "agent", "grader": "state_termination", "severity": "critical", "ok": True},\n    {"id": "agent-memory-scope", "family": "agent", "grader": "state_termination", "severity": "major", "ok": True},\n    {"id": "agent-fallback-bound", "family": "agent", "grader": "state_termination", "severity": "major", "ok": True},\n    {"id": "agent-degraded-path", "family": "agent", "grader": "state_termination", "severity": "minor", "ok": True},\n    {"id": "tool-schema-extra", "family": "tool", "grader": "tool_schema", "severity": "critical", "ok": True},\n    {"id": "tool-idempotency-replay", "family": "tool", "grader": "idempotency", "severity": "critical", "ok": True},\n    {"id": "tool-schema-ok", "family": "tool", "grader": "tool_schema", "severity": "minor", "ok": True},\n    {"id": "rag-grounded-hours", "family": "rag", "grader": "citation_support", "severity": "minor", "ok": True},\n]\ndefect = str(parameters.get("defect") or "unsupported_citation")\ngraded = []\nfor row in cases:\n    item = dict(row)\n    if defect == "unsupported_citation" and item["id"] == "rag-grounded-reset":\n        item["ok"] = False\n        item["localized_failure"] = "citation_support:rag-grounded-reset:doc-account-access::c1"\n    graded.append(item)\nn = len(graded)\nn_ok = sum(1 for item in graded if item["ok"])\ntask_success_rate = n_ok / n\nn_critical_fail = sum(1 for item in graded if item["severity"] == "critical" and not item["ok"])\ncritical_fail_rate = n_critical_fail / n\ncitation_fail = sum(1 for item in graded if item["grader"] == "citation_support" and not item["ok"]) / n\nfail_reasons = []\nif critical_fail_rate > 0:\n    fail_reasons.append(f"critical_fail_rate:{critical_fail_rate:.3f}>0.000")\nif citation_fail > 0:\n    fail_reasons.append(f"slice:citation_support:{citation_fail:.3f}>0.000")\npayload = {\n    "eval_version": EVAL_VERSION,\n    "n": n,\n    "task_success_rate": task_success_rate,\n    "critical_fail_rate": critical_fail_rate,\n    "citation_slice_fail": citation_fail,\n    "aggregate_only_passed": task_success_rate >= 0.80,\n    "canonical_gate_passed": len(fail_reasons) == 0,\n    "fail_reasons": fail_reasons,\n    "hidden_critical_case": "rag-grounded-reset",\n    "llm_judge_sole_grader": False,\n}\nprint(payload)\n[\n    {"type": "metric", "title": "eval_suite", "payload": payload},\n    {\n        "type": "table",\n        "title": "grader_slices",\n        "payload": {\n            "columns": ["case_id", "grader", "severity", "ok"],\n            "rows": [[item["id"], item["grader"], item["severity"], item["ok"]] for item in graded],\n        },\n    },\n]'

EXECUTE_BODY = {"code": EXECUTE_CODE, "parameters": {'defect': 'unsupported_citation'}}

SUBMIT_PLAN = [
    (STAGE_ORIENT, "framed versioned eval pack and V11 opening", [{"artifact_type": "markdown"}]),
    (STAGE_CODE, "traced load_eval_pack, graders, and decide_release_gate", [{"artifact_type": "trace"}]),
    (STAGE_EXPERIMENT, "canonical slice gates failed while aggregate-only passed", [{"artifact_type": "metric"}]),
    (STAGE_FAILURE, "hidden-critical citation is a governance defect; repair the gate", [{"artifact_type": "table"}]),
    (STAGE_TRANSFER, "fresh kiosk and warehouse cases designed without assistance", [{"artifact_type": "artifact"}]),
    (STAGE_ADR, "chose versioned deterministic graders; rejected judge-only scoring", [{"artifact_type": "markdown"}]),
    (STAGE_FLAGSHIP, "V11 evaluation is open; architecture remains M41", [{"artifact_type": "markdown"}]),
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
def m40_env(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    home = Path(tempfile.mkdtemp(prefix="los-g6-m40-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m40-{uuid.uuid4().hex}.sock")
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
        "execution_id": "exec-m40-sample",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [
            {
                "type": "metric",
                "title": "eval_suite",
                "payload": {
                    "eval_version": "m40.eval.v1",
                    "n": 12,
                    "aggregate_only_passed": True,
                    "canonical_gate_passed": False,
                },
            },
            {
                "type": "table",
                "title": "grader_slices",
                "payload": {
                    "columns": ["case_id", "grader", "ok"],
                    "rows": [["rag-grounded-reset", "citation_support", False]],
                },
            },
        ],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert types == ['metric', 'table']
    assert set(types) <= WP137_BLOCK_TYPES


def test_m40_runtime_roundtrip(m40_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m40_env["home"]
    sock: Path = m40_env["sock"]
    env: dict[str, str] = m40_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M40_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g6.reference.M40"
            assert loaded_body.get("version") == "6.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M40" in mission_ids
            assert "g6.reference.M40" not in mission_ids

            mission = client.get("/api/v1/missions/M40", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M40"
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
                json={"username": "m40-learner", "display_name": "M40 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M40", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M40"
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
            assert 'kn.m40.deterministic_graders' in nodes
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
                for required in ['metric', 'table']:
                    assert required in types
                assert set(types) <= WP137_BLOCK_TYPES
                metric = next(block for block in blocks if block["type"] == "metric")["payload"]
                assert metric['eval_version'] == 'm40.eval.v1'
                assert metric['n'] == 12
                assert metric['aggregate_only_passed'] == True
                assert metric['canonical_gate_passed'] == False
                assert metric['hidden_critical_case'] == 'rag-grounded-reset'
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
            assert increments == {'comp.eval.slice_gates', 'comp.eval.unassisted_transfer', 'comp.eval.deterministic_graders', 'comp.eval.suite_versioning'}

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
