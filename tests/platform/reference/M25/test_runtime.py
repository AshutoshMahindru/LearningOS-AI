"""G5 M25 on the frozen generic runtime: predict → WP-137 loop metric → gate."""

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
M25_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M25"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_EXPERIMENT = "stage_02_experiment"
STAGE_CODE = "stage_03_code_reading"
STAGE_FAILURE = "stage_04_controlled_failure"
STAGE_TRANSFER = "stage_05_transfer"
STAGE_SIX = "stage_06_adr"
STAGE_GATE = "stage_07_gate"

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

PREDICT_BODY = {'hypothesis': 'zero_grad → forward → loss → backward → step moves W; eval/no_grad does not', 'expected_values': {'moved': True, 'eval_mode': True}}

EXECUTE_CODE = "\nparams = {'W2_00': 1.0, 'b2_0': 0.0}\ngrad = {'W2_00': 0.4, 'b2_0': 0.1}\nlr = 0.1\nlog = []\nzero_grad_called = True\nif zero_grad_called:\n    log.append('zero_grad')\nlog.extend(['forward', 'loss', 'backward'])\nbefore = dict(params)\nparams['W2_00'] = params['W2_00'] - lr * grad['W2_00']\nparams['b2_0'] = params['b2_0'] - lr * grad['b2_0']\nlog.append('step')\neval_mode = True\nno_grad = True\npayload = {\n    'loop_order': log,\n    'W2_00_before': before['W2_00'],\n    'W2_00_after': params['W2_00'],\n    'moved': params['W2_00'] != before['W2_00'],\n    'eval_mode': eval_mode,\n    'no_grad': no_grad,\n    'autograd_vs_m24': 0.4,\n    'layout_note': 'nn.Linear W is (out, in); compare W.T with M24',\n}\nprint(payload)\n[\n    {'type': 'metric', 'title': 'canonical-step', 'payload': payload},\n    {\n        'type': 'trace',\n        'title': 'loop-order',\n        'payload': {'order': log, 'eval': ['eval', 'no_grad']},\n    },\n    {\n        'type': 'table',\n        'title': 'splits',\n        'payload': {\n            'columns': ['split', 'n'],\n            'rows': [['train', 24], ['val', 6], ['held_out', 6]],\n        },\n    },\n]\n"

EXECUTE_BODY = {
    "code": EXECUTE_CODE,
    "parameters": {},
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


def _bootstrap(client):
    boot = client.post("/api/v1/auth/bootstrap")
    assert boot.status_code == 200, boot.text
    token = boot.json()["token"]
    assert boot.json()["token_type"] == "bearer"
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


@pytest.fixture
def m25_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-m25-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m25-{uuid.uuid4().hex}.sock")
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


def test_sample_result_matches_frozen_wp137() -> None:
    from app.execution.result_schema import validate_structured_result

    sample = {
        "execution_id": "exec-m25-ref",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [{'type': 'metric', 'title': 'canonical-step', 'payload': {'moved': True, 'eval_mode': True}}, {'type': 'trace', 'title': 'loop-order', 'payload': {'order': ['zero_grad', 'forward', 'loss', 'backward', 'step']}}, {'type': 'table', 'title': 'splits', 'payload': {'columns': ['split', 'n'], 'rows': [['train', 24]]}}],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert set(types) <= WP137_BLOCK_TYPES
    assert "metric" in types


def test_m25_runtime_roundtrip_emits_wp137(m25_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = m25_env["home"]
    sock: Path = m25_env["sock"]
    env: dict[str, str] = m25_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M25_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M25"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M25" in mission_ids

            mission = client.get("/api/v1/missions/M25", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M25"
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == [
                STAGE_ORIENT,
                STAGE_EXPERIMENT,
                STAGE_CODE,
                STAGE_FAILURE,
                STAGE_TRANSFER,
                STAGE_SIX,
                STAGE_GATE,
            ]
            experiment = next(stage for stage in spec["stages"] if stage["id"] == STAGE_EXPERIMENT)
            assert experiment["type"] == "experiment"
            assert experiment["validation_rubric"]["required_evidence_type"] == "metric"

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m25-learner", "display_name": "M25 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M25", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M25"
            assert session_body["status"] == "ACTIVE"
            assert session_body["current_stage_id"] == STAGE_ORIENT
            session_id = session_body["session_id"]

            entered = _enter(client, headers, session_id, STAGE_ORIENT)
            assert entered.json()["current_stage_id"] == STAGE_ORIENT

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            got = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert got.status_code == 200, got.text
            assert got.json()["current_stage_id"] == STAGE_ORIENT
            assert got.json()["status"] == "ACTIVE"

            submitted = _submit(
                resumed,
                headers,
                session_id,
                STAGE_ORIENT,
                'framed CPU autograd module and loop order from M24',
                [{"artifact_type": "markdown"}],
            )
            assert submitted.json()["current_stage_id"] == STAGE_EXPERIMENT

            early_gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert early_gate.status_code == 200, early_gate.text
            early_body = early_gate.json()
            assert early_body["status"] == "REPAIR_REQUIRED"
            assert early_body["reason"] == "GATE_CRITERIA_UNMET"
            plan = early_body["repair_plan"]
            actions = {item.get("action") for item in plan.get("drills") or []}
            assert "targeted_repair" in actions
            target_stages = set(plan.get("target_stage_ids") or [])
            assert STAGE_EXPERIMENT in target_stages
            assert STAGE_FAILURE in target_stages
            assert STAGE_TRANSFER in target_stages

            lab_enter = _enter(resumed, headers, session_id, STAGE_EXPERIMENT)
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
            assert set(block["type"] for block in blocks) <= WP137_BLOCK_TYPES
            assert "dummy_hash" not in json.dumps(exec_body)
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)

            types = [block["type"] for block in blocks]
            assert "metric" in types
            metric = next(block for block in blocks if block["type"] == "metric")
            assert metric["payload"]["moved"] is True
            assert metric["payload"]["eval_mode"] is True
            assert metric["payload"]["loop_order"] == ["zero_grad", "forward", "loss", "backward", "step"]

            from app.execution.result_schema import validate_structured_result

            validate_structured_result(structured)

            lab_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_EXPERIMENT,
                "observed WP-137 blocks after a sealed prediction",
                [{"artifact_type": "metric"}],
            )
            assert lab_submit.json()["current_stage_id"] == STAGE_CODE

            _enter(resumed, headers, session_id, STAGE_CODE)
            code_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_CODE,
                'registration, zero_grad, mode, checkpoint, held-out boundary',
                [{"artifact_type": "artifact"}],
            )
            assert code_submit.json()["current_stage_id"] == STAGE_FAILURE

            fail_enter = _enter(resumed, headers, session_id, STAGE_FAILURE)
            assert fail_enter.json()["stage_type"] == "controlled_failure"
            fail_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_FAILURE,
                'stale .grad or train-mode eval; restore zero_grad and eval/no_grad',
                [{"artifact_type": "table"}],
            )
            assert fail_submit.json()["current_stage_id"] == STAGE_TRANSFER

            _enter(resumed, headers, session_id, STAGE_TRANSFER)
            transfer_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_TRANSFER,
                'fresh loop records named controls and frozen evaluation parameters',
                [{"artifact_type": "metric"}],
            )
            assert transfer_submit.json()["current_stage_id"] == STAGE_SIX

            _enter(resumed, headers, session_id, STAGE_SIX)
            six_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_SIX,
                'checkpoint ADR records seeds, optimizer state, rollback',
                [{"artifact_type": "markdown"}],
            )
            assert six_submit.json()["current_stage_id"] == STAGE_GATE

            _enter(resumed, headers, session_id, STAGE_GATE)
            gate_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_GATE,
                'ready for gate evaluation',
                [],
            )
            assert gate_submit.status_code == 200, gate_submit.text

            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = gate_body.get("competency_increments") or []
            assert {item["competency_id"] for item in increments} == {'comp.nn.autograd_mechanics', 'comp.nn.gradient_debugging', 'comp.nn.train_eval_boundary', 'comp.nn.training_loop'}

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
