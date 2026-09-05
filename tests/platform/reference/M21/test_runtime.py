"""G5 M21 on the frozen generic runtime: predict → WP-137 chart execute → gate."""

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
M21_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M21"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_EXPERIMENT = "stage_02_experiment"
STAGE_CODE = "stage_03_code_reading"
STAGE_FAILURE = "stage_04_controlled_failure"
STAGE_TRANSFER = "stage_05_transfer"
STAGE_SIX = "stage_06_flagship"
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

PREDICT_BODY = {'hypothesis': 'reference budget and intact labels beat majority baseline; max_iter=1 and shuffled labels do not', 'expected_values': {'beats_baseline': True, 'max_iter': 60}}

EXECUTE_CODE = "\nimport math\nfrom collections import Counter\n\nrows = [\n    ((0.0, 0.0), 0), ((0.1, 0.0), 0), ((0.0, 0.1), 0),\n    ((1.0, 1.0), 1), ((0.9, 1.0), 1), ((1.0, 0.9), 1),\n    ((0.05, 0.05), 0), ((0.95, 0.95), 1),\n]\nmax_iter = int(parameters.get('max_iter') or 60)\nshuffle = int(parameters.get('shuffle_labels') or 0)\nseed = int(parameters.get('model_seed') or 2101)\ntrain, test = rows[:6], rows[6:]\ny_train = [y for _, y in train]\nif shuffle:\n    y_train = y_train[1:] + y_train[:1]\nmaj = Counter(y for _, y in test).most_common(1)[0][0]\nbaseline = sum(1 for _, y in test if y == maj) / len(test)\n\ndef centroids(data, labels):\n    sums, counts = {}, {}\n    for (feat, _), label in zip(data, labels):\n        sums.setdefault(label, [0.0, 0.0])\n        counts[label] = counts.get(label, 0) + 1\n        sums[label][0] += feat[0]\n        sums[label][1] += feat[1]\n    return {k: [s / counts[k] for s in v] for k, v in sums.items()}\n\ncenters = centroids(train, y_train)\n\ndef predict(feat):\n    best, best_d = None, None\n    for label, center in centers.items():\n        dist = (feat[0] - center[0]) ** 2 + (feat[1] - center[1]) ** 2\n        if best_d is None or dist < best_d:\n            best, best_d = label, dist\n    return best\n\ntest_acc = sum(predict(f) == y for f, y in test) / len(test)\ntrain_acc = sum(predict(f) == y for (f, _), y in zip(train, y_train)) / len(train)\nif max_iter <= 1:\n    loss_curve = [2.3]\n    test_acc = min(test_acc, baseline)\nelif shuffle:\n    loss_curve = [1.2 - 0.01 * i for i in range(5)]\n    test_acc = baseline\nelse:\n    loss_curve = [1.5 * math.exp(-0.08 * i) for i in range(8)]\npayload = {\n    'model_seed': seed,\n    'max_iter': max_iter,\n    'shuffled_labels': bool(shuffle),\n    'majority_baseline_accuracy': baseline,\n    'train_accuracy': train_acc,\n    'test_accuracy': test_acc,\n    'beats_baseline': test_acc > baseline + 1e-9,\n}\nprint(payload)\n[\n    {\n        'type': 'chart',\n        'title': 'training-loss',\n        'payload': {\n            'chart_type': 'line',\n            'series': [{'name': 'loss', 'x': list(range(len(loss_curve))), 'y': loss_curve}],\n        },\n    },\n    {'type': 'metric', 'title': 'holdout', 'payload': payload},\n    {\n        'type': 'table',\n        'title': 'run-config',\n        'payload': {\n            'columns': ['field', 'value'],\n            'rows': [['max_iter', max_iter], ['seed', seed], ['test_accuracy', test_acc]],\n        },\n    },\n]\n"

EXECUTE_BODY = {
    "code": EXECUTE_CODE,
    "parameters": {"max_iter": 60, "model_seed": 2101, "shuffle_labels": 0},
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
def m21_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-m21-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m21-{uuid.uuid4().hex}.sock")
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
        "execution_id": "exec-m21-ref",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [{'type': 'chart', 'title': 'training-loss', 'payload': {'chart_type': 'line', 'series': [{'name': 'loss', 'x': [0, 1], 'y': [1.2, 0.4]}]}}, {'type': 'metric', 'title': 'holdout', 'payload': {'test_accuracy': 0.9, 'majority_baseline_accuracy': 0.1}}, {'type': 'table', 'title': 'run-config', 'payload': {'columns': ['field', 'value'], 'rows': [['max_iter', 60]]}}],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert set(types) <= WP137_BLOCK_TYPES
    assert "chart" in types


def test_m21_runtime_roundtrip_emits_wp137(m21_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = m21_env["home"]
    sock: Path = m21_env["sock"]
    env: dict[str, str] = m21_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M21_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M21"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M21" in mission_ids

            mission = client.get("/api/v1/missions/M21", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M21"
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
            assert experiment["validation_rubric"]["required_evidence_type"] == "chart"

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m21-learner", "display_name": "M21 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M21", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M21"
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
                'framed the black-box digits MLP whole; gate needs chart, table, metric',
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
            assert "chart" in types
            assert "metric" in types
            chart = next(block for block in blocks if block["type"] == "chart")
            assert chart["payload"].get("chart_type") == "line"
            metric = next(block for block in blocks if block["type"] == "metric")
            assert metric["payload"]["max_iter"] == 60
            assert metric["payload"]["beats_baseline"] is True
            assert metric["payload"]["shuffled_labels"] is False

            from app.execution.result_schema import validate_structured_result

            validate_structured_result(structured)

            lab_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_EXPERIMENT,
                "observed WP-137 blocks after a sealed prediction",
                [{"artifact_type": "chart"}],
            )
            assert lab_submit.json()["current_stage_id"] == STAGE_CODE

            _enter(resumed, headers, session_id, STAGE_CODE)
            code_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_CODE,
                'split, training-only scaler, fit, held-out eval; no coefs_',
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
                'max_iter=1 undertrains; shuffled labels break the target map; restore the named contract',
                [{"artifact_type": "table"}],
            )
            assert fail_submit.json()["current_stage_id"] == STAGE_TRANSFER

            _enter(resumed, headers, session_id, STAGE_TRANSFER)
            transfer_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_TRANSFER,
                'fresh run beats baseline; most-confused pair read from the matrix without internals',
                [{"artifact_type": "metric"}],
            )
            assert transfer_submit.json()["current_stage_id"] == STAGE_SIX

            _enter(resumed, headers, session_id, STAGE_SIX)
            six_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_SIX,
                'V05 starts from an accepted black-box run; M22 may not rewrite held-out evidence',
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
            assert {item["competency_id"] for item in increments} == {'comp.nn.black_box_training', 'comp.nn.holdout_baseline', 'comp.nn.seed_replay', 'comp.nn.training_failure_diagnosis'}

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
