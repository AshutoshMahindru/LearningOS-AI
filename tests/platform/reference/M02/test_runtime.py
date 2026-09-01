"""G5 M02 on the generic runtime: predict seal, no-AI, evidence, gate, repair, failure."""

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
M02_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M02"
DAEMON_PATH = REPO_ROOT / "platform" / "worker" / "daemon.py"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

STAGE_ORIENT = "stage_01_orientation"
STAGE_TRACE = "stage_02_trace_map"
STAGE_INTERROGATE = "stage_03_interrogate"
STAGE_CODE = "stage_04_code_reading"
STAGE_EXPERIMENT = "stage_05_experiment"
STAGE_FAILURE = "stage_06_controlled_failure"
STAGE_TRANSFER = "stage_07_transfer"
STAGE_FLAGSHIP = "stage_08_flagship"
STAGE_GATE = "stage_09_gate"

PREDICT_BODY = {
    "hypothesis": (
        "Rotating training labels breaks row-class alignment; honest held-out "
        "accuracy should fall toward chance while self-comparison of predictions "
        "still reports 1.0."
    ),
    "expected_values": {
        "honest_intact_min": 1.0,
        "honest_shuffled_max": 0.34,
        "self_comparison": 1.0,
    },
}

_SUPERVISED_SNIPPET = """
rows = [
    ((14.2, 3.1), 0), ((14.0, 2.9), 0), ((14.1, 3.0), 0), ((14.3, 3.2), 0),
    ((12.2, 2.0), 1), ((12.0, 1.9), 1), ((12.1, 2.1), 1), ((12.3, 2.0), 1),
    ((13.1, 0.6), 2), ((12.9, 0.5), 2), ((13.0, 0.7), 2), ((13.2, 0.6), 2),
]
rotate = int(parameters.get("rotate_labels") or 0) % 3
seed = int(parameters.get("seed") or 7)

def split(items):
    train, test = [], []
    for index, item in enumerate(items):
        (test if (index % 4) == 3 else train).append(item)
    return train, test

def centroids(train):
    sums = {}
    counts = {}
    for features, label in train:
        if label not in sums:
            sums[label] = [0.0] * len(features)
            counts[label] = 0
        for axis, value in enumerate(features):
            sums[label][axis] += value
        counts[label] += 1
    return {label: [total / counts[label] for total in sums[label]] for label in sums}

def predict(centers, features):
    assigned = []
    for row in features:
        best_label, best_distance = None, None
        for label, center in centers.items():
            distance = sum((left - right) ** 2 for left, right in zip(row, center))
            if best_distance is None or distance < best_distance:
                best_label, best_distance = label, distance
        assigned.append(best_label)
    return assigned

def accuracy(truth, guessed):
    return sum(left == right for left, right in zip(truth, guessed)) / len(truth)

train, test = split(rows)
corrupted = [(features, (label + rotate) % 3) for features, label in train]
centers = centroids(corrupted)
x_test = [features for features, _label in test]
y_test = [label for _features, label in test]
guessed = predict(centers, x_test)
honest = accuracy(y_test, guessed)
invalid = accuracy(guessed, guessed)
payload = {
    "seed": seed,
    "rotate_labels": rotate,
    "n_train": len(train),
    "n_test": len(test),
    "honest_accuracy": honest,
    "self_comparison": invalid,
    "disagreement_rate": sum(a != b for a, b in zip([y for _, y in train], [y for _, y in corrupted])) / len(train),
}
print(payload)
{"type": "metric", "title": "supervised-interrogation", "payload": payload}
"""

EXECUTE_INTACT = {"code": _SUPERVISED_SNIPPET, "parameters": {"seed": 7, "rotate_labels": 0}}
EXECUTE_CORRUPTED = {"code": _SUPERVISED_SNIPPET, "parameters": {"seed": 7, "rotate_labels": 1}}
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
def m02_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g5-m02-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g5-m02-{uuid.uuid4().hex}.sock")
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


def test_m02_generic_runtime_roundtrip(m02_env):
    from fastapi.testclient import TestClient

    from app.core.mdl_types import STAGE_TYPES
    from app.main import app

    home: Path = m02_env["home"]
    sock: Path = m02_env["sock"]
    env: dict[str, str] = m02_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M02_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M02"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M02" in mission_ids

            mission = client.get("/api/v1/missions/M02", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M02"
            types = [stage["type"] for stage in spec["stages"]]
            assert set(types) <= set(STAGE_TYPES)
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == [
                STAGE_ORIENT,
                STAGE_TRACE,
                STAGE_INTERROGATE,
                STAGE_CODE,
                STAGE_EXPERIMENT,
                STAGE_FAILURE,
                STAGE_TRANSFER,
                STAGE_FLAGSHIP,
                STAGE_GATE,
            ]

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m02-learner", "display_name": "M02 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M02", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M02"
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
                "framed the wine vertical slice and gate evidence",
                [{"artifact_type": "markdown"}],
            )
            assert submitted.json()["current_stage_id"] == STAGE_TRACE

            early_gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert early_gate.status_code == 200, early_gate.text
            early_body = early_gate.json()
            assert early_body["status"] == "REPAIR_REQUIRED"
            assert early_body["reason"] == "GATE_CRITERIA_UNMET"
            plan = early_body["repair_plan"]
            actions = {item.get("action") for item in plan.get("drills") or []}
            assert "targeted_repair" in actions
            failed_nodes = set(plan.get("failed_knowledge_nodes") or [])
            assert "kn.m02.predict_execute" in failed_nodes
            assert "kn.m02.failure_diagnosis" in failed_nodes
            assert "kn.m02.no_ai_transfer" in failed_nodes
            target_stages = set(plan.get("target_stage_ids") or [])
            assert STAGE_EXPERIMENT in target_stages
            assert STAGE_FAILURE in target_stages
            assert STAGE_TRANSFER in target_stages

            for stage_id, explanation, artifacts in (
                (STAGE_TRACE, "mapped raw csv through interrogation", [{"artifact_type": "diagram"}]),
                (STAGE_INTERROGATE, "test labels must not enter fit or predict", [{"artifact_type": "markdown"}]),
                (STAGE_CODE, "fit changes state; predict consumes it; evaluation reintroduces truth", [{"artifact_type": "markdown"}]),
            ):
                _enter(resumed, headers, session_id, stage_id)
                submitted = _submit(resumed, headers, session_id, stage_id, explanation, artifacts)
            assert submitted.json()["current_stage_id"] == STAGE_EXPERIMENT

            lab_enter = _enter(resumed, headers, session_id, STAGE_EXPERIMENT)
            assert lab_enter.json()["stage_type"] == "experiment"

            blocked = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXECUTE_INTACT,
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
                json=EXECUTE_INTACT,
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
            assert isinstance(blocks, list) and blocks
            assert blocks[0]["type"] == "metric"
            metric = blocks[0]["payload"]
            assert metric["honest_accuracy"] == 1.0
            assert metric["self_comparison"] == 1.0
            assert metric["rotate_labels"] == 0
            assert "dummy_hash" not in json.dumps(exec_body)
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)

            lab_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_EXPERIMENT,
                "intact labels preserve the honest 1.0; constants were dataset, model, and metric",
                [{"artifact_type": "metric"}],
            )
            assert lab_submit.json()["current_stage_id"] == STAGE_FAILURE
            assert lab_submit.json()["payload_hash"] != "dummy_hash"

            failure_enter = _enter(resumed, headers, session_id, STAGE_FAILURE)
            assert failure_enter.json()["stage_type"] == "controlled_failure"
            failure_exec = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_FAILURE}/execute",
                json=EXECUTE_CORRUPTED,
                headers=headers,
            )
            assert failure_exec.status_code == 200, failure_exec.text
            failure_body = failure_exec.json()
            assert failure_body["status"] == "SUCCESS", failure_body
            failure_blocks = (failure_body.get("structured_result") or {}).get("blocks") or []
            assert failure_blocks
            failure_metric = failure_blocks[0]["payload"]
            assert failure_metric["self_comparison"] == 1.0
            assert failure_metric["honest_accuracy"] < 1.0
            assert failure_metric["disagreement_rate"] == 1.0
            failure_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_FAILURE,
                "root cause is corrupted supervision and self-comparison; repair restores labels and y_test alignment",
                [{"artifact_type": "trace"}],
            )
            assert failure_submit.json()["current_stage_id"] == STAGE_TRANSFER

            transfer_enter = _enter(resumed, headers, session_id, STAGE_TRANSFER)
            assert transfer_enter.json()["stage_type"] == "transfer_assessment"
            transfer_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_TRANSFER,
                "fresh no-AI supervised run with a different seed and two-feature subset",
                [{"artifact_type": "artifact"}],
            )
            assert transfer_submit.json()["current_stage_id"] == STAGE_FLAGSHIP

            _enter(resumed, headers, session_id, STAGE_FLAGSHIP)
            flagship_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_FLAGSHIP,
                "V00 gains an interrogable vertical slice and an evaluation-integrity guardrail",
                [{"artifact_type": "markdown"}],
            )
            assert flagship_submit.json()["current_stage_id"] == STAGE_GATE

            gate_enter = _enter(resumed, headers, session_id, STAGE_GATE)
            assert gate_enter.json()["stage_type"] == "competency_gate"
            gate_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_GATE,
                "ready for gate evaluation",
                [],
            )
            assert gate_submit.status_code == 200, gate_submit.text

            tutor = resumed.post(
                "/api/v1/tutor/chat",
                json={
                    "session_id": session_id,
                    "stage_id": STAGE_TRANSFER,
                    "role": "learner",
                    "prompt": "please complete the no-AI transfer",
                },
                headers=headers,
            )
            assert tutor.status_code == 501, tutor.text
            assert tutor.json()["error"]["code"] == "TUTOR_NOT_AVAILABLE"

            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = gate_body.get("competency_increments") or []
            assert {item["competency_id"] for item in increments} == {
                "comp.m02.system_orientation",
                "comp.m02.experimental_reasoning",
                "comp.m02.evaluation_integrity",
                "comp.m02.unassisted_transfer",
            }

            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            evidence_body = evidence.json()
            claims = evidence_body.get("evidence") or []
            assert claims
            assert "dummy_hash" not in json.dumps(evidence_body)
            assistance = {claim.get("stage_id"): claim.get("assistance_level") for claim in claims}
            assert assistance.get(STAGE_TRANSFER) == "NO_AI_CERTIFIED"
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
                "comp.m02.system_orientation",
                "comp.m02.experimental_reasoning",
                "comp.m02.evaluation_integrity",
                "comp.m02.unassisted_transfer",
            }

            final = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert final.status_code == 200, final.text
            assert final.json()["status"] == "COMPLETED"
            assert "dummy_hash" not in json.dumps(final.json())
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
