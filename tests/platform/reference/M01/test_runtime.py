"""Load G5 M01 on the frozen generic runtime without writing ~/.learningos."""

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
M01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M01"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
REAL_HOME_DB = Path.home() / ".learningos" / "learningos.db"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_MAP = "stage_02_system_map"
STAGE_TRACE = "stage_03_system_trace"
STAGE_INTERROGATE = "stage_04_interrogate"
STAGE_E1 = "stage_05_experiment_e1"
STAGE_E2 = "stage_06_experiment_e2"
STAGE_E3 = "stage_07_experiment_e3"
STAGE_E4 = "stage_08_experiment_e4"
STAGE_E5 = "stage_09_experiment_e5"
STAGE_FAILURE = "stage_10_controlled_failure"
STAGE_NO_AI = "stage_11_no_ai_gate"
STAGE_TRANSFER = "stage_12_transfer"
STAGE_GATE = "stage_13_gate"

PREDICT_BODY = {
    "hypothesis": "one inference call leaves learned classifier state unchanged",
    "expected_values": {"unchanged": True},
}
EXECUTE_CODE = """
from collections import Counter, defaultdict
from hashlib import sha256
import json

def tok(s):
    return [w.strip('.,!?;:').lower() for w in s.split() if w.strip('.,!?;:')]

def train_classifier(rows):
    docs = Counter()
    words = defaultdict(Counter)
    vocab = set()
    for row in rows:
        docs[row['label']] += 1
        for word in tok(row['text']):
            words[row['label']][word] += 1
            vocab.add(word)
    return {
        'labels': sorted(docs),
        'docs': dict(docs),
        'words': {label: dict(words[label]) for label in sorted(docs)},
        'vocab': sorted(vocab),
    }

def digest(model):
    return sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()[:12]

train = [
    {'id': 1, 'text': 'reset account password', 'label': 'account'},
    {'id': 2, 'text': 'duplicate invoice charge', 'label': 'billing'},
]
model = train_classifier(train)
before = digest(model)
text = 'duplicate invoice charge'
scores = {}
n = sum(model['docs'].values())
v = max(1, len(model['vocab']))
for label in model['labels']:
    counts = model['words'][label]
    total = sum(counts.values())
    prior = model['docs'][label] / n
    mass = prior
    for word in tok(text):
        mass *= (counts.get(word, 0) + 1) / (total + v)
    scores[label] = mass
label = max(scores, key=scores.get)
after = digest(model)
payload = {
    'digest_before': before,
    'digest_after': after,
    'unchanged': before == after,
    'predicted_label': label,
}
print(payload)
{'type': 'metric', 'title': 'inference_state', 'payload': payload}
"""
EXECUTE_BODY = {"code": EXECUTE_CODE, "parameters": {}}
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}

SUBMIT_PLAN = [
    (STAGE_ORIENT, "framed the whole-system map", []),
    (STAGE_MAP, "mapped layers with data-flow and control-flow", [{"artifact_type": "diagram"}]),
    (STAGE_TRACE, "traced train/predict/retrieve/controller", [{"artifact_type": "trace"}]),
    (STAGE_INTERROGATE, "probed training versus inference evidence", []),
    (STAGE_E1, "inference left digest unchanged", [{"artifact_type": "metric"}]),
    (STAGE_E2, "retraining changed digest", [{"artifact_type": "metric"}]),
    (STAGE_E3, "retrieval changed context not weights", [{"artifact_type": "metric"}]),
    (STAGE_E4, "controller selected tool on urgent path", [{"artifact_type": "trace"}]),
    (STAGE_E5, "four layer labels were wrong", [{"artifact_type": "metric"}]),
    (STAGE_FAILURE, "repaired conflated training/inference/retrieval/tool/memory", [{"artifact_type": "markdown"}]),
    (STAGE_NO_AI, "reconstructed map without inventing training", [{"artifact_type": "diagram"}]),
    (STAGE_TRANSFER, "mapped unseen systems with calibrated uncertainty", [{"artifact_type": "artifact"}]),
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


@pytest.fixture
def m01_env(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    home = Path(tempfile.mkdtemp(prefix="los-g5-m01-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g5-m01-{uuid.uuid4().hex}.sock")
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
        json={"package_dir": str(M01_PACKAGE)},
        headers=headers,
    )
    assert loaded.status_code == 200, loaded.text
    body = loaded.json()
    assert body.get("id") == "g5.reference.M01"
    learner = client.post(
        "/api/v1/learners",
        json={"username": "m01-learner", "display_name": "M01 Learner"},
        headers=headers,
    )
    assert learner.status_code == 200, learner.text
    learner_id = learner.json()["learner_id"]
    session = client.post(
        "/api/v1/sessions",
        json={"mission_id": "M01", "learner_id": learner_id},
        headers=headers,
    )
    assert session.status_code == 200, session.text
    session_body = session.json()
    assert session_body["mission_id"] == "M01"
    assert session_body["status"] == "ACTIVE"
    assert session_body["current_stage_id"] == STAGE_ORIENT
    return learner_id, session_body["session_id"]


def _enter_submit(client, headers, session_id, stage_id, explanation, artifacts):
    entered = client.post(
        f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
        headers=headers,
    )
    assert entered.status_code == 200, entered.text
    submitted = client.post(
        f"/api/v1/sessions/{session_id}/stages/{stage_id}/submit",
        json={"explanation": explanation, "artifacts": artifacts},
        headers=headers,
    )
    assert submitted.status_code == 200, submitted.text
    return submitted.json()


def test_load_package_and_create_session_without_writing_home(m01_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m01_env["home"]
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        headers = _bootstrap(client)
        learner_id, session_id = _load_and_start(client, headers)
        missions = client.get("/api/v1/missions", headers=headers)
        assert missions.status_code == 200, missions.text
        mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
        assert "M01" in mission_ids
        mission = client.get("/api/v1/missions/M01", headers=headers)
        assert mission.status_code == 200, mission.text
        spec = mission.json()
        assert spec["id"] == "M01"
        types = [stage["type"] for stage in spec["stages"]]
        assert "trace_map" in types
        assert "experiment" in types
        assert "controlled_failure" in types
        assert "transfer_assessment" in types
        got = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        assert got.status_code == 200, got.text
        assert got.json()["current_stage_id"] == STAGE_ORIENT
        assert learner_id
    assert home.exists()
    assert str(home).startswith("/tmp/")
    _assert_home_untouched(before)


def test_gate_targeted_repair_when_evidence_missing(m01_env):
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
        assert "kn.m01.system_map" in nodes
        assert "kn.m01.training_inference_boundary" in nodes
        assert "kn.m01.no_ai_reconstruction" in nodes
        assert "kn.m01.transfer" in nodes
        stages = set(plan["target_stage_ids"])
        assert STAGE_MAP in stages
        assert STAGE_E1 in stages
        assert STAGE_NO_AI in stages
    _assert_home_untouched(before)


def test_prediction_seal_structured_output_and_full_gate(m01_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m01_env["home"]
    sock: Path = m01_env["sock"]
    env: dict[str, str] = m01_env["env"]
    worker: subprocess.Popen[bytes] | None = None
    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            learner_id, session_id = _load_and_start(client, headers)

            for stage_id, explanation, artifacts in SUBMIT_PLAN:
                if stage_id not in {STAGE_E1, STAGE_E2}:
                    submitted = _enter_submit(
                        client, headers, session_id, stage_id, explanation, artifacts
                    )
                    continue
                if stage_id == STAGE_E2:
                    entered = client.post(
                        f"/api/v1/sessions/{session_id}/stages/{STAGE_E2}/enter",
                        headers=headers,
                    )
                    assert entered.status_code == 200, entered.text
                    blocked_e2 = client.post(
                        f"/api/v1/sessions/{session_id}/stages/{STAGE_E2}/execute",
                        json=EXECUTE_BODY,
                        headers=headers,
                    )
                    assert blocked_e2.status_code == 409, blocked_e2.text
                    assert blocked_e2.json()["error"]["details"].get("reason") == "PREDICTION_REQUIRED"
                    predicted_e2 = client.post(
                        f"/api/v1/sessions/{session_id}/stages/{STAGE_E2}/predict",
                        json={
                            "hypothesis": "adding a labelled example and retraining changes digest",
                            "expected_values": {"digest_changed": True},
                        },
                        headers=headers,
                    )
                    assert predicted_e2.status_code == 200, predicted_e2.text
                    assert predicted_e2.json()["is_sealed"] is True
                    submitted = _enter_submit(
                        client, headers, session_id, STAGE_E2, explanation, artifacts
                    )
                    continue

                entered = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{STAGE_E1}/enter",
                    headers=headers,
                )
                assert entered.status_code == 200, entered.text
                assert entered.json()["stage_type"] == "experiment"

                blocked = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{STAGE_E1}/execute",
                    json=EXECUTE_BODY,
                    headers=headers,
                )
                assert blocked.status_code == 409, blocked.text
                assert blocked.json()["error"]["code"] == "CONFLICT"
                assert blocked.json()["error"]["details"].get("reason") == "PREDICTION_REQUIRED"

                predicted = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{STAGE_E1}/predict",
                    json=PREDICT_BODY,
                    headers=headers,
                )
                assert predicted.status_code == 200, predicted.text
                assert predicted.json()["is_sealed"] is True
                assert predicted.json()["prediction_hash"] != "dummy_hash"
                assert len(predicted.json()["prediction_hash"]) == 64

                executed = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{STAGE_E1}/execute",
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
                assert blocks[0]["type"] == "metric"
                assert blocks[0]["payload"]["unchanged"] is True
                assert "sk-test-should-never-leak" not in json.dumps(exec_body)

                submitted = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{STAGE_E1}/submit",
                    json={"explanation": explanation, "artifacts": artifacts},
                    headers=headers,
                )
                assert submitted.status_code == 200, submitted.text
                assert submitted.json()["current_stage_id"] == STAGE_E2

            assert submitted["current_stage_id"] == STAGE_GATE

            gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = {item["competency_id"] for item in (gate_body.get("competency_increments") or [])}
            assert "comp.sys.whole_trace" in increments
            assert "comp.sys.hypothesis_testing" in increments
            assert "comp.sys.unassisted_transfer" in increments

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
