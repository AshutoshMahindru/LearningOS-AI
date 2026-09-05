"""G5 M31 on the frozen generic runtime: predict → WP-137 execute → gate."""

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
M31_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M31"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_EXPERIMENT = "stage_02_experiment"
STAGE_CODE = "stage_03_code_reading"
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

PREDICT_BODY = {"hypothesis": "targets equal tokens[1:]; train loss falls; held-out NLL is a separate claim", "expected_values": {"targets": [20, 30, 40], "seed": 3101}}

EXECUTE_CODE = "tokens = [10, 20, 30, 40]\ninputs = tokens[:-1]\ntargets = tokens[1:]\nseed = int(parameters.get(\"seed\") or 3101)\nsteps = int(parameters.get(\"steps\") or 60)\nleak = int(parameters.get(\"leak\") or 0)\ntrain_ids = [\"t01\", \"t02\"]\neval_ids = [\"e01\", \"e02\"]\nif leak:\n    train_ids = train_ids + [\"e02\"]\noverlap = sorted(set(train_ids) & set(eval_ids))\nloss_curve = [2.2 * (0.92 ** i) for i in range(6)]\npayload = {\n    \"inputs\": inputs,\n    \"targets\": targets,\n    \"shift_ok\": targets == tokens[1:],\n    \"seed\": seed,\n    \"steps\": steps,\n    \"dataset_version\": \"v07-teaching-corpus-1\",\n    \"held_out_nll\": 1.8,\n    \"overlap\": overlap,\n    \"eval_valid\": overlap == [],\n}\nprint(payload)\n[\n    {\"type\": \"chart\", \"title\": \"train-loss\", \"payload\": {\"chart_type\": \"line\", \"series\": [{\"name\": \"loss\", \"x\": list(range(len(loss_curve))), \"y\": loss_curve}]}},\n    {\"type\": \"table\", \"title\": \"pairs\", \"payload\": {\"columns\": [\"input\", \"target\"], \"rows\": list(zip(inputs, targets))}},\n    {\"type\": \"metric\", \"title\": \"run\", \"payload\": payload},\n]"

EXECUTE_BODY = {
    "code": EXECUTE_CODE,
    "parameters": {"seed": 3101, "steps": 60, "leak": 0},
}

SUBMIT_PLAN = [
    (STAGE_ORIENT, "framed next-token pairs, protected eval, and a stage map", [{"artifact_type": "markdown"}]),
    (STAGE_EXPERIMENT, "observed WP-137 train-loss chart after a sealed prediction", [{"artifact_type": "chart"}]),
    (STAGE_CODE, "traced encode, window, tokens[1:] shift, split hash, eval alignment", [{"artifact_type": "artifact"}]),
    (STAGE_FAILURE, "e02 leaked into train; restore authored membership, keep eval shift", [{"artifact_type": "table"}]),
    (STAGE_TRANSFER, "pairs (10,20,30)\u2192(20,30,40); inference is not a weight update", [{"artifact_type": "metric"}]),
    (STAGE_ADR, "locked v07-teaching-corpus-1, split hash, checkpoint, audit fields", [{"artifact_type": "markdown"}]),
    (STAGE_FLAGSHIP, "M32 consumes a stage-aware checkpoint with frozen weights", [{"artifact_type": "markdown"}]),
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


@pytest.fixture
def m31_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-m31-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m31-{uuid.uuid4().hex}.sock")
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
        "execution_id": "exec-m31-ref",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [{"type": "chart", "title": "train-loss", "payload": {"chart_type": "line", "series": [{"name": "loss", "x": [0, 1], "y": [2.1, 0.7]}]}}, {"type": "metric", "title": "pairs", "payload": {"targets": [20, 30, 40], "seed": 3101}}],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert set(types) <= WP137_BLOCK_TYPES
    assert "chart" in types


def test_m31_runtime_roundtrip_emits_wp137(m31_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = m31_env["home"]
    sock: Path = m31_env["sock"]
    env: dict[str, str] = m31_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M31_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M31"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M31" in mission_ids
            assert "g5.reference.M31" not in mission_ids

            mission = client.get("/api/v1/missions/M31", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M31"
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == [
                STAGE_ORIENT,
                STAGE_EXPERIMENT,
                STAGE_CODE,
                STAGE_FAILURE,
                STAGE_TRANSFER,
                STAGE_ADR,
                STAGE_FLAGSHIP,
                STAGE_GATE,
            ]
            experiment = next(stage for stage in spec["stages"] if stage["id"] == STAGE_EXPERIMENT)
            assert experiment["type"] == "experiment"
            assert experiment["validation_rubric"]["required_evidence_type"] == "chart"

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m31-learner", "display_name": "M31 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M31", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M31"
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
                assert set(block["type"] for block in blocks) <= WP137_BLOCK_TYPES
                assert "dummy_hash" not in json.dumps(exec_body)
                assert "sk-test-should-never-leak" not in json.dumps(exec_body)

                types = [block["type"] for block in blocks]
                assert "chart" in types
                chart = next(block for block in blocks if block["type"] == "chart")
                assert chart["payload"].get("chart_type") == "line"
                metric = next(block for block in blocks if block["type"] == "metric")
                assert metric["payload"]["targets"] == [20, 30, 40]
                assert metric["payload"]["seed"] == 3101
                assert metric["payload"]["shift_ok"] is True

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
            increments = gate_body.get("competency_increments") or []
            assert {item["competency_id"] for item in increments} == {"comp.lm.next_token_pairs", "comp.lm.train_eval_boundary", "comp.lm.contamination", "comp.lm.lifecycle_stages"}

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
