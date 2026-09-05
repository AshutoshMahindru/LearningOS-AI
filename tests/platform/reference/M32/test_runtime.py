"""G5 M32 on the frozen generic runtime: predict → WP-137 execute → gate."""

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
M32_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M32"
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

PREDICT_BODY = {"hypothesis": "greedy index is 1; T>0 does not change argmax; (log3, 0) at T=1 is 0.75/0.25", "expected_values": {"greedy_index": 1, "p0": 0.75, "seed": 3201}}

EXECUTE_CODE = "import math\n\nlogits = [1.0, 3.0, 2.0, 0.0]\ngreedy_index = max(range(len(logits)), key=lambda i: (logits[i], -i))\npair = [math.log(3.0), 0.0]\ntemp = float(parameters.get(\"temperature\") or 1.0)\nscaled = [v / temp for v in pair]\npeak = max(scaled)\nexps = [math.exp(v - peak) for v in scaled]\ntotal = sum(exps)\nprobs = [e / total for e in exps]\nseed = int(parameters.get(\"seed\") or 3201)\npayload = {\n    \"greedy_index\": greedy_index,\n    \"p0\": probs[0],\n    \"p1\": probs[1],\n    \"temperature\": temp,\n    \"greedy_unchanged\": greedy_index == 1,\n    \"seed\": seed,\n    \"checkpoint_id\": \"v07-teaching-lm-1\",\n    \"filter_order\": [\"temperature\", \"top_k\", \"top_p\", \"softmax\"],\n}\nprint(payload)\n[\n    {\"type\": \"table\", \"title\": \"temperature\", \"payload\": {\"columns\": [\"T\", \"p0\", \"p1\"], \"rows\": [[temp, probs[0], probs[1]]]}},\n    {\"type\": \"metric\", \"title\": \"greedy\", \"payload\": payload},\n    {\"type\": \"trace\", \"title\": \"config\", \"payload\": {\"checkpoint_id\": \"v07-teaching-lm-1\", \"seed\": seed, \"temperature\": temp, \"stop\": \"stop\", \"max_tokens\": 8}},\n]"

EXECUTE_BODY = {
    "code": EXECUTE_CODE,
    "parameters": {"temperature": 1.0, "seed": 3201},
}

SUBMIT_PLAN = [
    (STAGE_ORIENT, "framed frozen-checkpoint decoding and the adaptation hierarchy", [{"artifact_type": "markdown"}]),
    (STAGE_EXPERIMENT, "observed WP-137 temperature table after a sealed prediction", [{"artifact_type": "table"}]),
    (STAGE_CODE, "traced prompt, filters, greedy, stop, fail-closed adapter, fingerprint", [{"artifact_type": "artifact"}]),
    (STAGE_FAILURE, "naive compare blamed the model; restore config control, not weights", [{"artifact_type": "table"}]),
    (STAGE_TRANSFER, "greedy 1, softmax 0.75/0.25, freshness is retrieval not fine-tune", [{"artifact_type": "metric"}]),
    (STAGE_ADR, "locked prompt\u2192retrieval\u2192tools\u2192parameters and required InferenceConfig logs", [{"artifact_type": "markdown"}]),
    (STAGE_FLAGSHIP, "P5/V07 ends with a provider contract, not a search service", [{"artifact_type": "markdown"}]),
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
def m32_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-m32-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m32-{uuid.uuid4().hex}.sock")
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
        "execution_id": "exec-m32-ref",
        "status": "SUCCESS",
        "exit_code": 0,
        "duration_ms": 12,
        "blocks": [{"type": "table", "title": "temperature", "payload": {"columns": ["T", "p0", "p1"], "rows": [[1.0, 0.75, 0.25]]}}, {"type": "metric", "title": "greedy", "payload": {"greedy_index": 1, "seed": 3201}}],
    }
    validated = validate_structured_result(sample)
    types = [block["type"] for block in validated["blocks"]]
    assert set(types) <= WP137_BLOCK_TYPES
    assert "table" in types


def test_m32_runtime_roundtrip_emits_wp137(m32_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = m32_env["home"]
    sock: Path = m32_env["sock"]
    env: dict[str, str] = m32_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M32_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M32"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M32" in mission_ids
            assert "g5.reference.M32" not in mission_ids

            mission = client.get("/api/v1/missions/M32", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M32"
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
            assert experiment["validation_rubric"]["required_evidence_type"] == "table"

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m32-learner", "display_name": "M32 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M32", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M32"
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
                assert "table" in types
                table = next(block for block in blocks if block["type"] == "table")
                assert table["payload"]["columns"][0] == "T"
                metric = next(block for block in blocks if block["type"] == "metric")
                assert metric["payload"]["greedy_index"] == 1
                assert abs(metric["payload"]["p0"] - 0.75) < 1e-9
                assert metric["payload"]["seed"] == 3201

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
            assert {item["competency_id"] for item in increments} == {"comp.inf.greedy_temperature", "comp.inf.filters_and_seed", "comp.inf.config_evidence", "comp.inf.adaptation_hierarchy"}

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
