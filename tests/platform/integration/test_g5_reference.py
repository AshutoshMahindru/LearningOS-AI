"""G5 M01–M05 on the frozen generic runtime: load, resume, evidence, gate, reload."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
DAEMON_PATH = REPO_ROOT / "platform" / "worker" / "daemon.py"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
FIXTURES = REPO_ROOT / "platform" / "fixtures"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MISSION_IDS = ("M01", "M02", "M03", "M04", "M05")
PACKAGE_IDS = {mid: f"g5.reference.{mid}" for mid in MISSION_IDS}
EXPERIMENT_TYPE = "experiment"
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}

PREDICT_BODY = {
    "hypothesis": "generic G5 integration execute succeeds on the frozen runtime",
    "expected_values": {"ok": True},
}
EXECUTE_BODY = {
    "code": (
        "payload = {'ok': True}\n"
        "print(payload)\n"
        "{'type': 'metric', 'title': 'ok', 'payload': payload}"
    ),
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


@pytest.fixture
def g5_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g5-int-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g5-int-{uuid.uuid4().hex}.sock")
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


def _package_dir(mission_id: str) -> Path:
    return FIXTURES / mission_id


def _load_package(client, headers, mission_id: str) -> dict:
    loaded = client.post(
        "/api/v1/curriculum/packages/load",
        json={"package_dir": str(_package_dir(mission_id))},
        headers=headers,
    )
    assert loaded.status_code == 200, loaded.text
    body = loaded.json()
    assert body.get("id") == PACKAGE_IDS[mission_id]
    assert body.get("version") == "5.0.0"
    return body


def _row_dicts(conn: sqlite3.Connection, sql: str) -> list[dict]:
    rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def _learner_session_snapshot() -> dict:
    from app.db.database import get_connection

    conn = get_connection()
    try:
        return {
            "learners": _row_dicts(
                conn, "SELECT id, username, display_name FROM learners ORDER BY id"
            ),
            "sessions": _row_dicts(
                conn,
                "SELECT id, learner_id, mission_id, status, current_stage_id "
                "FROM mission_sessions ORDER BY id",
            ),
        }
    finally:
        conn.close()


def _artifacts_for_stage(spec: dict, stage: dict) -> list[dict]:
    stage_id = stage["id"]
    contract = spec.get("gate_contract") if isinstance(spec.get("gate_contract"), dict) else {}
    required = contract.get("required_evidence") if isinstance(contract.get("required_evidence"), list) else []
    for item in required:
        if isinstance(item, dict) and item.get("stage_id") == stage_id and item.get("artifact_type"):
            return [{"artifact_type": item["artifact_type"]}]
    rubric = stage.get("validation_rubric") if isinstance(stage.get("validation_rubric"), dict) else {}
    required_type = rubric.get("required_evidence_type") or rubric.get("artifact_type")
    if isinstance(required_type, str) and required_type:
        return [{"artifact_type": required_type}]
    return [{"artifact_type": "markdown"}]


def _assert_provenance(claims: list[dict]) -> None:
    assert claims
    blob = json.dumps(claims)
    assert "dummy_hash" not in blob
    hashes: list[str] = []
    for claim in claims:
        provenance = claim.get("provenance") or claim
        for key in ("artifact_hash", "runner_hash", "curriculum_sha"):
            digest = provenance.get(key) or claim.get(key)
            if digest:
                hashes.append(str(digest))
                assert digest != "dummy_hash"
                assert len(str(digest)) == 64
    assert hashes


def test_platform_has_no_mission_specific_ui_or_api():
    """Grep platform/ excluding fixtures for M01–M05 special-case UI/API."""
    platform = REPO_ROOT / "platform"
    skip_dirs = {"node_modules", "dist", "__pycache__", ".git", "fixtures", "tests"}
    eq_mission = re.compile(r"""(?:mission_id\s*==\s*|==\s*)["']M0[1-5]["']""")
    route_mission = re.compile(r"/missions/M0[1-5]\b")
    custom_route = re.compile(
        r"/m0[1-5][-_/]|array-vectorization|ai-ml-landscape|messy-csv",
        re.IGNORECASE,
    )
    hits: list[str] = []
    for path in platform.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name.endswith((".test.ts", ".test.tsx", ".test.js", ".test.jsx")):
            continue
        if path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".css", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        if eq_mission.search(text) or route_mission.search(text) or custom_route.search(text):
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_g5_reference_packages_resume_gate_and_curriculum_reload(g5_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = g5_env["home"]
    sock: Path = g5_env["sock"]
    env: dict[str, str] = g5_env["env"]
    worker: subprocess.Popen[bytes] | None = None
    sessions: dict[str, str] = {}
    learner_id = ""
    predicted_once = False

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            for mission_id in MISSION_IDS:
                _load_package(client, headers, mission_id)

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            listed = [item.get("id") for item in missions.json().get("missions", [])]
            assert list(MISSION_IDS) == listed or set(MISSION_IDS) <= set(listed)
            for mission_id in MISSION_IDS:
                assert mission_id in listed
                assert PACKAGE_IDS[mission_id] not in listed

            learner = client.post(
                "/api/v1/learners",
                json={"username": "g5-learner", "display_name": "G5 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            for mission_id in MISSION_IDS:
                created = client.post(
                    "/api/v1/sessions",
                    json={"mission_id": mission_id, "learner_id": learner_id},
                    headers=headers,
                )
                assert created.status_code == 200, created.text
                body = created.json()
                assert body["mission_id"] == mission_id
                assert body["learner_id"] == learner_id
                assert body["status"] == "ACTIVE"
                assert body["current_stage_id"]
                sessions[mission_id] = body["session_id"]

            before_reload = _learner_session_snapshot()
            assert len(before_reload["learners"]) == 1
            assert {row["id"] for row in before_reload["learners"]} == {learner_id}
            assert {row["mission_id"] for row in before_reload["sessions"]} == set(MISSION_IDS)

            for mission_id in MISSION_IDS:
                _load_package(client, headers, mission_id)
            after_reload = _learner_session_snapshot()
            assert after_reload == before_reload

            still_learner = client.get(f"/api/v1/learners/{learner_id}", headers=headers)
            assert still_learner.status_code == 200, still_learner.text
            assert still_learner.json().get("username") == "g5-learner"
            for mission_id, session_id in sessions.items():
                got = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
                assert got.status_code == 200, got.text
                assert got.json()["mission_id"] == mission_id
                assert got.json()["learner_id"] == learner_id
                assert got.json()["status"] == "ACTIVE"
                stage_id = got.json()["current_stage_id"]
                entered = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
                    headers=headers,
                )
                assert entered.status_code == 200, entered.text
                assert entered.json().get("resumed") is not True

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            for mission_id, session_id in sessions.items():
                got = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
                assert got.status_code == 200, got.text
                assert got.json()["status"] == "ACTIVE"
                assert got.json()["mission_id"] == mission_id
                stage_id = got.json()["current_stage_id"]
                again = resumed.post(
                    f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
                    headers=headers,
                )
                assert again.status_code == 200, again.text
                assert again.json()["resumed"] is True

            awarded: set[str] = set()
            for mission_id, session_id in sessions.items():
                spec_resp = resumed.get(f"/api/v1/missions/{mission_id}", headers=headers)
                assert spec_resp.status_code == 200, spec_resp.text
                spec = spec_resp.json()
                assert spec.get("id") == mission_id
                stages = spec["stages"]
                assert stages

                for stage in stages:
                    stage_id = stage["id"]
                    entered = resumed.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
                        headers=headers,
                    )
                    assert entered.status_code == 200, entered.text
                    assert entered.json()["current_stage_id"] == stage_id

                    if stage.get("type") == EXPERIMENT_TYPE:
                        if not predicted_once:
                            blocked = resumed.post(
                                f"/api/v1/sessions/{session_id}/stages/{stage_id}/execute",
                                json=EXECUTE_BODY,
                                headers=headers,
                            )
                            assert blocked.status_code == 409, blocked.text
                            assert blocked.json()["error"]["code"] == "CONFLICT"
                            assert blocked.json()["error"]["details"].get("reason") == "PREDICTION_REQUIRED"
                            predicted_once = True
                        predicted = resumed.post(
                            f"/api/v1/sessions/{session_id}/stages/{stage_id}/predict",
                            json=PREDICT_BODY,
                            headers=headers,
                        )
                        assert predicted.status_code == 200, predicted.text
                        assert predicted.json()["is_sealed"] is True
                        assert predicted.json()["prediction_hash"] != "dummy_hash"
                        assert len(predicted.json()["prediction_hash"]) == 64
                        executed = resumed.post(
                            f"/api/v1/sessions/{session_id}/stages/{stage_id}/execute",
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

                    submitted = resumed.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/submit",
                        json={
                            "explanation": f"completed {stage_id} on generic runtime",
                            "artifacts": _artifacts_for_stage(spec, stage),
                        },
                        headers=headers,
                    )
                    assert submitted.status_code == 200, submitted.text
                    assert submitted.json()["payload_hash"] != "dummy_hash"

                gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
                assert gate.status_code == 200, gate.text
                gate_body = gate.json()
                assert gate_body["status"] == "PASSED", gate_body
                assert gate_body["reason"] == "GATE_CRITERIA_MET"
                increments = gate_body.get("competency_increments") or []
                assert increments
                awarded.update(item["competency_id"] for item in increments)
                assert all(str(item.get("competency_id") or "").startswith("comp.") for item in increments)

                final = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
                assert final.status_code == 200, final.text
                assert final.json()["status"] == "COMPLETED"

            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            claims = evidence.json().get("evidence") or []
            _assert_provenance(claims)
            claim_missions = {claim.get("mission_id") for claim in claims}
            assert set(MISSION_IDS) <= claim_missions

            today = resumed.get(f"/api/v1/learners/{learner_id}/next-action", headers=headers)
            assert today.status_code == 200, today.text
            action_body = today.json()
            assert action_body["action"] == "IDLE"
            assert action_body["reason"] == "ALL_MISSIONS_COMPLETE"
            competencies = {item["competency_id"] for item in (action_body.get("competencies") or [])}
            assert awarded <= competencies

            post_gate = _learner_session_snapshot()
            for mission_id in MISSION_IDS:
                _load_package(resumed, headers, mission_id)
            after_final_reload = _learner_session_snapshot()
            assert after_final_reload["learners"] == post_gate["learners"]
            assert after_final_reload["sessions"] == post_gate["sessions"]
            assert {row["status"] for row in after_final_reload["sessions"]} == {"COMPLETED"}
            assert {row["id"] for row in after_final_reload["sessions"]} == set(sessions.values())

            still = resumed.get(f"/api/v1/learners/{learner_id}", headers=headers)
            assert still.status_code == 200, still.text
            assert still.json().get("username") == "g5-learner"
            assert "openai" not in sys.modules
            assert "dummy_hash" not in json.dumps(action_body)
            assert "sk-test-should-never-leak" not in json.dumps(action_body)

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
