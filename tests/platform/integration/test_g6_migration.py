"""G6 M01–M42 on the frozen generic runtime: load, execute, gate, flagship, tutor."""

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
REQUIREMENTS = BACKEND_ROOT / "requirements.txt"
ROUTES = BACKEND_ROOT / "app" / "api" / "routes.py"
FRONTEND_SRC = REPO_ROOT / "platform" / "frontend" / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MISSION_IDS = tuple(f"M{i:02d}" for i in range(1, 43))
EXPERIMENT_TYPE = "experiment"
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}
NO_AI_POLICY = "NO_AI_REQUIRED"
FLAGSHIP_MISSIONS = {
    "V00": ["M01", "M02"],
    "V01": ["M03", "M04", "M05", "M06", "M07"],
    "V02": ["M08", "M09", "M10"],
    "V03": ["M11", "M12", "M13", "M14"],
    "V04": ["M15", "M16", "M17", "M18", "M19", "M20"],
    "V05": ["M21", "M22", "M23", "M24", "M25", "M26"],
    "V06": ["M27", "M28", "M29", "M30"],
    "V07": ["M31", "M32"],
    "V08": ["M33"],
    "V09": ["M34", "M35", "M36"],
    "V10": ["M37", "M38", "M39"],
    "V11": ["M40", "M41"],
    "V12": ["M42"],
}

PREDICT_BODY = {
    "hypothesis": "generic G6 integration execute succeeds on the frozen runtime",
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
def g6_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-int-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-int-{uuid.uuid4().hex}.sock")
    sock.unlink(missing_ok=True)
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-never-leak")
    monkeypatch.delenv("LEARNINGOS_TUTOR_PROVIDER", raising=False)
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(home)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("LEARNINGOS_TUTOR_PROVIDER", None)
    try:
        yield {"home": home, "sock": sock, "env": env, "monkeypatch": monkeypatch}
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


def _manifest(mission_id: str) -> dict:
    path = _package_dir(mission_id) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_package(client, headers, mission_id: str) -> dict:
    manifest = _manifest(mission_id)
    loaded = client.post(
        "/api/v1/curriculum/packages/load",
        json={"package_dir": str(_package_dir(mission_id))},
        headers=headers,
    )
    assert loaded.status_code == 200, loaded.text
    body = loaded.json()
    assert body.get("id") == manifest["id"]
    assert body.get("version") == manifest["version"]
    mission_ids = [item.get("id") for item in manifest.get("missions") or [] if isinstance(item, dict)]
    assert mission_id in mission_ids
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
    """Grep platform/ excluding fixtures for M01–M42 special-case UI/API."""
    platform = REPO_ROOT / "platform"
    skip_dirs = {"node_modules", "dist", "__pycache__", ".git", "fixtures", "tests"}
    eq_mission = re.compile(
        r"""(?:mission_id\s*==\s*|==\s*)["']M(?:0[1-9]|[1-3][0-9]|4[0-2])["']"""
    )
    route_mission = re.compile(r"/missions/M(?:0[1-9]|[1-3][0-9]|4[0-2])\b")
    custom_route = re.compile(
        r"/m(?:0[1-9]|[1-3][0-9]|4[0-2])[-_/]|array-vectorization|ai-ml-landscape|messy-csv",
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


def test_routes_union_keeps_flagship_and_real_tutor():
    text = ROUTES.read_text(encoding="utf-8")
    assert '@protected_router.get("/flagship")' in text
    assert "async def get_flagship" in text
    assert '@protected_router.post("/tutor/chat")' in text
    assert "from app.core.tutor import handle_tutor_chat, provider_configured" in text
    assert "Tutor is not available in G3" not in text
    assert "/missions/M42" not in text
    assert "openai" not in REQUIREMENTS.read_text(encoding="utf-8").lower()


def test_frontend_has_no_openai_or_vite_secrets():
    skip_dirs = {"node_modules", "dist", "__pycache__"}
    hits: list[str] = []
    for path in FRONTEND_SRC.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name.endswith((".test.ts", ".test.tsx", ".test.js", ".test.jsx")):
            continue
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "OPENAI_API_KEY" in text or "VITE_" in text:
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_reference_collects_without_pandas(tmp_path):
    (tmp_path / "pandas.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'pandas'\")\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(REPO_ROOT), str(BACKEND_ROOT), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--import-mode=importlib",
            "--collect-only",
            "-q",
            "tests/platform/reference",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    blob = completed.stdout + completed.stderr
    assert completed.returncode == 0, blob
    assert "ModuleNotFoundError" not in blob
    assert "No module named 'pandas'" not in blob


def test_g6_all_missions_load_execute_gate_flagship_and_tutor(g6_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = g6_env["home"]
    sock: Path = g6_env["sock"]
    env: dict[str, str] = g6_env["env"]
    monkeypatch = g6_env["monkeypatch"]
    worker: subprocess.Popen[bytes] | None = None
    sessions: dict[str, str] = {}
    learner_id = ""
    predicted_once = False
    no_ai_checked = 0

    for mission_id in MISSION_IDS:
        package_dir = _package_dir(mission_id)
        assert package_dir.is_dir(), mission_id
        assert (package_dir / "manifest.json").is_file(), mission_id
        assert (package_dir / "missions" / f"{mission_id}.json").is_file(), mission_id
        spec = json.loads((package_dir / "missions" / f"{mission_id}.json").read_text(encoding="utf-8"))
        assert spec.get("id") == mission_id
        manifest = _manifest(mission_id)
        assert mission_id in [
            item.get("id") for item in manifest.get("missions") or [] if isinstance(item, dict)
        ]

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded_ids: list[str] = []
            for mission_id in MISSION_IDS:
                body = _load_package(client, headers, mission_id)
                loaded_ids.append(body["id"])
            assert len(loaded_ids) == 42

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            listed = [item.get("id") for item in missions.json().get("missions", [])]
            assert set(MISSION_IDS) <= set(listed)
            for mission_id in MISSION_IDS:
                assert mission_id in listed
                assert _manifest(mission_id)["id"] not in listed

            flagship = client.get("/api/v1/flagship", headers=headers)
            assert flagship.status_code == 200, flagship.text
            flagship_body = flagship.json()
            assert flagship_body["schema"] == "learningos.flagship.v1"
            versions = flagship_body["versions"]
            assert [item["id"] for item in versions] == list(FLAGSHIP_MISSIONS)
            for version_id, expected in FLAGSHIP_MISSIONS.items():
                found = next(item for item in versions if item["id"] == version_id)
                assert found["missions"] == expected, version_id
            assert flagship_body.get("mission_count") == 42
            assert flagship_body.get("version_count") == 13

            missing_tutor = client.post(
                "/api/v1/tutor/chat",
                json={
                    "session_id": "sess-missing",
                    "stage_id": "stage-1",
                    "role": "SOCRATIC",
                    "prompt": "help",
                },
                headers=headers,
            )
            assert missing_tutor.status_code == 501, missing_tutor.text
            assert missing_tutor.json()["error"]["code"] == "TUTOR_NOT_AVAILABLE"
            assert "openai" not in sys.modules
            assert "sk-test-should-never-leak" not in missing_tutor.text

            config = client.get("/api/v1/system/config")
            assert config.status_code == 200, config.text
            assert "sk-test-should-never-leak" not in config.text
            assert "OPENAI_API_KEY" not in config.text
            assert "openai" not in config.text.lower()

            monkeypatch.setenv("LEARNINGOS_TUTOR_PROVIDER", "heuristic")

            learner = client.post(
                "/api/v1/learners",
                json={"username": "g6-learner", "display_name": "G6 Learner"},
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
            assert still_learner.json().get("username") == "g6-learner"
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
                    assert entered.status_code == 200, f"{mission_id} {stage_id}: {entered.text}"
                    assert entered.json()["current_stage_id"] == stage_id

                    if stage.get("assistance_policy") == NO_AI_POLICY:
                        locked = resumed.post(
                            "/api/v1/tutor/chat",
                            json={
                                "session_id": session_id,
                                "stage_id": stage_id,
                                "role": "SOCRATIC",
                                "prompt": "please complete this stage",
                            },
                            headers=headers,
                        )
                        assert locked.status_code == 403, f"{mission_id} {stage_id}: {locked.text}"
                        assert locked.json()["error"]["code"] == "ASSISTANCE_PROHIBITED"
                        assert locked.json()["error"]["details"].get("assistance_policy") == NO_AI_POLICY
                        no_ai_checked += 1

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
                        assert predicted.status_code == 200, f"{mission_id} {stage_id}: {predicted.text}"
                        assert predicted.json()["is_sealed"] is True
                        assert predicted.json()["prediction_hash"] != "dummy_hash"
                        assert len(predicted.json()["prediction_hash"]) == 64
                        executed = resumed.post(
                            f"/api/v1/sessions/{session_id}/stages/{stage_id}/execute",
                            json=EXECUTE_BODY,
                            headers=headers,
                        )
                        assert executed.status_code == 200, f"{mission_id} {stage_id}: {executed.text}"
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
                    assert submitted.status_code == 200, f"{mission_id} {stage_id}: {submitted.text}"
                    assert submitted.json()["payload_hash"] != "dummy_hash"

                gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
                assert gate.status_code == 200, f"{mission_id}: {gate.text}"
                gate_body = gate.json()
                assert gate_body["status"] == "PASSED", gate_body
                assert gate_body["reason"] == "GATE_CRITERIA_MET"
                increments = gate_body.get("competency_increments") or []
                assert increments, mission_id
                awarded.update(item["competency_id"] for item in increments)
                assert all(str(item.get("competency_id") or "").startswith("comp.") for item in increments)

                final = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
                assert final.status_code == 200, final.text
                assert final.json()["status"] == "COMPLETED"

            assert predicted_once
            assert no_ai_checked >= 42

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

            progress = resumed.get(f"/api/v1/flagship?learner_id={learner_id}", headers=headers)
            assert progress.status_code == 200, progress.text
            progress_body = progress.json()
            by_id = {item["id"]: item for item in progress_body.get("progress") or []}
            for version_id in FLAGSHIP_MISSIONS:
                assert by_id[version_id]["status"] == "COMPLETE", version_id

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
            assert still.json().get("username") == "g6-learner"
            assert "openai" not in sys.modules
            assert "dummy_hash" not in json.dumps(action_body)
            assert "sk-test-should-never-leak" not in json.dumps(action_body)
            assert "sk-test-should-never-leak" not in json.dumps(progress_body)

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
