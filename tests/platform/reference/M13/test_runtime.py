"""G6 M13 on the frozen generic runtime: predict → WP-137 neighbor metric → gate."""

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
M13_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M13"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
REAL_HOME_DB = Path.home() / ".learningos" / "learningos.db"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_experiment():
    import importlib.util

    path = M13_PACKAGE / "g5" / "reference" / "M13" / "experiment.py"
    spec = importlib.util.spec_from_file_location("m13_fixture_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXPERIMENT = _load_experiment()
KNN_EXECUTE_SOURCE = _EXPERIMENT.KNN_EXECUTE_SOURCE
default_parameters = _EXPERIMENT.default_parameters

STAGE_ORIENT = "stage_01_orientation"
STAGE_TRACE = "stage_02_trace_map"
STAGE_CODE = "stage_03_code_reading"
STAGE_EXPERIMENT = "stage_04_experiment"
STAGE_FAILURE = "stage_05_controlled_failure"
STAGE_TRANSFER = "stage_06_transfer"
STAGE_ADR = "stage_07_adr"
STAGE_FLAGSHIP = "stage_08_flagship"
STAGE_GATE = "stage_09_gate"
STAGE_SEQUENCE = [
    STAGE_ORIENT, STAGE_TRACE, STAGE_CODE, STAGE_EXPERIMENT, STAGE_FAILURE,
    STAGE_TRANSFER, STAGE_ADR, STAGE_FLAGSHIP, STAGE_GATE,
]
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}
PREDICT_BODY = {
    "hypothesis": "query (6,6) with k=3 is a mixed neighborhood; identity of neighbors matters",
    "expected_values": {"k": 3},
}
EXECUTE_BODY = {"code": KNN_EXECUTE_SOURCE, "parameters": default_parameters()}


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _start_worker(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(DAEMON_PATH)], env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
    with pytest.raises(ValueError):
        path.resolve().relative_to(REPO_ROOT.resolve())


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
def m13_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-m13-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m13-{uuid.uuid4().hex}.sock")
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
    return {"Authorization": f"Bearer {boot.json()['token']}"}


def _enter(client, headers, session_id: str, stage_id: str):
    response = client.post(f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter", headers=headers)
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


def test_m13_runtime_knn_roundtrip(m13_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m13_env["home"]
    sock: Path = m13_env["sock"]
    env: dict[str, str] = m13_env["env"]
    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M13_PACKAGE)}, headers=headers,
            )
            assert loaded.json().get("id") == "g5.reference.M13"
            spec = client.get("/api/v1/missions/M13", headers=headers).json()
            assert [stage["id"] for stage in spec["stages"]] == STAGE_SEQUENCE
            learner_id = client.post(
                "/api/v1/learners",
                json={"username": "m13-learner", "display_name": "M13 Learner"},
                headers=headers,
            ).json()["learner_id"]
            session_id = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M13", "learner_id": learner_id},
                headers=headers,
            ).json()["session_id"]
            _enter(client, headers, session_id, STAGE_ORIENT)

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            _submit(
                resumed, headers, session_id, STAGE_ORIENT,
                "framed instance-based votes and the high-scale trap",
                [{"artifact_type": "markdown"}],
            )
            early = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers).json()
            assert early["status"] == "REPAIR_REQUIRED"
            nodes = set(early["repair_plan"]["failed_knowledge_nodes"])
            assert "kn.m13.query_neighbors" in nodes
            assert "kn.m13.scale_failure" in nodes
            assert "kn.m13.no_ai_transfer" in nodes

            _enter(resumed, headers, session_id, STAGE_TRACE)
            _submit(
                resumed, headers, session_id, STAGE_TRACE,
                "mapped query to distances to neighbor ids to votes",
                [{"artifact_type": "diagram"}],
            )
            _enter(resumed, headers, session_id, STAGE_CODE)
            _submit(
                resumed, headers, session_id, STAGE_CODE,
                "scaler fits on train only; raw query against scaled model fails",
                [{"artifact_type": "trace"}],
            )
            lab_enter = _enter(resumed, headers, session_id, STAGE_EXPERIMENT)
            assert lab_enter.json()["stage_type"] == "experiment"
            blocked = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXECUTE_BODY, headers=headers,
            )
            assert blocked.status_code == 409
            predicted = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/predict",
                json=PREDICT_BODY, headers=headers,
            )
            assert predicted.json()["is_sealed"] is True
            executed = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXECUTE_BODY, headers=headers,
            )
            assert executed.status_code == 200, executed.text
            exec_body = executed.json()
            assert exec_body["status"] == "SUCCESS"
            structured = exec_body.get("structured_result") or {}
            assert structured["status"] in WP137_STATUSES
            blocks = structured.get("blocks") or exec_body.get("blocks") or []
            metric = next(block for block in blocks if block["type"] == "metric")
            assert metric["payload"]["k"] == 3
            assert metric["payload"]["neighbor_ids"]
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)
            _submit(
                resumed, headers, session_id, STAGE_EXPERIMENT,
                "neighbor identities and distances recorded, not accuracy alone",
                [{"artifact_type": "metric"}],
            )
            _enter(resumed, headers, session_id, STAGE_FAILURE)
            _submit(
                resumed, headers, session_id, STAGE_FAILURE,
                "interface_event_count dominates squared distance; scale on train rows",
                [{"artifact_type": "metric"}],
            )
            _enter(resumed, headers, session_id, STAGE_TRANSFER)
            _submit(
                resumed, headers, session_id, STAGE_TRANSFER,
                "hand Euclidean table for (6,6) and scale-repair without AI",
                [{"artifact_type": "artifact"}],
            )
            _enter(resumed, headers, session_id, STAGE_ADR)
            _submit(
                resumed, headers, session_id, STAGE_ADR,
                "ADR: training-only scaler, drop the weak high-scale feature",
                [{"artifact_type": "markdown"}],
            )
            _enter(resumed, headers, session_id, STAGE_FLAGSHIP)
            _submit(
                resumed, headers, session_id, STAGE_FLAGSHIP,
                "V03 keeps neighbor diagnostics beside tree and ensemble explanations",
                [{"artifact_type": "markdown"}],
            )
            _enter(resumed, headers, session_id, STAGE_GATE)
            _submit(resumed, headers, session_id, STAGE_GATE, "ready for gate evaluation", [])
            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers).json()
            assert gate["status"] == "PASSED"
            increments = {item["competency_id"] for item in (gate.get("competency_increments") or [])}
            assert "comp.knn.neighbor_trace" in increments
            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert "dummy_hash" not in json.dumps(evidence.json())
            today = resumed.get(f"/api/v1/learners/{learner_id}/next-action", headers=headers).json()
            assert today["action"] == "IDLE"
            final = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
            assert final["status"] == "COMPLETED"

        completed = subprocess.run(
            [sys.executable, str(STATE_GUARD), "--repo", str(REPO_ROOT)],
            check=False, capture_output=True, text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        _assert_home_untouched(before)
    finally:
        _stop_worker(worker)
