"""G6 M12 on the frozen generic runtime: predict → WP-137 ensemble metric → gate."""

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
M12_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M12"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
REAL_HOME_DB = Path.home() / ".learningos" / "learningos.db"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_experiment():
    import importlib.util

    path = M12_PACKAGE / "g5" / "reference" / "M12" / "experiment.py"
    spec = importlib.util.spec_from_file_location("m12_fixture_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXPERIMENT = _load_experiment()
ENSEMBLE_EXECUTE_SOURCE = _EXPERIMENT.ENSEMBLE_EXECUTE_SOURCE
default_parameters = _EXPERIMENT.default_parameters

STAGE_ORIENT = "stage_01_orientation"
STAGE_INTERROGATE = "stage_02_interrogate"
STAGE_CODE = "stage_03_code_reading"
STAGE_EXPERIMENT = "stage_04_experiment"
STAGE_FAILURE = "stage_05_controlled_failure"
STAGE_TRANSFER = "stage_06_transfer"
STAGE_ADR = "stage_07_adr"
STAGE_FLAGSHIP = "stage_08_flagship"
STAGE_GATE = "stage_09_gate"
STAGE_SEQUENCE = [
    STAGE_ORIENT,
    STAGE_INTERROGATE,
    STAGE_CODE,
    STAGE_EXPERIMENT,
    STAGE_FAILURE,
    STAGE_TRANSFER,
    STAGE_ADR,
    STAGE_FLAGSHIP,
    STAGE_GATE,
]
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}
PREDICT_BODY = {
    "hypothesis": "a depth-limited stump underfits; bagging votes can disagree without repairing labels",
    "expected_values": {"more_trees_repair_corrupted_labels": False},
}
EXECUTE_BODY = {"code": ENSEMBLE_EXECUTE_SOURCE, "parameters": default_parameters()}


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
def m12_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g6-m12-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g6-m12-{uuid.uuid4().hex}.sock")
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


def test_m12_runtime_ensemble_roundtrip(m12_env):
    from fastapi.testclient import TestClient

    from app.main import app

    before = _home_mtime()
    home: Path = m12_env["home"]
    sock: Path = m12_env["sock"]
    env: dict[str, str] = m12_env["env"]
    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M12_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            assert loaded.json().get("id") == "g5.reference.M12"
            spec = client.get("/api/v1/missions/M12", headers=headers).json()
            assert [stage["id"] for stage in spec["stages"]] == STAGE_SEQUENCE
            learner_id = client.post(
                "/api/v1/learners",
                json={"username": "m12-learner", "display_name": "M12 Learner"},
                headers=headers,
            ).json()["learner_id"]
            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M12", "learner_id": learner_id},
                headers=headers,
            )
            session_id = session.json()["session_id"]
            _enter(client, headers, session_id, STAGE_ORIENT)

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            _submit(
                resumed, headers, session_id, STAGE_ORIENT,
                "framed limited-tree baseline and ensemble comparison",
                [{"artifact_type": "markdown"}],
            )
            early = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers).json()
            assert early["status"] == "REPAIR_REQUIRED"
            nodes = set(early["repair_plan"]["failed_knowledge_nodes"])
            assert "kn.m12.limited_tree" in nodes
            assert "kn.m12.more_trees_limit" in nodes
            assert "kn.m12.no_ai_transfer" in nodes

            _enter(resumed, headers, session_id, STAGE_INTERROGATE)
            _submit(
                resumed, headers, session_id, STAGE_INTERROGATE,
                "bagging is parallel averaging; boosting is sequential correction",
                [{"artifact_type": "markdown"}],
            )
            _enter(resumed, headers, session_id, STAGE_CODE)
            _submit(
                resumed, headers, session_id, STAGE_CODE,
                "traced bootstrap votes and staged_predict corrections",
                [{"artifact_type": "trace"}],
            )

            lab_enter = _enter(resumed, headers, session_id, STAGE_EXPERIMENT)
            assert lab_enter.json()["stage_type"] == "experiment"
            blocked = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXECUTE_BODY, headers=headers,
            )
            assert blocked.status_code == 409
            assert blocked.json()["error"]["details"].get("reason") == "PREDICTION_REQUIRED"
            predicted = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/predict",
                json=PREDICT_BODY, headers=headers,
            )
            assert predicted.json()["is_sealed"] is True
            assert predicted.json()["prediction_hash"] != "dummy_hash"
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
            assert metric["payload"]["more_trees_repair_corrupted_labels"] is False
            assert metric["payload"]["n_bootstraps"] == 4
            assert "sk-test-should-never-leak" not in json.dumps(exec_body)

            _submit(
                resumed, headers, session_id, STAGE_EXPERIMENT,
                "observed bootstrap disagreement; split held fixed",
                [{"artifact_type": "metric"}],
            )
            _enter(resumed, headers, session_id, STAGE_FAILURE)
            _submit(
                resumed, headers, session_id, STAGE_FAILURE,
                "added trees average the flipped labels; they do not repair them",
                [{"artifact_type": "metric"}],
            )
            _enter(resumed, headers, session_id, STAGE_TRANSFER)
            _submit(
                resumed, headers, session_id, STAGE_TRANSFER,
                "latency-bounded bagging trial with a limited-tree baseline",
                [{"artifact_type": "artifact"}],
            )
            _enter(resumed, headers, session_id, STAGE_ADR)
            _submit(
                resumed, headers, session_id, STAGE_ADR,
                "ADR: bounded forest under latency and noisy-label constraints",
                [{"artifact_type": "markdown"}],
            )
            _enter(resumed, headers, session_id, STAGE_FLAGSHIP)
            _submit(
                resumed, headers, session_id, STAGE_FLAGSHIP,
                "V03 keeps the comparison workflow, not a frozen winner",
                [{"artifact_type": "markdown"}],
            )
            _enter(resumed, headers, session_id, STAGE_GATE)
            _submit(resumed, headers, session_id, STAGE_GATE, "ready for gate evaluation", [])
            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers).json()
            assert gate["status"] == "PASSED"
            increments = {item["competency_id"] for item in (gate.get("competency_increments") or [])}
            assert "comp.ensemble.limited_baseline" in increments
            assert "comp.ensemble.unassisted_transfer" in increments
            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.json().get("evidence")
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
