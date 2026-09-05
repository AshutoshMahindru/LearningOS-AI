"""G5 M03 on generic runtime: predict → execute WP-137 → debug tests → gate."""

from __future__ import annotations

import json
import os
import re
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
M03_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M03"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

STAGE_ORIENT = "stage_01_orientation"
STAGE_EXPERIMENT = "stage_02_experiment"
STAGE_CODE_READING = "stage_03_code_reading"
STAGE_REBUILD = "stage_04_rebuild_debug"
STAGE_FAILURE = "stage_05_controlled_failure"
STAGE_TRANSFER = "stage_06_transfer"
STAGE_GATE = "stage_07_gate"

STAGE_SEQUENCE = [
    STAGE_ORIENT,
    STAGE_EXPERIMENT,
    STAGE_CODE_READING,
    STAGE_REBUILD,
    STAGE_FAILURE,
    STAGE_TRANSFER,
    STAGE_GATE,
]

PREDICT_BODY = {
    "hypothesis": "member discount 0.1 on subtotal 40 plus shipping 5 yields 41.0",
    "expected_values": {"after_total": 41.0, "discount": 0.1},
}

EXPERIMENT_CODE = """\
orders = parameters['orders']
member = parameters['member']
MEMBER_DISCOUNT = 0.1
FREE_SHIPPING_THRESHOLD = 50
SHIPPING = 5

def subtotal(basket):
    return sum(item['qty'] * item['price'] for item in basket)

def final_total(basket, is_member):
    amount = subtotal(basket)
    if is_member:
        amount = round(amount * (1 - MEMBER_DISCOUNT), 2)
    shipping = 0 if amount >= FREE_SHIPPING_THRESHOLD else SHIPPING
    return round(amount + shipping, 2)

before = parameters['before_total']
after = final_total(orders, member)
print(after)
{
    'type': 'state_diff',
    'title': 'order-total-modification',
    'payload': {
        'before': {'total': before},
        'after': {'total': after, 'member': member, 'discount': MEMBER_DISCOUNT},
    },
}
"""

EXPERIMENT_EXECUTE = {
    "code": EXPERIMENT_CODE,
    "parameters": {
        "orders": [{"qty": 2, "price": 10}, {"qty": 1, "price": 20}],
        "member": True,
        "before_total": 45.0,
    },
}

REBUILD_CODE = """\
def last_id(items):
    if not items:
        return None
    return items[-1]['id']

def lookup(record, key):
    if key not in record:
        return None
    return record[key]

passed = 0
assert last_id([{'id': 1}, {'id': 2}]) == 2
passed += 1
assert last_id([]) is None
passed += 1
assert lookup({'sku': 'A'}, 'qty') is None
passed += 1
print(passed)
{'type': 'metric', 'title': 'debug-tests', 'payload': {'passed': passed, 'failed': 0}}
"""

FAILURE_CODE = """\
def buggy_final_total(subtotal, member):
    shipping = 0 if subtotal >= 50 else 5
    amount = subtotal + shipping
    if member:
        amount = round(amount * 0.9, 2)
    return amount

def repaired_final_total(subtotal, member):
    amount = round(subtotal * 0.9, 2) if member else subtotal
    shipping = 0 if amount >= 50 else 5
    return round(amount + shipping, 2)

observed = buggy_final_total(40, True)
expected = 41.0
repaired = repaired_final_total(40, True)
assert observed == 40.5
assert repaired == expected
print(repaired)
{
    'type': 'trace',
    'title': 'controlled-failure',
    'payload': {
        'symptom': observed,
        'expected': expected,
        'root_cause': 'discount_applied_after_shipping',
        'repaired': repaired,
    },
}
"""

TRANSFER_CODE = """\
stock = {'dough': 12, 'cheese': 8, 'tomato': 10}

def consume(inventory, ingredient, quantity):
    if ingredient not in inventory:
        return 0
    remaining = inventory[ingredient] - quantity
    if remaining < 0:
        remaining = 0
    inventory[ingredient] = remaining
    return remaining

def below_reorder(inventory, threshold):
    return [name for name, qty in inventory.items() if qty < threshold]

assert consume(stock, 'cheese', 3) == 5
assert consume(stock, 'cheese', 10) == 0
assert consume(stock, 'olive', 1) == 0
assert 'dough' not in below_reorder(stock, 5)
print(stock)
{
    'type': 'artifact',
    'title': 'inventory-transfer',
    'payload': {'stock': stock, 'reorder': below_reorder(stock, 5)},
}
"""

WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}
COMPETENCIES = {
    "comp.py.experiment",
    "comp.py.code_reading",
    "comp.py.test_debugging",
    "comp.py.diff_verification",
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
def m03_env(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="los-g5-m03-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g5-m03-{uuid.uuid4().hex}.sock")
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


def _enter(client, headers, session_id: str, stage_id: str, expected_type: str | None = None):
    entered = client.post(
        f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
        headers=headers,
    )
    assert entered.status_code == 200, entered.text
    body = entered.json()
    assert body["current_stage_id"] == stage_id
    if expected_type is not None:
        assert body["stage_type"] == expected_type
    return body


def _submit(client, headers, session_id: str, stage_id: str, explanation: str, artifact_type: str):
    submitted = client.post(
        f"/api/v1/sessions/{session_id}/stages/{stage_id}/submit",
        json={
            "explanation": explanation,
            "artifacts": [{"artifact_type": artifact_type}],
        },
        headers=headers,
    )
    assert submitted.status_code == 200, submitted.text
    return submitted.json()


def _execute(client, headers, session_id: str, stage_id: str, body: dict):
    executed = client.post(
        f"/api/v1/sessions/{session_id}/stages/{stage_id}/execute",
        json=body,
        headers=headers,
    )
    assert executed.status_code == 200, executed.text
    return executed.json()


def test_platform_has_no_m03_special_case_routes_or_conditionals():
    platform = REPO_ROOT / "platform"
    skip_dirs = {"node_modules", "dist", "__pycache__", ".git"}
    eq_m03 = re.compile(r"""mission_id\s*==\s*["']M03["']""")
    route_m03 = re.compile(r"/missions/M03")
    hits: list[str] = []
    for path in platform.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if "fixtures" in path.parts and "M03" in path.parts:
            continue
        if path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".css", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        if eq_m03.search(text) or route_m03.search(text):
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_m03_runtime_roundtrip(m03_env):
    from fastapi.testclient import TestClient

    from app.main import app

    home: Path = m03_env["home"]
    sock: Path = m03_env["sock"]
    env: dict[str, str] = m03_env["env"]
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(home)

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(M03_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g5.reference.M03"
            assert loaded_body.get("version") == "5.0.0"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "M03" in mission_ids
            assert "g5.reference.M03" not in mission_ids

            mission = client.get("/api/v1/missions/M03", headers=headers)
            assert mission.status_code == 200, mission.text
            spec = mission.json()
            assert spec.get("id") == "M03"
            stage_ids = [stage["id"] for stage in spec["stages"]]
            assert stage_ids == STAGE_SEQUENCE
            types = [stage["type"] for stage in spec["stages"]]
            assert "code_reading" in types
            assert "rebuild_debug" in types
            assert "experiment" in types

            learner = client.post(
                "/api/v1/learners",
                json={"username": "m03-learner", "display_name": "M03 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M03", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            session_body = session.json()
            assert session_body["mission_id"] == "M03"
            assert session_body["status"] == "ACTIVE"
            assert session_body["current_stage_id"] == STAGE_ORIENT
            session_id = session_body["session_id"]

            entered = client.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/enter",
                headers=headers,
            )
            assert entered.status_code == 200, entered.text
            assert entered.json()["stage_type"] == "orientation"

        with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
            headers = _bootstrap(resumed)
            got = resumed.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert got.status_code == 200, got.text
            assert got.json()["current_stage_id"] == STAGE_ORIENT
            assert got.json().get("current_stage", {}).get("id") == STAGE_ORIENT

            again = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_ORIENT}/enter",
                headers=headers,
            )
            assert again.status_code == 200, again.text
            assert again.json()["resumed"] is True

            submitted = _submit(
                resumed,
                headers,
                session_id,
                STAGE_ORIENT,
                "framed the working program before formal language study",
                "markdown",
            )
            assert submitted["current_stage_id"] == STAGE_EXPERIMENT

            lab_enter = _enter(resumed, headers, session_id, STAGE_EXPERIMENT, "experiment")
            assert lab_enter["stage_type"] == "experiment"

            blocked = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_EXPERIMENT}/execute",
                json=EXPERIMENT_EXECUTE,
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

            executed = _execute(resumed, headers, session_id, STAGE_EXPERIMENT, EXPERIMENT_EXECUTE)
            assert executed["status"] == "SUCCESS", executed
            assert executed["status"] != "UNSUPPORTED"
            assert executed["code_hash"] != "dummy_hash"
            structured = executed.get("structured_result") or {}
            for key in ("execution_id", "status", "exit_code", "duration_ms", "blocks"):
                assert key in structured, key
            assert structured["status"] in WP137_STATUSES
            assert structured["status"] == "SUCCESS"
            blocks = structured.get("blocks") or executed.get("blocks") or []
            assert isinstance(blocks, list) and blocks
            assert blocks[0]["type"] == "state_diff"
            assert blocks[0]["payload"]["after"]["total"] == 41.0
            diagnostics = structured.get("diagnostics") or executed.get("diagnostics") or {}
            assert "41" in str(diagnostics.get("stdout") or "")
            assert "dummy_hash" not in json.dumps(executed)
            assert "sk-test-should-never-leak" not in json.dumps(executed)

            lab_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_EXPERIMENT,
                "observed after total 41.0 vs before 45.0; member discount applied before shipping",
                "state_diff",
            )
            assert lab_submit["current_stage_id"] == STAGE_CODE_READING
            assert lab_submit["payload_hash"] != "dummy_hash"

            _enter(resumed, headers, session_id, STAGE_CODE_READING, "code_reading")
            reading = _submit(
                resumed,
                headers,
                session_id,
                STAGE_CODE_READING,
                "traced consume(): stock is mutable, missing keys raise KeyError, remaining can go negative",
                "artifact",
            )
            assert reading["current_stage_id"] == STAGE_REBUILD

            _enter(resumed, headers, session_id, STAGE_REBUILD, "rebuild_debug")
            rebuild_exec = _execute(
                resumed,
                headers,
                session_id,
                STAGE_REBUILD,
                {"code": REBUILD_CODE, "parameters": {}},
            )
            assert rebuild_exec["status"] == "SUCCESS", rebuild_exec
            rebuild_blocks = (rebuild_exec.get("structured_result") or {}).get("blocks") or []
            assert rebuild_blocks
            assert rebuild_blocks[0]["type"] == "metric"
            assert rebuild_blocks[0]["payload"]["passed"] == 3
            rebuild_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_REBUILD,
                "off-by-one used items[-1]; KeyError repaired with membership test; three assertions pass",
                "metric",
            )
            assert rebuild_submit["current_stage_id"] == STAGE_FAILURE

            _enter(resumed, headers, session_id, STAGE_FAILURE, "controlled_failure")
            failure_exec = _execute(
                resumed,
                headers,
                session_id,
                STAGE_FAILURE,
                {"code": FAILURE_CODE, "parameters": {}},
            )
            assert failure_exec["status"] == "SUCCESS", failure_exec
            failure_blocks = (failure_exec.get("structured_result") or {}).get("blocks") or []
            assert failure_blocks
            assert failure_blocks[0]["type"] == "trace"
            assert failure_blocks[0]["payload"]["root_cause"] == "discount_applied_after_shipping"
            failure_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_FAILURE,
                "seeded cause was discount after shipping; repair applies discount first; expected 41.0",
                "trace",
            )
            assert failure_submit["current_stage_id"] == STAGE_TRANSFER

            _enter(resumed, headers, session_id, STAGE_TRANSFER, "transfer_assessment")
            transfer_exec = _execute(
                resumed,
                headers,
                session_id,
                STAGE_TRANSFER,
                {"code": TRANSFER_CODE, "parameters": {}},
            )
            assert transfer_exec["status"] == "SUCCESS", transfer_exec
            transfer_blocks = (transfer_exec.get("structured_result") or {}).get("blocks") or []
            assert transfer_blocks
            assert transfer_blocks[0]["type"] == "artifact"
            transfer_submit = _submit(
                resumed,
                headers,
                session_id,
                STAGE_TRANSFER,
                "fresh inventory consume never goes negative; unknown ingredient returns 0; reorder helper added",
                "artifact",
            )
            assert transfer_submit["current_stage_id"] == STAGE_GATE

            _enter(resumed, headers, session_id, STAGE_GATE, "competency_gate")
            gate_submit = resumed.post(
                f"/api/v1/sessions/{session_id}/stages/{STAGE_GATE}/submit",
                json={"explanation": "ready for gate evaluation"},
                headers=headers,
            )
            assert gate_submit.status_code == 200, gate_submit.text

            gate = resumed.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            gate_body = gate.json()
            assert gate_body["status"] == "PASSED"
            assert gate_body["reason"] == "GATE_CRITERIA_MET"
            increments = gate_body.get("competency_increments") or []
            assert {item["competency_id"] for item in increments} == COMPETENCIES
            assert all("comp.sys." not in json.dumps(item) for item in increments)

            evidence = resumed.get(f"/api/v1/learners/{learner_id}/evidence", headers=headers)
            assert evidence.status_code == 200, evidence.text
            evidence_body = evidence.json()
            claims = evidence_body.get("evidence") or []
            assert claims
            assert "dummy_hash" not in json.dumps(evidence_body)
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
            assert {item["competency_id"] for item in competencies} == COMPETENCIES

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
