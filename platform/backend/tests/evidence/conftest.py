from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

PACKAGE_ID = "g4.evidence.fixture"
MISSION_A = "GX01"
MISSION_B = "GX02"


def canonical_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def generic_mission_spec(
    mission_id: str,
    *,
    order_index: int = 1,
    include_gate: bool = True,
    two_criteria: bool = False,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": mission_id,
        "title": f"Generic evidence fixture {mission_id}",
        "phase": {"id": "phase_evidence", "title": "Evidence"},
        "order_index": order_index,
        "core_invariant": "Sequencing and gates are driven by spec_json only.",
        "competencies": ["comp.generic.alpha", "comp.generic.beta"],
        "knowledge_nodes": ["kn.generic.alpha", "kn.generic.beta"],
        "prerequisites": [],
        "stages": [
            {
                "id": "s1_orient",
                "title": "Orientation",
                "type": "orientation",
                "assistance_policy": "UNRESTRICTED",
            },
            {
                "id": "s2_lab",
                "title": "Experiment",
                "type": "experiment",
                "assistance_policy": "SOCRATIC_ONLY",
                "runner": {"module": "generic.lab", "entrypoint": "run", "timeout_sec": 5},
            },
            {
                "id": "s3_wrap",
                "title": "Gate",
                "type": "competency_gate",
                "assistance_policy": "NO_AI_REQUIRED",
            },
        ],
    }
    if include_gate:
        evidence = [
            {
                "competency_id": "comp.generic.alpha",
                "stage_id": "s1_orient",
                "artifact_type": "markdown",
                "knowledge_node_id": "kn.generic.alpha",
            }
        ]
        if two_criteria:
            evidence.append(
                {
                    "competency_id": "comp.generic.beta",
                    "stage_id": "s2_lab",
                    "artifact_type": "trace",
                    "knowledge_node_id": "kn.generic.beta",
                }
            )
        spec["gate_contract"] = {
            "required_evidence": evidence,
            "pass_threshold": 1.0,
            "repair_policy": {"allow_targeted_repair": True, "max_repair_attempts": 3},
        }
    return spec


def seed_mission(
    conn: sqlite3.Connection,
    mission_id: str = MISSION_A,
    *,
    order_index: int = 1,
    include_gate: bool = True,
    two_criteria: bool = False,
    git_commit_sha: str | None = None,
) -> dict[str, Any]:
    spec = generic_mission_spec(
        mission_id,
        order_index=order_index,
        include_gate=include_gate,
        two_criteria=two_criteria,
    )
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    digest = git_commit_sha or hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT INTO curriculum_packages (id, version, git_commit_sha, manifest_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            version = excluded.version,
            git_commit_sha = excluded.git_commit_sha,
            manifest_json = excluded.manifest_json
        """,
        (
            PACKAGE_ID,
            "3.0.0",
            digest,
            json.dumps({"id": PACKAGE_ID, "version": "3.0.0"}, sort_keys=True, separators=(",", ":")),
        ),
    )
    conn.execute(
        """
        INSERT INTO missions (id, package_id, title, phase_id, order_index, schema_version, spec_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            package_id = excluded.package_id,
            title = excluded.title,
            phase_id = excluded.phase_id,
            order_index = excluded.order_index,
            schema_version = excluded.schema_version,
            spec_json = excluded.spec_json
        """,
        (
            spec["id"],
            PACKAGE_ID,
            spec["title"],
            spec["phase"]["id"],
            spec["order_index"],
            "v1",
            spec_json,
        ),
    )
    conn.commit()
    return spec


def insert_learner(conn: sqlite3.Connection, username: str = "evidence-learner") -> str:
    learner_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
        (learner_id, username, username),
    )
    conn.commit()
    return learner_id


def insert_session(
    conn: sqlite3.Connection,
    learner_id: str,
    mission_id: str,
    spec: dict[str, Any],
) -> str:
    from app.api import runtime

    session_id = str(uuid.uuid4())
    current = runtime.first_stage_id(spec)
    conn.execute(
        """
        INSERT INTO mission_sessions (id, learner_id, mission_id, status, current_stage_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, learner_id, mission_id, runtime.SESSION_ACTIVE, current),
    )
    runtime.ensure_ready_attempt(conn, session_id, runtime.get_stage(spec, current))
    conn.commit()
    return session_id


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.delenv("LEARNINGOS_WORKER_SOCKET", raising=False)
    return home


@pytest.fixture
def conn(data_home):
    from app.db.database import get_connection, init_db

    init_db()
    connection = get_connection()
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def client(data_home):
    from app.main import app

    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    response = client.post("/api/v1/auth/bootstrap")
    assert response.status_code == 200, response.text
    payload = response.json()
    return {"Authorization": f"Bearer {payload['token']}"}
