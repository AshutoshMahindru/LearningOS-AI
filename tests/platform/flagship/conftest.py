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

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
FLAGSHIP_FIXTURE = REPO_ROOT / "platform" / "fixtures" / "flagship"
FROZEN_BASE = "f7926e661a955f2d78bd8584877815825c5ef047"

for candidate in (REPO_ROOT, BACKEND_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)


def generic_mission_spec(
    mission_id: str,
    *,
    order_index: int = 1,
    flagship_version: str | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": mission_id,
        "title": f"Generic fixture {mission_id}",
        "phase": {"id": "phase_flagship", "title": "Flagship"},
        "order_index": order_index,
        "core_invariant": "Sequencing is driven by spec_json and the flagship index.",
        "competencies": ["comp.generic.flagship"],
        "knowledge_nodes": ["kn.generic.flagship"],
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
            },
            {
                "id": "s3_wrap",
                "title": "Gate",
                "type": "competency_gate",
                "assistance_policy": "NO_AI_REQUIRED",
            },
        ],
        "gate_contract": {
            "required_evidence": [
                {
                    "competency_id": "comp.generic.flagship",
                    "stage_id": "s1_orient",
                    "artifact_type": "markdown",
                }
            ],
            "pass_threshold": 1.0,
            "repair_policy": {"allow_targeted_repair": True, "max_repair_attempts": 3},
        },
    }
    if flagship_version:
        spec["flagship_version"] = flagship_version
    return spec


def seed_mission(
    conn: sqlite3.Connection,
    mission_id: str,
    *,
    order_index: int = 1,
    flagship_version: str | None = None,
) -> dict[str, Any]:
    spec = generic_mission_spec(
        mission_id, order_index=order_index, flagship_version=flagship_version
    )
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
    package_id = "g6.flagship.test"
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
            package_id,
            "6.0.0",
            digest,
            json.dumps({"id": package_id, "version": "6.0.0"}, sort_keys=True, separators=(",", ":")),
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
            package_id,
            spec["title"],
            spec["phase"]["id"],
            spec["order_index"],
            "v1",
            spec_json,
        ),
    )
    conn.commit()
    return spec


def insert_learner(conn: sqlite3.Connection, username: str = "flagship-learner") -> str:
    learner_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
        (learner_id, username, username),
    )
    conn.commit()
    return learner_id


def complete_mission(conn: sqlite3.Connection, learner_id: str, mission_id: str) -> str:
    session_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO mission_sessions (id, learner_id, mission_id, status, current_stage_id)
        VALUES (?, ?, ?, 'COMPLETED', 's3_wrap')
        """,
        (session_id, learner_id, mission_id),
    )
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
