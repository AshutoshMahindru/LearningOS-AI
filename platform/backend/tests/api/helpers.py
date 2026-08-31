from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

RUNTIME_MISSION_ID = "M00"
RUNTIME_PACKAGE_ID = "g4.runtime.fixture"


def assert_typed_error(response, status_code: int, code: str) -> dict:
    assert response.status_code == status_code, response.text
    body = response.json()
    assert "error" in body
    error = body["error"]
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error.get("details", {}), dict)
    return error


def assert_no_forbidden_payload(response) -> None:
    text = response.text
    assert "dummy_hash" not in text.lower()
    assert "comp.sys.hypothesis_testing" not in text
    assert "G3_QUARANTINED" not in text


def canonical_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def runtime_mission_spec(*, include_gate: bool = True) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": RUNTIME_MISSION_ID,
        "title": "Generic runtime fixture",
        "phase": {"id": "phase_runtime", "title": "Runtime"},
        "order_index": 1,
        "core_invariant": "Stage sequencing is driven by spec_json, never by mission id.",
        "competencies": ["comp.generic.runtime"],
        "knowledge_nodes": ["kn.generic.runtime"],
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
    }
    if include_gate:
        spec["gate_contract"] = {
            "required_evidence": [
                {
                    "competency_id": "comp.generic.runtime",
                    "stage_id": "s2_lab",
                    "artifact_type": "trace",
                }
            ],
            "pass_threshold": 1.0,
            "repair_policy": {"allow_targeted_repair": True, "max_repair_attempts": 3},
        }
    return spec


def seed_runtime_mission(conn: sqlite3.Connection, *, include_gate: bool = True) -> dict[str, Any]:
    spec = runtime_mission_spec(include_gate=include_gate)
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
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
            RUNTIME_PACKAGE_ID,
            "3.0.0",
            digest,
            json.dumps({"id": RUNTIME_PACKAGE_ID, "version": "3.0.0"}, sort_keys=True, separators=(",", ":")),
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
            RUNTIME_PACKAGE_ID,
            spec["title"],
            spec["phase"]["id"],
            spec["order_index"],
            "v1",
            spec_json,
        ),
    )
    conn.commit()
    return spec
