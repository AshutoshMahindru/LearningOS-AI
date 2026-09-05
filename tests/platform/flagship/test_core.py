from __future__ import annotations

from pathlib import Path

from app.core.flagship import (
    VERSION_IDS,
    blocking_prerequisites,
    current_version_id,
    learner_progress,
    load_index,
    mission_unlocked,
    unmet_blocking,
    version_for_mission,
    versions,
)
from tests.platform.flagship.conftest import complete_mission, insert_learner, seed_mission

CORE_PATH = Path(__file__).resolve().parents[3] / "platform" / "backend" / "app" / "core" / "flagship.py"


def test_load_index_and_version_lookup() -> None:
    index = load_index()
    assert [item["id"] for item in versions(index)] == list(VERSION_IDS)
    v00 = version_for_mission("M01")
    assert v00 is not None
    assert v00["id"] == "V00"
    assert version_for_mission("M42")["id"] == "V12"
    assert version_for_mission("GX01") is None
    assert blocking_prerequisites("M08") == ["M07"]
    assert blocking_prerequisites("GX01") == []


def test_partial_catalog_ignores_missing_prereqs() -> None:
    assert mission_unlocked("M07", completed_missions=[], catalog_ids=["M07"])
    assert not mission_unlocked(
        "M07",
        completed_missions=["M04"],
        catalog_ids=["M04", "M05", "M06", "M07"],
    )
    assert unmet_blocking(
        "M07",
        ["M04", "M05"],
        ["M04", "M05", "M06", "M07"],
    ) == ["M06"]
    assert mission_unlocked("M15", completed_missions=["M05"], catalog_ids=["M05", "M15"])


def test_progress_statuses(conn) -> None:
    for mission_id, order in (("M01", 1), ("M02", 2), ("M03", 3)):
        seed_mission(conn, mission_id, order_index=order)
    learner_id = insert_learner(conn, "progress-learner")
    empty = {item["id"]: item for item in learner_progress(conn, learner_id)}
    assert empty["V00"]["status"] == "AVAILABLE"
    assert empty["V01"]["status"] == "LOCKED"
    assert empty["V12"]["status"] == "ABSENT"
    complete_mission(conn, learner_id, "M01")
    mid = {item["id"]: item for item in learner_progress(conn, learner_id)}
    assert mid["V00"]["status"] == "IN_PROGRESS"
    complete_mission(conn, learner_id, "M02")
    done = {item["id"]: item for item in learner_progress(conn, learner_id)}
    assert done["V00"]["status"] == "COMPLETE"
    assert done["V01"]["status"] == "AVAILABLE"
    assert current_version_id(list(done.values())) == "V01"


def test_core_has_no_mission_id_branch() -> None:
    source = CORE_PATH.read_text(encoding="utf-8")
    assert "if mission_id == \"M42\"" not in source
    assert "if mission_id == 'M42'" not in source
    assert "/missions/M42" not in source
    assert version_for_mission("M41")["id"] == "V11"
    assert version_for_mission("M42")["id"] == "V12"


def test_artifact_type_metadata_omits_mission_ids() -> None:
    payload = load_index()["learner_artifact_types"]
    blob = str(payload)
    assert "M01" not in blob
    assert "M42" not in blob
