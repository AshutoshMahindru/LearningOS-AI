from __future__ import annotations

import json

from app.core.projection import next_action
from tests.platform.flagship.conftest import complete_mission, insert_learner, seed_mission


def test_next_action_uses_flagship_prereqs_among_loaded_missions(conn) -> None:
    seed_mission(conn, "M05", order_index=5, flagship_version="V01")
    seed_mission(conn, "M07", order_index=7, flagship_version="V01")
    learner_id = insert_learner(conn, "prereq-learner")
    first = next_action(conn, learner_id)
    assert first["action"] == "START_MISSION"
    assert first["mission_id"] == "M05"
    assert first["flagship_version"] == "V01"
    assert first["flagship"]["version_id"] == "V01"
    complete_mission(conn, learner_id, "M05")
    following = next_action(conn, learner_id)
    assert following["action"] == "START_MISSION"
    assert following["mission_id"] == "M07"
    assert following["reason"] == "NEXT_MISSION"


def test_next_action_blocks_when_loaded_prereqs_unmet(conn) -> None:
    seed_mission(conn, "M07", order_index=7, flagship_version="V01")
    seed_mission(conn, "M06", order_index=6, flagship_version="V01")
    learner_id = insert_learner(conn, "blocked-learner")
    today = next_action(conn, learner_id)
    assert today["mission_id"] == "M06"
    complete_mission(conn, learner_id, "M07")
    still = next_action(conn, learner_id)
    assert still["mission_id"] == "M06"


def test_generic_catalog_is_unchanged_and_omits_flagship_spine(conn) -> None:
    seed_mission(conn, "GX01", order_index=1)
    seed_mission(conn, "GX02", order_index=2)
    learner_id = insert_learner(conn, "generic-learner")
    today = next_action(conn, learner_id)
    assert today["action"] == "START_MISSION"
    assert today["mission_id"] == "GX01"
    assert "flagship_version" not in today
    blob = json.dumps(today)
    assert "M01" not in blob
    assert "M42" not in blob


def test_math_track_unblocks_without_completing_intervening_versions(conn) -> None:
    seed_mission(conn, "M05", order_index=5, flagship_version="V01")
    seed_mission(conn, "M15", order_index=15, flagship_version="V04")
    learner_id = insert_learner(conn, "math-learner")
    first = next_action(conn, learner_id)
    assert first["mission_id"] == "M05"
    complete_mission(conn, learner_id, "M05")
    following = next_action(conn, learner_id)
    assert following["mission_id"] == "M15"
    assert following["flagship_version"] == "V04"
