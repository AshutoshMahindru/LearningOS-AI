from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlite3

from app.db.database import get_connection, init_db
from app.db.ledger import GENESIS_PREV_HASH, EventLedger, compute_event_hash


def _insert_learner(conn: sqlite3.Connection, learner_id: str = "learner_1") -> None:
    conn.execute(
        "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
        (learner_id, f"user_{learner_id}", learner_id),
    )
    conn.commit()


def test_append_hash_chain(data_home: Path) -> None:
    init_db()
    conn = get_connection()
    try:
        _insert_learner(conn)
        ledger = EventLedger(conn)
        first_id = ledger.append("learner_1", "session.started", {"mission_id": "g3.fixture"})
        second_id = ledger.append("learner_1", "artifact.stored", {"size": 4})
        assert first_id != second_id

        rows = conn.execute(
            """
            SELECT id, event_type, payload_json, prev_hash, event_hash, created_at
            FROM learning_events
            ORDER BY rowid
            """
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["prev_hash"] == GENESIS_PREV_HASH
        assert rows[1]["prev_hash"] == rows[0]["event_hash"]

        for row in rows:
            payload = json.loads(row["payload_json"])
            expected = compute_event_hash(
                learner_id="learner_1",
                event_type=row["event_type"],
                payload=payload,
                prev_hash=row["prev_hash"],
                created_at=row["created_at"],
            )
            assert row["event_hash"] == expected
    finally:
        conn.close()

    conn2 = get_connection()
    try:
        count = conn2.execute("SELECT COUNT(*) AS n FROM learning_events").fetchone()["n"]
        assert count == 2
    finally:
        conn2.close()


def test_update_and_delete_are_refused(data_home: Path) -> None:
    init_db()
    conn = get_connection()
    try:
        _insert_learner(conn)
        event_id = EventLedger(conn).append("learner_1", "ping", {"ok": True})
        with pytest.raises(sqlite3.Error):
            conn.execute(
                "UPDATE learning_events SET event_type = ? WHERE id = ?",
                ("mutated", event_id),
            )
        with pytest.raises(sqlite3.Error):
            conn.execute("DELETE FROM learning_events WHERE id = ?", (event_id,))
        conn.rollback()
        row = conn.execute(
            "SELECT event_type FROM learning_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        assert row["event_type"] == "ping"
        assert conn.execute("SELECT COUNT(*) AS n FROM learning_events").fetchone()["n"] == 1
        assert not hasattr(EventLedger, "update")
        assert not hasattr(EventLedger, "delete")
    finally:
        conn.close()
