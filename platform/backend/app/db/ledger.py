"""Append-only learning event ledger with SHA-256 hash chaining."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone

GENESIS_PREV_HASH = "0" * 64


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_event_hash(
    *,
    learner_id: str,
    event_type: str,
    payload: dict,
    prev_hash: str,
    created_at: str,
) -> str:
    """SHA-256 of the canonical payload envelope chained to prev_hash."""
    envelope = {
        "created_at": created_at,
        "event_type": event_type,
        "learner_id": learner_id,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


class EventLedger:
    """INSERT-only event log. UPDATE/DELETE are not offered and are blocked by SQL triggers."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def append(self, learner_id: str, event_type: str, payload: dict) -> str:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        event_id = str(uuid.uuid4())
        prev_row = self._conn.execute(
            """
            SELECT event_hash
            FROM learning_events
            WHERE learner_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (learner_id,),
        ).fetchone()
        prev_hash = prev_row["event_hash"] if prev_row is not None else GENESIS_PREV_HASH
        event_hash = compute_event_hash(
            learner_id=learner_id,
            event_type=event_type,
            payload=payload,
            prev_hash=prev_hash,
            created_at=created_at,
        )
        try:
            self._conn.execute(
                """
                INSERT INTO learning_events (
                    id, learner_id, event_type, payload_json, prev_hash, event_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    learner_id,
                    event_type,
                    canonical_json(payload),
                    prev_hash,
                    event_hash,
                    created_at,
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return event_id
