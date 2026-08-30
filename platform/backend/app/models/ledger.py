import uuid
import json
from app.db.database import get_connection

class EventLedger:
    """WP-244: Append-only event ledger for all learning activities."""
    
    @staticmethod
    def append_event(learner_id: str, event_type: str, payload: dict):
        conn = get_connection()
        try:
            event_id = str(uuid.uuid4())
            cursor = conn.cursor()
            
            # Note: requires an events table in the schema.
            # Assuming a generic events table for WP-244
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_events (
                    id TEXT PRIMARY KEY,
                    learner_id TEXT,
                    event_type TEXT,
                    payload_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("""
                INSERT INTO learning_events (id, learner_id, event_type, payload_json)
                VALUES (?, ?, ?, ?)
            """, (event_id, learner_id, event_type, json.dumps(payload)))
            conn.commit()
            return event_id
        finally:
            conn.close()
