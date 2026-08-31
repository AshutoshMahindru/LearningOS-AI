-- ============================================================================
-- 0002_event_ledger.sql
-- Append-only learning_events ledger with hash chaining.
-- UPDATE/DELETE are refused by ABORT triggers.
-- ============================================================================

CREATE TABLE IF NOT EXISTS learning_events (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_learning_events_learner_created
    ON learning_events(learner_id, created_at);

CREATE INDEX IF NOT EXISTS idx_learning_events_hash
    ON learning_events(event_hash);

CREATE TRIGGER IF NOT EXISTS trg_learning_events_no_update
BEFORE UPDATE ON learning_events
BEGIN
    SELECT RAISE(ABORT, 'learning_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_events_no_delete
BEFORE DELETE ON learning_events
BEGIN
    SELECT RAISE(ABORT, 'learning_events is append-only');
END;
