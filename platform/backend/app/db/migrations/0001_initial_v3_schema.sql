-- ============================================================================
-- LearningOS V3 Database Schema & Initial Migration (0001_initial_v3_schema.sql)
-- Target Database: SQLite 3.38+ with WAL mode & Foreign Key Enforcement
-- Location: ~/.learningos/learningos.db
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- 1. Learner Profile & Configuration
CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY, -- e.g. "learner_default"
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    autonomy_tier INTEGER NOT NULL DEFAULT 0, -- 0 (None) to 4 (Full Autonomous)
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. Curriculum Package Registry
CREATE TABLE IF NOT EXISTS curriculum_packages (
    id TEXT PRIMARY KEY, -- e.g. "curriculum_core_v3"
    version TEXT NOT NULL,
    git_commit_sha TEXT NOT NULL,
    manifest_json TEXT NOT NULL, -- Full package metadata JSON
    installed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 3. Mission Definitions & Status
CREATE TABLE IF NOT EXISTS missions (
    id TEXT PRIMARY KEY, -- e.g. "M01", "M25", "M42"
    package_id TEXT NOT NULL,
    title TEXT NOT NULL,
    phase_id TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    schema_version TEXT NOT NULL DEFAULT "v1",
    spec_json TEXT NOT NULL, -- Full validated MDL v1 JSON
    FOREIGN KEY (package_id) REFERENCES curriculum_packages(id) ON DELETE CASCADE
);

-- 4. Learner Mission Sessions
CREATE TABLE IF NOT EXISTS mission_sessions (
    id TEXT PRIMARY KEY, -- UUIDv4
    learner_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT "ACTIVE", -- ACTIVE, PAUSED, COMPLETED, ABANDONED
    current_stage_id TEXT NOT NULL,
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    paused_at DATETIME,
    completed_at DATETIME,
    FOREIGN KEY (learner_id) REFERENCES learners(id) ON DELETE CASCADE,
    FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE RESTRICT
);

-- 5. Stage Attempts & Execution State
CREATE TABLE IF NOT EXISTS stage_attempts (
    id TEXT PRIMARY KEY, -- UUIDv4
    session_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    stage_type TEXT NOT NULL, -- orientation, experiment, transfer_assessment, etc.
    attempt_number INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT "ACTIVE", -- READY, ACTIVE, SUBMITTED, PASSED, REPAIR_REQUIRED
    assistance_level TEXT NOT NULL DEFAULT "UNASSISTED", -- UNASSISTED, SOCRATIC, NO_AI_CERTIFIED
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    FOREIGN KEY (session_id) REFERENCES mission_sessions(id) ON DELETE CASCADE
);

-- 6. Predictions (Predict-Commit-Run Contract)
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY, -- UUIDv4
    stage_attempt_id TEXT NOT NULL,
    hypothesis_text TEXT NOT NULL,
    expected_values_json TEXT NOT NULL, -- JSON structured prediction data
    prediction_hash TEXT NOT NULL, -- SHA-256 hash of prediction payload
    committed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_sealed BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (stage_attempt_id) REFERENCES stage_attempts(id) ON DELETE CASCADE
);

-- 7. Executions & Structured Results
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY, -- UUIDv4
    stage_attempt_id TEXT NOT NULL,
    runner_id TEXT NOT NULL,
    input_code TEXT,
    code_hash TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    structured_result_json TEXT NOT NULL, -- Validated WP-137 Structured Result JSON
    diagnostics_log TEXT, -- Raw stdout / stderr
    executed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stage_attempt_id) REFERENCES stage_attempts(id) ON DELETE CASCADE
);

-- 8. Evidence Items & Cryptographic Provenance
CREATE TABLE IF NOT EXISTS evidence_items (
    id TEXT PRIMARY KEY, -- UUIDv4
    learner_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    stage_attempt_id TEXT NOT NULL,
    competency_id TEXT NOT NULL, -- e.g. "comp.ml.system_mapping"
    knowledge_node_id TEXT NOT NULL, -- e.g. "kn.m01.pipeline_trace"
    artifact_type TEXT NOT NULL, -- chart, code, adr, metric, trace
    artifact_path TEXT, -- relative path inside ~/.learningos/artifacts/
    artifact_hash TEXT NOT NULL, -- SHA-256 hash
    assessment_status TEXT NOT NULL DEFAULT "ACCEPTED", -- ACCEPTED, REJECTED, SUPERSEDED
    assistance_level TEXT NOT NULL, -- UNASSISTED, SOCRATIC, NO_AI_CERTIFIED
    curriculum_sha TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(id) ON DELETE CASCADE,
    FOREIGN KEY (stage_attempt_id) REFERENCES stage_attempts(id) ON DELETE CASCADE
);

-- 9. Competency Mastery Graph
CREATE TABLE IF NOT EXISTS competency_mastery (
    learner_id TEXT NOT NULL,
    competency_id TEXT NOT NULL,
    level INTEGER NOT NULL DEFAULT 0, -- 0 (L0 Unexposed) to 5 (L5 Master/Author)
    decay_score REAL NOT NULL DEFAULT 1.0, -- 1.0 = Fresh, 0.0 = Decayed
    last_evaluated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_evidence_item_id TEXT,
    PRIMARY KEY (learner_id, competency_id),
    FOREIGN KEY (learner_id) REFERENCES learners(id) ON DELETE CASCADE,
    FOREIGN KEY (last_evidence_item_id) REFERENCES evidence_items(id) ON DELETE SET NULL
);

-- 10. Architectural Decision Records (ADRs)
CREATE TABLE IF NOT EXISTS adrs (
    id TEXT PRIMARY KEY, -- e.g. "ADR-001"
    learner_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT "PROPOSED", -- PROPOSED, ACCEPTED, SUPERSEDED
    context_text TEXT NOT NULL,
    decision_text TEXT NOT NULL,
    consequences_text TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(id) ON DELETE CASCADE
);

-- Indexes for Fast Querying
CREATE INDEX IF NOT EXISTS idx_sessions_learner ON mission_sessions(learner_id, status);
CREATE INDEX IF NOT EXISTS idx_attempts_session ON stage_attempts(session_id, stage_id);
CREATE INDEX IF NOT EXISTS idx_evidence_learner_comp ON evidence_items(learner_id, competency_id);
CREATE INDEX IF NOT EXISTS idx_executions_attempt ON executions(stage_attempt_id);
