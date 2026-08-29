# WP-133: Local-First Storage Architecture

## 1. Storage Hierarchy under `~/.learningos/`

```
~/.learningos/
├── config.json                     # Local preferences, provider endpoints, active profile
├── learningos.db                   # Main SQLite database (WAL mode, foreign keys ON)
├── learningos.db-wal               # Write-Ahead Log
├── artifacts/                      # Checksummed binary and structured outputs
│   └── sha256/
│       ├── ab/cd1234...            # Saved plots, model weights, trace dumps
│       └── ...
├── sessions/                       # Raw session execution traces and replays
│   └── sess_20260829_001.jsonl
└── backups/                        # Automatic pre-migration snapshot dumps
    └── backup_v3_20260829.sql.gz
```

## 2. Concurrency & Integrity
- SQLite is configured with `PRAGMA journal_mode = WAL;`, `PRAGMA synchronous = NORMAL;`, `PRAGMA foreign_keys = ON;`, and a 5000ms busy timeout.
- File writes to `artifacts/` use atomic tempfile-and-rename patterns to prevent corrupt artifacts during sudden shutdowns.
