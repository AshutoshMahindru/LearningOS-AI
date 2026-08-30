import sqlite3
import os
from pathlib import Path

# Paths
HOME_DIR = Path.home()
DB_DIR = HOME_DIR / ".learningos"
DB_PATH = DB_DIR / "learningos.db"

# We assume the backend is run from the repo root or we find it relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = PROJECT_ROOT / "architecture" / "learningos-v3" / "03_technical_architecture" / "WP-134_sqlite_data_model_and_migrations.sql"

def get_connection() -> sqlite3.Connection:
    """Returns a connected SQLite database connection with WAL mode enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Enforce WAL mode as per WP-133
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    """Initializes the database schema if it doesn't exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")
        
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
        
    conn = get_connection()
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
