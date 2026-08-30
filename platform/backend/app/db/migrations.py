import os
import sqlite3
from pathlib import Path
from app.db.database import get_connection, DB_PATH

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

def run_migrations():
    """WP-242: Simple forward migration framework."""
    conn = get_connection()
    try:
        # Create migrations table if not exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS _schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Get applied versions
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM _schema_migrations")
        applied = {row["version"] for row in cursor.fetchall()}
        
        if not MIGRATIONS_DIR.exists():
            return
            
        migrations = sorted([f for f in os.listdir(MIGRATIONS_DIR) if f.endswith('.sql')])
        for filename in migrations:
            version = int(filename.split('_')[0])
            if version not in applied:
                filepath = MIGRATIONS_DIR / filename
                with open(filepath, 'r') as f:
                    conn.executescript(f.read())
                conn.execute("INSERT INTO _schema_migrations (version) VALUES (?)", (version,))
                conn.commit()
                print(f"Applied migration {filename}")
                
    finally:
        conn.close()

if __name__ == "__main__":
    run_migrations()
