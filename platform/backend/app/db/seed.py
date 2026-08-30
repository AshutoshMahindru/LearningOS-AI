import sqlite3
import json
import os
from pathlib import Path
from app.db.database import get_connection, PROJECT_ROOT

FIXTURES_DIR = PROJECT_ROOT / "architecture" / "learningos-v3" / "04_cross_mission_proof" / "fixtures"

def seed_database():
    conn = get_connection()
    try:
        # 1. Insert default learner
        learner_id = "learner_default"
        conn.execute("""
            INSERT OR IGNORE INTO learners (id, username, display_name)
            VALUES (?, ?, ?)
        """, (learner_id, "default_user", "Default Learner"))
        
        # 2. Insert dummy curriculum package
        package_id = "curriculum_core_v3"
        conn.execute("""
            INSERT OR REPLACE INTO curriculum_packages (id, version, git_commit_sha, manifest_json)
            VALUES (?, ?, ?, ?)
        """, (package_id, "3.0.0", "HEAD", "{}"))
        
        # 2. Iterate and insert missions
        if not FIXTURES_DIR.exists():
            print(f"Error: Fixtures dir not found at {FIXTURES_DIR}")
            return
            
        missions_added = 0
        order_index = 0
        
        for file_path in FIXTURES_DIR.glob("*.json"):
            with open(file_path, 'r') as f:
                spec = json.load(f)
                
            mission_id = spec.get("id")
            title = spec.get("title", "Untitled")
            phase_id = spec.get("phase", {}).get("id", "unknown")
            spec_json = json.dumps(spec)
            
            if not mission_id:
                print(f"Skipping {file_path.name}: No mission_id found")
                continue
                
            conn.execute("""
                INSERT OR REPLACE INTO missions (id, package_id, title, phase_id, order_index, schema_version, spec_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (mission_id, package_id, title, phase_id, order_index, "v1", spec_json))
            
            order_index += 1
            missions_added += 1
            print(f"Seeded mission: {mission_id} ({title})")
            
        conn.commit()
        print(f"Successfully seeded {missions_added} missions.")
        
    except Exception as e:
        conn.rollback()
        print(f"Failed to seed database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_database()
