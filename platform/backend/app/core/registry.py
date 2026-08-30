import json
from pathlib import Path
from typing import Dict, Any

class CurriculumRegistry:
    """WP-251 & WP-252: Curriculum package registry and discovery."""
    
    _missions: Dict[str, Any] = {}
    
    @classmethod
    def load_all_missions(cls, missions_dir: Path):
        """Scans the missions directory and loads definition schemas."""
        if not missions_dir.exists():
            return
            
        for path in missions_dir.glob("M*/mission.json"):
            try:
                with open(path, "r") as f:
                    spec = json.load(f)
                    mission_id = spec.get("id")
                    if mission_id:
                        cls._missions[mission_id] = spec
            except Exception as e:
                print(f"Failed to load mission {path}: {e}")
                
    @classmethod
    def get_mission(cls, mission_id: str) -> dict:
        return cls._missions.get(mission_id)
        
    @classmethod
    def list_missions(cls) -> list:
        return list(cls._missions.values())
