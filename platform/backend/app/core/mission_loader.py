import json
from pathlib import Path
from jsonschema import validate, ValidationError
from typing import Dict, Any

# We assume the backend is run from the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = PROJECT_ROOT / "architecture" / "learningos-v3" / "03_technical_architecture" / "WP-136_mission_definition_schema.json"

class MissionLoader:
    def __init__(self):
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict[str, Any]:
        if not SCHEMA_PATH.exists():
            return {}
            
        with open(SCHEMA_PATH, 'r') as f:
            return json.load(f)
            
    def validate_mission(self, mission_data: Dict[str, Any]) -> bool:
        try:
            validate(instance=mission_data, schema=self.schema)
            return True
        except ValidationError as e:
            print(f"Mission validation failed: {e.message}")
            return False
            
    # In the future, this class will load actual YAML missions from a missions/ directory.
