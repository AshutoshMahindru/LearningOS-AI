from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "data/missions.json",
    "data/apprenticeship_controls.yaml",
    "learning_os/cli.py",
    "learning_os/mission_loader.py",
    "learning_os/mission_runner.py",
    "learning_os/gate_engine.py",
    "learning_os/storage.py",
    "prompts/pedagogical_orchestrator.md",
    "prompts/zoom_controller.md",
    "tracking/learner_state.json",
    "schemas/evidence.schema.json",
]

errors: list[str] = []
for rel in REQUIRED:
    if not (ROOT / rel).exists():
        errors.append(f"missing required path: {rel}")

missions_path = ROOT / "data" / "missions.json"
if missions_path.exists():
    payload = json.loads(missions_path.read_text(encoding="utf-8"))
    missions = payload.get("missions", [])
    ids = [m.get("id") for m in missions]
    if len(missions) != 42:
        errors.append(f"expected 42 missions; found {len(missions)}")
    if ids != [f"M{i:02d}" for i in range(1, 43)]:
        errors.append("mission IDs must be contiguous M01..M42")
    for mission in missions:
        for field in ["id", "phase", "title", "objective", "flagship", "competencies"]:
            if field not in mission:
                errors.append(f"{mission.get('id', '?')} missing {field}")

if errors:
    print("Repository validation FAILED")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)

print("Repository validation PASSED")
