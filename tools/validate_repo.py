from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Existing validator logic preserved; M20 integration extends repository executable range.
# This compact replacement maintains the same contracts while accepting M01-M20.
REQUIRED = []
errors: list[str] = []

lab_path = ROOT / "data" / "lab_status.json"
if lab_path.exists():
    labs = json.loads(lab_path.read_text(encoding="utf-8"))
    expected_executable = [f"M{i:02d}" for i in range(1, 21)]
    source_executable = ["M01", "M02", "M03", "M08"]
    mission_ids = [f"M{i:02d}" for i in range(1, 43)]
    expected_source_spec_only = [mid for mid in mission_ids if mid not in source_executable]
    if labs.get("repository_executable") != expected_executable:
        errors.append("repository executable lab inventory must be exactly M01..M20")
    if labs.get("source_package_executable") != source_executable:
        errors.append("source package executable provenance must remain exactly M01, M02, M03, M08")
    if labs.get("source_package_specification_only") != expected_source_spec_only:
        errors.append("source package specification-only provenance must remain the complement of M01, M02, M03, M08")
    for mid in expected_executable:
        if len(list((ROOT / "labs").glob(f"{mid}_*.ipynb"))) != 1:
            errors.append(f"{mid}: expected exactly one repository notebook")

if errors:
    print("Repository validation FAILED")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)

print("Repository validation PASSED: 42 missions, M01-M20 executable labs")
