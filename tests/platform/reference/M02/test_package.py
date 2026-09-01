from __future__ import annotations

import json
from pathlib import Path

from app.core.mdl_types import STAGE_TYPES
from app.core.mdl_validator import validate_mission, validate_package as validate_mdl_package
from tools.authoring.validate import validate_package

REPO_ROOT = Path(__file__).resolve().parents[4]
M02_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M02"
FROZEN_STAGE_TYPES = set(STAGE_TYPES)


def test_package_identity_is_g5_reference_m02() -> None:
    result = validate_package(M02_PACKAGE)
    assert result.ok, result.errors
    assert result.package_id == "g5.reference.M02"
    assert result.version == "5.0.0"
    assert result.mission_ids == ["M02"]
    assert result.schema_engine in {"jsonschema", "mdl_validator"}
    loaded = validate_mdl_package(M02_PACKAGE)
    assert loaded.id == "g5.reference.M02"
    assert [mission["id"] for mission in loaded.missions] == ["M02"]


def test_mission_satisfies_wp136_and_uses_frozen_stage_types() -> None:
    result = validate_package(M02_PACKAGE)
    assert result.ok, result.errors
    mission = result.package.missions[0]  # type: ignore[union-attr]
    payload = validate_mission(mission)
    assert payload["id"] == "M02"
    assert payload["order_index"] == 2
    assert payload["flagship_version"] == "V00"
    types = [stage["type"] for stage in payload["stages"]]
    assert set(types) <= FROZEN_STAGE_TYPES
    assert "orientation" in types
    assert "experiment" in types
    assert "controlled_failure" in types
    assert "transfer_assessment" in types
    assert "competency_gate" in types
    assert "custom_m02_stage" not in types
    experiment = next(stage for stage in payload["stages"] if stage["type"] == "experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text
    policies = {stage["id"]: stage["assistance_policy"] for stage in payload["stages"]}
    assert policies["stage_05_experiment"] == "RESTRICTED_HINTS"
    assert policies["stage_07_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_09_gate"] == "NO_AI_REQUIRED"
    contract = payload["gate_contract"]
    assert contract["pass_threshold"] == 1.0
    assert contract["repair_policy"]["allow_targeted_repair"] is True
    assert contract["repair_policy"]["max_repair_attempts"] == 3
    required_stages = {item["stage_id"] for item in contract["required_evidence"]}
    assert "stage_05_experiment" in required_stages
    assert "stage_06_controlled_failure" in required_stages
    assert "stage_07_transfer" in required_stages


def test_fixture_files_live_under_platform_fixtures_m02() -> None:
    mission_path = M02_PACKAGE / "missions" / "M02.json"
    assert M02_PACKAGE == REPO_ROOT / "platform" / "fixtures" / "M02"
    assert (M02_PACKAGE / "manifest.json").is_file()
    assert (M02_PACKAGE / "SHA256SUMS").is_file()
    assert mission_path.is_file()
    spec = json.loads(mission_path.read_text(encoding="utf-8"))
    blob = json.dumps(spec)
    assert spec["id"] == "M02"
    assert "g5.reference.M02" not in spec["id"]
    assert all(stage["type"] in FROZEN_STAGE_TYPES for stage in spec["stages"])
    assert "prediction" in blob.lower()
    assert "NO_AI_REQUIRED" in blob
    assert "controlled_failure" in blob
    assert "allow_targeted_repair" in blob
