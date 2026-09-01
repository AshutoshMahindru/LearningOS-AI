from __future__ import annotations

import json
import re
from pathlib import Path

from tools.authoring.package import (
    F01_PACKAGE_ID,
    F01_PACKAGE_VERSION,
    SYNTHETIC_MISSION_ID,
    default_package_path,
    load_package,
)
from tools.authoring.paths import WP136_SCHEMA_PATH
from tools.authoring.validate import validate_mission_document, validate_package

REPO_ROOT = Path(__file__).resolve().parents[3]
F01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "f01"


def test_default_package_path_is_platform_fixtures_f01() -> None:
    assert default_package_path() == F01_PACKAGE
    assert default_package_path() == REPO_ROOT / "platform" / "fixtures" / "f01"


def test_package_identity_keeps_f01_in_metadata_only() -> None:
    package = load_package(F01_PACKAGE)
    assert package.id == F01_PACKAGE_ID
    assert package.version == F01_PACKAGE_VERSION
    assert "F01" in package.title
    assert package.missions[0]["id"] == SYNTHETIC_MISSION_ID
    assert package.missions[0]["id"] != "F01"
    assert re.match(r"^M[0-9]{2}$", package.missions[0]["id"])


def test_mission_satisfies_wp136_and_g4_journey() -> None:
    result = validate_package(F01_PACKAGE)
    assert result.ok, result.errors
    assert result.schema_engine in {"jsonschema", "mdl_validator"}
    mission = result.package.missions[0]  # type: ignore[union-attr]
    types = [stage["type"] for stage in mission["stages"]]
    assert "orientation" in types
    assert "experiment" in types
    assert "transfer_assessment" in types or "competency_gate" in types
    assert "gate_contract" in mission
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_02_experiment"] == "RESTRICTED_HINTS"
    assert policies["stage_03_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_04_gate"] == "NO_AI_REQUIRED"


def test_mission_is_generic_not_m01_system_mapping() -> None:
    mission = json.loads((F01_PACKAGE / "missions" / "M00.json").read_text(encoding="utf-8"))
    blob = json.dumps(mission).lower()
    assert "system mapping" not in blob
    assert "m01" not in blob
    assert mission["id"] != "M01"
    assert "Operations Intelligence" not in json.dumps(mission)


def test_wp136_schema_rejects_f01_as_mission_id() -> None:
    mission = json.loads((F01_PACKAGE / "missions" / "M00.json").read_text(encoding="utf-8"))
    mission["id"] = "F01"
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors
    assert any("F01" in item or "M[0-9]" in item or "does not match" in item for item in errors)


def test_wp136_schema_file_is_readable() -> None:
    assert WP136_SCHEMA_PATH.is_file()
    schema = json.loads(WP136_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["id"]["pattern"] == "^M[0-9]{2}$"


def test_authoring_tree_has_no_mission_id_f01_conditionals() -> None:
    roots = [
        REPO_ROOT / "tools" / "authoring",
        REPO_ROOT / "platform" / "fixtures" / "f01",
        REPO_ROOT / "tests" / "platform" / "authoring",
    ]
    pattern = re.compile(r"mission_id\s*==\s*[\"']F01[\"']")
    hits: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == []
