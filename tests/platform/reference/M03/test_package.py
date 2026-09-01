from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.mdl_types import STAGE_TYPES
from app.core.mdl_validator import validate_mission, validate_package as validate_mdl_package
from app.core.mission_loader import load_package as g3_load
from tools.authoring.package import load_package, sha256_file
from tools.authoring.validate import validate_package

REPO_ROOT = Path(__file__).resolve().parents[4]
M03_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M03"
MISSION_PATH = M03_PACKAGE / "missions" / "M03.json"

PACKAGE_ID = "g5.reference.M03"
PACKAGE_VERSION = "5.0.0"
MISSION_ID = "M03"

REQUIRED_STAGE_TYPES = {
    "orientation",
    "experiment",
    "code_reading",
    "rebuild_debug",
    "controlled_failure",
    "transfer_assessment",
    "competency_gate",
}


def test_package_identity_keeps_m03_as_wp136_mission_id() -> None:
    package = load_package(M03_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.version == PACKAGE_VERSION
    assert package.missions[0]["id"] == MISSION_ID
    assert re.match(r"^M[0-9]{2}$", package.missions[0]["id"])
    assert package.id != MISSION_ID
    assert "g5.reference.M03" in json.dumps(package.manifest)


def test_package_validates_against_wp136_and_g4_journey() -> None:
    result = validate_package(M03_PACKAGE)
    assert result.ok, result.errors
    assert result.schema_engine in {"jsonschema", "mdl_validator"}
    assert result.mission_ids == [MISSION_ID]
    mdl = validate_mdl_package(M03_PACKAGE)
    assert mdl.id == PACKAGE_ID
    spec = validate_mission(MISSION_PATH)
    assert spec["id"] == MISSION_ID
    assert spec["order_index"] == 3
    assert spec["flagship_version"] == "V01"


def test_g3_loader_accepts_sealed_package() -> None:
    package = g3_load(M03_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.missions[0]["id"] == MISSION_ID
    assert package.digest == load_package(M03_PACKAGE).digest
    assert len(package.digest) == 64


def test_integrity_files_match_payload() -> None:
    package = load_package(M03_PACKAGE)
    checksums = (M03_PACKAGE / "SHA256SUMS").read_text(encoding="utf-8")
    assert "manifest.json" in checksums
    assert "missions/M03.json" in checksums
    assert sha256_file(M03_PACKAGE / "manifest.json") in checksums
    assert sha256_file(MISSION_PATH) in checksums
    assert package.digest == json.loads((M03_PACKAGE / "manifest.json").read_text(encoding="utf-8"))["digest"]


def test_stages_are_generic_catalogue_types_for_code_edit_debug_test_diff() -> None:
    mission = json.loads(MISSION_PATH.read_text(encoding="utf-8"))
    types = [stage["type"] for stage in mission["stages"]]
    assert set(types) <= set(STAGE_TYPES)
    for required in REQUIRED_STAGE_TYPES:
        assert required in types, required
    assert types[0] == "orientation"
    assert types[-1] == "competency_gate"
    assert "experiment" in types
    assert "code_reading" in types
    assert "rebuild_debug" in types

    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text
    assert experiment["validation_rubric"]["required_evidence_type"] == "state_diff"

    rebuild = next(stage for stage in mission["stages"] if stage["type"] == "rebuild_debug")
    assert "test" in rebuild["instructions"].lower()
    assert rebuild["validation_rubric"]["required_evidence_type"] == "metric"

    transfer = next(stage for stage in mission["stages"] if stage["type"] == "transfer_assessment")
    assert transfer["assistance_policy"] == "NO_AI_REQUIRED"

    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_02_experiment"] == "RESTRICTED_HINTS"
    assert policies["stage_03_code_reading"] == "SOCRATIC_ONLY"
    assert policies["stage_04_rebuild_debug"] == "RESTRICTED_HINTS"
    assert policies["stage_06_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_07_gate"] == "NO_AI_REQUIRED"


def test_gate_contract_covers_experiment_debug_and_transfer() -> None:
    mission = json.loads(MISSION_PATH.read_text(encoding="utf-8"))
    required = mission["gate_contract"]["required_evidence"]
    by_stage = {item["stage_id"]: item for item in required}
    assert by_stage["stage_02_experiment"]["artifact_type"] == "state_diff"
    assert by_stage["stage_03_code_reading"]["artifact_type"] == "artifact"
    assert by_stage["stage_04_rebuild_debug"]["artifact_type"] == "metric"
    assert by_stage["stage_06_transfer"]["artifact_type"] == "artifact"
    assert {item["competency_id"] for item in required} == {
        "comp.py.experiment",
        "comp.py.code_reading",
        "comp.py.test_debugging",
        "comp.py.diff_verification",
    }
    assert mission["gate_contract"]["pass_threshold"] == 1.0


def test_wp136_schema_rejects_package_id_as_mission_id() -> None:
    from tools.authoring.validate import validate_mission_document

    mission = json.loads(MISSION_PATH.read_text(encoding="utf-8"))
    mission["id"] = "g5.reference.M03"
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors
    assert any("does not match" in item or "g5.reference.M03" in item or "M[0-9]" in item for item in errors)


def test_source_mission_content_is_represented() -> None:
    mission = json.dumps(json.loads(MISSION_PATH.read_text(encoding="utf-8"))).lower()
    for token in (
        "predict",
        "nameerror",
        "typeerror",
        "keyerror",
        "off-by-one",
        "no-ai",
        "inventory",
        "order-total",
        "v01",
    ):
        assert token in mission, token
