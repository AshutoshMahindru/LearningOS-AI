from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.core.mdl_types import (
    ASSISTANCE_POLICIES,
    MDL_SCHEMA_VERSION,
    MISSION_REQUIRED_FIELDS,
    STAGE_TYPES,
    ValidationError,
)
from app.core.mdl_validator import (
    frozen_mission_fixtures_dir,
    load_mdl_schema,
    validate_mission,
    validate_package,
    wp136_schema_path,
)
from app.core.mission_loader import compute_payload_digest, fixture_package_path


def _minimal_mission(**overrides) -> dict:
    mission = {
        "id": "M00",
        "title": "Synthetic fixture mission",
        "phase": {"id": "phase_fixture", "title": "Fixture"},
        "order_index": 1,
        "core_invariant": "Validate MDL v1 without mission-specific platform branches.",
        "competencies": ["comp.fixture.schema"],
        "knowledge_nodes": ["kn.fixture.schema"],
        "stages": [
            {
                "id": "stage_01_orientation",
                "title": "Orientation",
                "type": "orientation",
                "assistance_policy": "UNRESTRICTED",
            }
        ],
        "gate_contract": {
            "required_evidence": [
                {
                    "competency_id": "comp.fixture.schema",
                    "stage_id": "stage_01_orientation",
                    "artifact_type": "artifact",
                }
            ],
            "pass_threshold": 1.0,
        },
    }
    mission.update(overrides)
    return mission


def _write_package(
    root: Path,
    missions: list[dict],
    *,
    digest: str | None = None,
    manifest_extra: dict | None = None,
) -> Path:
    pkg = root / "pkg"
    (pkg / "missions").mkdir(parents=True)
    entries: list[dict[str, str]] = []
    rels: list[str] = []
    for spec in missions:
        rel = f"missions/{spec['id']}.json"
        (pkg / rel).write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        entries.append({"id": spec["id"], "path": rel})
        rels.append(rel)
    manifest = {
        "id": "g4.fixture.curriculum",
        "version": "4.0.0",
        "title": "G4 schema validator fixture",
        "missions": entries,
        "digest": compute_payload_digest(pkg, rels),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    if digest is not None:
        manifest["digest"] = digest
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pkg


def test_wp136_is_loaded_from_frozen_architecture():
    path = wp136_schema_path()
    assert path.is_file()
    assert path.name == "WP-136_mission_definition_schema.json"
    assert "architecture/learningos-v3/03_technical_architecture" in path.as_posix()
    schema = load_mdl_schema()
    assert tuple(schema["required"]) == MISSION_REQUIRED_FIELDS
    assert tuple(schema["properties"]["stages"]["items"]["properties"]["type"]["enum"]) == STAGE_TYPES
    assert (
        tuple(schema["properties"]["stages"]["items"]["properties"]["assistance_policy"]["enum"])
        == ASSISTANCE_POLICIES
    )


@pytest.mark.parametrize(
    "name",
    [
        "M01_system_mapping.json",
        "M03_code_modification.json",
        "M04_data_cleaning.json",
        "M25_neural_network.json",
        "M42_agentic_capstone.json",
    ],
)
def test_valid_frozen_mission_fixtures(name):
    path = frozen_mission_fixtures_dir() / name
    payload = validate_mission(path)
    assert payload["id"].startswith("M")
    assert payload["stages"]
    assert payload["gate_contract"]
    assert "schema_version" not in payload or str(payload["schema_version"]).startswith("v1")


def test_valid_minimal_mission_defaults_to_v1():
    payload = validate_mission(_minimal_mission())
    assert payload["id"] == "M00"
    assert MDL_SCHEMA_VERSION == "v1"


def test_invalid_missing_stages():
    mission = _minimal_mission()
    del mission["stages"]
    with pytest.raises(ValidationError) as raised:
        validate_mission(mission)
    assert raised.value.path == "stages"
    assert "stages" in raised.value.message
    assert raised.value.errors
    assert all("path" in item and "message" in item for item in raised.value.errors)


def test_invalid_empty_stages():
    mission = _minimal_mission(stages=[])
    with pytest.raises(ValidationError) as raised:
        validate_mission(mission)
    assert "stages" in raised.value.path
    assert raised.value.message


def test_invalid_stage_type():
    mission = _minimal_mission()
    mission["stages"][0]["type"] = "custom_m01_stage"
    with pytest.raises(ValidationError) as raised:
        validate_mission(mission)
    assert raised.value.path == "stages[0].type"
    assert "custom_m01_stage" in raised.value.message
    assert "orientation" in raised.value.message or "enum" in raised.value.message.lower() or "not one of" in raised.value.message


def test_invalid_assistance_policy():
    mission = _minimal_mission()
    mission["stages"][0]["assistance_policy"] = "FULL_AUTOCOMPLETE"
    with pytest.raises(ValidationError) as raised:
        validate_mission(mission)
    assert raised.value.path == "stages[0].assistance_policy"
    assert raised.value.message


def test_invalid_missing_gate_contract():
    mission = _minimal_mission()
    del mission["gate_contract"]
    with pytest.raises(ValidationError) as raised:
        validate_mission(mission)
    assert raised.value.path == "gate_contract"
    assert "gate_contract" in raised.value.message


def test_invalid_mission_id_f01_is_not_special_cased():
    mission = _minimal_mission(id="F01")
    with pytest.raises(ValidationError) as raised:
        validate_mission(mission)
    assert raised.value.path == "id"
    assert "F01" in raised.value.message or "pattern" in raised.value.message.lower() or "does not match" in raised.value.message


def test_schema_version_v1_aliases_accepted():
    for value in ("v1", "1", 1, "v1.0.0", "schema/v1", "learningos.mission.v1.json"):
        payload = validate_mission(_minimal_mission(schema_version=value))
        assert payload["schema_version"] == value


def test_version_mismatch():
    mission = _minimal_mission(schema_version="v2")
    with pytest.raises(ValidationError) as raised:
        validate_mission(mission)
    assert raised.value.code == "VERSION_MISMATCH"
    assert raised.value.path == "schema_version"
    assert "v1" in raised.value.message
    assert raised.value.message  # never a silent pass


def test_validate_package_accepts_tmp_wp136_package(tmp_path):
    pkg = _write_package(tmp_path, [_minimal_mission()])
    package = validate_package(pkg)
    assert package.id == "g4.fixture.curriculum"
    assert len(package.missions) == 1
    assert package.missions[0]["id"] == "M00"
    assert package.source_path == pkg.resolve()


def test_package_with_bad_digest(tmp_path):
    pkg = _write_package(tmp_path, [_minimal_mission()], digest="0" * 64)
    with pytest.raises(ValidationError) as raised:
        validate_package(pkg)
    assert raised.value.code == "BAD_DIGEST"
    assert "digest" in raised.value.message.lower()
    assert raised.value.path
    assert raised.value.message


def test_package_version_mismatch(tmp_path):
    pkg = _write_package(
        tmp_path,
        [_minimal_mission()],
        manifest_extra={"schema_version": "v2"},
    )
    with pytest.raises(ValidationError) as raised:
        validate_package(pkg)
    assert raised.value.code == "VERSION_MISMATCH"
    assert "schema_version" in raised.value.path
    assert "v1" in raised.value.message


def test_g3_fixture_package_fails_wp136():
    with pytest.raises(ValidationError) as raised:
        validate_package(fixture_package_path())
    paths = {item["path"] for item in raised.value.errors}
    assert paths
    assert raised.value.message
    assert "id" in paths or "gate_contract" in paths or "stages[0].assistance_policy" in paths


def test_does_not_write_developer_home(isolated_home):
    real_home_db = Path.home() / ".learningos" / "learningos.db"
    before = real_home_db.stat().st_mtime_ns if real_home_db.exists() else None
    validate_mission(_minimal_mission())
    if before is None:
        assert not real_home_db.exists()
    else:
        assert real_home_db.stat().st_mtime_ns == before
    assert isolated_home.resolve() != (Path.home() / ".learningos").resolve()


def test_validate_mission_does_not_mutate_input():
    mission = _minimal_mission()
    snapshot = copy.deepcopy(mission)
    validate_mission(mission)
    assert mission == snapshot
