from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
M01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M01"

MISSION_ID_RE = re.compile(r"^M[0-9]{2}$")
FROZEN_STAGE_TYPES = {
    "orientation",
    "trace_map",
    "interrogate",
    "experiment",
    "code_reading",
    "rebuild_debug",
    "controlled_failure",
    "transfer_assessment",
    "competency_gate",
    "reflection_adr",
    "flagship_integration",
}
REQUIRED_COVERAGE = {
    "orientation",
    "trace_map",
    "experiment",
    "controlled_failure",
    "transfer_assessment",
    "competency_gate",
}


def _mission() -> dict:
    return json.loads((M01_PACKAGE / "missions" / "M01.json").read_text(encoding="utf-8"))


def test_package_layout_and_identity() -> None:
    assert (M01_PACKAGE / "manifest.json").is_file()
    assert (M01_PACKAGE / "SHA256SUMS").is_file()
    assert (M01_PACKAGE / "missions" / "M01.json").is_file()
    manifest = json.loads((M01_PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "g5.reference.M01"
    assert manifest["version"] == "5.0.0"
    assert manifest["missions"][0]["id"] == "M01"
    assert manifest["missions"][0]["path"] == "missions/M01.json"
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["digest"])


def test_mission_id_matches_wp136_pattern() -> None:
    mission = _mission()
    assert mission["id"] == "M01"
    assert MISSION_ID_RE.fullmatch(mission["id"])
    assert mission["title"]
    assert mission["order_index"] == 1
    assert mission["flagship_version"] == "V00"


def test_mdl_validator_accepts_package() -> None:
    from app.core.mdl_validator import validate_mission, validate_package

    payload = validate_mission(M01_PACKAGE / "missions" / "M01.json", expected_id="M01")
    assert payload["id"] == "M01"
    package = validate_package(M01_PACKAGE)
    assert package.id == "g5.reference.M01"
    assert package.version == "5.0.0"
    assert len(package.missions) == 1
    assert package.missions[0]["id"] == "M01"


def test_authoring_validate_accepts_package() -> None:
    from tools.authoring.validate import validate_package

    result = validate_package(M01_PACKAGE)
    assert result.ok, result.errors
    assert result.schema_engine == "mdl_validator"
    assert result.package_id == "g5.reference.M01"
    assert result.mission_ids == ["M01"]


def test_stage_catalogue_sequencing_and_coverage() -> None:
    mission = _mission()
    stages = mission["stages"]
    ids = [stage["id"] for stage in stages]
    types = [stage["type"] for stage in stages]
    assert ids[0] == "stage_01_orientation"
    assert types[0] == "orientation"
    assert types[-1] == "competency_gate"
    assert set(types) <= FROZEN_STAGE_TYPES
    assert REQUIRED_COVERAGE <= set(types)
    assert "code_reading" in types
    assert types.count("experiment") >= 2
    assert ids.index("stage_02_system_map") < ids.index("stage_03_system_trace")
    assert ids.index("stage_03_system_trace") < ids.index("stage_05_experiment_e1")
    assert ids.index("stage_05_experiment_e1") < ids.index("stage_10_controlled_failure")
    assert ids.index("stage_10_controlled_failure") < ids.index("stage_11_no_ai_gate")
    assert ids.index("stage_11_no_ai_gate") < ids.index("stage_12_transfer")
    assert ids.index("stage_12_transfer") < ids.index("stage_13_gate")


def test_prediction_cycle_on_every_experiment_stage() -> None:
    mission = _mission()
    experiments = [stage for stage in mission["stages"] if stage["type"] == "experiment"]
    assert [stage["id"] for stage in experiments] == [
        "stage_05_experiment_e1",
        "stage_06_experiment_e2",
        "stage_07_experiment_e3",
        "stage_08_experiment_e4",
        "stage_09_experiment_e5",
    ]
    for stage in experiments:
        text = stage["instructions"].lower()
        assert "predict" in text, stage["id"]
        assert "execute" in text or "run" in text, stage["id"]
        assert "submit" in text or "explain" in text, stage["id"]
        assert "commit" in text, stage["id"]
        assert stage["assistance_policy"] == "RESTRICTED_HINTS"
        runner = stage.get("runner") or {}
        assert runner.get("timeout_sec", 0) >= 1


def test_no_ai_required_on_unassisted_stages() -> None:
    mission = _mission()
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_11_no_ai_gate"] == "NO_AI_REQUIRED"
    assert policies["stage_12_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_13_gate"] == "NO_AI_REQUIRED"
    assert "NO_AI_REQUIRED" in policies.values()
    no_ai_blob = json.dumps(
        [stage for stage in mission["stages"] if stage["id"] == "stage_11_no_ai_gate"]
    ).lower()
    assert "invent" in no_ai_blob
    assert "training" in no_ai_blob


def test_gate_contract_evidence_and_targeted_repair() -> None:
    mission = _mission()
    contract = mission["gate_contract"]
    required = contract["required_evidence"]
    assert contract["pass_threshold"] == 1.0
    repair = contract["repair_policy"]
    assert repair["allow_targeted_repair"] is True
    assert repair["max_repair_attempts"] >= 1
    by_stage = {item["stage_id"]: item for item in required}
    assert by_stage["stage_02_system_map"]["artifact_type"] == "diagram"
    assert by_stage["stage_03_system_trace"]["artifact_type"] == "trace"
    assert by_stage["stage_05_experiment_e1"]["artifact_type"] == "metric"
    assert by_stage["stage_10_controlled_failure"]["artifact_type"] == "markdown"
    assert by_stage["stage_11_no_ai_gate"]["artifact_type"] == "diagram"
    assert by_stage["stage_12_transfer"]["artifact_type"] == "artifact"
    comps = mission["competencies"]
    nodes = mission["knowledge_nodes"]
    assert len(comps) == len(nodes)
    for item in required:
        assert item["competency_id"] in comps
        assert item["stage_id"] in {stage["id"] for stage in mission["stages"]}


def test_package_has_no_prefilled_learner_evidence() -> None:
    blob = (M01_PACKAGE / "missions" / "M01.json").read_text(encoding="utf-8").lower()
    for forbidden in (
        "learner_evidence",
        "learner_response",
        "fabricated learner",
        "prefilled",
    ):
        assert forbidden not in blob
    mission = _mission()
    assert "system map" in json.dumps(mission).lower()
    assert "trace" in json.dumps(mission).lower()
    assert "transfer" in json.dumps(mission).lower()


def test_fixture_lives_only_in_allowed_g5_paths() -> None:
    relative = M01_PACKAGE.relative_to(REPO_ROOT)
    assert relative == Path("platform/fixtures/M01")
    tests = REPO_ROOT / "tests" / "platform" / "reference" / "M01"
    assert tests.is_dir()
