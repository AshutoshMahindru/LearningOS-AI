from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tools.authoring.package import load_package, sha256_file
from tools.authoring.validate import validate_mission_document, validate_package

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M09"
MISSION_PATH = PACKAGE / "missions" / "M09.json"
PACKAGE_ID = "g6.reference.M09"
PACKAGE_VERSION = "6.0.0"
MISSION_ID = "M09"
FROZEN_BASE = "f7926e661a955f2d78bd8584877815825c5ef047"
ORDER_INDEX = 9
FLAGSHIP = "V02"
PHASE_ID = "P2"
TITLE = "Make Binary Decisions"
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
    "experiment",
    "controlled_failure",
    "transfer_assessment",
    "competency_gate",
}
WP137_BLOCK_TYPES = {
    "table",
    "chart",
    "trace",
    "state_diff",
    "diagram",
    "markdown",
    "metric",
    "artifact",
}
LANE_PREFIXES = (
    "platform/fixtures/M06/",
    "platform/fixtures/M07/",
    "platform/fixtures/M08/",
    "platform/fixtures/M09/",
    "platform/fixtures/M10/",
    "tests/platform/reference/M06/",
    "tests/platform/reference/M07/",
    "tests/platform/reference/M08/",
    "tests/platform/reference/M09/",
    "tests/platform/reference/M10/",
)
OWN_PREFIXES = (
    "platform/fixtures/M09/",
    "tests/platform/reference/M09/",
)
SOURCE_TOKENS = ["threshold", "precision", "recall", "confusion", "probability", "0.50", "disengage", "v02"]


def _mission() -> dict:
    return json.loads(MISSION_PATH.read_text(encoding="utf-8"))


def test_package_layout_and_identity() -> None:
    assert (PACKAGE / "manifest.json").is_file()
    assert (PACKAGE / "SHA256SUMS").is_file()
    assert MISSION_PATH.is_file()
    package = load_package(PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.version == PACKAGE_VERSION
    assert package.missions[0]["id"] == MISSION_ID
    assert re.fullmatch(r"^M[0-9]{2}$", package.missions[0]["id"])
    assert package.id != MISSION_ID
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == PACKAGE_ID
    assert manifest["version"] == PACKAGE_VERSION
    assert manifest["schema"] == "learningos.mission.v1"
    assert manifest["missions"][0]["id"] == MISSION_ID
    assert manifest["missions"][0]["path"] == "missions/M09.json"
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["digest"])
    assert package.digest == manifest["digest"]


def test_mission_id_matches_wp136_pattern() -> None:
    mission = _mission()
    assert mission["id"] == MISSION_ID
    assert mission["title"] == TITLE
    assert mission["order_index"] == ORDER_INDEX
    assert mission["flagship_version"] == FLAGSHIP
    assert mission["phase"]["id"] == PHASE_ID
    assert mission["id"] != PACKAGE_ID


def test_mdl_validator_accepts_package() -> None:
    from app.core.mdl_validator import validate_mission, validate_package as mdl_validate_package
    from app.core.mission_loader import load_package as g3_load

    payload = validate_mission(MISSION_PATH, expected_id=MISSION_ID)
    assert payload["id"] == MISSION_ID
    package = mdl_validate_package(PACKAGE)
    assert package.id == PACKAGE_ID
    loaded = g3_load(PACKAGE)
    assert loaded.id == PACKAGE_ID
    assert loaded.missions[0]["id"] == MISSION_ID


def test_authoring_validate_accepts_package() -> None:
    result = validate_package(PACKAGE)
    assert result.ok, result.errors
    assert result.schema_engine in {"jsonschema", "mdl_validator"}
    assert result.package_id == PACKAGE_ID
    assert result.mission_ids == [MISSION_ID]


def test_integrity_files_match_payload() -> None:
    checksums = (PACKAGE / "SHA256SUMS").read_text(encoding="utf-8")
    assert "manifest.json" in checksums
    assert "missions/M09.json" in checksums
    assert sha256_file(PACKAGE / "manifest.json") in checksums
    assert sha256_file(MISSION_PATH) in checksums


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
    assert "experiment" in types
    assert "controlled_failure" in types
    assert "transfer_assessment" in types
    experiment_ids = [stage["id"] for stage in stages if stage["type"] == "experiment"]
    failure_ids = [stage["id"] for stage in stages if stage["type"] == "controlled_failure"]
    transfer_ids = [stage["id"] for stage in stages if stage["type"] == "transfer_assessment"]
    gate_ids = [stage["id"] for stage in stages if stage["type"] == "competency_gate"]
    assert ids.index(experiment_ids[0]) < ids.index(failure_ids[0])
    assert ids.index(failure_ids[0]) < ids.index(transfer_ids[0])
    assert ids.index(transfer_ids[0]) < ids.index(gate_ids[0])


def test_prediction_cycle_on_every_experiment_stage() -> None:
    mission = _mission()
    experiments = [stage for stage in mission["stages"] if stage["type"] == "experiment"]
    assert experiments
    for stage in experiments:
        text = stage["instructions"].lower()
        assert "predict" in text, stage["id"]
        assert "execute" in text or "run" in text, stage["id"]
        assert "submit" in text or "explain" in text, stage["id"]
        assert "commit" in text, stage["id"]
        assert stage["assistance_policy"] == "RESTRICTED_HINTS"
        runner = stage.get("runner") or {}
        assert runner.get("module", "").startswith("g6.reference.M09.")
        assert runner.get("timeout_sec", 0) >= 1
        rubric = stage.get("validation_rubric") or {}
        assert rubric.get("required_evidence_type") in WP137_BLOCK_TYPES


def test_no_ai_required_on_unassisted_stages() -> None:
    mission = _mission()
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert "NO_AI_REQUIRED" in policies.values()
    for stage in mission["stages"]:
        if stage["type"] in {"transfer_assessment", "competency_gate"}:
            assert stage["assistance_policy"] == "NO_AI_REQUIRED", stage["id"]
    transfer = next(stage for stage in mission["stages"] if stage["type"] == "transfer_assessment")
    blob = json.dumps(transfer).lower()
    assert "invent" in blob or "without" in blob


def test_gate_contract_evidence_and_targeted_repair() -> None:
    mission = _mission()
    contract = mission["gate_contract"]
    required = contract["required_evidence"]
    assert contract["pass_threshold"] == 1.0
    repair = contract["repair_policy"]
    assert repair["allow_targeted_repair"] is True
    assert repair["max_repair_attempts"] >= 1
    stage_ids = {stage["id"] for stage in mission["stages"]}
    comps = mission["competencies"]
    nodes = mission["knowledge_nodes"]
    assert len(comps) == len(nodes)
    for item in required:
        assert item["competency_id"] in comps
        assert item["stage_id"] in stage_ids
        assert item["artifact_type"] in WP137_BLOCK_TYPES


def test_spec_and_rubric_use_wp137_block_types() -> None:
    mission = _mission()
    for stage in mission["stages"]:
        rubric = stage.get("validation_rubric") or {}
        evidence_type = rubric.get("required_evidence_type")
        if evidence_type:
            assert evidence_type in WP137_BLOCK_TYPES, evidence_type


def test_package_has_no_prefilled_learner_evidence() -> None:
    blob = MISSION_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "learner_evidence",
        "learner_response",
        "fabricated learner",
        "prefilled",
    ):
        assert forbidden not in blob
    mission = json.dumps(_mission()).lower()
    for token in SOURCE_TOKENS:
        assert token in mission, token


def test_wp136_schema_rejects_package_id_as_mission_id() -> None:
    mission = _mission()
    mission["id"] = PACKAGE_ID
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors
    assert any("does not match" in item or "M[0-9]" in item or PACKAGE_ID in item for item in errors)


def test_package_has_no_custom_frontend_or_runtime_payload() -> None:
    for path in PACKAGE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in {".tsx", ".ts", ".jsx", ".js", ".css", ".html", ".py"}
    mission = json.dumps(_mission()).lower()
    assert "/missions/m09/" not in mission
    assert "bespoke" not in mission


def test_platform_has_no_mission_special_case_routes() -> None:
    roots = [
        REPO_ROOT / "platform" / "backend" / "app",
        REPO_ROOT / "platform" / "frontend" / "src",
    ]
    skip_dirs = {"node_modules", "dist", "__pycache__"}
    eq_mission = re.compile(r"""==\s*["']M09["']""")
    route_mission = re.compile(r"/missions/M09")
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or any(part in skip_dirs for part in path.parts):
                continue
            if path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                continue
            text = path.read_text(encoding="utf-8")
            if eq_mission.search(text) or route_mission.search(text):
                hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_fixture_lives_only_in_allowed_g6_paths() -> None:
    relative = PACKAGE.relative_to(REPO_ROOT)
    assert relative == Path("platform/fixtures/M09")
    tests = REPO_ROOT / "tests" / "platform" / "reference" / "M09"
    assert tests.is_dir()


def test_reference_tests_collect_without_pandas(tmp_path: Path) -> None:
    (tmp_path / "pandas.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'pandas'\")\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(REPO_ROOT), str(REPO_ROOT / "platform" / "backend")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/platform/reference/M09",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ModuleNotFoundError" not in completed.stdout + completed.stderr
    assert "test_runtime.py" in completed.stdout


def _changed_paths() -> list[str]:
    import pytest

    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{FROZEN_BASE}^{{commit}}"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"frozen base {FROZEN_BASE} is not in this clone (shallow checkout)")

    def _lines(args: list[str]) -> list[str]:
        completed = subprocess.run(
            args,
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(
                "git history for path-ownership check is unavailable "
                f"(exit {completed.returncode}: {(completed.stderr or completed.stdout).strip()[:200]})"
            )
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    tracked = _lines(["git", "diff", "--name-only", FROZEN_BASE])
    untracked = _lines(["git", "ls-files", "--others", "--exclude-standard"])
    return sorted(set(tracked + untracked))


def test_allowed_diff_paths_are_fixtures_and_reference_tests_only() -> None:
    paths = _changed_paths()
    if not paths:
        import pytest

        pytest.skip("no diff versus frozen base yet")
    unexpected = [
        path
        for path in paths
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in LANE_PREFIXES)
    ]
    assert unexpected == [], unexpected
    assert any(path.startswith(OWN_PREFIXES[0]) for path in paths)
    assert any(path.startswith(OWN_PREFIXES[1]) for path in paths)
