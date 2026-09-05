"""G5 M05 reference package: WP-136 spec, WP-137 rubric, generic runtime only."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tools.authoring.package import load_package
from tools.authoring.validate import validate_mission_document, validate_package

REPO_ROOT = Path(__file__).resolve().parents[4]
M05_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M05"
FROZEN_BASE = "407e8199d457c57bcb3b5703add7872ddc8d7854"
M05_LANE_SHA = "8f0ceac5508075d1e7a0587aca0898bf83487f0d"
PACKAGE_ID = "g5.reference.M05"
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
CATALOGUE_STAGE_TYPES = {
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


def _mission() -> dict:
    return json.loads((M05_PACKAGE / "missions" / "M05.json").read_text(encoding="utf-8"))


def test_package_identity_is_g5_reference_m05() -> None:
    package = load_package(M05_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.version == "5.0.0"
    assert package.missions[0]["id"] == "M05"
    assert re.match(r"^M[0-9]{2}$", package.missions[0]["id"])
    assert package.manifest.get("schema") == "learningos.mission.v1"
    assert "custom frontend" in (package.manifest.get("description") or "").lower()


def test_wp136_id_is_m05_not_package_id() -> None:
    mission = _mission()
    assert mission["id"] == "M05"
    assert mission["id"] != PACKAGE_ID
    assert mission["order_index"] == 5
    assert mission["flagship_version"] == "V01"
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors == []


def test_package_validates_against_wp136_and_g4_journey() -> None:
    result = validate_package(M05_PACKAGE)
    assert result.ok, result.errors
    assert result.package_id == PACKAGE_ID
    assert result.mission_ids == ["M05"]
    mission = result.package.missions[0]  # type: ignore[union-attr]
    types = [stage["type"] for stage in mission["stages"]]
    assert "orientation" in types
    assert "experiment" in types
    assert "transfer_assessment" in types
    assert "competency_gate" in types
    assert set(types) <= CATALOGUE_STAGE_TYPES
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text
    assert "chart" in text


def test_mdl_validator_accepts_sealed_package() -> None:
    from app.core.mdl_validator import validate_mission, validate_package as mdl_validate_package
    from app.core.mission_loader import load_package as g3_load

    payload = validate_mission(M05_PACKAGE / "missions" / "M05.json")
    assert payload["id"] == "M05"
    package = mdl_validate_package(M05_PACKAGE)
    assert package.id == PACKAGE_ID
    loaded = g3_load(M05_PACKAGE)
    assert loaded.id == PACKAGE_ID
    assert loaded.missions[0]["id"] == "M05"


def test_spec_and_rubric_use_wp137_block_types() -> None:
    mission = _mission()
    required: list[str] = []
    for stage in mission["stages"]:
        rubric = stage.get("validation_rubric") or {}
        evidence_type = rubric.get("required_evidence_type")
        if evidence_type:
            required.append(evidence_type)
            assert evidence_type in WP137_BLOCK_TYPES, evidence_type
    for item in mission["gate_contract"]["required_evidence"]:
        artifact_type = item["artifact_type"]
        required.append(artifact_type)
        assert artifact_type in WP137_BLOCK_TYPES, artifact_type
    assert "chart" in required
    assert "metric" in required
    assert "table" in required
    blob = json.dumps(mission).lower()
    assert "chart" in blob
    assert "timing" in blob
    assert "benchmark" in blob


def test_experiment_is_chart_oriented_on_generic_runner() -> None:
    mission = _mission()
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    runner = experiment["runner"]
    assert runner["module"] == "g5.reference.M05.experiment"
    assert runner["entrypoint"] == "run_timing_benchmark"
    assert experiment["validation_rubric"]["required_evidence_type"] == "chart"
    assert experiment["assistance_policy"] == "RESTRICTED_HINTS"
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_03_controlled_failure"] == "SOCRATIC_ONLY"
    assert policies["stage_04_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_05_gate"] == "NO_AI_REQUIRED"


def test_package_has_no_custom_frontend_or_runtime_payload() -> None:
    for path in M05_PACKAGE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in {".tsx", ".ts", ".jsx", ".js", ".css", ".html", ".py"}
        assert path.name not in {"index.html", "App.tsx"}
    mission = json.dumps(_mission()).lower()
    assert "/missions/m05/chart" not in mission
    assert "custom frontend" not in mission or "no custom" in mission
    assert "bespoke" not in mission


def test_platform_has_no_m05_special_case_routes_or_conditionals() -> None:
    platform = REPO_ROOT / "platform"
    skip_dirs = {"node_modules", "dist", "__pycache__", ".git"}
    eq_m05 = re.compile(r"""mission_id\s*==\s*["']M05["']""")
    custom_route = re.compile(r"/m05[-_/]|/array-vectorization", re.IGNORECASE)
    hits: list[str] = []
    for path in platform.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if "fixtures" in path.parts and "M05" in path.parts:
            continue
        if path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".css", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        if eq_m05.search(text) or custom_route.search(text):
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_frontend_registry_stays_generic() -> None:
    registry = (REPO_ROOT / "platform" / "frontend" / "src" / "player" / "stageRegistry.ts").read_text(
        encoding="utf-8"
    )
    assert "experiment" in registry
    assert "M05" not in registry
    assert "array_vectorization" not in registry.lower()
    workbench = (REPO_ROOT / "platform" / "frontend" / "src" / "workbench" / "types.ts").read_text(
        encoding="utf-8"
    )
    for block_type in ("chart", "metric", "table"):
        assert f"'{block_type}'" in workbench or f'"{block_type}"' in workbench


def test_wp136_schema_rejects_package_id_as_mission_id() -> None:
    mission = _mission()
    mission["id"] = PACKAGE_ID
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors
    assert any("does not match" in item or "M[0-9]" in item or "id" in item.lower() for item in errors)


def _changed_paths() -> list[str]:
    import subprocess

    import pytest

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

    # Shallow CI checkouts (fetch-depth: 1) do not contain FROZEN_BASE.
    for sha, label in ((FROZEN_BASE, "frozen base"), (M05_LANE_SHA, "M05 lane")):
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            pytest.skip(f"{label} {sha} is not in this clone (shallow checkout)")

    # Bind the exclusive-write check to the M05 lane commit, not HEAD, so the
    # combined G5 integration tree can load M01–M04 without failing this invariant.
    return sorted(_lines(["git", "diff", "--name-only", FROZEN_BASE, M05_LANE_SHA]))


def test_allowed_diff_paths_are_fixtures_and_reference_tests_only() -> None:
    paths = _changed_paths()
    allowed_prefixes = (
        "platform/fixtures/M05/",
        "tests/platform/reference/M05/",
    )
    unexpected = [
        path
        for path in paths
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in allowed_prefixes)
    ]
    assert unexpected == [], unexpected
    assert any(path.startswith("platform/fixtures/M05/") for path in paths)
    assert any(path.startswith("tests/platform/reference/M05/") for path in paths)
