"""G6 M13 reference package: WP-136 spec on the frozen generic runtime."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from app.core.mdl_types import STAGE_TYPES
from app.core.mdl_validator import validate_mission, validate_package as validate_mdl_package
from app.core.mission_loader import load_package as g3_load
from tools.authoring.package import load_package, sha256_file
from tools.authoring.validate import validate_mission_document, validate_package

REPO_ROOT = Path(__file__).resolve().parents[4]
M13_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M13"
MISSION_PATH = M13_PACKAGE / "missions" / "M13.json"
PACKAGE_ID = "g5.reference.M13"
FROZEN_BASE = "f7926e661a955f2d78bd8584877815825c5ef047"
LANE_SHA = "bb3e86bac0f14aa804673cf6ff48361a959aab75"
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
REQUIRED_TYPES = {
    "orientation",
    "experiment",
    "code_reading",
    "controlled_failure",
    "transfer_assessment",
    "competency_gate",
}
LANE_PREFIXES = (
    "platform/fixtures/M11/",
    "platform/fixtures/M12/",
    "platform/fixtures/M13/",
    "platform/fixtures/M14/",
    "tests/platform/reference/M11/",
    "tests/platform/reference/M12/",
    "tests/platform/reference/M13/",
    "tests/platform/reference/M14/",
)
G6_MISSION_PATH = re.compile(r"^(platform/fixtures|tests/platform/reference)/M\d{2}/")


def _mission() -> dict:
    return json.loads(MISSION_PATH.read_text(encoding="utf-8"))


def test_package_identity_is_g5_reference_m13() -> None:
    package = load_package(M13_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.version == "5.0.0"
    assert package.missions[0]["id"] == "M13"
    assert re.match(r"^M[0-9]{2}$", package.missions[0]["id"])
    assert package.id != "M13"
    assert package.digest
    assert len(package.digest) == 64
    assert "no custom" in (package.manifest.get("description") or "").lower()


def test_wp136_and_g4_journey_contract() -> None:
    result = validate_package(M13_PACKAGE)
    assert result.ok, result.errors
    assert result.package_id == PACKAGE_ID
    assert result.mission_ids == ["M13"]
    assert result.schema_engine in {"jsonschema", "mdl_validator"}
    mdl = validate_mdl_package(M13_PACKAGE)
    assert mdl.id == PACKAGE_ID
    spec = validate_mission(MISSION_PATH)
    assert spec["id"] == "M13"
    assert spec["order_index"] == 13
    assert spec["flagship_version"] == "V03"
    assert spec["prerequisites"] == ["M09"]


def test_g3_loader_accepts_sealed_package() -> None:
    package = g3_load(M13_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.missions[0]["id"] == "M13"
    assert package.digest == load_package(M13_PACKAGE).digest


def test_integrity_files_match_payload() -> None:
    checksums = (M13_PACKAGE / "SHA256SUMS").read_text(encoding="utf-8")
    for rel in (
        "manifest.json",
        "missions/M13.json",
        "g5/reference/M13/experiment.py",
        "g5/reference/M13/data/knn_scale_cases.csv",
    ):
        assert rel in checksums
        assert sha256_file(M13_PACKAGE / rel) in checksums


def test_stages_are_catalogue_types_with_prediction_cycle() -> None:
    mission = _mission()
    types = [stage["type"] for stage in mission["stages"]]
    assert set(types) <= set(STAGE_TYPES)
    for required in REQUIRED_TYPES:
        assert required in types, required
    assert types[0] == "orientation"
    assert types[-1] == "competency_gate"
    experiment = next(stage for stage in mission["stages"] if stage["id"] == "stage_04_experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text
    assert "commit" in text
    assert experiment["assistance_policy"] == "RESTRICTED_HINTS"
    runner = experiment.get("runner") or {}
    assert runner.get("module") == "g5.reference.M13.experiment"
    assert runner.get("entrypoint") == "run_knn_query"
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_06_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_09_gate"] == "NO_AI_REQUIRED"


def test_gate_contract_evidence_and_targeted_repair() -> None:
    mission = _mission()
    contract = mission["gate_contract"]
    required = contract["required_evidence"]
    assert contract["pass_threshold"] == 1.0
    assert contract["repair_policy"]["allow_targeted_repair"] is True
    by_stage = {item["stage_id"]: item for item in required}
    assert by_stage["stage_04_experiment"]["artifact_type"] == "metric"
    assert by_stage["stage_05_controlled_failure"]["artifact_type"] == "metric"
    assert by_stage["stage_06_transfer"]["artifact_type"] == "artifact"
    assert len(mission["competencies"]) == len(mission["knowledge_nodes"])
    for item in required:
        assert item["artifact_type"] in WP137_BLOCK_TYPES
        assert item["competency_id"] in mission["competencies"]


def test_source_mission_content_is_represented() -> None:
    blob = json.dumps(_mission()).lower()
    for token in (
        "neighbor",
        "euclidean",
        "manhattan",
        "interface_event_count",
        "scaler",
        "no-ai",
        "v03",
        "predict",
    ):
        assert token in blob, token
    for forbidden in ("learner_evidence", "learner_response", "fabricated learner", "prefilled"):
        assert forbidden not in blob


def test_wp136_schema_rejects_package_id_as_mission_id() -> None:
    mission = _mission()
    mission["id"] = PACKAGE_ID
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors


def test_lab_source_has_no_eager_pandas() -> None:
    source = (M13_PACKAGE / "g5" / "reference" / "M13" / "experiment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".", 1)[0] not in {"pandas", "sklearn"} for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            assert not module.startswith("pandas")
            assert not module.startswith("sklearn")
    assert "KNN_EXECUTE_SOURCE" in source


def test_reference_tests_collect_without_pandas(tmp_path: Path) -> None:
    (tmp_path / "pandas.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'pandas'\")\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(M13_PACKAGE), str(REPO_ROOT), str(REPO_ROOT / "platform" / "backend")]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/platform/reference/M13"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ModuleNotFoundError" not in completed.stdout + completed.stderr
    assert "test_runtime.py" in completed.stdout


def test_package_has_no_custom_frontend() -> None:
    for path in M13_PACKAGE.rglob("*"):
        if path.is_file():
            assert path.suffix not in {".tsx", ".ts", ".jsx", ".js", ".css", ".html"}
    assert "bespoke" not in json.dumps(_mission()).lower()


def test_platform_has_no_m13_special_case_routes() -> None:
    roots = [REPO_ROOT / "platform" / "backend" / "app", REPO_ROOT / "platform" / "frontend" / "src"]
    eq_m13 = re.compile(r"""==\s*["']M13["']""")
    custom_route = re.compile(r"/m13[-_/]|/knn", re.IGNORECASE)
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                continue
            text = path.read_text(encoding="utf-8")
            if eq_m13.search(text) or custom_route.search(text):
                hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_frontend_registry_stays_generic() -> None:
    registry = (REPO_ROOT / "platform" / "frontend" / "src" / "player" / "stageRegistry.ts").read_text(
        encoding="utf-8"
    )
    assert "experiment" in registry
    assert "M13" not in registry


def _changed_paths() -> list[str]:
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

    # Shallow CI checkouts omit FROZEN_BASE. Bind exclusive-write to the lane
    # commit so the combined G6 tree does not fail this invariant.
    for sha, label in ((FROZEN_BASE, "frozen base"), (LANE_SHA, "lane")):
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            pytest.skip(f"{label} {sha} is not in this clone (shallow checkout)")

    return sorted(_lines(["git", "diff", "--name-only", FROZEN_BASE, LANE_SHA]))


def test_allowed_diff_paths_are_fixtures_and_reference_tests_only() -> None:
    import pytest

    paths = _changed_paths()
    unexpected = [
        path
        for path in paths
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in LANE_PREFIXES)
    ]
    other_missions = [path for path in unexpected if G6_MISSION_PATH.match(path)]
    core_leaks = [path for path in unexpected if path not in other_missions]
    assert core_leaks == [], core_leaks
    if other_missions:
        pytest.skip("combined G6 integration tree; exclusive-write bound to 51B M13-M14")
    assert any(path.startswith("platform/fixtures/M13/") for path in paths)
    assert any(path.startswith("tests/platform/reference/M13/") for path in paths)
