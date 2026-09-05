"""G5 M25 reference package: WP-136 spec, WP-137 rubric, generic runtime only."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tools.authoring.package import load_package
from tools.authoring.validate import validate_mission_document, validate_package

REPO_ROOT = Path(__file__).resolve().parents[4]
M25_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M25"
FROZEN_BASE = "f7926e661a955f2d78bd8584877815825c5ef047"
PACKAGE_ID = "g5.reference.M25"
MISSION_ID = "M25"
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
LANE_PREFIXES = tuple(
    f"platform/fixtures/M{n:02d}/" for n in range(21, 27)
) + tuple(f"tests/platform/reference/M{n:02d}/" for n in range(21, 27))


def _mission() -> dict:
    return json.loads((M25_PACKAGE / "missions" / "M25.json").read_text(encoding="utf-8"))


def test_package_identity_is_g5_reference_m25() -> None:
    package = load_package(M25_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.version == "5.0.0"
    assert package.missions[0]["id"] == MISSION_ID
    assert re.match(r"^M[0-9]{2}$", package.missions[0]["id"])
    assert package.manifest.get("schema") == "learningos.mission.v1"
    assert "custom frontend" in (package.manifest.get("description") or "").lower()
    assert package.id != MISSION_ID
    assert len(package.digest) == 64


def test_wp136_id_is_m25_not_package_id() -> None:
    mission = _mission()
    assert mission["id"] == MISSION_ID
    assert mission["id"] != PACKAGE_ID
    assert mission["order_index"] == 25
    assert mission["flagship_version"] == "V05"
    assert mission["phase"]["id"] == "P4"
    assert mission["prerequisites"] == ['M24']
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors == []


def test_package_validates_against_wp136_and_g4_journey() -> None:
    result = validate_package(M25_PACKAGE)
    assert result.ok, result.errors
    assert result.package_id == PACKAGE_ID
    assert result.mission_ids == [MISSION_ID]
    mission = result.package.missions[0]  # type: ignore[union-attr]
    types = [stage["type"] for stage in mission["stages"]]
    assert types[0] == "orientation"
    assert types[-1] == "competency_gate"
    assert "experiment" in types
    assert "controlled_failure" in types
    assert "transfer_assessment" in types
    assert "code_reading" in types
    assert set(types) <= CATALOGUE_STAGE_TYPES
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text


def test_mdl_validator_accepts_sealed_package() -> None:
    from app.core.mdl_validator import validate_mission, validate_package as mdl_validate_package
    from app.core.mission_loader import load_package as g3_load

    payload = validate_mission(M25_PACKAGE / "missions" / "M25.json")
    assert payload["id"] == MISSION_ID
    package = mdl_validate_package(M25_PACKAGE)
    assert package.id == PACKAGE_ID
    loaded = g3_load(M25_PACKAGE)
    assert loaded.id == PACKAGE_ID
    assert loaded.missions[0]["id"] == MISSION_ID


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
    assert "metric" in required
    assert "table" in required


def test_experiment_is_catalogue_oriented_on_generic_runner() -> None:
    mission = _mission()
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    runner = experiment["runner"]
    assert runner["module"] == "g5.reference.M25.experiment"
    assert runner["entrypoint"] == "run_pytorch_training"
    assert experiment["validation_rubric"]["required_evidence_type"] == "metric"
    assert experiment["assistance_policy"] == "RESTRICTED_HINTS"
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_04_controlled_failure"] == "SOCRATIC_ONLY"
    assert policies["stage_05_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_07_gate"] == "NO_AI_REQUIRED"
    ids = [stage["id"] for stage in mission["stages"]]
    assert ids == [
        "stage_01_orientation",
        "stage_02_experiment",
        "stage_03_code_reading",
        "stage_04_controlled_failure",
        "stage_05_transfer",
        "stage_06_adr",
        "stage_07_gate",
    ]
    assert mission["stages"][5]["type"] == "reflection_adr"


def test_gate_contract_evidence_and_targeted_repair() -> None:
    mission = _mission()
    contract = mission["gate_contract"]
    required = contract["required_evidence"]
    assert contract["pass_threshold"] == 1.0
    repair = contract["repair_policy"]
    assert repair["allow_targeted_repair"] is True
    assert repair["max_repair_attempts"] >= 1
    by_stage = {item["stage_id"]: item for item in required}
    assert by_stage["stage_01_orientation"]["artifact_type"] == "markdown"
    assert by_stage["stage_02_experiment"]["artifact_type"] == "metric"
    assert by_stage["stage_04_controlled_failure"]["artifact_type"] == "table"
    assert by_stage["stage_05_transfer"]["artifact_type"] == "metric"
    comps = mission["competencies"]
    nodes = mission["knowledge_nodes"]
    assert len(comps) == len(nodes)
    assert set(comps) == {'comp.nn.autograd_mechanics', 'comp.nn.gradient_debugging', 'comp.nn.train_eval_boundary', 'comp.nn.training_loop'}
    for item in required:
        assert item["competency_id"] in comps
        assert item["stage_id"] in {stage["id"] for stage in mission["stages"]}


def test_package_has_no_prefilled_learner_evidence() -> None:
    blob = (M25_PACKAGE / "missions" / "M25.json").read_text(encoding="utf-8").lower()
    for forbidden in (
        "learner_evidence",
        "learner_response",
        "fabricated learner",
        "prefilled",
    ):
        assert forbidden not in blob


def test_source_mission_content_is_represented() -> None:
    mission = json.dumps(_mission()).lower()
    for token in ('predict', 'zero_grad', 'autograd', 'eval', 'checkpoint', 'no custom', 'v05', 'pytorch'):
        assert token in mission, token


def test_package_has_no_custom_frontend_or_runtime_payload() -> None:
    for path in M25_PACKAGE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in {".tsx", ".ts", ".jsx", ".js", ".css", ".html", ".py"}
        assert path.name not in {"index.html", "App.tsx"}
        tree_ok = True
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    assert all(alias.name.split(".", 1)[0] != "pandas" for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    assert not str(node.module or "").startswith("pandas")
            tree_ok = tree_ok and True
        assert tree_ok
    mission = json.dumps(_mission()).lower()
    assert "/missions/m25/" not in mission
    assert "bespoke" not in mission


def test_no_eager_pandas_import_in_fixture_or_tests() -> None:
    roots = [M25_PACKAGE, Path(__file__).resolve().parent]
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    assert all(alias.name.split(".", 1)[0] != "pandas" for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    assert not str(node.module or "").startswith("pandas")


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
            "tests/platform/reference/M25",
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


def test_platform_has_no_m25_special_case_routes_or_conditionals() -> None:
    platform = REPO_ROOT / "platform"
    skip_dirs = {"node_modules", "dist", "__pycache__", ".git"}
    eq_mid = re.compile(r"""mission_id\s*==\s*["']M25["']""")
    custom_route = re.compile(r"/m25[-_/]|/pytorch-training", re.IGNORECASE)
    hits: list[str] = []
    for path in platform.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if "fixtures" in path.parts and "M25" in path.parts:
            continue
        if path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".css", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        if eq_mid.search(text) or custom_route.search(text):
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_frontend_registry_stays_generic() -> None:
    registry = (REPO_ROOT / "platform" / "frontend" / "src" / "player" / "stageRegistry.ts").read_text(
        encoding="utf-8"
    )
    assert "experiment" in registry
    assert "M25" not in registry
    assert "pytorch_training_loop" not in registry.lower()
    workbench = (REPO_ROOT / "platform" / "frontend" / "src" / "workbench" / "types.ts").read_text(
        encoding="utf-8"
    )
    for block_type in ("chart", "metric", "table", "trace"):
        assert f"'{block_type}'" in workbench or f'"{block_type}"' in workbench


def test_wp136_schema_rejects_package_id_as_mission_id() -> None:
    mission = _mission()
    mission["id"] = PACKAGE_ID
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors
    assert any("does not match" in item or "M[0-9]" in item or "id" in item.lower() for item in errors)


def _lane_commits() -> list[str]:
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

    listed = subprocess.run(
        [
            "git",
            "rev-list",
            "--no-merges",
            f"{FROZEN_BASE}..HEAD",
            "--",
            f"platform/fixtures/M25",
            f"tests/platform/reference/M25",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        pytest.skip(
            "git history for path-ownership check is unavailable "
            f"(exit {listed.returncode}: {(listed.stderr or listed.stdout).strip()[:200]})"
        )
    shas = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not shas:
        pytest.skip(f"no lane commits for M25 versus frozen base in this clone")
    return shas


def test_allowed_diff_paths_are_fixtures_and_reference_tests_only() -> None:
    shas = _lane_commits()
    unexpected: list[str] = []
    saw_fixture = False
    saw_tests = False
    for sha in shas:
        completed = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            import pytest

            pytest.skip(
                "git diff-tree for path-ownership check is unavailable "
                f"(exit {completed.returncode})"
            )
        for path in completed.stdout.splitlines():
            path = path.strip()
            if not path:
                continue
            if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in LANE_PREFIXES):
                unexpected.append(path)
            if path.startswith("platform/fixtures/M25/"):
                saw_fixture = True
            if path.startswith("tests/platform/reference/M25/"):
                saw_tests = True
    assert unexpected == [], unexpected
    assert saw_fixture
    assert saw_tests
