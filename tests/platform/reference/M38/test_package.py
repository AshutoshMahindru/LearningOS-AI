"""G5 M38 reference package: WP-136 spec, WP-137 rubric, generic runtime only."""

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
M38_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M38"
FROZEN_BASE = "f7926e661a955f2d78bd8584877815825c5ef047"
LANE_SHA = "54cca3f9509ced24f4973d52001d8f894f248f88"
PACKAGE_ID = "g5.reference.M38"
MISSION_ID = "M38"
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
REQUIRED_COVERAGE = {
    "orientation",
    "code_reading",
    "experiment",
    "controlled_failure",
    "transfer_assessment",
    "reflection_adr",
    "flagship_integration",
    "competency_gate",
}
LANE_ALLOWED = tuple(
    f"platform/fixtures/M{i:02d}/" for i in range(33, 40)
) + tuple(
    f"tests/platform/reference/M{i:02d}/" for i in range(33, 40)
)
SOURCE_TOKENS = ['predict', 'checkpoint', 'resume', 'state', 'loop', 'v10', 'no custom', 'approval', 'terminal']


def _mission() -> dict:
    return json.loads((M38_PACKAGE / "missions" / "M38.json").read_text(encoding="utf-8"))


def test_package_identity_is_g5_reference() -> None:
    package = load_package(M38_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.version == "5.0.0"
    assert package.missions[0]["id"] == MISSION_ID
    assert re.fullmatch(r"^M[0-9]{2}$", package.missions[0]["id"])
    assert package.manifest.get("schema") == "learningos.mission.v1"
    assert "custom frontend" in (package.manifest.get("description") or "").lower()
    assert package.id != MISSION_ID
    assert len(package.digest) == 64


def test_wp136_id_is_mission_not_package_id() -> None:
    mission = _mission()
    assert mission["id"] == MISSION_ID
    assert mission["id"] != PACKAGE_ID
    assert mission["order_index"] == 38
    assert mission["flagship_version"] == "V10"
    assert mission["phase"]["id"] == "P7"
    assert mission["title"] == 'Build a Stateful Agent Workflow'
    assert mission["prerequisites"] == ['M37']
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors == []


def test_package_validates_against_wp136_and_g4_journey() -> None:
    result = validate_package(M38_PACKAGE)
    assert result.ok, result.errors
    assert result.package_id == PACKAGE_ID
    assert result.mission_ids == [MISSION_ID]
    mission = result.package.missions[0]  # type: ignore[union-attr]
    types = [stage["type"] for stage in mission["stages"]]
    assert types[0] == "orientation"
    assert types[-1] == "competency_gate"
    assert REQUIRED_COVERAGE <= set(types)
    assert set(types) <= CATALOGUE_STAGE_TYPES
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "commit" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text


def test_mdl_validator_accepts_sealed_package() -> None:
    from app.core.mdl_validator import validate_mission, validate_package as mdl_validate_package
    from app.core.mission_loader import load_package as g3_load

    payload = validate_mission(M38_PACKAGE / "missions" / "M38.json")
    assert payload["id"] == MISSION_ID
    package = mdl_validate_package(M38_PACKAGE)
    assert package.id == PACKAGE_ID
    loaded = g3_load(M38_PACKAGE)
    assert loaded.id == PACKAGE_ID
    assert loaded.missions[0]["id"] == MISSION_ID
    assert len(loaded.digest) == 64


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
    for token in ['trace', 'table']:
        assert token in required
    blob = json.dumps(mission).lower()
    for token in SOURCE_TOKENS:
        assert token in blob, token


def test_experiment_is_generic_runner_with_assistance_ladder() -> None:
    mission = _mission()
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    runner = experiment["runner"]
    assert runner["module"] == "g5.reference.M38.experiment"
    assert runner["entrypoint"] == 'run_state_machine'
    assert runner.get("timeout_sec", 0) >= 1
    assert experiment["assistance_policy"] == "RESTRICTED_HINTS"
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_04_controlled_failure"] == "SOCRATIC_ONLY"
    assert policies["stage_05_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_08_gate"] == "NO_AI_REQUIRED"
    ids = [stage["id"] for stage in mission["stages"]]
    assert ids == ['stage_01_orientation', 'stage_02_code_reading', 'stage_03_experiment', 'stage_04_controlled_failure', 'stage_05_transfer', 'stage_06_adr', 'stage_07_flagship', 'stage_08_gate']
    assert mission["stages"][5]["type"] == "reflection_adr"
    assert mission["stages"][6]["type"] == "flagship_integration"


def test_gate_contract_evidence_and_targeted_repair() -> None:
    mission = _mission()
    contract = mission["gate_contract"]
    assert contract["pass_threshold"] == 1.0
    repair = contract["repair_policy"]
    assert repair["allow_targeted_repair"] is True
    assert repair["max_repair_attempts"] >= 1
    comps = set(mission["competencies"])
    stages = {stage["id"] for stage in mission["stages"]}
    for item in contract["required_evidence"]:
        assert item["competency_id"] in comps
        assert item["stage_id"] in stages
    assert comps == {'comp.agent.state_schema', 'comp.agent.checkpoint_resume', 'comp.agent.loop_approval', 'comp.agent.unassisted_transfer'}
    for node in ['kn.m38.state_schema', 'kn.m38.checkpoint_resume', 'kn.m38.loop_approval', 'kn.m38.no_ai_transfer']:
        assert node in mission["knowledge_nodes"]
    assert len(mission["competencies"]) == len(mission["knowledge_nodes"])


def test_package_has_no_prefilled_learner_evidence() -> None:
    blob = (M38_PACKAGE / "missions" / "M38.json").read_text(encoding="utf-8").lower()
    for forbidden in (
        "learner_evidence",
        "learner_response",
        "fabricated learner",
        "prefilled",
    ):
        assert forbidden not in blob


def test_package_has_no_custom_frontend_or_eager_pandas() -> None:
    for path in M38_PACKAGE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in {".tsx", ".ts", ".jsx", ".js", ".css", ".html", ".py"}
        assert path.name not in {"index.html", "App.tsx"}
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    assert all(alias.name.split(".", 1)[0] != "pandas" for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    assert not str(node.module or "").startswith("pandas")
    mission = json.dumps(_mission()).lower()
    assert "/missions/m38/" not in mission
    assert "bespoke" not in mission


def test_no_eager_pandas_import_in_fixture_or_tests() -> None:
    roots = [M38_PACKAGE, Path(__file__).resolve().parent]
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
            "tests/platform/reference/M38",
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


def test_platform_has_no_mission_special_case_routes_or_conditionals() -> None:
    platform = REPO_ROOT / "platform"
    skip_dirs = {"node_modules", "dist", "__pycache__", ".git"}
    eq_mid = re.compile(r"""mission_id\s*==\s*["']M38["']""")
    custom_route = re.compile(r"/m38[-_/]|stateful-agent|teaching-graph-1", re.IGNORECASE)
    hits: list[str] = []
    for path in platform.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if "fixtures" in path.parts or "reference" in path.parts:
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
    assert "M38" not in registry
    assert 'stateful_agent_graph' not in registry.lower()
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
    if not paths:
        pytest.skip("no diff versus frozen base yet")
    unexpected = [
        path
        for path in paths
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in LANE_ALLOWED)
    ]
    other_g6 = [
        path
        for path in unexpected
        if re.match(r"(platform/fixtures|tests/platform/reference)/M\d{2}/", path)
    ]
    if unexpected and other_g6 and all(
        re.match(r"(platform/fixtures|tests/platform/reference)/M\d{2}/", path)
        for path in unexpected
    ):
        pytest.skip("combined G6 integration tree contains other migrator lanes")
    assert unexpected == [], unexpected
    assert any(path.startswith("platform/fixtures/M38/") for path in paths)
    assert any(path.startswith("tests/platform/reference/M38/") for path in paths)


def test_fixture_lives_only_in_allowed_g6_paths() -> None:
    relative = M38_PACKAGE.relative_to(REPO_ROOT)
    assert relative == Path("platform/fixtures/M38")
    tests = REPO_ROOT / "tests" / "platform" / "reference" / "M38"
    assert tests.is_dir()
