from __future__ import annotations

import json
import re
from pathlib import Path

from tools.authoring.package import load_package
from tools.authoring.paths import WP136_SCHEMA_PATH
from tools.authoring.validate import validate_mission_document, validate_package

REPO_ROOT = Path(__file__).resolve().parents[4]
M04_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M04"
PACKAGE_ID = "g5.reference.M04"
MISSION_ID = "M04"


def test_package_identity_is_g5_reference_m04() -> None:
    package = load_package(M04_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.version == "5.0.0"
    assert package.missions[0]["id"] == MISSION_ID
    assert re.match(r"^M[0-9]{2}$", package.missions[0]["id"])
    assert package.digest
    assert len(package.digest) == 64


def test_wp136_and_g4_journey_contract() -> None:
    result = validate_package(M04_PACKAGE)
    assert result.ok, result.errors
    assert result.package_id == PACKAGE_ID
    assert result.mission_ids == [MISSION_ID]
    mission = result.package.missions[0]  # type: ignore[union-attr]
    types = [stage["type"] for stage in mission["stages"]]
    assert types == [
        "orientation",
        "experiment",
        "controlled_failure",
        "transfer_assessment",
        "competency_gate",
    ]
    experiment = next(stage for stage in mission["stages"] if stage["type"] == "experiment")
    text = experiment["instructions"].lower()
    assert "predict" in text
    assert "execute" in text or "run" in text
    assert "submit" in text or "explain" in text
    runner = experiment["runner"]
    assert runner["module"] == "g5.reference.M04.experiment"
    assert runner["entrypoint"] == "run_quality_pipeline"
    policies = {stage["id"]: stage["assistance_policy"] for stage in mission["stages"]}
    assert policies["stage_01_orientation"] == "UNRESTRICTED"
    assert policies["stage_02_experiment"] == "RESTRICTED_HINTS"
    assert policies["stage_03_controlled_failure"] == "SOCRATIC_ONLY"
    assert policies["stage_04_transfer"] == "NO_AI_REQUIRED"
    assert policies["stage_05_gate"] == "NO_AI_REQUIRED"
    assert mission["flagship_version"] == "V01"
    assert mission["order_index"] == 4
    assert mission["prerequisites"] == ["M03"]
    assert WP136_SCHEMA_PATH.is_file()


def test_mission_is_tabular_cleaning_not_sensor_imputation_stub() -> None:
    mission = json.loads((M04_PACKAGE / "missions" / "M04.json").read_text(encoding="utf-8"))
    blob = json.dumps(mission).lower()
    assert "customer" in blob
    assert "duplicate" in blob
    assert "outlier" in blob
    assert "telemetry" not in blob
    assert "sensor dataset" not in blob
    assert mission["id"] != "M00"


def test_lab_source_and_data_are_fixture_local() -> None:
    cleaning = M04_PACKAGE / "g5" / "reference" / "M04" / "cleaning.py"
    orders = M04_PACKAGE / "g5" / "reference" / "M04" / "data" / "customer_orders_dirty.csv"
    inventory = M04_PACKAGE / "g5" / "reference" / "M04" / "data" / "inventory_transfer.csv"
    assert cleaning.is_file()
    assert orders.is_file()
    assert inventory.is_file()
    mission_cleaning = REPO_ROOT / "missions" / "M04" / "cleaning.py"
    assert cleaning.read_text(encoding="utf-8") == mission_cleaning.read_text(encoding="utf-8")
    checksums = (M04_PACKAGE / "SHA256SUMS").read_text(encoding="utf-8")
    for rel in (
        "manifest.json",
        "missions/M04.json",
        "g5/reference/M04/cleaning.py",
        "g5/reference/M04/experiment.py",
        "g5/reference/M04/data/customer_orders_dirty.csv",
        "g5/reference/M04/data/inventory_transfer.csv",
    ):
        assert rel in checksums


def test_wp136_schema_rejects_non_m_id() -> None:
    mission = json.loads((M04_PACKAGE / "missions" / "M04.json").read_text(encoding="utf-8"))
    mission["id"] = "F01"
    engine, errors = validate_mission_document(mission)
    assert engine in {"jsonschema", "mdl_validator"}
    assert errors
    assert any("F01" in item or "M[0-9]" in item or "does not match" in item for item in errors)


def test_g3_loader_accepts_sealed_package() -> None:
    from app.core.mission_loader import load_package as g3_load

    package = g3_load(M04_PACKAGE)
    assert package.id == PACKAGE_ID
    assert package.missions[0]["id"] == MISSION_ID


def test_platform_has_no_m04_special_case_routes() -> None:
    roots = [
        REPO_ROOT / "platform" / "backend" / "app",
        REPO_ROOT / "platform" / "frontend" / "src",
    ]
    skip_dirs = {"node_modules", "dist", "__pycache__"}
    eq_m04 = re.compile(r"""==\s*["']M04["']""")
    route_m04 = re.compile(r"/missions/M04")
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
            if eq_m04.search(text) or route_m04.search(text):
                hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []
