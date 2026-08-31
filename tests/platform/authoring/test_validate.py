from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from tools.authoring.package import load_package, rewrite_integrity
from tools.authoring.validate import format_validate_result, validate_package

REPO_ROOT = Path(__file__).resolve().parents[3]
F01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "f01"
CLI_PATH = REPO_ROOT / "tools" / "authoring" / "cli.py"


def _copy_package(tmp_path: Path) -> Path:
    dest = tmp_path / "pkg"
    shutil.copytree(F01_PACKAGE, dest)
    return dest


def test_validate_default_package_succeeds() -> None:
    result = validate_package(F01_PACKAGE)
    assert result.ok, result.errors
    assert result.package_id == "g4.fixture.f01"
    assert result.mission_ids == ["M00"]
    assert "VALID" in format_validate_result(result)


def test_cli_validate_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "validate"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "VALID" in completed.stdout
    assert "g4.fixture.f01" in completed.stdout


def test_cli_preview_lists_stages_and_policies() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "preview"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    assert "stage_01_orientation" in stdout
    assert "orientation" in stdout
    assert "UNRESTRICTED" in stdout
    assert "experiment" in stdout
    assert "RESTRICTED_HINTS" in stdout
    assert "NO_AI_REQUIRED" in stdout
    assert "predict -> execute -> submit" in stdout


def test_tampered_payload_digest_fails(tmp_path: Path) -> None:
    dest = _copy_package(tmp_path)
    (dest / "SHA256SUMS").unlink()
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = validate_package(dest)
    assert not result.ok
    assert result.code == "BAD_DIGEST"
    assert any("digest" in item.lower() or "SHA256SUMS" in item for item in result.errors)


def test_tampered_mission_file_fails_checksum(tmp_path: Path) -> None:
    dest = _copy_package(tmp_path)
    mission_path = dest / "missions" / "M00.json"
    spec = json.loads(mission_path.read_text(encoding="utf-8"))
    spec["title"] = "tampered"
    mission_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    result = validate_package(dest)
    assert not result.ok
    assert result.code == "BAD_DIGEST"
    assert any("Checksum mismatch" in item or "digest" in item.lower() for item in result.errors)


def test_schema_invalid_mission_fails_after_reseal(tmp_path: Path) -> None:
    dest = _copy_package(tmp_path)
    mission_path = dest / "missions" / "M00.json"
    spec = json.loads(mission_path.read_text(encoding="utf-8"))
    spec["id"] = "F01"
    mission_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["missions"][0]["id"] = "F01"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rewrite_integrity(dest)
    result = validate_package(dest)
    assert not result.ok
    assert any("F01" in item or "WP-136" in item for item in result.errors)


def test_g3_loader_accepts_sealed_package() -> None:
    from app.core.mission_loader import load_package as g3_load

    package = g3_load(F01_PACKAGE)
    assert package.id == "g4.fixture.f01"
    assert package.missions[0]["id"] == "M00"


def test_load_package_round_trip() -> None:
    package = load_package(F01_PACKAGE)
    assert package.digest
    assert len(package.digest) == 64
    assert (package.source_path / "SHA256SUMS").is_file()
