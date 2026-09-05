from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from app.core.mission_loader import CurriculumPackageError, fixture_package_path, load_package
from app.core.secrets import IntegrityError, verify_package_checksums

REPO_ROOT = Path(__file__).resolve().parents[4]
CLI = REPO_ROOT / "tools" / "platform" / "secrets.py"
G3_PACKAGE = REPO_ROOT / "platform" / "worker" / "fixtures" / "g3_curriculum"
F01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "f01"


def _copy_package(tmp_path: Path, source: Path) -> Path:
    dest = tmp_path / source.name
    shutil.copytree(source, dest)
    return dest


def test_fixture_packages_verify_closed() -> None:
    assert verify_package_checksums(G3_PACKAGE, required=True)
    assert verify_package_checksums(F01_PACKAGE, required=True)
    package = load_package(fixture_package_path())
    assert package.digest
    assert (package.source_path / "SHA256SUMS").is_file()


def test_tampered_sha256sums_is_rejected(tmp_path: Path) -> None:
    dest = _copy_package(tmp_path, G3_PACKAGE)
    sums = dest / "SHA256SUMS"
    original = sums.read_text(encoding="utf-8")
    tampered = []
    for line in original.splitlines():
        if line and not line.startswith("#"):
            digest, rel = line.split(None, 1)
            digest = ("0" if digest[0] != "0" else "1") + digest[1:]
            tampered.append(f"{digest}  {rel}")
        else:
            tampered.append(line)
    sums.write_text("\n".join(tampered) + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError) as integrity:
        verify_package_checksums(dest, required=True)
    assert integrity.value.code == "BAD_DIGEST"
    with pytest.raises(CurriculumPackageError) as loaded:
        load_package(dest)
    assert loaded.value.code == "BAD_DIGEST"


def test_tampered_payload_without_updating_sums_is_rejected(tmp_path: Path) -> None:
    dest = _copy_package(tmp_path, G3_PACKAGE)
    mission = dest / "missions" / "g3.fixture.orientation.json"
    spec = json.loads(mission.read_text(encoding="utf-8"))
    spec["title"] = "tampered-title"
    mission.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError) as integrity:
        verify_package_checksums(dest, required=True)
    assert integrity.value.code == "BAD_DIGEST"
    with pytest.raises(CurriculumPackageError) as loaded:
        load_package(dest)
    assert loaded.value.code == "BAD_DIGEST"


def test_missing_sha256sums_fails_closed_when_required(tmp_path: Path) -> None:
    dest = _copy_package(tmp_path, G3_PACKAGE)
    (dest / "SHA256SUMS").unlink()
    with pytest.raises(IntegrityError) as raised:
        verify_package_checksums(dest, required=True)
    assert raised.value.code == "MISSING_CHECKSUMS"


def test_path_escape_and_invalid_line_fail_closed(tmp_path: Path) -> None:
    dest = _copy_package(tmp_path, G3_PACKAGE)
    (dest / "SHA256SUMS").write_text("not-a-digest  manifest.json\n", encoding="utf-8")
    with pytest.raises(IntegrityError):
        verify_package_checksums(dest, required=True)
    digest = hashlib.sha256(b"x").hexdigest()
    (dest / "SHA256SUMS").write_text(f"{digest}  ../escape.txt\n", encoding="utf-8")
    with pytest.raises(IntegrityError):
        verify_package_checksums(dest, required=True)
