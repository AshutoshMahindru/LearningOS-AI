"""Fixture package load and integrity (SHA256SUMS + payload digest)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import AuthoringError
from .paths import F01_PACKAGE_DIR, F01_PACKAGE_ID, F01_PACKAGE_VERSION, SYNTHETIC_MISSION_ID

__all__ = [
    "F01_PACKAGE_DIR",
    "F01_PACKAGE_ID",
    "F01_PACKAGE_VERSION",
    "SYNTHETIC_MISSION_ID",
    "Package",
    "canonical_json_bytes",
    "compute_payload_digest",
    "default_package_path",
    "load_package",
    "rewrite_integrity",
    "sha256_file",
]


def default_package_path() -> Path:
    return F01_PACKAGE_DIR


def canonical_json_bytes(value: Any) -> bytes:
    # Must match app.core.mission_loader.canonical_json_bytes so G3 can install this package.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_payload_digest(package_dir: Path, mission_relpaths: list[str]) -> str:
    hasher = hashlib.sha256()
    for rel in sorted(mission_relpaths):
        path = package_dir / rel
        payload = json.loads(path.read_text(encoding="utf-8"))
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(canonical_json_bytes(payload))
        hasher.update(b"\n")
    return hasher.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise AuthoringError(
                f"Invalid SHA256SUMS line {line_no}",
                code="BAD_DIGEST",
                details={"path": str(path), "line": line_no},
            )
        digest, rel = parts
        entries[rel.strip()] = digest.strip()
    return entries


def _mission_relpath(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        return item, f"missions/{item}.json"
    if isinstance(item, dict) and isinstance(item.get("id"), str):
        rel = item.get("path") or f"missions/{item['id']}.json"
        return item["id"], str(rel)
    raise AuthoringError(
        "Each mission entry must be an id string or object with id",
        code="MISSING_METADATA",
        details={"entry": item},
    )


@dataclass(frozen=True)
class Package:
    id: str
    version: str
    title: str
    digest: str
    manifest: dict[str, Any]
    missions: tuple[dict[str, Any], ...]
    mission_paths: tuple[str, ...]
    source_path: Path


def load_package(path: Path | str | None = None, *, verify: bool = True) -> Package:
    package_dir = Path(path or default_package_path()).expanduser().resolve()
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        raise AuthoringError(
            f"Missing manifest.json under {package_dir}",
            code="MISSING_MANIFEST",
            details={"path": str(package_dir)},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuthoringError(
            f"Invalid manifest.json: {exc}",
            code="INVALID_MANIFEST",
            details={"path": str(manifest_path)},
        ) from exc
    if not isinstance(manifest, dict):
        raise AuthoringError("manifest.json must be an object", code="INVALID_MANIFEST")

    missing = [key for key in ("id", "version", "missions") if key not in manifest]
    if missing:
        raise AuthoringError(
            f"Manifest missing required fields: {', '.join(missing)}",
            code="MISSING_METADATA",
            details={"missing": missing},
        )
    package_id = manifest["id"]
    version = manifest["version"]
    missions_field = manifest["missions"]
    digest = manifest.get("digest") or manifest.get("sha256")
    if not isinstance(package_id, str) or not package_id:
        raise AuthoringError("Manifest id is required", code="MISSING_METADATA")
    if not isinstance(version, str) or not version:
        raise AuthoringError("Manifest version is required", code="MISSING_METADATA")
    if not isinstance(missions_field, list) or not missions_field:
        raise AuthoringError("Manifest must list at least one mission", code="MISSING_METADATA")
    if not isinstance(digest, str) or not digest:
        raise AuthoringError("Manifest digest/sha256 is required", code="BAD_DIGEST")
    digest = digest.removeprefix("sha256:")

    mission_ids: list[str] = []
    mission_relpaths: list[str] = []
    for entry in missions_field:
        mission_id, relpath = _mission_relpath(entry)
        mission_ids.append(mission_id)
        mission_relpaths.append(relpath)

    if verify:
        _verify_integrity(package_dir, digest, mission_relpaths)

    loaded: list[dict[str, Any]] = []
    for mission_id, rel in zip(mission_ids, mission_relpaths, strict=True):
        spec_path = package_dir / rel
        if not spec_path.is_file():
            raise AuthoringError(
                f"Mission file missing: {rel}",
                code="MISSING_METADATA",
                details={"path": rel},
            )
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise AuthoringError(f"Mission {rel} must be a JSON object", code="INVALID_MANIFEST")
        if spec.get("id") != mission_id:
            raise AuthoringError(
                f"Mission id mismatch: expected {mission_id}, found {spec.get('id')}",
                code="MISSING_METADATA",
                details={"expected": mission_id, "found": spec.get("id")},
            )
        loaded.append(spec)

    return Package(
        id=package_id,
        version=version,
        title=str(manifest.get("title") or package_id),
        digest=digest,
        manifest=manifest,
        missions=tuple(loaded),
        mission_paths=tuple(mission_relpaths),
        source_path=package_dir,
    )


def _verify_integrity(package_dir: Path, expected_digest: str, mission_relpaths: list[str]) -> None:
    checksums_path = package_dir / "SHA256SUMS"
    if not checksums_path.is_file():
        raise AuthoringError(
            f"Missing SHA256SUMS under {package_dir}",
            code="BAD_DIGEST",
            details={"path": str(checksums_path)},
        )
    listed = parse_sha256sums(checksums_path)
    if "manifest.json" not in listed:
        raise AuthoringError(
            "SHA256SUMS must include manifest.json",
            code="BAD_DIGEST",
            details={"path": "manifest.json"},
        )
    for rel in mission_relpaths:
        if rel not in listed:
            raise AuthoringError(
                f"SHA256SUMS missing {rel}",
                code="BAD_DIGEST",
                details={"path": rel},
            )
    for rel, expected in listed.items():
        file_path = package_dir / rel
        if not file_path.is_file():
            raise AuthoringError(
                f"SHA256SUMS references missing file {rel}",
                code="BAD_DIGEST",
                details={"path": rel},
            )
        actual = sha256_file(file_path)
        if actual != expected:
            raise AuthoringError(
                f"Checksum mismatch for {rel}",
                code="BAD_DIGEST",
                details={"path": rel, "expected": expected, "actual": actual},
            )

    actual_digest = compute_payload_digest(package_dir, mission_relpaths)
    if actual_digest != expected_digest:
        raise AuthoringError(
            "Package payload digest does not match manifest",
            code="BAD_DIGEST",
            details={"expected": expected_digest, "actual": actual_digest},
        )


def rewrite_integrity(package_dir: Path | str) -> str:
    """Recompute manifest digest and SHA256SUMS. Used when sealing a package copy in tests."""
    package_dir = Path(package_dir).expanduser().resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mission_relpaths = [_mission_relpath(entry)[1] for entry in manifest["missions"]]
    digest = compute_payload_digest(package_dir, mission_relpaths)
    manifest["digest"] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checksum_rels = ["manifest.json", *mission_relpaths]
    lines = [f"{sha256_file(package_dir / rel)}  {rel}" for rel in checksum_rels]
    (package_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return digest
