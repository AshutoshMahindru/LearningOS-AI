"""G3 curriculum package loader: required metadata plus payload integrity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.secrets import IntegrityError, verify_package_checksums

G3_FIXTURE_PACKAGE_ID = "g3.fixture.curriculum"
G3_FIXTURE_VERSION = "3.0.0"


class CurriculumPackageError(ValueError):
    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class CurriculumPackage:
    id: str
    version: str
    digest: str
    missions: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    source_path: Path
    mission_paths: tuple[str, ...] = field(default_factory=tuple)

    @property
    def identity(self) -> tuple[str, str]:
        return (self.id, self.version)


Package = CurriculumPackage


def fixture_package_path() -> Path:
    return Path(__file__).resolve().parents[3] / "worker" / "fixtures" / "g3_curriculum"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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


def _mission_relpath(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        return item, f"missions/{item}.json"
    if isinstance(item, dict) and isinstance(item.get("id"), str):
        rel = item.get("path") or f"missions/{item['id']}.json"
        return item["id"], str(rel)
    raise CurriculumPackageError(
        "Each mission entry must be an id string or object with id",
        code="MISSING_METADATA",
        details={"entry": item},
    )


def _validate_mission(spec: dict[str, Any], *, expected_id: str) -> None:
    missing = [key for key in ("id", "title", "stages") if key not in spec]
    if missing:
        raise CurriculumPackageError(
            f"Mission {expected_id} missing required fields: {', '.join(missing)}",
            code="MISSING_METADATA",
            details={"mission_id": expected_id, "missing": missing},
        )
    if spec["id"] != expected_id:
        raise CurriculumPackageError(
            f"Mission id mismatch: expected {expected_id}, found {spec['id']}",
            code="MISSING_METADATA",
            details={"expected": expected_id, "found": spec["id"]},
        )
    if not isinstance(spec["title"], str) or not spec["title"]:
        raise CurriculumPackageError(
            f"Mission {expected_id} title is required",
            code="MISSING_METADATA",
            details={"mission_id": expected_id},
        )
    stages = spec["stages"]
    if not isinstance(stages, list) or not stages:
        raise CurriculumPackageError(
            f"Mission {expected_id} must declare at least one stage",
            code="MISSING_METADATA",
            details={"mission_id": expected_id},
        )


class MissionLoader:
    def load_package(self, path: Path | str) -> CurriculumPackage:
        package_dir = Path(path).expanduser().resolve()
        manifest_path = package_dir / "manifest.json"
        if not manifest_path.is_file():
            raise CurriculumPackageError(
                f"Missing manifest.json under {package_dir}",
                code="MISSING_MANIFEST",
                details={"path": str(package_dir)},
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CurriculumPackageError(
                f"Invalid manifest.json: {exc}",
                code="INVALID_MANIFEST",
                details={"path": str(manifest_path)},
            ) from exc
        if not isinstance(manifest, dict):
            raise CurriculumPackageError(
                "manifest.json must be an object",
                code="INVALID_MANIFEST",
                details={"path": str(manifest_path)},
            )

        missing = [key for key in ("id", "version", "missions") if key not in manifest]
        if missing:
            raise CurriculumPackageError(
                f"Manifest missing required fields: {', '.join(missing)}",
                code="MISSING_METADATA",
                details={"missing": missing},
            )
        package_id = manifest["id"]
        version = manifest["version"]
        missions_field = manifest["missions"]
        digest = manifest.get("digest") or manifest.get("sha256")
        if not isinstance(package_id, str) or not package_id:
            raise CurriculumPackageError("Manifest id is required", code="MISSING_METADATA")
        if not isinstance(version, str) or not version:
            raise CurriculumPackageError("Manifest version is required", code="MISSING_METADATA")
        if not isinstance(missions_field, list) or not missions_field:
            raise CurriculumPackageError(
                "Manifest must list at least one mission",
                code="MISSING_METADATA",
                details={"id": package_id},
            )
        if not isinstance(digest, str) or not digest:
            raise CurriculumPackageError(
                "Manifest digest/sha256 is required",
                code="BAD_DIGEST",
                details={"id": package_id},
            )
        digest = digest.removeprefix("sha256:")

        mission_ids: list[str] = []
        mission_relpaths: list[str] = []
        for entry in missions_field:
            mission_id, relpath = _mission_relpath(entry)
            mission_ids.append(mission_id)
            mission_relpaths.append(relpath)

        try:
            verify_package_checksums(package_dir, required=False)
        except IntegrityError as exc:
            raise CurriculumPackageError(str(exc), code=exc.code, details=exc.details) from exc

        for rel in mission_relpaths:
            if not (package_dir / rel).is_file():
                raise CurriculumPackageError(
                    f"Mission file missing: {rel}",
                    code="MISSING_METADATA",
                    details={"path": rel},
                )

        actual_digest = compute_payload_digest(package_dir, mission_relpaths)
        if actual_digest != digest:
            raise CurriculumPackageError(
                "Package payload digest does not match manifest",
                code="BAD_DIGEST",
                details={"expected": digest, "actual": actual_digest},
            )

        loaded: list[dict[str, Any]] = []
        for mission_id, rel in zip(mission_ids, mission_relpaths, strict=True):
            spec = json.loads((package_dir / rel).read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                raise CurriculumPackageError(
                    f"Mission {rel} must be a JSON object",
                    code="INVALID_MANIFEST",
                    details={"path": rel},
                )
            _validate_mission(spec, expected_id=mission_id)
            loaded.append(spec)

        return CurriculumPackage(
            id=package_id,
            version=version,
            digest=digest,
            missions=tuple(loaded),
            manifest=manifest,
            source_path=package_dir,
            mission_paths=tuple(mission_relpaths),
        )

    def validate_mission(self, mission_data: dict[str, Any]) -> bool:
        try:
            mission_id = str(mission_data.get("id") or "")
            _validate_mission(mission_data, expected_id=mission_id)
            return True
        except CurriculumPackageError:
            return False


def load_package(path: Path | str) -> CurriculumPackage:
    return MissionLoader().load_package(path)
