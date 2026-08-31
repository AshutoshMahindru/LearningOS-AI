"""MDL v1 mission and curriculum-package validator.

Loads the frozen WP-136 schema from architecture/ at runtime. Mission ids must
match ``^M[0-9]{2}$`` (synthetic fixture ids such as M00). There is no F01
platform exception.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from app.core.mdl_types import (
    ASSISTANCE_POLICIES,
    GATE_CONTRACT_REQUIRED_FIELDS,
    MDL_SCHEMA_VERSION,
    MISSION_REQUIRED_FIELDS,
    SCHEMA_VERSION_FIELD_NAMES,
    STAGE_REQUIRED_FIELDS,
    STAGE_TYPES,
    ValidationError,
    ValidationIssue,
)
from app.core.mission_loader import CurriculumPackage, CurriculumPackageError, load_package

WP136_RELATIVE = Path(
    "architecture/learningos-v3/03_technical_architecture/WP-136_mission_definition_schema.json"
)

_MAJOR_VERSION_RE = re.compile(
    r"(?:^|schema/|mission\.)v?(?P<major>\d+)\b",
    re.IGNORECASE,
)


def wp136_schema_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / WP136_RELATIVE
        if candidate.is_file():
            return candidate
    raise ValidationError(
        f"Frozen WP-136 schema not found ({WP136_RELATIVE.as_posix()})",
        path="$",
        code="MISSING_SCHEMA",
    )


def frozen_mission_fixtures_dir() -> Path:
    return wp136_schema_path().resolve().parents[1] / "04_cross_mission_proof" / "fixtures"


def _require_schema_alignment(schema: dict[str, Any]) -> None:
    try:
        required = tuple(schema.get("required") or ())
        stage_item = schema["properties"]["stages"]["items"]
        stage_props = stage_item["properties"]
        stage_enum = tuple(stage_props["type"]["enum"])
        assistance_enum = tuple(stage_props["assistance_policy"]["enum"])
        stage_required = tuple(stage_item.get("required") or ())
        gate_required = tuple(schema["properties"]["gate_contract"].get("required") or ())
    except (KeyError, TypeError) as exc:
        raise ValidationError(
            f"WP-136 schema is missing expected catalogue structure: {exc}",
            path="$",
            code="INVALID_SCHEMA",
        ) from exc
    if required != MISSION_REQUIRED_FIELDS:
        raise ValidationError(
            "WP-136 required fields do not match MDL v1 catalogue",
            path="required",
            code="INVALID_SCHEMA",
            details={"expected": list(MISSION_REQUIRED_FIELDS), "found": list(required)},
        )
    if stage_enum != STAGE_TYPES:
        raise ValidationError(
            "WP-136 stage type enum does not match MDL v1 catalogue",
            path="stages.items.properties.type.enum",
            code="INVALID_SCHEMA",
            details={"expected": list(STAGE_TYPES), "found": list(stage_enum)},
        )
    if assistance_enum != ASSISTANCE_POLICIES:
        raise ValidationError(
            "WP-136 assistance_policy enum does not match MDL v1 catalogue",
            path="stages.items.properties.assistance_policy.enum",
            code="INVALID_SCHEMA",
            details={"expected": list(ASSISTANCE_POLICIES), "found": list(assistance_enum)},
        )
    if stage_required != STAGE_REQUIRED_FIELDS:
        raise ValidationError(
            "WP-136 stage required fields do not match MDL v1 catalogue",
            path="stages.items.required",
            code="INVALID_SCHEMA",
        )
    if gate_required != GATE_CONTRACT_REQUIRED_FIELDS:
        raise ValidationError(
            "WP-136 gate_contract required fields do not match MDL v1 catalogue",
            path="gate_contract.required",
            code="INVALID_SCHEMA",
        )


@lru_cache(maxsize=1)
def load_mdl_schema() -> dict[str, Any]:
    path = wp136_schema_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(
            f"Unable to load frozen WP-136 schema: {exc}",
            path=str(path),
            code="INVALID_SCHEMA",
        ) from exc
    if not isinstance(payload, dict):
        raise ValidationError(
            "WP-136 schema must be a JSON object",
            path=str(path),
            code="INVALID_SCHEMA",
        )
    _require_schema_alignment(payload)
    return payload


@lru_cache(maxsize=1)
def _mdl_validator() -> Any:
    schema = load_mdl_schema()
    validator_cls = validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except SchemaError as exc:
        raise ValidationError(
            f"WP-136 schema is itself invalid: {exc.message}",
            path=str(wp136_schema_path()),
            code="INVALID_SCHEMA",
        ) from exc
    return validator_cls(schema)


def _format_path(items: list[Any]) -> str:
    if not items:
        return "$"
    out = ""
    for item in items:
        if isinstance(item, int):
            out = f"{out}[{item}]" if out else f"$[{item}]"
        else:
            token = str(item)
            out = token if not out else f"{out}.{token}"
    return out


def _schema_version_value(payload: Mapping[str, Any]) -> tuple[str, Any] | None:
    for name in SCHEMA_VERSION_FIELD_NAMES:
        if name in payload:
            return name, payload[name]
    return None


def _mdl_major_version(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = _MAJOR_VERSION_RE.search(text)
    if match is None:
        return None
    return int(match.group("major"))


def _check_schema_version(payload: Mapping[str, Any], *, path_prefix: str = "") -> None:
    found = _schema_version_value(payload)
    if found is None:
        return
    field, value = found
    path = f"{path_prefix}.{field}" if path_prefix else field
    if value is None:
        return
    major = _mdl_major_version(value)
    if major == 1:
        return
    raise ValidationError(
        f"Unsupported MDL schema_version {value!r}; expected {MDL_SCHEMA_VERSION}",
        path=path,
        code="VERSION_MISMATCH",
        details={"schema_version": value, "expected": MDL_SCHEMA_VERSION},
    )


def _issues_from_jsonschema(mission: Mapping[str, Any]) -> list[ValidationIssue]:
    validator = _mdl_validator()
    issues: list[ValidationIssue] = []
    for error in validator.iter_errors(mission):
        path_items = list(error.absolute_path)
        if error.validator == "required":
            match = re.search(r"'([^']+)' is a required property", error.message)
            if match:
                path_items.append(match.group(1))
        issues.append(
            ValidationIssue(path=_format_path(path_items), message=error.message)
        )
    issues.sort(key=lambda issue: (issue.path, issue.message))
    return issues


def _load_mission_payload(mission: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(mission, Mapping):
        if not isinstance(mission, dict):
            return dict(mission)
        return mission
    path = Path(mission)
    if path.is_dir():
        raise ValidationError(
            "Expected a mission JSON object or file, not a package directory",
            path=str(path),
            code="VALIDATION_ERROR",
        )
    if not path.is_file():
        raise ValidationError(
            f"Mission file not found: {path}",
            path=str(path),
            code="MISSING_METADATA",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(
            f"Invalid mission JSON: {exc}",
            path=str(path),
            code="INVALID_MANIFEST",
        ) from exc
    if not isinstance(payload, dict):
        raise ValidationError(
            "Mission must be a JSON object",
            path=str(path),
            code="VALIDATION_ERROR",
        )
    return payload


def _from_package_error(exc: CurriculumPackageError) -> ValidationError:
    details = dict(exc.details or {})
    path = str(details.get("path") or "")
    if exc.code == "BAD_DIGEST":
        path = path or "manifest.digest"
    elif exc.code in {"MISSING_MANIFEST", "INVALID_MANIFEST"}:
        path = path or "manifest.json"
    else:
        path = path or "manifest"
    return ValidationError(
        str(exc),
        path=path,
        code=exc.code,
        details=details,
    )


def validate_mission(
    mission: Mapping[str, Any] | Path | str,
    *,
    expected_id: str | None = None,
) -> dict[str, Any]:
    """Validate a mission object or JSON file against frozen WP-136 / MDL v1.

    Raises ValidationError with path and message. Does not return True on failure.
    """
    payload = _load_mission_payload(mission)
    if not isinstance(payload, dict):
        raise ValidationError("Mission must be a JSON object", path="$")
    _check_schema_version(payload)
    if expected_id is not None and payload.get("id") != expected_id:
        raise ValidationError(
            f"Mission id mismatch: expected {expected_id}, found {payload.get('id')!r}",
            path="id",
            code="MISSING_METADATA",
            details={"expected": expected_id, "found": payload.get("id")},
        )
    issues = _issues_from_jsonschema(payload)
    if issues:
        raise ValidationError.from_issues(issues)
    return payload


def validate_package(path: Path | str) -> CurriculumPackage:
    """Validate a curriculum package directory, then each mission against WP-136.

    Package shape matches ``platform/worker/fixtures/g3_curriculum``:
    ``manifest.json`` plus mission files, with digest integrity.
    """
    package_dir = Path(path)
    try:
        package = load_package(package_dir)
    except CurriculumPackageError as exc:
        raise _from_package_error(exc) from exc
    _check_schema_version(package.manifest, path_prefix="manifest")
    for spec in package.missions:
        validate_mission(spec)
    return package


__all__ = [
    "ASSISTANCE_POLICIES",
    "MDL_SCHEMA_VERSION",
    "MISSION_REQUIRED_FIELDS",
    "STAGE_REQUIRED_FIELDS",
    "GATE_CONTRACT_REQUIRED_FIELDS",
    "STAGE_TYPES",
    "ValidationError",
    "ValidationIssue",
    "frozen_mission_fixtures_dir",
    "load_mdl_schema",
    "validate_mission",
    "validate_package",
    "wp136_schema_path",
]
