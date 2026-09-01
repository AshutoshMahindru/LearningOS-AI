"""Load frozen WP-137 at runtime and validate structured results.

The schema file is not copied into this package; it is read from architecture/.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from app.execution.contracts import (
    BLOCK_TYPES,
    EXECUTION_STATUSES,
    WP137_REQUIRED_FIELDS,
    ContractError,
    StructuredResult,
)

WP137_RELATIVE = Path(
    "architecture/learningos-v3/03_technical_architecture/WP-137_structured_result_schema.json"
)


class ResultSchemaError(ContractError):
    """WP-137 validation failure with JSON path and message."""

    def __init__(
        self,
        message: str,
        *,
        path: str = "$",
        code: str = "RESULT_SCHEMA_ERROR",
        errors: list[dict[str, Any]] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, path=path, code=code, details=details)
        self.errors: list[dict[str, Any]] = errors or [
            {"path": path, "message": message, "code": code}
        ]

    @classmethod
    def from_errors(cls, errors: list[dict[str, Any]]) -> ResultSchemaError:
        if not errors:
            return cls("Structured result validation failed", path="$")
        primary = errors[0]
        path = str(primary.get("path") or "$")
        message = str(primary.get("message") or "Structured result validation failed")
        if len(errors) == 1:
            text = message
        else:
            text = f"{len(errors)} validation errors; first at {path}: {message}"
        return cls(text, path=path, code="RESULT_SCHEMA_ERROR", errors=errors)


def wp137_schema_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / WP137_RELATIVE
        if candidate.is_file():
            return candidate
    raise ResultSchemaError(
        f"Frozen WP-137 schema not found ({WP137_RELATIVE.as_posix()})",
        path="$",
        code="MISSING_SCHEMA",
    )


def _require_schema_alignment(schema: dict[str, Any]) -> None:
    try:
        required = tuple(schema.get("required") or ())
        status_enum = tuple(schema["properties"]["status"]["enum"])
        block_enum = tuple(
            schema["properties"]["blocks"]["items"]["properties"]["type"]["enum"]
        )
        block_required = tuple(schema["properties"]["blocks"]["items"].get("required") or ())
    except (KeyError, TypeError) as exc:
        raise ResultSchemaError(
            f"WP-137 schema is missing expected result structure: {exc}",
            path="$",
            code="INVALID_SCHEMA",
        ) from exc
    if required != WP137_REQUIRED_FIELDS:
        raise ResultSchemaError(
            "WP-137 required fields do not match the frozen result catalogue",
            path="required",
            code="INVALID_SCHEMA",
            details={"expected": list(WP137_REQUIRED_FIELDS), "found": list(required)},
        )
    if status_enum != EXECUTION_STATUSES:
        raise ResultSchemaError(
            "WP-137 status enum does not match SUCCESS/FAILED/TIMEOUT/CRASHED",
            path="properties.status.enum",
            code="INVALID_SCHEMA",
            details={"expected": list(EXECUTION_STATUSES), "found": list(status_enum)},
        )
    if block_enum != BLOCK_TYPES:
        raise ResultSchemaError(
            "WP-137 block type enum does not match the frozen catalogue",
            path="properties.blocks.items.properties.type.enum",
            code="INVALID_SCHEMA",
            details={"expected": list(BLOCK_TYPES), "found": list(block_enum)},
        )
    if block_required != ("type", "payload"):
        raise ResultSchemaError(
            "WP-137 block required fields do not match type/payload",
            path="properties.blocks.items.required",
            code="INVALID_SCHEMA",
            details={"expected": ["type", "payload"], "found": list(block_required)},
        )


@lru_cache(maxsize=1)
def load_result_schema() -> dict[str, Any]:
    path = wp137_schema_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResultSchemaError(
            f"Unable to load frozen WP-137 schema: {exc}",
            path=str(path),
            code="INVALID_SCHEMA",
        ) from exc
    if not isinstance(payload, dict):
        raise ResultSchemaError(
            "WP-137 schema must be a JSON object",
            path=str(path),
            code="INVALID_SCHEMA",
        )
    _require_schema_alignment(payload)
    return payload


@lru_cache(maxsize=1)
def _result_validator() -> Any:
    schema = load_result_schema()
    validator_cls = validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except SchemaError as exc:
        raise ResultSchemaError(
            f"WP-137 schema is itself invalid: {exc.message}",
            path=str(wp137_schema_path()),
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


def _issues_from_jsonschema(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    validator = _result_validator()
    issues: list[dict[str, str]] = []
    for error in validator.iter_errors(payload):
        path_items = list(error.absolute_path)
        if error.validator == "required":
            match = re.search(r"'([^']+)' is a required property", error.message)
            if match:
                path_items.append(match.group(1))
        issues.append({"path": _format_path(path_items), "message": error.message, "code": "RESULT_SCHEMA_ERROR"})
    issues.sort(key=lambda item: (item["path"], item["message"]))
    return issues


def validate_structured_result(result: Mapping[str, Any] | StructuredResult) -> dict[str, Any]:
    """Validate a result mapping against frozen WP-137. Never silent-pass."""
    payload = result.to_dict() if isinstance(result, StructuredResult) else dict(result)
    if not isinstance(payload, dict):
        raise ResultSchemaError("Structured result must be a JSON object", path="$")
    issues = _issues_from_jsonschema(payload)
    if issues:
        raise ResultSchemaError.from_errors(issues)
    return payload
