"""MDL v1 types, catalogue constants, and validation error (WP-136 / WP-121)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

MDL_SCHEMA_VERSION: Final = "v1"
MISSION_ID_PATTERN: Final = r"^M[0-9]{2}$"

STAGE_TYPES: Final[tuple[str, ...]] = (
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
)

ASSISTANCE_POLICIES: Final[tuple[str, ...]] = (
    "UNRESTRICTED",
    "SOCRATIC_ONLY",
    "RESTRICTED_HINTS",
    "NO_AI_REQUIRED",
)

MISSION_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "title",
    "phase",
    "order_index",
    "core_invariant",
    "competencies",
    "knowledge_nodes",
    "stages",
    "gate_contract",
)

STAGE_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "title",
    "type",
    "assistance_policy",
)

GATE_CONTRACT_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "required_evidence",
    "pass_threshold",
)

SCHEMA_VERSION_FIELD_NAMES: Final[tuple[str, ...]] = (
    "schema_version",
    "mdl_version",
    "mdl_schema_version",
)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    code: str = "VALIDATION_ERROR"

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "code": self.code}


class ValidationError(ValueError):
    """MDL or curriculum-package validation failure with JSON path and message."""

    def __init__(
        self,
        message: str,
        *,
        path: str = "$",
        code: str = "VALIDATION_ERROR",
        errors: list[dict[str, Any]] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.code = code
        self.errors: list[dict[str, Any]] = errors or [
            {"path": path, "message": message, "code": code}
        ]
        self.details: dict[str, Any] = details or {}

    @classmethod
    def from_issues(
        cls,
        issues: list[ValidationIssue],
        *,
        details: dict[str, Any] | None = None,
    ) -> ValidationError:
        if not issues:
            return cls("Validation failed", path="$", details=details)
        primary = issues[0]
        payload = [issue.to_dict() for issue in issues]
        if len(issues) == 1:
            message = primary.message
        else:
            message = (
                f"{len(issues)} validation errors; first at {primary.path}: {primary.message}"
            )
        return cls(
            message,
            path=primary.path,
            code=primary.code,
            errors=payload,
            details=details,
        )
