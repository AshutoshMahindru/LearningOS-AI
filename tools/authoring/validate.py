"""Validate a fixture package against WP-136 and the G4 E2E journey contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import AuthoringError
from .package import Package, load_package
from .paths import SYNTHETIC_MISSION_ID, WP136_SCHEMA_PATH

MISSION_ID_PATTERN = re.compile(r"^M[0-9]{2}$")
EXPERIMENT_CYCLE_TOKENS = ("predict", "execute", "submit")
JOURNEY_REQUIRED_TYPES = ("orientation", "experiment")
JOURNEY_TERMINAL_TYPES = frozenset({"transfer_assessment", "competency_gate"})


@dataclass
class ValidateResult:
    ok: bool
    errors: list[str]
    package_id: str | None = None
    version: str | None = None
    digest: str | None = None
    mission_ids: list[str] = field(default_factory=list)
    schema_engine: str = "jsonschema"
    package: Package | None = None

    @property
    def code(self) -> str:
        if self.ok:
            return "OK"
        joined = " ".join(self.errors)
        if "Checksum" in joined or "digest" in joined.lower() or joined.startswith("BAD_DIGEST"):
            return "BAD_DIGEST"
        if "schema" in joined.lower() or "WP-136" in joined:
            return "SCHEMA"
        return "INVALID"


def _format_jsonschema_error(error: Any) -> str:
    path = "/".join(str(part) for part in error.absolute_path)
    where = path or "<root>"
    return f"WP-136 {where}: {error.message}"


def _validate_with_mdl_validator(mission: dict[str, Any]) -> tuple[str, list[str]] | None:
    """Use 21A mdl_validator when present. Return None to fall back to jsonschema."""
    try:
        from app.core import mdl_validator as mod
    except ImportError:
        return None

    fn: Callable[..., Any] | None = None
    for name in ("validate_mission", "validate_mdl", "validate"):
        candidate = getattr(mod, name, None)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        cls = getattr(mod, "MdlValidator", None)
        if cls is not None:
            try:
                instance = cls()
            except TypeError:
                instance = None
            if instance is not None:
                for name in ("validate_mission", "validate"):
                    candidate = getattr(instance, name, None)
                    if callable(candidate):
                        fn = candidate
                        break
    if fn is None:
        return None

    try:
        result = fn(mission)
    except TypeError:
        return None
    except Exception as exc:
        message = getattr(exc, "message", None) or str(exc)
        return "mdl_validator", [f"WP-136 mdl_validator: {message}"]

    if result is None or result is True:
        return "mdl_validator", []
    if result is False:
        return "mdl_validator", ["WP-136 mdl_validator rejected the mission"]
    if isinstance(result, (list, tuple)):
        errors = [str(item) for item in result if item]
        return "mdl_validator", errors
    ok_attr = getattr(result, "ok", None)
    if isinstance(ok_attr, bool):
        if ok_attr:
            return "mdl_validator", []
        extra = getattr(result, "errors", None) or getattr(result, "message", None) or "rejected"
        if isinstance(extra, (list, tuple)):
            return "mdl_validator", [str(item) for item in extra]
        return "mdl_validator", [f"WP-136 mdl_validator: {extra}"]
    return "mdl_validator", []


def _validate_with_jsonschema(mission: dict[str, Any]) -> list[str]:
    if not WP136_SCHEMA_PATH.is_file():
        return [f"WP-136 schema missing at {WP136_SCHEMA_PATH}"]
    schema = json.loads(WP136_SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        import jsonschema
    except ImportError:
        return _structural_fallback(mission)

    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    return [_format_jsonschema_error(error) for error in validator.iter_errors(mission)]


def _structural_fallback(mission: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in (
        "id",
        "title",
        "phase",
        "order_index",
        "core_invariant",
        "competencies",
        "knowledge_nodes",
        "stages",
        "gate_contract",
    ):
        if field_name not in mission:
            errors.append(f"WP-136 missing required field '{field_name}'")
    mission_id = mission.get("id")
    if isinstance(mission_id, str) and not MISSION_ID_PATTERN.match(mission_id):
        errors.append(f"WP-136 id {mission_id!r} does not match ^M[0-9]{{2}}$")
    return errors


def validate_mission_document(mission: dict[str, Any]) -> tuple[str, list[str]]:
    adapted = _validate_with_mdl_validator(mission)
    if adapted is not None:
        engine, errors = adapted
        if not errors:
            return engine, _journey_errors(mission)
        return engine, errors + _journey_errors(mission)
    errors = _validate_with_jsonschema(mission)
    return "jsonschema", errors + _journey_errors(mission)


def _journey_errors(mission: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    mission_id = mission.get("id")
    if mission_id in {"F01"}:
        errors.append(
            "Mission id 'F01' is not WP-136 valid; use a synthetic id such as "
            f"{SYNTHETIC_MISSION_ID} and keep F01 in package metadata only."
        )
    stages = mission.get("stages")
    if not isinstance(stages, list) or not stages:
        return errors
    types = {stage.get("type") for stage in stages if isinstance(stage, dict)}
    for required in JOURNEY_REQUIRED_TYPES:
        if required not in types:
            errors.append(f"G4 journey missing required stage type '{required}'")
    if types.isdisjoint(JOURNEY_TERMINAL_TYPES):
        errors.append("G4 journey requires transfer_assessment or competency_gate")
    if not isinstance(mission.get("gate_contract"), dict):
        errors.append("G4 journey missing gate_contract")
    else:
        contract = mission["gate_contract"]
        if not contract.get("required_evidence"):
            errors.append("gate_contract.required_evidence must be non-empty")
        if "pass_threshold" not in contract:
            errors.append("gate_contract.pass_threshold is required")

    for stage in stages:
        if not isinstance(stage, dict) or stage.get("type") != "experiment":
            continue
        text = str(stage.get("instructions") or "").lower()
        missing = [token for token in EXPERIMENT_CYCLE_TOKENS if token not in text]
        if "execute" in missing and "run" in text:
            missing = [token for token in missing if token != "execute"]
        if "submit" in missing and "explain" in text:
            missing = [token for token in missing if token != "submit"]
        if missing:
            errors.append(
                f"Experiment stage {stage.get('id')} must describe predict/execute/submit "
                f"(missing {', '.join(missing)})"
            )
    return errors


def validate_package(path: Path | str | None = None) -> ValidateResult:
    try:
        package = load_package(path, verify=True)
    except AuthoringError as exc:
        return ValidateResult(ok=False, errors=[f"{exc.code}: {exc}"])

    errors: list[str] = []
    engine = "jsonschema"
    for mission in package.missions:
        engine, mission_errors = validate_mission_document(mission)
        errors.extend(mission_errors)

    return ValidateResult(
        ok=not errors,
        errors=errors,
        package_id=package.id,
        version=package.version,
        digest=package.digest,
        mission_ids=[str(mission.get("id")) for mission in package.missions],
        schema_engine=engine,
        package=package,
    )


def format_validate_result(result: ValidateResult) -> str:
    if not result.ok:
        return "INVALID\n" + "\n".join(f"- {item}" for item in result.errors)
    mission = ",".join(result.mission_ids) or "-"
    return (
        f"VALID  {result.package_id}@{result.version}  "
        f"mission={mission}  engine={result.schema_engine}  digest={result.digest}"
    )
