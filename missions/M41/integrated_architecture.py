"""Integrated AI System Architecture instrumentation module for Mission M41.

Provides formal data contracts, boundary enforcers, decision classifiers,
degradation policy specifications, and observability budget validators for V11 AI systems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import time


class BoundaryType(str, Enum):
    SYSTEM = "SYSTEM"
    DATA = "DATA"
    CONTROL = "CONTROL"
    TRUST = "TRUST"


class DecisionKind(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    MODEL = "MODEL"


class DegradationMode(str, Enum):
    FULL_CAPABILITY = "FULL_CAPABILITY"
    REDUCED_RETRIEVAL = "REDUCED_RETRIEVAL"
    FALLBACK_CACHED = "FALLBACK_CACHED"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass
class SystemBoundary:
    boundary_id: str
    name: str
    boundary_type: BoundaryType
    description: str
    enforced_by: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "name": self.name,
            "boundary_type": self.boundary_type.value,
            "description": self.description,
            "enforced_by": self.enforced_by,
        }


@dataclass
class DecisionRule:
    rule_id: str
    name: str
    kind: DecisionKind
    condition: str
    handler: str
    fallback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "kind": self.kind.value,
            "condition": self.condition,
            "handler": self.handler,
            "fallback": self.fallback,
        }


@dataclass
class InterfaceContract:
    contract_id: str
    source_component: str
    target_component: str
    schema_version: str
    sla_ms: int
    validation_rule: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "source_component": self.source_component,
            "target_component": self.target_component,
            "schema_version": self.schema_version,
            "sla_ms": self.sla_ms,
            "validation_rule": self.validation_rule,
        }


@dataclass
class ObservabilityBudget:
    max_latency_ms: float = 2000.0
    max_cost_usd: float = 0.05
    min_confidence_score: float = 0.70
    max_tool_calls: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_latency_ms": self.max_latency_ms,
            "max_cost_usd": self.max_cost_usd,
            "min_confidence_score": self.min_confidence_score,
            "max_tool_calls": self.max_tool_calls,
        }


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SystemArchitectureConfig:
    name: str
    version: str
    boundaries: List[SystemBoundary] = field(default_factory=list)
    decision_rules: List[DecisionRule] = field(default_factory=list)
    interface_contracts: List[InterfaceContract] = field(default_factory=list)
    default_degradation: DegradationMode = DegradationMode.FULL_CAPABILITY
    budget: ObservabilityBudget = field(default_factory=ObservabilityBudget)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "decision_rules": [r.to_dict() for r in self.decision_rules],
            "interface_contracts": [c.to_dict() for c in self.interface_contracts],
            "default_degradation": self.default_degradation.value,
            "budget": self.budget.to_dict(),
        }


class ArchitectureValidator:
    """Validates system architecture configurations for consistency and safety."""

    @staticmethod
    def validate(config: SystemArchitectureConfig) -> ValidationResult:
        errors = []
        warnings = []

        if not config.name:
            errors.append("Architecture config must have a non-empty name.")

        if not config.boundaries:
            errors.append("Architecture config must define at least one SystemBoundary.")

        # Trust boundary check: Model decisions crossing trust boundary must have fallbacks
        trust_boundaries = {b.boundary_id for b in config.boundaries if b.boundary_type == BoundaryType.TRUST}
        for rule in config.decision_rules:
            if rule.kind == DecisionKind.MODEL and not rule.fallback:
                warnings.append(f"Model decision rule {rule.rule_id} has no fallback specified.")

        # Interface SLAs check
        for contract in config.interface_contracts:
            if contract.sla_ms <= 0:
                errors.append(f"Interface contract {contract.contract_id} has invalid SLA {contract.sla_ms}ms.")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


@dataclass
class TelemetryTrace:
    trace_id: str
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    boundary_violations: List[str] = field(default_factory=list)
    degradation_mode: DegradationMode = DegradationMode.FULL_CAPABILITY

    def evaluate_budget(self, budget: ObservabilityBudget) -> Dict[str, bool]:
        return {
            "latency_ok": self.latency_ms <= budget.max_latency_ms,
            "cost_ok": self.cost_usd <= budget.max_cost_usd,
            "violations_ok": len(self.boundary_violations) == 0,
        }


def build_default_v11_architecture() -> SystemArchitectureConfig:
    """Constructs the canonical V11 integrated AI system architecture configuration."""
    boundaries = [
        SystemBoundary(
            boundary_id="b-sys-01",
            name="External API Gateway Boundary",
            boundary_type=BoundaryType.SYSTEM,
            description="Isolates public ingress from internal services.",
            enforced_by="API Gateway / Auth Middleware",
        ),
        SystemBoundary(
            boundary_id="b-trust-01",
            name="Model Tool Execution Boundary",
            boundary_type=BoundaryType.TRUST,
            description="Isolates model generated tool parameters prior to side-effect execution.",
            enforced_by="Deterministic Tool Validator Guard",
        ),
        SystemBoundary(
            boundary_id="b-data-01",
            name="Vector Store Retrieval Boundary",
            boundary_type=BoundaryType.DATA,
            description="Enforces tenant data isolation and access controls during vector search.",
            enforced_by="Tenant ACL Filter",
        ),
    ]

    decision_rules = [
        DecisionRule(
            rule_id="r-auth-01",
            name="Authentication Guard",
            kind=DecisionKind.DETERMINISTIC,
            condition="request.header.has(Authorization)",
            handler="JWTValidator",
            fallback="FailClosed",
        ),
        DecisionRule(
            rule_id="r-route-01",
            name="Intent Classification Router",
            kind=DecisionKind.MODEL,
            condition="request.body.query != None",
            handler="LLMIntentClassifier",
            fallback="KeywordFallbackRouter",
        ),
    ]

    contracts = [
        InterfaceContract(
            contract_id="c-retrieval-01",
            source_component="RetrievalService",
            target_component="ContextAssembler",
            schema_version="v1.0",
            sla_ms=300,
            validation_rule="assert len(chunks) > 0 and all(c.score >= 0.5 for c in chunks)",
        ),
        InterfaceContract(
            contract_id="c-model-01",
            source_component="ContextAssembler",
            target_component="LLMInference",
            schema_version="v1.0",
            sla_ms=1200,
            validation_rule="assert prompt_tokens <= 4096",
        ),
    ]

    return SystemArchitectureConfig(
        name="V11 Integrated AI System Architecture",
        version="v11.0",
        boundaries=boundaries,
        decision_rules=decision_rules,
        interface_contracts=contracts,
        default_degradation=DegradationMode.FULL_CAPABILITY,
        budget=ObservabilityBudget(),
    )
