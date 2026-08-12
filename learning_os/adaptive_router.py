from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    action: str
    reason: str
    target: str | None = None


class AdaptiveRouter:
    """Routes among advance, repair, retention, zoom-in, and mission execution."""

    def decide(
        self,
        mission: dict[str, Any],
        blockers: list[str],
        gate_status: str | None,
        retention_due: bool = False,
        unmet_prerequisites: list[str] | None = None,
    ) -> RouteDecision:
        unmet_prerequisites = unmet_prerequisites or []
        if blockers:
            return RouteDecision("ZOOM_IN", "A named blocker prevents progress.", blockers[-1])
        if unmet_prerequisites:
            return RouteDecision("ZOOM_IN", "A blocking prerequisite is not yet evidenced as passed.", unmet_prerequisites[0])
        if retention_due:
            return RouteDecision("RETENTION", "Previously learned competency requires retrieval practice.", mission.get("id"))
        if gate_status == "PASS":
            return RouteDecision("ADVANCE", "Mission evidence satisfies the current gate.", mission.get("id"))
        if gate_status in {"PARTIAL", "FAIL"}:
            return RouteDecision("REPAIR", "Evidence is insufficient; target only the missing gate evidence.", mission.get("id"))
        return RouteDecision("CONTINUE", "Mission is active and requires execution.", mission.get("id"))
