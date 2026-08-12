from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RouteDecision:
    action: str
    reason: str
    target: str | None = None


class AdaptiveRouter:
    """Routes the learner between advance, repair, retention, and zoom-in."""

    def decide(
        self,
        mission: dict[str, Any],
        blockers: list[str],
        gate_status: str | None,
        retention_due: bool = False,
    ) -> RouteDecision:
        if blockers:
            return RouteDecision(
                "ZOOM_IN",
                "A blocker prevents progress; retrieve the narrowest prerequisite.",
                blockers[-1],
            )
        if retention_due:
            return RouteDecision(
                "RETENTION",
                "Previously learned competency requires retrieval practice.",
                mission.get("id"),
            )
        if gate_status == "PASS":
            return RouteDecision(
                "ADVANCE",
                "Mission evidence satisfies the current gate.",
                mission.get("id"),
            )
        if gate_status in {"PARTIAL", "FAIL"}:
            return RouteDecision(
                "REPAIR",
                "Evidence is insufficient; target the missing competency.",
                mission.get("id"),
            )
        return RouteDecision(
            "CONTINUE",
            "Mission is active and requires execution.",
            mission.get("id"),
        )
