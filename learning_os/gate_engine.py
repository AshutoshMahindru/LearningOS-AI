from __future__ import annotations

from dataclasses import dataclass

from .evidence_engine import EvidenceEngine


@dataclass(frozen=True)
class GateResult:
    status: str
    reasons: list[str]


class GateEngine:
    """Minimum apprenticeship gate: artifact + explanation + transfer + no-AI evidence."""

    def __init__(self, evidence: EvidenceEngine) -> None:
        self.evidence = evidence

    def evaluate(self, mission_id: str) -> GateResult:
        records = self.evidence.for_mission(mission_id)
        reasons: list[str] = []
        if not records:
            return GateResult("FAIL", ["No evidence recorded for mission."])
        if not any(r.get("type") in {"artifact", "lab", "build", "review"} for r in records):
            reasons.append("Missing independent deliverable/artifact evidence.")
        if not any(r.get("explanation") for r in records):
            reasons.append("Missing explanation evidence.")
        if not any(r.get("transfer") for r in records):
            reasons.append("Missing unseen transfer evidence.")
        if not any(r.get("no_ai") for r in records):
            reasons.append("Missing no-AI competency evidence.")
        if not reasons:
            return GateResult("PASS", [])
        if len(reasons) <= 2:
            return GateResult("PARTIAL", reasons)
        return GateResult("FAIL", reasons)
