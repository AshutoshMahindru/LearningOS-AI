from dataclasses import dataclass, field
from typing import List


@dataclass
class Competency:
    name: str
    level: int = 0
    evidence: List[str] = field(default_factory=list)


@dataclass
class MissionState:
    mission_id: str
    status: str = "not_started"
    blockers: List[str] = field(default_factory=list)


@dataclass
class LearnerState:
    learner: str
    current_mission: str
    competencies: List[Competency] = field(default_factory=list)

    def find_competency(self, name: str):
        return next((c for c in self.competencies if c.name == name), None)
