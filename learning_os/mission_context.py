from __future__ import annotations

from pathlib import Path
from typing import Any

from .content_router import ContentRouter
from .gate_engine import GateEngine
from .knowledge_graph import KnowledgeGraph
from .lab_registry import LabRegistry
from .mission_loader import MissionRepository
from .prerequisite_graph import PrerequisiteGraph
from .storage import StateStore


class MissionContextAssembler:
    def __init__(self, root: str | Path, gates: GateEngine) -> None:
        self.root = Path(root)
        self.missions = MissionRepository(root)
        self.store = StateStore(root)
        self.graph = PrerequisiteGraph(root)
        self.knowledge = KnowledgeGraph(root)
        self.content = ContentRouter(root)
        self.labs = LabRegistry(root)
        self.gates = gates

    def build(self, mission_id: str, blocker: str | None = None, max_depth: str | None = None) -> dict[str, Any]:
        mission = self.missions.get(mission_id)
        learner = self.store.learner()
        status = learner.get("mission_status", {})
        gate = self.gates.evaluate(mission["id"])
        unmet = self.graph.unmet(mission["id"], status)
        effective_blocker = blocker or (unmet[0] if unmet else None)

        knowledge_introduced = self.knowledge.nodes_for_mission(mission["id"])
        blocker_context = self.knowledge.context_for(blocker) if blocker else None

        return {
            "mission": mission,
            "gate": {"status": gate.status, "reasons": gate.reasons},
            "prerequisites": {
                "blocking": self.graph.blocking(mission["id"]),
                "helpful": self.graph.helpful(mission["id"]),
                "unmet": unmet,
                "zoom_target": effective_blocker,
            },
            "knowledge": {
                "introduced": knowledge_introduced,
                "blocker_context": blocker_context,
            },
            "lab": self.labs.status(mission["id"]),
            "content": self.content.route(mission, blocker=blocker, max_depth=max_depth),
            "learner": {
                "current_mission": learner.get("current_mission"),
                "blockers": learner.get("blockers", []),
            },
        }
