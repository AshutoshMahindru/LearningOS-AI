from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .closed_loop import LearningLoop
from .mission_context import MissionContextAssembler


PLAYER_STEPS: tuple[dict[str, str], ...] = (
    {"id": "whole", "title": "Whole first", "verb": "Run or inspect the useful whole before decomposing it."},
    {"id": "map", "title": "Map", "verb": "Name the important parts, boundaries, state and flows."},
    {"id": "interrogate", "title": "Interrogate", "verb": "Ask what changes, what stays fixed and what evidence would distinguish explanations."},
    {"id": "experiment", "title": "Experiment", "verb": "Predict first, manipulate one thing, observe, then explain the delta."},
    {"id": "break", "title": "Break it", "verb": "Trigger the controlled failure and diagnose it from evidence."},
    {"id": "explain", "title": "Explain", "verb": "Explain the mechanism and boundaries in your own words."},
    {"id": "no_ai", "title": "No-AI gate", "verb": "Work from memory without tutor assistance and record independent evidence."},
    {"id": "transfer", "title": "Transfer", "verb": "Apply the competency to a fresh scenario that was not used in the walkthrough."},
    {"id": "gate", "title": "Gate", "verb": "Submit the accumulated evidence to the formal mission gate."},
)


class MissionPlayer:
    """Guided projection over a mission's canonical apprenticeship sequence."""

    def __init__(self, root: str | Path, loop: LearningLoop) -> None:
        self.root = Path(root)
        self.loop = loop
        self.store = loop.store
        self.context = MissionContextAssembler(self.root, loop.gates)

    def _states(self) -> dict[str, Any]:
        payload = self.store.read("mission_player.json", {})
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _default_state(mission_id: str) -> dict[str, Any]:
        return {
            "mission_id": mission_id,
            "current_step": PLAYER_STEPS[0]["id"],
            "completed": [],
            "responses": {},
            "started_at": None,
            "updated_at": None,
        }

    def _state(self, mission_id: str) -> dict[str, Any]:
        states = self._states()
        raw = states.get(mission_id)
        state = dict(raw) if isinstance(raw, dict) else self._default_state(mission_id)
        state.setdefault("completed", [])
        state.setdefault("responses", {})
        state.setdefault("current_step", PLAYER_STEPS[0]["id"])
        return state

    def _save_state(self, mission_id: str, state: dict[str, Any]) -> None:
        states = self._states()
        states[mission_id] = state
        self.store.write("mission_player.json", states)

    def _read(self, mission_id: str, name: str, limit: int = 7000) -> str:
        path = self.root / "missions" / mission_id / name
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n\n[material truncated in player view]"

    def _material(self, mission_id: str, step_id: str, context: dict[str, Any]) -> dict[str, Any]:
        mission = context["mission"]
        competencies = mission.get("competencies", [])
        if step_id == "whole":
            return {
                "prompt": f"Start with the complete system for: {mission['objective']}",
                "material": self._read(mission_id, "README.md", 5000),
            }
        if step_id == "map":
            return {
                "prompt": "Draw the system before explaining individual mechanisms. Mark inputs, outputs, mutable state, boundaries and control/data flow.",
                "blocking_prerequisites": context["prerequisites"].get("blocking", []),
                "helpful_prerequisites": context["prerequisites"].get("helpful", []),
                "knowledge_introduced": context["knowledge"].get("introduced", []),
            }
        if step_id == "interrogate":
            questions = [
                f"What observable behavior would convince you that you understand {item}?"
                for item in competencies
            ]
            questions.extend([
                "Which state changes during the operation and which state must remain fixed?",
                "What is the smallest experiment that could falsify your current explanation?",
            ])
            return {
                "prompt": "Interrogate the system instead of memorising labels.",
                "questions": questions,
                "material": self._read(mission_id, "code_reading.md", 3500),
            }
        if step_id == "experiment":
            lab = context.get("lab", {})
            return {
                "prompt": "Write a prediction before each run. Change one consequential variable, inspect the output and explain the difference.",
                "lab": lab,
                "material": self._read(mission_id, "experiments.yaml", 6500),
            }
        if step_id == "break":
            return {
                "prompt": "Cause the intended failure. Diagnose it from observations before repairing it.",
                "material": self._read(mission_id, "controlled_failure.md", 6500),
            }
        if step_id == "explain":
            return {
                "prompt": (
                    f"Explain how the system achieves this objective without leaning on unexplained jargon: {mission['objective']} "
                    "State the mechanism, the evidence you observed, one limitation and one condition under which your explanation would fail."
                ),
                "competencies": competencies,
            }
        if step_id == "no_ai":
            return {
                "prompt": "Tutor assistance is locked for this step. Complete the task independently, then record no-AI evidence.",
                "material": self._read(mission_id, "no_ai_gate.md", 6500),
            }
        if step_id == "transfer":
            return {
                "prompt": "Apply the mission competencies to a fresh case. State assumptions and calibrated uncertainty rather than guessing.",
                "material": self._read(mission_id, "assessment.yaml", 6500),
            }
        gate = self.loop.gates.evaluate(mission_id)
        return {
            "prompt": "The formal gate is evidence-based. It cannot be completed by clicking through the player.",
            "gate": {"status": gate.status, "reasons": gate.reasons},
        }

    def _requirement_error(self, mission_id: str, step_id: str) -> str | None:
        records = self.loop.evidence.for_mission(mission_id)
        if step_id == "experiment" and not any(
            record.get("type") in {"artifact", "lab", "build", "review"} for record in records
        ):
            return "Record artifact, lab, build or review evidence before completing the experiment step."
        if step_id == "explain" and not any(record.get("explanation") for record in records):
            return "Record explanation evidence before completing the explain step."
        if step_id == "no_ai" and not any(record.get("no_ai") for record in records):
            return "Record evidence produced without AI assistance before completing the no-AI step."
        if step_id == "transfer" and not any(record.get("transfer") for record in records):
            return "Record unseen-transfer evidence before completing the transfer step."
        if step_id == "gate" and self.loop.gates.evaluate(mission_id).status != "PASS":
            return "The formal mission gate must PASS before the player can mark the gate step complete."
        return None

    def view(self, mission_id: str) -> dict[str, Any]:
        mid = self.loop.missions.get(mission_id)["id"]
        state = self._state(mid)
        context = self.context.build(mid)
        completed = set(state.get("completed", []))
        current = state.get("current_step")
        steps: list[dict[str, Any]] = []
        for spec in PLAYER_STEPS:
            step_id = spec["id"]
            steps.append({
                **spec,
                "complete": step_id in completed,
                "active": step_id == current,
                "requirement": self._requirement_error(mid, step_id),
                "response": state.get("responses", {}).get(step_id, ""),
                "content": self._material(mid, step_id, context),
            })
        return {
            "mission_id": mid,
            "current_step": current,
            "completed": list(state.get("completed", [])),
            "completion_ratio": round(len(completed) / len(PLAYER_STEPS), 4),
            "steps": steps,
        }

    def complete(self, mission_id: str, step_id: str, response: str = "") -> dict[str, Any]:
        mid = self.loop.missions.get(mission_id)["id"]
        wanted = str(step_id or "").strip().lower()
        known = [item["id"] for item in PLAYER_STEPS]
        if wanted not in known:
            raise ValueError(f"Unknown mission-player step: {step_id}")
        state = self._state(mid)
        current = state.get("current_step")
        if current == "complete":
            return self.view(mid)
        if wanted != current:
            raise ValueError(f"Complete the active step first: {current}")
        requirement = self._requirement_error(mid, wanted)
        if requirement:
            raise ValueError(requirement)

        now = datetime.now(timezone.utc).isoformat()
        if state.get("started_at") is None:
            state["started_at"] = now
        if wanted not in state["completed"]:
            state["completed"].append(wanted)
        cleaned = str(response or "").strip()
        if cleaned:
            state["responses"][wanted] = cleaned
        index = known.index(wanted)
        state["current_step"] = known[index + 1] if index + 1 < len(known) else "complete"
        state["updated_at"] = now
        self._save_state(mid, state)
        return self.view(mid)

    def reset(self, mission_id: str) -> dict[str, Any]:
        mid = self.loop.missions.get(mission_id)["id"]
        states = self._states()
        states.pop(mid, None)
        self.store.write("mission_player.json", states)
        return self.view(mid)
