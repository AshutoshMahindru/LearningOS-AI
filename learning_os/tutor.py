from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from .closed_loop import LearningLoop
from .mission_context import MissionContextAssembler
from .mission_player import MissionPlayer


class TutorEngine:
    """Context-aware Socratic tutor with an optional OpenAI Responses backend.

    The tutor is deliberately unavailable during the no-AI mission-player step.
    If OPENAI_API_KEY is not configured, a deterministic local Socratic coach
    remains available so the learning workflow never depends on network access.
    """

    def __init__(self, root: str | Path, loop: LearningLoop) -> None:
        self.root = Path(root)
        self.loop = loop
        self.context = MissionContextAssembler(self.root, loop.gates)
        self.player = MissionPlayer(self.root, loop)

    def _history(self) -> list[dict[str, Any]]:
        payload = self.loop.store.read("tutor_sessions.json", [])
        return payload if isinstance(payload, list) else []

    def _save_turn(self, turn: dict[str, Any]) -> None:
        history = self._history()
        history.append(turn)
        self.loop.store.write("tutor_sessions.json", history[-200:])

    def history(self, mission_id: str) -> list[dict[str, Any]]:
        mid = self.loop.missions.get(mission_id)["id"]
        return [item for item in self._history() if item.get("mission_id") == mid][-30:]

    def _fallback(self, mission: dict[str, Any], step: dict[str, Any], message: str) -> str:
        competencies = mission.get("competencies", [])
        focus = competencies[0] if competencies else "the mission objective"
        text = message.lower()
        if any(token in text for token in ("answer", "tell me", "solution")):
            return (
                f"I will not jump straight to the finished answer. For {focus}, state your current explanation in one or two sentences, "
                "then name one observation that would prove it wrong."
            )
        if any(token in text for token in ("stuck", "don't know", "dont know", "confused")):
            return (
                f"Use the current step — {step['title']} — to shrink the problem. What are the input, output and mutable state of the smallest component you can inspect?"
            )
        return (
            f"Before I add information, make a prediction. In the current step ({step['title']}), what do you expect to happen and what evidence would distinguish your explanation from the nearest alternative?"
        )

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        pieces: list[str] = []
        for item in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    pieces.append(content["text"].strip())
        return "\n".join(piece for piece in pieces if piece)

    def _remote(self, prompt: str) -> tuple[str, str]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return "", "local"
        model = os.getenv("LEARNINGOS_OPENAI_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
        body = json.dumps({"model": model, "input": prompt}).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = self._extract_text(payload)
            return (text, model) if text else ("", "local")
        except (error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return "", "local"

    def ask(self, mission_id: str, message: str) -> dict[str, Any]:
        mid = self.loop.missions.get(mission_id)["id"]
        question = str(message or "").strip()
        if not question:
            raise ValueError("Tutor message is required")
        player = self.player.view(mid)
        if player["current_step"] == "no_ai":
            raise ValueError("Tutor assistance is locked during the no-AI gate step")
        context = self.context.build(mid)
        mission = context["mission"]
        active = next((item for item in player["steps"] if item["active"]), player["steps"][0])
        evidence = self.loop.evidence.for_mission(mid)[-6:]
        prior = self.history(mid)[-6:]
        prompt = (
            "You are the LearningOS Socratic tutor. Help the learner build transferable competence, not merely finish a task. "
            "Do not claim evidence the learner has not produced. Prefer one precise question or a small diagnostic experiment over a lecture. "
            "Do not provide a complete no-AI gate answer. Calibrate depth to the mission and current step.\n\n"
            f"Mission: {mid} — {mission['title']}\nObjective: {mission['objective']}\n"
            f"Competencies: {', '.join(mission.get('competencies', []))}\nCurrent player step: {active['title']} — {active['verb']}\n"
            f"Gate: {context['gate']['status']} — {context['gate']['reasons']}\n"
            f"Recent evidence: {[item.get('summary') for item in evidence]}\n"
            f"Recent tutor turns: {[(item.get('user'), item.get('assistant')) for item in prior]}\n\nLearner: {question}\nTutor:"
        )
        answer, provider = self._remote(prompt)
        if not answer:
            answer = self._fallback(mission, active, question)
            provider = "local-socratic"
        turn = {
            "mission_id": mid,
            "step": active["id"],
            "user": question,
            "assistant": answer,
            "provider": provider,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_turn(turn)
        return {"turn": turn, "history": self.history(mid), "locked": False}
