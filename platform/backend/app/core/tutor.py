"""Provider-neutral tutor boundary. Secrets stay on the server."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from app.core.errors import AppError, NotFoundError, TutorUnavailableError, ValidationAppError
from app.core.security import is_secret_key_name

TUTOR_ROLES = ("NAVIGATOR", "SOCRATIC", "DEBUGGER", "FEYNMAN")
NO_AI_POLICY = "NO_AI_REQUIRED"
UNSTARTED_STAGE_SENTINELS = frozenset({"", "start"})
HEURISTIC_PROVIDERS = frozenset({"heuristic", "local", "stub"})
ROLE_ALIASES = {
    "LEARNER": "SOCRATIC",
    "TUTOR": "SOCRATIC",
    "SOCRATIC_TUTOR": "SOCRATIC",
    "NAV": "NAVIGATOR",
    "REVIEWER": "FEYNMAN",
    "FEYNMAN_REVIEWER": "FEYNMAN",
}

_ROLE_PROBES = {
    "NAVIGATOR": (
        "What is the smallest next evidence-producing action on this stage?",
        "Which mission invariant would fail if you skipped the current work?",
    ),
    "SOCRATIC": (
        "What do you predict will happen, and which assumption drives that prediction?",
        "What observation would falsify your current causal model?",
    ),
    "DEBUGGER": (
        "What is the exact symptom, and can you reproduce it with a smaller case?",
        "Which measurement would discriminate between your remaining hypotheses?",
    ),
    "FEYNMAN": (
        "Explain the mechanism in plain language, then give one concrete example.",
        "Which term is still jargon, and what would a newcomer need to see?",
    ),
}


class LLMProvider(ABC):
    """Abstract adapter. Callers must not bind this interface to a vendor SDK."""

    name: str = "none"

    @abstractmethod
    async def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        raise NotImplementedError


class HeuristicProvider(LLMProvider):
    """Local Socratic stub. No network, no vendor SDK, never a worked solution."""

    name = "heuristic"

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        del temperature
        user = ""
        for message in messages:
            if str(message.get("role") or "") == "user":
                user = str(message.get("content") or "")
        role = "SOCRATIC"
        for candidate in TUTOR_ROLES:
            if candidate in system_prompt:
                role = candidate
                break
        probes = _ROLE_PROBES[role]
        index = sum(ord(ch) for ch in user) % len(probes) if user else 0
        return (
            f"{role} guidance: I will not complete the exercise or reveal a worked solution. "
            f"{probes[index]} "
            "Name the evidence you already have and what observation would falsify your current model."
        )


def provider_name() -> str:
    return os.environ.get("LEARNINGOS_TUTOR_PROVIDER", "").strip().lower()


def provider_configured() -> bool:
    return provider_name() in HEURISTIC_PROVIDERS


def resolve_provider() -> LLMProvider | None:
    if provider_name() in HEURISTIC_PROVIDERS:
        return HeuristicProvider()
    return None


def normalize_role(raw: str | None) -> str:
    value = (raw or "").strip().upper()
    value = ROLE_ALIASES.get(value, value)
    if value in TUTOR_ROLES:
        return value
    return "SOCRATIC"


def assistance_policy_of(stage: dict[str, Any] | None) -> str:
    if not stage:
        return ""
    return str(stage.get("assistance_policy") or "").strip().upper()


def assistance_locked(stage: dict[str, Any] | None) -> bool:
    return assistance_policy_of(stage) == NO_AI_POLICY


def guidance_mode(policy: str) -> str:
    value = (policy or "").strip().upper()
    if value == NO_AI_POLICY:
        return "locked"
    if value in {"SOCRATIC_ONLY", "RESTRICTED_HINTS"}:
        return "socratic"
    return "unrestricted"


def system_prompt_for(role: str) -> str:
    return (
        f"You are the {role} tutor for a local learning runtime. "
        "Ask targeted questions. Never provide a copy-paste solution, completed code, "
        "or protected answers. Respect a locked no-assistance stage by refusing to guide."
    )


def scrub_secrets(text: str) -> str:
    if not text:
        return text
    redacted = text
    for key, value in os.environ.items():
        if not value or len(value) < 8 or not is_secret_key_name(key):
            continue
        if value in redacted:
            redacted = redacted.replace(value, "[redacted]")
    return redacted


def stage_by_id(spec: dict[str, Any], stage_id: str) -> dict[str, Any]:
    stages = spec.get("stages")
    if not isinstance(stages, list):
        raise NotFoundError("Stage not found in mission spec", details={"stage_id": stage_id})
    for item in stages:
        if isinstance(item, dict) and str(item.get("id") or "") == stage_id:
            return item
    raise NotFoundError("Stage not found in mission spec", details={"stage_id": stage_id})


def _current_stage(spec: dict[str, Any], session: dict[str, Any]) -> dict[str, Any] | None:
    current_id = str(session.get("current_stage_id") or "").strip()
    if current_id in UNSTARTED_STAGE_SENTINELS:
        return None
    try:
        return stage_by_id(spec, current_id)
    except NotFoundError:
        return None


def _refuse_locked(*, session_id: str, stage_id: str, policy: str) -> None:
    raise AppError(
        "ASSISTANCE_PROHIBITED",
        "Assistance is prohibited for this stage. Complete the no-AI attempt without the tutor.",
        403,
        {
            "session_id": session_id,
            "stage_id": stage_id,
            "assistance_policy": policy or NO_AI_POLICY,
        },
    )


async def handle_tutor_chat(
    *,
    session: dict[str, Any],
    spec: dict[str, Any],
    stage_id: str,
    role: str,
    prompt: str,
) -> dict[str, Any]:
    """Generate guidance only when a provider is configured and the stage is not locked."""
    if not provider_configured():
        raise TutorUnavailableError("No tutor provider is configured")

    text = (prompt or "").strip()
    if not text:
        raise ValidationAppError("prompt is required")

    session_id = str(session.get("session_id") or session.get("id") or "")
    requested = stage_by_id(spec, stage_id)
    current = _current_stage(spec, session)
    if assistance_locked(requested):
        _refuse_locked(
            session_id=session_id,
            stage_id=stage_id,
            policy=assistance_policy_of(requested),
        )
    if assistance_locked(current):
        _refuse_locked(
            session_id=session_id,
            stage_id=str(current.get("id") if current else stage_id),
            policy=assistance_policy_of(current),
        )

    provider = resolve_provider()
    if provider is None:
        raise TutorUnavailableError("No tutor provider is configured")

    chosen_role = normalize_role(role)
    reply = await provider.generate_response(
        messages=[{"role": "user", "content": text}],
        system_prompt=system_prompt_for(chosen_role),
    )
    policy = assistance_policy_of(requested)
    return {
        "role": chosen_role,
        "reply": scrub_secrets(reply),
        "provider": provider.name,
        "assistance_policy": policy,
        "learner": {
            "session_id": session_id,
            "stage_id": str(requested.get("id") or stage_id),
            "stage_type": str(requested.get("type") or ""),
            "assistance_policy": policy,
            "guidance_mode": guidance_mode(policy),
        },
    }
