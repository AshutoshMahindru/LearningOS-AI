from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from tests.api.helpers import (
    RUNTIME_MISSION_ID,
    assert_no_forbidden_payload,
    assert_typed_error,
    seed_runtime_mission,
)

TUTOR_BODY = {
    "session_id": "sess-1",
    "stage_id": "stage-1",
    "role": "SOCRATIC",
    "prompt": "Why does this fail?",
}

FORBIDDEN_SDK_ROOTS = frozenset({"openai", "anthropic", "google", "httpx", "requests"})


def _tutor_source_path() -> Path:
    return Path(__file__).resolve().parents[2] / "app" / "core" / "tutor.py"


def _start_session(client, auth_headers) -> dict:
    from app.db.database import get_connection

    conn = get_connection()
    try:
        seed_runtime_mission(conn)
    finally:
        conn.close()
    learner = client.post(
        "/api/v1/learners",
        json={"username": "tutor-learner", "display_name": "Tutor"},
        headers=auth_headers,
    )
    assert learner.status_code == 200, learner.text
    learner_id = learner.json()["learner_id"]
    session = client.post(
        "/api/v1/sessions",
        json={"mission_id": RUNTIME_MISSION_ID, "learner_id": learner_id},
        headers=auth_headers,
    )
    assert session.status_code == 200, session.text
    return {"learner_id": learner_id, "session_id": session.json()["session_id"], "headers": auth_headers}


def _chat(client, headers, session_id: str, stage_id: str, prompt: str = "What should I try next?", role: str = "SOCRATIC"):
    return client.post(
        "/api/v1/tutor/chat",
        json={"session_id": session_id, "stage_id": stage_id, "role": role, "prompt": prompt},
        headers=headers,
    )


def test_tutor_default_is_501_and_does_not_import_vendor_sdk(client, auth_headers, monkeypatch):
    monkeypatch.delenv("LEARNINGOS_TUTOR_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key-do-not-leak")
    sys.modules.pop("openai", None)
    response = client.post("/api/v1/tutor/chat", json=TUTOR_BODY, headers=auth_headers)
    error = assert_typed_error(response, 501, "TUTOR_NOT_AVAILABLE")
    assert_no_forbidden_payload(response)
    assert "openai" not in sys.modules
    assert "sk-test-openai-key-do-not-leak" not in response.text
    assert "TUTOR" in error["code"]


def test_tutor_unauthenticated_is_401(client):
    response = client.post("/api/v1/tutor/chat", json=TUTOR_BODY)
    assert_typed_error(response, 401, "UNAUTHORIZED")


def test_tutor_source_has_no_vendor_sdk_import():
    tree = ast.parse(_tutor_source_path().read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in FORBIDDEN_SDK_ROOTS
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in FORBIDDEN_SDK_ROOTS


def test_heuristic_tutor_guides_allowed_stage(client, auth_headers, monkeypatch):
    monkeypatch.setenv("LEARNINGOS_TUTOR_PROVIDER", "heuristic")
    sys.modules.pop("openai", None)
    ctx = _start_session(client, auth_headers)
    response = _chat(client, ctx["headers"], ctx["session_id"], "s1_orient", role="NAVIGATOR")
    assert response.status_code == 200, response.text
    assert_no_forbidden_payload(response)
    body = response.json()
    assert body["role"] == "NAVIGATOR"
    assert body["provider"] == "heuristic"
    assert body["assistance_policy"] == "UNRESTRICTED"
    assert "worked solution" in body["reply"].lower() or "not complete" in body["reply"].lower()
    assert body["learner"]["guidance_mode"] == "unrestricted"
    assert "openai" not in sys.modules
    assert "openai" not in json.dumps(body).lower()


def test_no_ai_stage_is_403_and_does_not_call_provider(client, auth_headers, monkeypatch):
    monkeypatch.setenv("LEARNINGOS_TUTOR_PROVIDER", "heuristic")
    called: list[str] = []

    async def forbidden(*_args, **_kwargs):
        called.append("generate")
        raise AssertionError("provider must not run on a locked stage")

    monkeypatch.setattr("app.core.tutor.HeuristicProvider.generate_response", forbidden)
    ctx = _start_session(client, auth_headers)
    response = _chat(client, ctx["headers"], ctx["session_id"], "s3_wrap")
    error = assert_typed_error(response, 403, "ASSISTANCE_PROHIBITED")
    assert_no_forbidden_payload(response)
    assert error["details"]["assistance_policy"] == "NO_AI_REQUIRED"
    assert called == []


def test_heuristic_unknown_session_is_404(client, auth_headers, monkeypatch):
    monkeypatch.setenv("LEARNINGOS_TUTOR_PROVIDER", "heuristic")
    response = client.post("/api/v1/tutor/chat", json=TUTOR_BODY, headers=auth_headers)
    assert_typed_error(response, 404, "NOT_FOUND")


def test_current_no_ai_stage_cannot_be_bypassed_with_another_stage_id(client, auth_headers, monkeypatch):
    monkeypatch.setenv("LEARNINGOS_TUTOR_PROVIDER", "heuristic")
    ctx = _start_session(client, auth_headers)
    session_id = ctx["session_id"]
    from app.db.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE mission_sessions SET current_stage_id = ? WHERE id = ?",
            ("s3_wrap", session_id),
        )
        conn.commit()
    finally:
        conn.close()

    response = _chat(client, ctx["headers"], session_id, "s1_orient")
    assert_typed_error(response, 403, "ASSISTANCE_PROHIBITED")


def test_tutor_reply_scrubs_provider_secrets(client, auth_headers, monkeypatch):
    secret = "sk-test-openai-key-do-not-leak"
    monkeypatch.setenv("LEARNINGOS_TUTOR_PROVIDER", "heuristic")
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    async def leak(*_args, **_kwargs):
        return f"contact support with {secret}"

    monkeypatch.setattr("app.core.tutor.HeuristicProvider.generate_response", leak)
    ctx = _start_session(client, auth_headers)
    response = _chat(client, ctx["headers"], ctx["session_id"], "s1_orient")
    assert response.status_code == 200, response.text
    assert secret not in response.text
    assert "[redacted]" in response.json()["reply"]


def test_system_config_still_omits_secrets_when_tutor_enabled(client, auth_headers, monkeypatch):
    secret = "sk-test-openai-key-do-not-leak"
    monkeypatch.setenv("LEARNINGOS_TUTOR_PROVIDER", "heuristic")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    response = client.get("/api/v1/system/config")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert secret not in response.text
    assert "OPENAI_API_KEY" not in response.text
    assert set(payload) == {
        "data_home",
        "database_path",
        "worker_socket",
        "bind_host",
        "api_prefix",
    }
    ctx = _start_session(client, auth_headers)
    chat = _chat(client, ctx["headers"], ctx["session_id"], "s1_orient")
    assert chat.status_code == 200, chat.text
    assert secret not in chat.text
