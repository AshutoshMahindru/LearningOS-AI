from __future__ import annotations

import json

from app.core.security import get_provider_secret, is_secret_key_name


def _assert_no_secret_keys(obj) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower = str(key).lower()
            assert "api_key" not in lower
            assert "apikey" not in lower
            assert "secret" not in lower
            assert "password" not in lower
            assert lower not in {"token", "auth_token", "access_token"}
            assert not is_secret_key_name(str(key)) or key in {
                "data_home",
                "database_path",
                "worker_socket",
                "bind_host",
                "api_prefix",
            }
            _assert_no_secret_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_secret_keys(item)


def test_config_omits_provider_secrets_even_if_env_set(client, data_home, monkeypatch):
    secret = "sk-test-openai-key-do-not-leak"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    (data_home / "config.json").write_text(
        json.dumps(
            {
                "theme": "dark",
                "openai_api_key": secret,
                "api_key": secret,
                "provider_secret": secret,
                "auth_token": "should-not-appear",
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/api/v1/system/config")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["bind_host"] == "127.0.0.1"
    assert payload["api_prefix"] == "/api/v1"
    assert payload["data_home"] == str(data_home.resolve())
    assert payload["database_path"] == str((data_home / "learningos.db").resolve())
    assert "worker_socket" in payload
    assert secret not in response.text
    assert "OPENAI_API_KEY" not in response.text
    assert "sk-test" not in response.text
    _assert_no_secret_keys(payload)
    assert set(payload) == {
        "data_home",
        "database_path",
        "worker_socket",
        "bind_host",
        "api_prefix",
    }
    assert get_provider_secret("OPENAI") == secret
    assert get_provider_secret("OPENAI_API_KEY") == secret
