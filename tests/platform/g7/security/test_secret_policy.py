from __future__ import annotations

import shutil
import stat
import tempfile
from pathlib import Path

import pytest

from app.core.secrets import (
    MemoryKeychain,
    NullKeychain,
    SecretPolicyError,
    SecretStore,
    delete_secret,
    persist_secret,
    public_payload,
    resolve_secret,
    secret_policy,
)
from app.core.security import get_provider_secret

REPO_ROOT = Path(__file__).resolve().parents[4]
FRONTEND_ROOT = REPO_ROOT / "platform" / "frontend"


def test_policy_documents_server_side_locations_only() -> None:
    policy = secret_policy()
    assert policy["preferred"] == "macos_keychain"
    assert policy["fallback"] == "learningos_home_file"
    assert policy["file_mode"] == "0600"
    assert policy["vite_bindings"] is False
    assert policy["public_config_omits_secrets"] is True
    assert "process_environment" in policy["allowed_locations"]
    for forbidden in ("git_worktree", "frontend_source", "vite_env", "http_responses"):
        assert forbidden in policy["forbidden_locations"]


def test_file_fallback_is_mode_0600(isolated_home: Path) -> None:
    store = SecretStore(keychain=NullKeychain(), data_home=isolated_home, repo_root=REPO_ROOT)
    location = persist_secret("OPENAI_API_KEY", "sk-test-file-mode", store=store)
    assert location == "learningos_home_file"
    path = isolated_home / "secrets" / "OPENAI_API_KEY"
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert resolve_secret("OPENAI", store=store) == "sk-test-file-mode"
    assert path.read_text(encoding="utf-8") == "sk-test-file-mode"
    assert not (REPO_ROOT / "secrets").exists()
    assert not (FRONTEND_ROOT / "secrets").exists()


def test_prefers_mocked_keychain_and_skips_home_file(isolated_home: Path) -> None:
    keychain = MemoryKeychain()
    store = SecretStore(keychain=keychain, data_home=isolated_home, repo_root=REPO_ROOT)
    location = persist_secret("OPENAI_API_KEY", "sk-test-keychain", store=store)
    assert location == "macos_keychain"
    assert keychain.get("OPENAI_API_KEY") == "sk-test-keychain"
    secrets_dir = isolated_home / "secrets"
    assert not secrets_dir.exists() or not any(secrets_dir.iterdir())
    assert resolve_secret("OPENAI_API_KEY", store=store) == "sk-test-keychain"


def test_disabled_keychain_falls_back_to_file(isolated_home: Path) -> None:
    keychain = MemoryKeychain(enabled=False)
    store = SecretStore(keychain=keychain, data_home=isolated_home, repo_root=REPO_ROOT)
    location = persist_secret("ANTHROPIC_API_KEY", "sk-ant-file", store=store)
    assert location == "learningos_home_file"
    assert not keychain.items
    path = isolated_home / "secrets" / "ANTHROPIC_API_KEY"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_refuses_vite_bindings_and_worktree_home(isolated_home: Path) -> None:
    store = SecretStore(keychain=NullKeychain(), data_home=isolated_home, repo_root=REPO_ROOT)
    with pytest.raises(SecretPolicyError) as vite:
        persist_secret("VITE_OPENAI_API_KEY", "sk-nope", store=store)
    assert vite.value.code == "FORBIDDEN_NAME"

    fake_root = Path(tempfile.mkdtemp(prefix="learningos-g7-repo-", dir="/tmp"))
    try:
        (fake_root / ".git").mkdir()
        (fake_root / "platform" / "frontend" / "src").mkdir(parents=True)
        home = fake_root / "learningos-home"
        home.mkdir()
        nested = SecretStore(keychain=NullKeychain(), data_home=home, repo_root=fake_root)
        with pytest.raises(SecretPolicyError) as worktree:
            persist_secret("OPENAI_API_KEY", "sk-nope", store=nested)
        assert worktree.value.code == "FORBIDDEN_LOCATION"
        assert not (home / "secrets").exists()
    finally:
        shutil.rmtree(fake_root, ignore_errors=True)


def test_env_overrides_persisted_and_config_redacts_secret_keys(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SecretStore(keychain=NullKeychain(), data_home=isolated_home, repo_root=REPO_ROOT)
    persist_secret("OPENAI_API_KEY", "sk-persisted-value", store=store)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-value")
    assert resolve_secret("OPENAI_API_KEY", store=store) == "sk-env-value"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert get_provider_secret("OPENAI") == "sk-persisted-value"
    delete_secret("OPENAI_API_KEY", store=store)
    assert get_provider_secret("OPENAI") is None

    cleaned = public_payload(
        {
            "data_home": str(isolated_home),
            "openai_api_key": "sk-should-drop",
            "VITE_OPENAI_API_KEY": "sk-should-drop",
            "api_prefix": "/api/v1",
        }
    )
    assert cleaned == {"data_home": str(isolated_home), "api_prefix": "/api/v1"}


def test_system_config_omits_persisted_secrets(isolated_home: Path) -> None:
    store = SecretStore(keychain=NullKeychain(), data_home=isolated_home, repo_root=REPO_ROOT)
    persist_secret("OPENAI_API_KEY", "sk-must-not-leak-config", store=store)
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.get("/api/v1/system/config")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["data_home"] == str(isolated_home.resolve())
    assert "sk-must-not-leak-config" not in response.text
    assert "OPENAI_API_KEY" not in response.text
    assert "VITE_" not in response.text
    assert set(payload) == {
        "data_home",
        "database_path",
        "worker_socket",
        "bind_host",
        "api_prefix",
    }


def test_policy_module_has_no_vendor_or_pandas_import() -> None:
    source = (REPO_ROOT / "platform" / "backend" / "app" / "core" / "secrets.py").read_text(
        encoding="utf-8"
    )
    assert "import openai" not in source
    assert "import pandas" not in source
    assert "from openai" not in source
    assert "from pandas" not in source
    req = (REPO_ROOT / "platform" / "backend" / "requirements.txt").read_text(encoding="utf-8")
    assert "openai" not in req
