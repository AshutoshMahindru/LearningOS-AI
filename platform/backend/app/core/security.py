from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

from fastapi import Request

from app.core.config import get_settings
from app.core.errors import UnauthorizedError

AUTH_TOKEN_FILENAME = ".auth_token"
TOKEN_BYTES = 32
LOOPBACK_HOSTS = frozenset(
    {
        "127.0.0.1",
        "::1",
        "localhost",
        "testclient",
        "::ffff:127.0.0.1",
    }
)
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
    "private_key",
)


def auth_token_path() -> Path:
    return get_settings().data_home / AUTH_TOKEN_FILENAME


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip().lower().strip("[]")
    if normalized in LOOPBACK_HOSTS:
        return True
    if normalized.startswith("127."):
        return True
    return False


def require_loopback(request: Request) -> None:
    host = request.client.host if request.client else None
    if not is_loopback_host(host):
        raise UnauthorizedError(
            "This endpoint is only available from loopback",
            status_code=403,
            details={"client_host": host},
        )


def _write_token_file(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def read_auth_token() -> str | None:
    path = auth_token_path()
    try:
        if not path.is_file():
            return None
        token = path.read_text(encoding="ascii").strip()
        return token or None
    except OSError:
        return None


def ensure_auth_token() -> str:
    path = auth_token_path()
    existing = read_auth_token()
    if existing:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return existing
    token = secrets.token_hex(TOKEN_BYTES)
    _write_token_file(path, token)
    return token


def rotate_auth_token() -> str:
    token = secrets.token_hex(TOKEN_BYTES)
    _write_token_file(auth_token_path(), token)
    return token


def require_auth(request: Request) -> str:
    header = request.headers.get("authorization")
    if not header:
        raise UnauthorizedError("Missing Authorization header")
    scheme, separator, credentials = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not credentials.strip():
        raise UnauthorizedError("Invalid Authorization header")
    expected = read_auth_token()
    provided = credentials.strip()
    if not expected or not hmac.compare_digest(provided, expected):
        raise UnauthorizedError("Invalid bearer token")
    return provided


def is_secret_key_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def get_provider_secret(provider: str) -> str | None:
    """Read a provider secret from the environment. Never return this over HTTP."""
    raw = (provider or "").strip()
    if not raw:
        return None
    candidates = [raw, raw.upper(), raw.lower()]
    key = raw.upper().replace("-", "_")
    if key.endswith("_API_KEY"):
        candidates.append(key)
    else:
        candidates.append(f"{key}_API_KEY")
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        value = os.environ.get(name)
        if value:
            return value
    return None
