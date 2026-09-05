"""G7 secret policy and package integrity.

Provider secrets may live in:
  - the process environment (runtime only, never HTTP)
  - the macOS keychain (preferred when available)
  - ``$LEARNINGOS_HOME/secrets/<NAME>`` with mode 0600 (fallback)

They must never be written into the Git worktree, frontend source, ``VITE_*``
bindings, or ``GET /system/config``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

KEYCHAIN_SERVICE = "dev.learningos.secrets"
SECRET_DIRNAME = "secrets"
SECRET_FILE_MODE = 0o600
SECRET_DIR_MODE = 0o700
SHA256SUMS_NAME = "SHA256SUMS"
SHA256_HEX_LEN = 64

ALLOWED_SECRET_LOCATIONS = (
    "process_environment",
    "macos_keychain",
    "learningos_home_file",
)
FORBIDDEN_SECRET_LOCATIONS = (
    "git_worktree",
    "frontend_source",
    "vite_env",
    "http_responses",
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
_FALSEY = frozenset({"0", "false", "no", "off"})


class SecretPolicyError(ValueError):
    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class IntegrityError(SecretPolicyError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "BAD_DIGEST",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class KeychainBackend(Protocol):
    def available(self) -> bool: ...
    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class NullKeychain:
    """Used when macOS keychain is missing (CI) or explicitly disabled."""

    def available(self) -> bool:
        return False

    def get(self, name: str) -> str | None:
        del name
        return None

    def set(self, name: str, value: str) -> None:
        del name, value
        raise SecretPolicyError("macOS keychain is not available", code="KEYCHAIN_UNAVAILABLE")

    def delete(self, name: str) -> None:
        del name


@dataclass
class MemoryKeychain:
    """In-memory keychain double for tests."""

    items: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def available(self) -> bool:
        return self.enabled

    def get(self, name: str) -> str | None:
        return self.items.get(name)

    def set(self, name: str, value: str) -> None:
        if not self.enabled:
            raise SecretPolicyError("macOS keychain is not available", code="KEYCHAIN_UNAVAILABLE")
        self.items[name] = value

    def delete(self, name: str) -> None:
        self.items.pop(name, None)


class MacOSKeychain:
    """Generic-password items via the ``security`` CLI. Never log values."""

    service = KEYCHAIN_SERVICE

    def available(self) -> bool:
        if os.environ.get("LEARNINGOS_USE_KEYCHAIN", "1").strip().lower() in _FALSEY:
            return False
        if sys.platform != "darwin":
            return False
        return shutil.which("security") is not None

    def get(self, name: str) -> str | None:
        try:
            completed = subprocess.run(
                ["security", "find-generic-password", "-s", self.service, "-a", name, "-w"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        value = completed.stdout.rstrip("\n")
        return value or None

    def set(self, name: str, value: str) -> None:
        subprocess.run(
            ["security", "delete-generic-password", "-s", self.service, "-a", name],
            check=False,
            capture_output=True,
            timeout=5,
        )
        try:
            completed = subprocess.run(
                [
                    "security",
                    "add-generic-password",
                    "-s",
                    self.service,
                    "-a",
                    name,
                    "-w",
                    value,
                    "-U",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SecretPolicyError(
                "macOS keychain write failed",
                code="KEYCHAIN_UNAVAILABLE",
                details={"reason": str(exc)},
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise SecretPolicyError(
                "macOS keychain write failed",
                code="KEYCHAIN_UNAVAILABLE",
                details={"reason": detail or f"exit {completed.returncode}"},
            )

    def delete(self, name: str) -> None:
        subprocess.run(
            ["security", "delete-generic-password", "-s", self.service, "-a", name],
            check=False,
            capture_output=True,
            timeout=5,
        )


def is_secret_key_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def secret_policy() -> dict[str, Any]:
    return {
        "allowed_locations": list(ALLOWED_SECRET_LOCATIONS),
        "forbidden_locations": list(FORBIDDEN_SECRET_LOCATIONS),
        "preferred": "macos_keychain",
        "fallback": "learningos_home_file",
        "file_mode": format(SECRET_FILE_MODE, "o").zfill(4),
        "public_config_omits_secrets": True,
        "vite_bindings": False,
    }


def detect_repo_root(start: Path | None = None) -> Path | None:
    origin = start if start is not None else Path(__file__).resolve()
    for candidate in origin.parents:
        has_platform = (candidate / "platform" / "backend" / "app").is_dir()
        has_git = (candidate / ".git").exists() or (candidate / ".git").is_file()
        has_arch = (candidate / "architecture" / "learningos-v3").is_dir()
        if has_platform and (has_git or has_arch):
            return candidate
    return None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def resolve_data_home() -> Path:
    raw = os.environ.get("LEARNINGOS_HOME") or "~/.learningos"
    return Path(raw).expanduser().resolve()


def normalize_secret_name(name: str) -> str:
    text = (name or "").strip().replace("-", "_").upper()
    if not text or not all(ch.isalnum() or ch == "_" for ch in text) or text[0].isdigit():
        raise SecretPolicyError(
            "Secret names must be environment-style identifiers",
            code="FORBIDDEN_NAME",
            details={"name": name},
        )
    if text.startswith("VITE_"):
        raise SecretPolicyError(
            "Provider secrets must not use VITE_ bindings",
            code="FORBIDDEN_NAME",
            details={"name": name, "location": "vite_env"},
        )
    if not is_secret_key_name(text):
        raise SecretPolicyError(
            "Name is not a provider secret identifier",
            code="FORBIDDEN_NAME",
            details={"name": name},
        )
    return text


def _secret_name_candidates(provider: str) -> list[str]:
    raw = (provider or "").strip()
    if not raw:
        return []
    names = [raw, raw.upper().replace("-", "_")]
    key = names[-1]
    if not key.endswith("_API_KEY"):
        names.append(f"{key}_API_KEY")
    seen: set[str] = set()
    ordered: list[str] = []
    for item in names:
        if item in seen:
            continue
        seen.add(item)
        try:
            ordered.append(normalize_secret_name(item))
        except SecretPolicyError:
            continue
    return ordered


def _public_key_allowed(name: str) -> bool:
    if name.upper().startswith("VITE_"):
        return False
    return not is_secret_key_name(name)


def public_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop secret-named keys. ``GET /system/config`` must never include them."""
    return {str(key): value for key, value in payload.items() if _public_key_allowed(str(key))}


def default_keychain() -> KeychainBackend:
    backend = MacOSKeychain()
    if backend.available():
        return backend
    return NullKeychain()


class SecretStore:
    def __init__(
        self,
        *,
        keychain: KeychainBackend | None = None,
        repo_root: Path | None = None,
        data_home: Path | None = None,
    ) -> None:
        self.keychain = keychain if keychain is not None else default_keychain()
        self.repo_root = repo_root if repo_root is not None else detect_repo_root()
        self._data_home = data_home

    def data_home(self) -> Path:
        if self._data_home is not None:
            return Path(self._data_home).expanduser().resolve()
        return resolve_data_home()

    def secrets_dir(self) -> Path:
        return self.data_home() / SECRET_DIRNAME

    def file_path(self, name: str) -> Path:
        return self.secrets_dir() / normalize_secret_name(name)

    def _assert_file_location_allowed(self, path: Path) -> None:
        home = self.data_home()
        resolved = path.resolve()
        if self.repo_root is not None and (
            _is_within(home, self.repo_root) or _is_within(resolved, self.repo_root)
        ):
            raise SecretPolicyError(
                "Refusing to store secrets inside the Git worktree",
                code="FORBIDDEN_LOCATION",
                details={
                    "path": str(resolved),
                    "data_home": str(home),
                    "repo_root": str(self.repo_root),
                    "location": "git_worktree",
                },
            )
        if "frontend" in resolved.parts:
            raise SecretPolicyError(
                "Refusing to store secrets in frontend source",
                code="FORBIDDEN_LOCATION",
                details={"path": str(resolved), "location": "frontend_source"},
            )

    def _write_file(self, path: Path, value: str) -> None:
        self._assert_file_location_allowed(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, SECRET_DIR_MODE)
        tmp = path.with_name(f".{path.name}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, SECRET_FILE_MODE)
        try:
            os.write(fd, value.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, SECRET_FILE_MODE)
        os.replace(tmp, path)
        os.chmod(path, SECRET_FILE_MODE)

    def persist(self, name: str, value: str) -> str:
        normalized = normalize_secret_name(name)
        if not value:
            raise SecretPolicyError("Secret value is required", code="EMPTY_SECRET", details={"name": normalized})
        if self.keychain.available():
            try:
                self.keychain.set(normalized, value)
                return "macos_keychain"
            except SecretPolicyError:
                pass
        path = self.file_path(normalized)
        self._write_file(path, value)
        return "learningos_home_file"

    def lookup_persisted(self, name: str) -> str | None:
        for candidate in _secret_name_candidates(name) or [normalize_secret_name(name)]:
            if self.keychain.available():
                value = self.keychain.get(candidate)
                if value:
                    return value
            path = self.file_path(candidate)
            try:
                if not path.is_file():
                    continue
                try:
                    os.chmod(path, SECRET_FILE_MODE)
                except OSError:
                    pass
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                return text
        return None

    def delete(self, name: str) -> None:
        normalized = normalize_secret_name(name)
        if self.keychain.available():
            self.keychain.delete(normalized)
        path = self.file_path(normalized)
        try:
            if path.is_file():
                self._assert_file_location_allowed(path)
                path.unlink()
        except SecretPolicyError:
            raise
        except OSError:
            return


def get_store() -> SecretStore:
    return SecretStore()


def persist_secret(name: str, value: str, *, store: SecretStore | None = None) -> str:
    return (store or get_store()).persist(name, value)


def lookup_persisted_secret(name: str, *, store: SecretStore | None = None) -> str | None:
    try:
        return (store or get_store()).lookup_persisted(name)
    except SecretPolicyError:
        return None


def resolve_secret(name: str, *, store: SecretStore | None = None) -> str | None:
    for candidate in _secret_name_candidates(name):
        value = os.environ.get(candidate) or os.environ.get(name)
        if value:
            return value
    env_direct = os.environ.get((name or "").strip())
    if env_direct:
        return env_direct
    return lookup_persisted_secret(name, store=store)


def delete_secret(name: str, *, store: SecretStore | None = None) -> None:
    (store or get_store()).delete(name)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256_hex(value: str) -> bool:
    if len(value) != SHA256_HEX_LEN:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise IntegrityError(
            f"Unable to read SHA256SUMS at {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise IntegrityError(
                f"Invalid SHA256SUMS line {line_no}",
                details={"path": str(path), "line": line_no},
            )
        digest, rel = parts[0].strip().lower(), parts[1].strip()
        if rel.startswith("*"):
            rel = rel[1:]
        if not _is_sha256_hex(digest):
            raise IntegrityError(
                f"Invalid SHA256 digest on line {line_no}",
                details={"path": str(path), "line": line_no},
            )
        if not rel or rel.startswith("/") or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise IntegrityError(
                f"SHA256SUMS path is not a safe package-relative file on line {line_no}",
                details={"path": rel, "line": line_no},
            )
        entries[rel] = digest
    if not entries:
        raise IntegrityError(
            f"SHA256SUMS has no checksum entries under {path}",
            details={"path": str(path)},
        )
    return entries


def verify_package_checksums(package_dir: Path | str, *, required: bool = True) -> dict[str, str]:
    """Fail closed on digest mismatch. ``required=True`` also rejects a missing SHA256SUMS."""
    root = Path(package_dir).expanduser().resolve()
    checksums_path = root / SHA256SUMS_NAME
    if not checksums_path.is_file():
        if required:
            raise IntegrityError(
                f"Missing SHA256SUMS under {root}",
                code="MISSING_CHECKSUMS",
                details={"path": str(checksums_path)},
            )
        return {}
    listed = parse_sha256sums(checksums_path)
    for rel, expected in listed.items():
        file_path = (root / rel).resolve()
        if not _is_within(file_path, root):
            raise IntegrityError(
                f"SHA256SUMS path escapes package directory: {rel}",
                details={"path": rel},
            )
        if not file_path.is_file():
            raise IntegrityError(
                f"SHA256SUMS references missing file {rel}",
                details={"path": rel},
            )
        actual = sha256_file(file_path)
        if actual != expected:
            raise IntegrityError(
                f"Checksum mismatch for {rel}",
                details={"path": rel, "expected": expected, "actual": actual},
            )
    return listed
