from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CLI = REPO_ROOT / "tools" / "platform" / "secrets.py"
G3_PACKAGE = REPO_ROOT / "platform" / "worker" / "fixtures" / "g3_curriculum"


def _run(args: list[str], *, env: dict[str, str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=str(REPO_ROOT),
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_policy_set_get_and_verify_package(isolated_home: Path) -> None:
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(isolated_home)
    env["LEARNINGOS_USE_KEYCHAIN"] = "0"
    policy = _run(["policy"], env=env)
    assert policy.returncode == 0, policy.stderr
    payload = json.loads(policy.stdout)
    assert payload["preferred"] == "macos_keychain"
    assert payload["fallback"] == "learningos_home_file"
    assert "vite_env" in payload["forbidden_locations"]

    stored = _run(["set", "OPENAI_API_KEY", "--value", "sk-cli-secret"], env=env)
    assert stored.returncode == 0, stored.stderr
    assert "learningos_home_file" in stored.stdout
    path = isolated_home / "secrets" / "OPENAI_API_KEY"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    got = _run(["get", "OPENAI_API_KEY"], env=env)
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == "sk-cli-secret"

    ok = _run(["verify-package", str(G3_PACKAGE)], env=env)
    assert ok.returncode == 0, ok.stderr
    assert "SHA256SUMS ok" in ok.stdout

    missing = _run(["verify-package", str(isolated_home)], env=env)
    assert missing.returncode == 1
    assert "SHA256SUMS" in missing.stderr or "integrity" in missing.stderr.lower()
