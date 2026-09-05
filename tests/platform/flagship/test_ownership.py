from __future__ import annotations

import subprocess

import pytest

from tests.platform.flagship.conftest import FROZEN_BASE, REPO_ROOT

ALLOWED_PREFIXES = (
    "platform/fixtures/flagship/",
    "platform/backend/app/core/flagship.py",
    "platform/backend/app/core/projection.py",
    "platform/backend/app/api/routes.py",
    "tests/platform/flagship/",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_diff_paths_stay_inside_lane_ownership() -> None:
    probe = _git("cat-file", "-e", f"{FROZEN_BASE}^{{commit}}")
    if probe.returncode != 0:
        pytest.skip(f"FROZEN_BASE {FROZEN_BASE} is not in this clone (shallow checkout)")
    diff = _git("diff", "--name-only", FROZEN_BASE)
    assert diff.returncode == 0, diff.stderr
    changed = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
    assert changed, "lane must add flagship index, core, and tests"
    forbidden = [
        path
        for path in changed
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    ]
    assert forbidden == [], f"paths outside G6 53 ownership: {forbidden}"
