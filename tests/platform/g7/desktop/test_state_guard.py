from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"


def test_state_guard_clean_on_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, str(STATE_GUARD)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PASSED" in completed.stdout
