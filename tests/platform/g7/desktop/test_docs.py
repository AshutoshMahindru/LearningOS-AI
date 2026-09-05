from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from tools.desktop import launch
from tools.platform import install

REPO_ROOT = Path(__file__).resolve().parents[4]
DESKTOP_README = REPO_ROOT / "tools" / "desktop" / "README.md"
PACKAGING_LAUNCHER = REPO_ROOT / "packaging" / "learningos"

FORBIDDEN = (
    "jupyter",
    "pip install",
    "npm ci",
    "npm install",
    "python3 -m venv",
    "python -m venv",
    "virtualenv",
    "create a venv",
    "create a virtualenv",
)


def _help_text(main) -> str:
    buffer = io.StringIO()
    with pytest.raises(SystemExit) as exited:
        with redirect_stdout(buffer):
            main(["--help"])
    assert exited.value.code == 0
    return buffer.getvalue()


def test_learner_facing_install_path_does_not_require_jupyter_or_manual_venv() -> None:
    readme = DESKTOP_README.read_text(encoding="utf-8")
    assert DESKTOP_README.is_file()
    assert "python3 tools/desktop/launch.py" in readme
    assert "LEARNINGOS_HOME" in readme
    lowered = "\n".join(
        [
            readme,
            _help_text(install.main),
            _help_text(launch.main),
            PACKAGING_LAUNCHER.read_text(encoding="utf-8"),
        ]
    ).lower()
    for phrase in FORBIDDEN:
        assert phrase not in lowered, f"learner-facing install path mentions {phrase!r}"
