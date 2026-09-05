"""Learner docs and README must stay on the V3 product path."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DOC_SUFFIXES = {".md", ".txt", ".rst"}

BROWSER_OPENAI_INSTRUCTION = re.compile(
    r"(set|export|put|paste|add|store)\s+`?OPENAI_API_KEY"
    r"|OPENAI_API_KEY\s*(in|=|to)\s*(the\s+)?(browser|frontend|vite|env)"
    r"|VITE_OPENAI",
    re.IGNORECASE,
)
SANDBOX_LATER = re.compile(r"sandbox is later", re.IGNORECASE)
JUPYTER_NOT_REQUIRED = re.compile(
    r"jupyter is not required|not required.{0,40}jupyter|jupyter.{0,40}not required",
    re.IGNORECASE | re.DOTALL,
)


def documentation_files() -> list[Path]:
    files = [REPO_ROOT / "README.md"]
    docs = REPO_ROOT / "docs"
    files.extend(
        path for path in sorted(docs.rglob("*")) if path.is_file() and path.suffix.lower() in DOC_SUFFIXES
    )
    return files


def combined_docs() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in documentation_files())


def test_docs_do_not_mention_worker_daemon() -> None:
    hits = [
        str(path.relative_to(REPO_ROOT))
        for path in documentation_files()
        if "worker_daemon" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_docs_do_not_instruct_openai_in_the_browser() -> None:
    hits = []
    for path in documentation_files():
        text = path.read_text(encoding="utf-8")
        if BROWSER_OPENAI_INSTRUCTION.search(text):
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == []


def test_readme_points_at_canonical_worker_not_later_sandbox() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "platform/worker/daemon.py" in readme
    assert SANDBOX_LATER.search(readme) is None
    assert "platform/backend/worker_daemon.py" not in readme


def test_readme_learner_path_is_installer_not_pip_npm() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:bash)?\n(.*?)```", readme, re.DOTALL)
    assert blocks, "README should include a launch command"
    first = blocks[0]
    assert "tools/platform/install.py" in first
    assert "pip " not in first
    assert "npm " not in first


def test_learner_docs_cover_install_data_backup_and_offline() -> None:
    text = combined_docs()
    assert "tools/platform/install.py" in text
    assert "tools/desktop" in text
    assert "LEARNINGOS_HOME" in text
    assert "dest_home" in text
    assert JUPYTER_NOT_REQUIRED.search(text)
    assert re.search(r"offline", text, re.IGNORECASE)
    assert "Git worktree" in text or "git worktree" in text.lower()
