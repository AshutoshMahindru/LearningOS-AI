from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
FRONTEND_ROOT = REPO_ROOT / "platform" / "frontend"
SKIP_DIRS = {".git", "dist", "node_modules"}
SKIP_FILES = {"package-lock.json"}
SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".json", ".env"}


def _walk(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os_walk(root):
        directories[:] = [name for name in directories if name not in SKIP_DIRS]
        for name in names:
            if name in SKIP_FILES:
                continue
            path = current / name
            if path.suffix not in SOURCE_SUFFIXES:
                continue
            if path.name.endswith(".test.ts") or path.name.endswith(".test.tsx"):
                continue
            if path.name in {"setup.ts", "http.ts"}:
                continue
            files.append(path)
    return files


def os_walk(root: Path):
    for current, directories, names in os.walk(root):
        yield Path(current), directories, names


def test_frontend_source_has_no_provider_or_vite_secrets() -> None:
    openai = "_".join(("OPENAI", "API", "KEY"))
    anthropic = "_".join(("ANTHROPIC", "API", "KEY"))
    vite = "VITE" + "_"
    hits: list[str] = []
    for path in _walk(FRONTEND_ROOT):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(FRONTEND_ROOT).as_posix()
        if openai in text:
            hits.append(f"{relative}: {openai}")
        if anthropic in text:
            hits.append(f"{relative}: {anthropic}")
        if vite in text:
            hits.append(f"{relative}: {vite} binding")
        if "sk-" in text:
            hits.append(f"{relative}: sk- prefix")
    assert hits == []
