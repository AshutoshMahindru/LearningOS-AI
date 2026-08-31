from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
F01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "f01"
CLI_PATH = REPO_ROOT / "tools" / "authoring" / "cli.py"

for candidate in (REPO_ROOT, BACKEND_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)
