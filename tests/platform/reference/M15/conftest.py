from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
M15_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M15"

for candidate in (REPO_ROOT, BACKEND_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)
