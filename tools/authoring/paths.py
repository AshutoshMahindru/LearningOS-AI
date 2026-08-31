"""Repository and fixture locations for G4 authoring tools."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _pin_stdlib_platform() -> None:
    """Bind stdlib `platform` before jsonschema/attrs import it.

    The repository has a `platform/` tree. Running from the repo root would
    otherwise make `import platform` resolve to that tree.
    """
    current = sys.modules.get("platform")
    if current is not None and hasattr(current, "python_implementation"):
        return
    stdlib_path = os.path.join(os.path.dirname(os.__file__), "platform.py")
    spec = importlib.util.spec_from_file_location("platform", stdlib_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["platform"] = module
    spec.loader.exec_module(module)


_pin_stdlib_platform()

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
WP136_SCHEMA_PATH = (
    REPO_ROOT
    / "architecture"
    / "learningos-v3"
    / "03_technical_architecture"
    / "WP-136_mission_definition_schema.json"
)
F01_PACKAGE_DIR = REPO_ROOT / "platform" / "fixtures" / "f01"

F01_PACKAGE_ID = "g4.fixture.f01"
F01_PACKAGE_VERSION = "4.0.0"
SYNTHETIC_MISSION_ID = "M00"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(parent.expanduser().resolve(strict=False))
        return True
    except ValueError:
        return False
