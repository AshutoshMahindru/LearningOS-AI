from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .closed_loop import LearningLoop


class LabRunner:
    """Inspect and execute the canonical mission notebook from the local app."""

    def __init__(self, root: str | Path, loop: LearningLoop) -> None:
        self.root = Path(root).resolve()
        self.loop = loop

    def _notebook(self, mission_id: str) -> Path:
        mid = self.loop.missions.get(mission_id)["id"]
        matches = sorted((self.root / "labs").glob(f"{mid}_*.ipynb"))
        if not matches:
            raise ValueError(f"No repository notebook found for {mid}")
        return matches[0].resolve()

    def inspect(self, mission_id: str) -> dict[str, Any]:
        path = self._notebook(mission_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = payload.get("cells", []) if isinstance(payload, dict) else []
        code = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "code"]
        markdown = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "markdown"]
        executed = [cell for cell in code if cell.get("execution_count") is not None]
        return {
            "mission_id": self.loop.missions.get(mission_id)["id"],
            "path": path.relative_to(self.root).as_posix(),
            "cells": len(cells),
            "code_cells": len(code),
            "markdown_cells": len(markdown),
            "executed_cells": len(executed),
        }

    def run(self, mission_id: str, timeout_seconds: int = 240) -> dict[str, Any]:
        path = self._notebook(mission_id)
        mid = self.loop.missions.get(mission_id)["id"]
        timeout = max(30, min(int(timeout_seconds), 600))
        output_dir = self.root / "tracking" / "lab_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_name = f"{mid}_{stamp}.ipynb"
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(path),
            "--output",
            output_name,
            "--output-dir",
            str(output_dir),
            f"--ExecutePreprocessor.timeout={timeout}",
        ]
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout + 30,
                check=False,
            )
            status = "PASS" if completed.returncode == 0 else "FAIL"
            result = {
                "mission_id": mid,
                "status": status,
                "returncode": completed.returncode,
                "source": path.relative_to(self.root).as_posix(),
                "executed_notebook": (Path("tracking") / "lab_runs" / output_name).as_posix() if status == "PASS" else None,
                "stdout": completed.stdout[-6000:],
                "stderr": completed.stderr[-6000:],
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "mission_id": mid,
                "status": "TIMEOUT",
                "returncode": None,
                "source": path.relative_to(self.root).as_posix(),
                "executed_notebook": None,
                "stdout": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        runs = self.loop.store.read("lab_runs.json", [])
        if not isinstance(runs, list):
            runs = []
        runs.append(result)
        self.loop.store.write("lab_runs.json", runs[-100:])
        return result

    def recent(self, mission_id: str) -> list[dict[str, Any]]:
        mid = self.loop.missions.get(mission_id)["id"]
        runs = self.loop.store.read("lab_runs.json", [])
        if not isinstance(runs, list):
            return []
        return [item for item in runs if item.get("mission_id") == mid][-10:]
