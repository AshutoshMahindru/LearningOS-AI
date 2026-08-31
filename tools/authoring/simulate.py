"""Dry-run a fixture stage sequence using a temporary LEARNINGOS_HOME."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .errors import AuthoringError
from .package import Package, load_package
from .paths import REPO_ROOT, is_within
from .validate import validate_package


@dataclass
class SimulateResult:
    ok: bool
    home: Path
    trace_path: Path
    trace: dict[str, Any]
    text: str


def _assert_external_home(home: Path, repo_root: Path) -> Path:
    resolved = home.expanduser().resolve(strict=False)
    if is_within(resolved, repo_root):
        raise AuthoringError(
            f"LEARNINGOS_HOME {resolved} is inside the Git worktree; refusing to simulate",
            code="DATA_HOME",
            details={"home": str(resolved), "repo": str(repo_root)},
        )
    return resolved


def allocate_home(repo_root: Path | None = None, home: Path | str | None = None) -> Path:
    repo = (repo_root or REPO_ROOT).resolve()
    if home is not None:
        target = _assert_external_home(Path(home), repo)
        target.mkdir(parents=True, exist_ok=True)
        return target
    created = Path(tempfile.mkdtemp(prefix="learningos-authoring-"))
    return _assert_external_home(created, repo)


@contextmanager
def isolated_learningos_home(home: Path) -> Iterator[Path]:
    previous = os.environ.get("LEARNINGOS_HOME")
    os.environ["LEARNINGOS_HOME"] = str(home)
    try:
        yield home
    finally:
        if previous is None:
            os.environ.pop("LEARNINGOS_HOME", None)
        else:
            os.environ["LEARNINGOS_HOME"] = previous


def _stage_actions(stage: dict[str, Any]) -> tuple[str, ...]:
    stage_type = stage.get("type")
    if stage_type == "experiment":
        return ("predict", "execute", "submit")
    if stage_type == "competency_gate":
        return ("enter", "evaluate_gate")
    return ("enter", "submit")


def _dry_run_mission(package: Package, mission: dict[str, Any], home: Path) -> dict[str, Any]:
    stages_out: list[dict[str, Any]] = []
    for stage in mission.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        actions = list(_stage_actions(stage))
        record: dict[str, Any] = {
            "id": stage.get("id"),
            "type": stage.get("type"),
            "assistance_policy": stage.get("assistance_policy"),
            "state": "COMPLETED",
            "actions": actions,
            "worker_invoked": False,
        }
        if stage.get("type") == "experiment":
            record["cycle"] = {
                "predict": {"status": "sealed", "mutable": False},
                "execute": {"status": "dry_run", "worker": False},
                "submit": {"status": "accepted"},
            }
        stages_out.append(record)

    contract = mission.get("gate_contract") if isinstance(mission.get("gate_contract"), dict) else {}
    evidence_count = len(contract.get("required_evidence") or [])
    return {
        "package_id": package.id,
        "package_version": package.version,
        "mission_id": mission.get("id"),
        "learningos_home": str(home),
        "worktree_writes": False,
        "simulated_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages_out,
        "gate": {
            "status": "PASSED",
            "pass_threshold": contract.get("pass_threshold"),
            "evidence_count": evidence_count,
            "mode": "dry_run",
        },
        "mission": "COMPLETED",
    }


def _format_trace(trace: dict[str, Any]) -> str:
    lines = [
        f"LEARNINGOS_HOME  {trace['learningos_home']}",
        f"Package          {trace['package_id']}@{trace['package_version']}",
        f"Mission          {trace['mission_id']}",
        "Worktree writes  no",
        "",
        "Stage sequence",
    ]
    for index, stage in enumerate(trace.get("stages") or [], start=1):
        actions = " -> ".join(stage.get("actions") or [])
        lines.append(
            f"  {index}. {stage.get('id')}  type={stage.get('type')}  "
            f"assistance={stage.get('assistance_policy')}  {actions}  {stage.get('state')}"
        )
    gate = trace.get("gate") or {}
    lines.append("")
    lines.append(
        f"Gate             {gate.get('status')}  "
        f"threshold={gate.get('pass_threshold')}  evidence={gate.get('evidence_count')}"
    )
    lines.append(f"Mission          {trace.get('mission')}")
    lines.append("Learner state stayed under LEARNINGOS_HOME (tmp); Git worktree untouched.")
    return "\n".join(lines) + "\n"


def simulate_package(
    path: Path | str | None = None,
    *,
    home: Path | str | None = None,
    repo_root: Path | None = None,
) -> SimulateResult:
    repo = (repo_root or REPO_ROOT).resolve()
    validation = validate_package(path)
    if not validation.ok or validation.package is None:
        raise AuthoringError(
            "Refusing to simulate an invalid package: " + "; ".join(validation.errors),
            code=validation.code,
            details={"errors": validation.errors},
        )
    package = validation.package
    target_home = allocate_home(repo, home)
    (target_home / "authoring").mkdir(parents=True, exist_ok=True)

    traces = [_dry_run_mission(package, mission, target_home) for mission in package.missions]
    payload = traces[0] if len(traces) == 1 else {"missions": traces, "learningos_home": str(target_home)}
    text = "".join(_format_trace(item) for item in traces)
    trace_path = target_home / "authoring" / "simulate-trace.json"

    with isolated_learningos_home(target_home):
        if is_within(Path(os.environ["LEARNINGOS_HOME"]), repo):
            raise AuthoringError(
                "LEARNINGOS_HOME resolved inside the Git worktree",
                code="DATA_HOME",
            )
        trace_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        # Intentionally do not create a .learningos directory; home *is* LEARNINGOS_HOME.

    return SimulateResult(ok=True, home=target_home, trace_path=trace_path, trace=payload, text=text)
