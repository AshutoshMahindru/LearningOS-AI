from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evidence_engine import EvidenceEngine
from .gate_engine import GateEngine
from .mission_loader import MissionRepository
from .mission_runner import MissionRunner
from .storage import StateStore


def runtime(root: Path) -> tuple[MissionRunner, EvidenceEngine]:
    store = StateStore(root)
    evidence = EvidenceEngine(store)
    gates = GateEngine(evidence)
    runner = MissionRunner(MissionRepository(root), store, gates)
    return runner, evidence


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="learning-os", description="Local-first AI Learning OS runtime")
    p.add_argument("--root", default=os.environ.get("LEARNING_OS_ROOT", "."), help="Repository root")
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("mission")
    sub.add_parser("status")
    sub.add_parser("next")
    gate = sub.add_parser("gate")
    gate.add_argument("mission", nargs="?")
    review = sub.add_parser("review")
    review.add_argument("mission", nargs="?")
    ev = sub.add_parser("evidence")
    ev.add_argument("mission")
    ev.add_argument("--type", default="artifact")
    ev.add_argument("--summary", required=True)
    ev.add_argument("--competency", action="append", default=[])
    ev.add_argument("--no-ai", action="store_true")
    ev.add_argument("--transfer", action="store_true")
    ev.add_argument("--explanation", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    root = Path(args.root).resolve()
    runner, evidence = runtime(root)
    if args.command == "start":
        mission = runner.start(args.mission)
        print(f"MISSION {mission['id']} - {mission['title']}")
        print(f"Objective: {mission['objective']}")
        print("Runtime: Whole -> Map -> Interrogate -> Manipulate -> Zoom if blocked -> Decompose -> Rebuild -> Break -> Explain -> No-AI Transfer -> Gate")
        print("Evidence: artifact + explanation + unseen transfer + no-AI execution")
    elif args.command == "status":
        print(json.dumps(runner.status(), indent=2))
    elif args.command == "next":
        print(runner.next_action())
    elif args.command in {"gate", "review"}:
        status = runner.status()
        mission = args.mission or (status["mission"] or {}).get("id")
        if not mission:
            raise SystemExit("No mission selected. Start a mission first.")
        print(json.dumps(runner.gate(mission), indent=2))
    elif args.command == "evidence":
        record = evidence.add(args.mission, args.type, args.summary, args.competency, args.no_ai, args.transfer, args.explanation)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
