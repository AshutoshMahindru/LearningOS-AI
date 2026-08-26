from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .closed_loop import LearningLoop
from .content_router import ContentRouter
from .dashboard import DashboardService
from .dashboard_server import serve_app, serve_dashboard
from .lab_registry import LabRegistry
from .mission_context import MissionContextAssembler
from .mission_loader import MissionRepository
from .prerequisite_graph import PrerequisiteGraph
from .storage import StateStore


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="learning-os", description="Local-first AI Learning OS runtime")
    p.add_argument("--root", default=os.environ.get("LEARNING_OS_ROOT", "."), help="Repository root")
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start"); start.add_argument("mission")
    sub.add_parser("status"); sub.add_parser("next"); sub.add_parser("step")
    gate = sub.add_parser("gate"); gate.add_argument("mission", nargs="?")
    review = sub.add_parser("review"); review.add_argument("mission", nargs="?")
    ev = sub.add_parser("evidence")
    ev.add_argument("mission"); ev.add_argument("--type", default="artifact"); ev.add_argument("--summary", required=True)
    ev.add_argument("--competency", action="append", default=[]); ev.add_argument("--no-ai", action="store_true")
    ev.add_argument("--transfer", action="store_true"); ev.add_argument("--explanation", action="store_true")
    context = sub.add_parser("context"); context.add_argument("mission", nargs="?"); context.add_argument("--blocker"); context.add_argument("--depth", choices=["L1","L2","L3","L4","L5"])
    resources = sub.add_parser("resources"); resources.add_argument("mission"); resources.add_argument("--blocker"); resources.add_argument("--depth", choices=["L1","L2","L3","L4","L5"])
    prereqs = sub.add_parser("prereqs"); prereqs.add_argument("mission")
    lab = sub.add_parser("lab-status"); lab.add_argument("mission")
    sub.add_parser("learner-model")
    retention = sub.add_parser("retention"); retention.add_argument("action", choices=["due", "complete"]); retention.add_argument("event_id", nargs="?"); retention.add_argument("--fail", action="store_true")
    autonomy = sub.add_parser("autonomy"); autonomy.add_argument("action", choices=["status", "evaluate"]); autonomy.add_argument("--signal", action="append", default=[])
    sq = sub.add_parser("sidequest"); sq.add_argument("action", choices=["list", "open", "close"]); sq.add_argument("--mission"); sq.add_argument("--target"); sq.add_argument("--reason"); sq.add_argument("--return-target"); sq.add_argument("--minutes", type=int, default=60); sq.add_argument("--id"); sq.add_argument("--assessment", choices=["PASS","PARTIAL","FAIL"]); sq.add_argument("--outcome", default="")
    app = sub.add_parser("app", help="Launch the interactive Learning OS learner app"); app.add_argument("--host", default="127.0.0.1"); app.add_argument("--port", type=int, default=8765)
    dash = sub.add_parser("dashboard", help="Legacy dashboard command and JSON projection"); dash.add_argument("--mission"); dash.add_argument("--serve", action="store_true"); dash.add_argument("--host", default="127.0.0.1"); dash.add_argument("--port", type=int, default=8765)
    return p


def main() -> None:
    args = parser().parse_args()
    root = Path(args.root).resolve()
    loop = LearningLoop(root)
    if args.command == "start":
        mission = loop.start(args.mission)
        print(f"MISSION {mission['id']} - {mission['title']}")
        print(f"Objective: {mission['objective']}")
        print("Runtime: Whole -> Map -> Interrogate -> Manipulate -> Zoom if blocked -> Decompose -> Rebuild -> Break -> Explain -> No-AI Transfer -> Gate")
    elif args.command == "status": print(json.dumps(loop.runner.status(), indent=2))
    elif args.command in {"next", "step"}: print(json.dumps(loop.step(), indent=2))
    elif args.command in {"gate", "review"}:
        current = loop.runner.status(); mission = args.mission or (current["mission"] or {}).get("id")
        if not mission: raise SystemExit("No mission selected. Start a mission first.")
        print(json.dumps(loop.gate(mission), indent=2))
    elif args.command == "evidence": print(json.dumps(loop.record_evidence(args.mission, args.type, args.summary, args.competency, args.no_ai, args.transfer, args.explanation), indent=2))
    elif args.command == "context":
        current = loop.runner.status(); mission = args.mission or (current["mission"] or {}).get("id")
        if not mission: raise SystemExit("No mission selected. Start a mission first.")
        print(json.dumps(MissionContextAssembler(root, loop.gates).build(mission, args.blocker, args.depth), indent=2))
    elif args.command == "resources": print(json.dumps(ContentRouter(root).route(MissionRepository(root).get(args.mission), args.blocker, args.depth), indent=2))
    elif args.command == "prereqs":
        graph = PrerequisiteGraph(root); raw = StateStore(root).learner()
        print(json.dumps({"mission_id": args.mission.upper(), "blocking": graph.blocking(args.mission), "helpful": graph.helpful(args.mission), "unmet": graph.unmet(args.mission, raw.get("mission_status", {}))}, indent=2))
    elif args.command == "lab-status": print(json.dumps(LabRegistry(root).status(args.mission), indent=2))
    elif args.command == "learner-model": print(json.dumps(loop.learner.get(), indent=2))
    elif args.command == "retention":
        if args.action == "due": print(json.dumps(loop.retention.due(), indent=2))
        else:
            if not args.event_id: raise SystemExit("retention complete requires event_id")
            print(json.dumps(loop.retention.complete(args.event_id, not args.fail), indent=2))
    elif args.command == "autonomy": print(json.dumps(loop.autonomy.status() if args.action == "status" else loop.autonomy.evaluate(set(args.signal)), indent=2))
    elif args.command == "sidequest":
        if args.action == "list": print(json.dumps(loop.side_quests.active(args.mission), indent=2))
        elif args.action == "open":
            if not all([args.mission, args.target, args.reason, args.return_target]): raise SystemExit("sidequest open requires --mission --target --reason --return-target")
            print(json.dumps(loop.side_quests.open(args.mission, args.target, args.reason, args.return_target, args.minutes), indent=2))
        else:
            if not args.id or not args.assessment: raise SystemExit("sidequest close requires --id --assessment")
            print(json.dumps(loop.side_quests.close(args.id, args.assessment, args.outcome), indent=2))
    elif args.command == "app":
        serve_app(root, args.host, args.port)
    elif args.command == "dashboard":
        if args.serve:
            serve_dashboard(root, args.host, args.port)
        else:
            print(json.dumps(DashboardService(root).snapshot(args.mission), indent=2))


if __name__ == "__main__":
    main()
