from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "data/missions.json", "data/mission_dependencies.json", "data/source_registry.json",
    "data/content_registry.json", "data/lab_status.json", "data/autonomy_policy.json", "data/apprenticeship_controls.yaml",
    "data/knowledge_graph.yaml", "data/knowledge_graph.csv", "learning_os/knowledge_graph.py",
    "learning_os/cli.py", "learning_os/closed_loop.py", "learning_os/learner_model.py", "learning_os/retention_engine.py",
    "learning_os/autonomy_engine.py", "learning_os/side_quest_engine.py", "learning_os/decision_engine.py", "learning_os/dashboard.py",
    "learning_os/dashboard_server.py", "learning_os/mission_loader.py", "learning_os/mission_runner.py", "learning_os/gate_engine.py",
    "learning_os/storage.py", "learning_os/retrieval.py", "learning_os/content_router.py", "learning_os/prerequisite_graph.py",
    "learning_os/mission_context.py", "web/dashboard.html", "prompts/pedagogical_orchestrator.md", "prompts/zoom_controller.md",
    "tracking/learner_state.json", "tracking/learner_model.json", "tracking/retention_events.json", "tracking/side_quests.json",
    "tracking/autonomy_events.json", "schemas/evidence.schema.json"
]
errors: list[str] = []
for rel in REQUIRED:
    if not (ROOT / rel).exists(): errors.append(f"missing required path: {rel}")

mission_ids = [f"M{i:02d}" for i in range(1, 43)]
missions_path = ROOT / "data" / "missions.json"
if missions_path.exists():
    missions = json.loads(missions_path.read_text(encoding="utf-8")).get("missions", [])
    ids = [m.get("id") for m in missions]
    if len(missions) != 42: errors.append(f"expected 42 missions; found {len(missions)}")
    if ids != mission_ids: errors.append("mission IDs must be contiguous M01..M42")

sources_path, content_path = ROOT / "data" / "source_registry.json", ROOT / "data" / "content_registry.json"
if sources_path.exists() and content_path.exists():
    sources = {s["id"] for s in json.loads(sources_path.read_text(encoding="utf-8"))["sources"]}
    for item in json.loads(content_path.read_text(encoding="utf-8"))["content"]:
        missing = set(item.get("source_ids", [])) - sources
        if missing: errors.append(f"content {item['id']} references missing sources: {sorted(missing)}")
        unknown = set(item.get("missions", [])) - set(mission_ids)
        if unknown: errors.append(f"content {item['id']} references unknown missions: {sorted(unknown)}")

dep_path = ROOT / "data" / "mission_dependencies.json"
if dep_path.exists():
    deps = json.loads(dep_path.read_text(encoding="utf-8"))["dependencies"]
    if set(deps) != set(mission_ids): errors.append("dependency graph must cover M01..M42 exactly")
    for mid, spec in deps.items():
        for dep in spec.get("blocking", []) + spec.get("helpful", []):
            if dep not in mission_ids: errors.append(f"{mid} references unknown dependency {dep}")
        for dep in spec.get("blocking", []):
            if int(dep[1:]) >= int(mid[1:]): errors.append(f"blocking dependency must precede mission: {mid} -> {dep}")

# Canonical concept dependency graph integrity.
kg_path = ROOT / "data" / "knowledge_graph.csv"
if kg_path.exists():
    with kg_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("id", "").strip() for row in rows]
    concepts = [row.get("concept", "").strip() for row in rows]
    concept_keys = [concept.casefold() for concept in concepts]
    concept_set = set(concept_keys)

    if len(rows) != 253: errors.append(f"canonical knowledge graph must contain 253 nodes; found {len(rows)}")
    if len(set(ids)) != len(ids): errors.append("canonical knowledge graph IDs must be unique")
    if len(concept_set) != len(concepts): errors.append("canonical knowledge graph concepts must be unique case-insensitively")
    for required_id in ["K001", "K252", "KTXT"]:
        if required_id not in set(ids): errors.append(f"canonical knowledge graph missing required node {required_id}")

    def refs(value: str | None) -> list[str]:
        return [part.strip() for part in (value or "").split("|") if part.strip()]

    for row in rows:
        node_id = row.get("id", "").strip() or "<missing-id>"
        if not row.get("domain", "").strip(): errors.append(f"{node_id}: missing domain")
        if not row.get("concept", "").strip(): errors.append(f"{node_id}: missing concept")
        first_mission = row.get("first_mission", "").strip()
        if first_mission and first_mission not in mission_ids:
            errors.append(f"{node_id}: invalid first_mission {first_mission}")
        for relation_name in ["prerequisites", "enables"]:
            for concept in refs(row.get(relation_name)):
                if concept.casefold() not in concept_set:
                    errors.append(f"{node_id}: unresolved {relation_name} concept {concept!r}")

if (ROOT / "data" / "knowledge_graph.bootstrap.json").exists():
    errors.append("obsolete data/knowledge_graph.bootstrap.json must not remain after canonical graph migration")

lab_path = ROOT / "data" / "lab_status.json"
if lab_path.exists():
    labs = json.loads(lab_path.read_text(encoding="utf-8"))
    expected_executable = [f"M{i:02d}" for i in range(1, 42)]
    source_executable = ["M01", "M02", "M03", "M08"]
    expected_source_spec_only = [mid for mid in mission_ids if mid not in source_executable]
    if labs.get("repository_executable") != expected_executable:
        errors.append("repository executable lab inventory must be exactly M01..M41")
    if labs.get("source_package_executable") != source_executable:
        errors.append("source package executable provenance must remain exactly M01, M02, M03, M08")
    if labs.get("source_package_specification_only") != expected_source_spec_only:
        errors.append("source package specification-only provenance must remain the complement of M01, M02, M03, M08")
    for mid in expected_executable:
        matches = list((ROOT / "labs").glob(f"{mid}_*.ipynb"))
        if len(matches) != 1:
            errors.append(f"{mid}: expected exactly one repository notebook; found {len(matches)}")

policy_path = ROOT / "data" / "autonomy_policy.json"
if policy_path.exists():
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    levels = [item["level"] for item in policy.get("levels", [])]
    if levels != ["A1", "A2", "A3", "A4"]: errors.append("autonomy policy must define A1..A4 in order")

html_path = ROOT / "web" / "dashboard.html"
if html_path.exists():
    html = html_path.read_text(encoding="utf-8")
    if "/api/dashboard" not in html: errors.append("dashboard must consume the live dashboard API")
    if "Read-only dashboard" not in html: errors.append("dashboard must make its read-only mutation boundary explicit")

if errors:
    print("Repository validation FAILED")
    for error in errors: print(f"- {error}")
    sys.exit(1)
print("Repository validation PASSED: 42 missions, M01-M41 executable labs, and canonical 253-node knowledge graph")
