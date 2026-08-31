"""Print stage list and assistance policies for a fixture package."""

from __future__ import annotations

from pathlib import Path

from .package import Package, load_package


def preview_package(path: Path | str | None = None, *, verify: bool = True) -> str:
    package = load_package(path, verify=verify)
    return format_preview(package)


def format_preview(package: Package) -> str:
    lines: list[str] = [
        f"Package           {package.id}  v{package.version}",
        f"Title             {package.title}",
        f"Digest            {package.digest}",
        f"Source            {package.source_path}",
        "",
    ]
    for mission in package.missions:
        phase = mission.get("phase") if isinstance(mission.get("phase"), dict) else {}
        lines.append(f"Mission           {mission.get('id')}  {mission.get('title')}")
        lines.append(f"Phase             {phase.get('id', '-')} / {phase.get('title', '-')}")
        lines.append(f"Order             {mission.get('order_index')}")
        lines.append(f"Invariant         {mission.get('core_invariant')}")
        lines.append("")
        lines.append("Stages")
        lines.append(f"  {'#':<4}{'id':<28}{'type':<24}{'assistance'}")
        for index, stage in enumerate(mission.get("stages") or [], start=1):
            if not isinstance(stage, dict):
                continue
            lines.append(
                f"  {index:<4}{str(stage.get('id', '-')):<28}"
                f"{str(stage.get('type', '-')):<24}{str(stage.get('assistance_policy', '-'))}"
            )
            if stage.get("type") == "experiment":
                lines.append("       cycle               predict -> execute -> submit (dry-run in simulate)")
        lines.append("")
        contract = mission.get("gate_contract") if isinstance(mission.get("gate_contract"), dict) else {}
        lines.append("Gate contract")
        lines.append(f"  pass_threshold   {contract.get('pass_threshold')}")
        repair = contract.get("repair_policy") if isinstance(contract.get("repair_policy"), dict) else {}
        if repair:
            lines.append(
                "  repair           "
                f"allow_targeted_repair={repair.get('allow_targeted_repair')}  "
                f"max_repair_attempts={repair.get('max_repair_attempts')}"
            )
        lines.append("  required_evidence")
        for item in contract.get("required_evidence") or []:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"    - {item.get('competency_id')}  "
                f"{item.get('stage_id')}  {item.get('artifact_type')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
