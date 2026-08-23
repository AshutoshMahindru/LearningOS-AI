from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"


def mission_notebooks() -> list[Path]:
    notebooks = []
    for number in range(1, 37):
        prefix = f"M{number:02d}_"
        matches = sorted(path for path in LABS.glob(f"{prefix}*.ipynb") if path.is_file())
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one notebook for M{number:02d}; found {len(matches)}")
        notebooks.append(matches[0])
    return notebooks


def validate_source(path: Path) -> tuple[int, int]:
    notebook = nbformat.read(path, as_version=4)
    ids = [cell.get("id") for cell in notebook.cells]
    if not ids or any(not cell_id for cell_id in ids):
        raise AssertionError(f"{path}: every cell must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{path}: cell ids must be unique")

    code_cells = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        code_cells += 1
        if cell.get("execution_count") is not None:
            raise AssertionError(f"{path}: source code cell {cell.id} has an execution count")
        if cell.get("outputs"):
            raise AssertionError(f"{path}: source code cell {cell.id} has committed outputs")
    return len(notebook.cells), code_cells


def execute(path: Path) -> int:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    return sum(1 for cell in notebook.cells if cell.cell_type == "code")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate M01-M36 source notebooks and optionally execute them.")
    parser.add_argument("--execute", action="store_true", help="Execute each notebook in a fresh kernel after source validation.")
    args = parser.parse_args()

    total_cells = 0
    total_code = 0
    executed_code = 0
    notebooks = mission_notebooks()
    for path in notebooks:
        cells, code = validate_source(path)
        total_cells += cells
        total_code += code
        if args.execute:
            executed_code += execute(path)
        print(f"PASS {path.relative_to(ROOT)}: {cells} cells, {code} code")

    print(f"Validated {len(notebooks)} notebooks: {total_cells} cells, {total_code} code cells")
    if args.execute:
        print(f"Executed {executed_code}/{total_code} code cells without error")


if __name__ == "__main__":
    main()
