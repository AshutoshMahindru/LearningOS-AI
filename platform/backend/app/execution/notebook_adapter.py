"""Notebook-compatibility adapter. No Jupyter server is started."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.execution.contracts import (
    DEFAULT_TIMEOUT_SEC,
    NOTEBOOK_RUNNER_ID,
    StructuredResult,
    hash_payload,
)
from app.execution.python_runner import ExecSession, run_session


def _cell_source(cell: Mapping[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source or "")


def _strip_ipython_lines(source: str) -> str:
    kept: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        kept.append(line)
    return "".join(kept)


def normalize_cells(notebook: Mapping[str, Any] | Sequence[Any]) -> list[dict[str, Any]]:
    if isinstance(notebook, Mapping):
        raw_cells = notebook.get("cells")
        if raw_cells is None:
            raw_cells = notebook.get("nb_cells")
        if raw_cells is None:
            raw_cells = []
    else:
        raw_cells = notebook
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
        raise TypeError("notebook cells must be a sequence")
    cells: list[dict[str, Any]] = []
    for index, item in enumerate(raw_cells):
        if not isinstance(item, Mapping):
            raise TypeError(f"notebook cell {index} must be an object")
        cell_type = str(item.get("cell_type") or item.get("type") or "code")
        cells.append(
            {
                "cell_type": cell_type,
                "source": _cell_source(item),
                "id": item.get("id"),
                "metadata": dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), Mapping) else {},
            }
        )
    return cells


def _notebook_code_hash(cells: Sequence[Mapping[str, Any]]) -> str:
    material = [{"cell_type": cell.get("cell_type"), "source": cell.get("source")} for cell in cells]
    return hash_payload(material)


def run_notebook(
    notebook: Mapping[str, Any] | Sequence[Any],
    *,
    parameters: Mapping[str, Any] | None = None,
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC,
    execution_id: str | None = None,
    workdir: str | Path | None = None,
    runner_id: str = NOTEBOOK_RUNNER_ID,
) -> StructuredResult:
    cells = normalize_cells(notebook)
    session = ExecSession(
        parameters=parameters,
        runner_id=runner_id,
        workdir=workdir,
        code_hash=_notebook_code_hash(cells),
    )

    def body() -> None:
        for index, cell in enumerate(cells):
            cell_type = str(cell.get("cell_type") or "code")
            source = str(cell.get("source") or "")
            title = cell.get("id")
            title_text = str(title) if isinstance(title, str) and title else f"cell_{index}"
            if cell_type == "markdown":
                if source.strip():
                    session.emit("markdown", {"text": source}, title=title_text)
                continue
            if cell_type in {"raw", "none"}:
                continue
            python_source = _strip_ipython_lines(source)
            if not python_source.strip():
                continue
            filename = f"<learningos:notebook:cell:{index}>"
            value = session.run_source(python_source, filename=filename)
            session.ingest(value)

    return run_session(session, body, timeout_sec=timeout_sec, execution_id=execution_id)
