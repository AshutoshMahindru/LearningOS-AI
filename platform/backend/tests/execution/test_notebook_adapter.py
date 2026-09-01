from __future__ import annotations

from pathlib import Path

from app.execution.contracts import NOTEBOOK_RUNNER_ID, ExecutionJob
from app.execution.notebook_adapter import run_notebook
from app.execution.python_runner import run_job
from app.execution.result_schema import validate_structured_result


def test_notebook_markdown_and_code_cells(workdir: Path):
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "intro"]},
            {"cell_type": "code", "source": ["x = 1\n", "x + 1"]},
            {"cell_type": "code", "source": "emit('metric', {'x': x}, title='x')"},
        ],
    }
    result = run_notebook(notebook, workdir=workdir)
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.reproducibility is not None
    assert result.reproducibility.runner_id == NOTEBOOK_RUNNER_ID
    types = [block.type for block in result.blocks]
    assert types[0] == "markdown"
    assert result.blocks[0].payload["text"].startswith("# Title")
    assert "metric" in types


def test_notebook_cell_source_as_string_list(workdir: Path):
    result = run_notebook(
        [{"cell_type": "code", "source": ["a = 2\n", "a * 3"]}],
        workdir=workdir,
    )
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.blocks[0].payload["value"] == 6


def test_notebook_stops_on_failure(workdir: Path):
    notebook = {
        "cells": [
            {"cell_type": "code", "source": "emit('metric', {'ok': 1}, title='ok')"},
            {"cell_type": "code", "source": "raise RuntimeError('cell-fail')"},
            {"cell_type": "markdown", "source": "should not render"},
        ]
    }
    result = run_notebook(notebook, workdir=workdir)
    validate_structured_result(result)
    assert result.status == "FAILED"
    texts = [str(block.payload) for block in result.blocks]
    assert all("should not render" not in text for text in texts)
    assert any(block.type == "trace" for block in result.blocks)


def test_notebook_strips_magics(workdir: Path):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "%matplotlib inline\n!echo hi\nemit('metric', {'ok': 1}, title='ok')\n",
            }
        ]
    }
    result = run_notebook(notebook, workdir=workdir)
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.blocks[0].payload["ok"] == 1


def test_notebook_adapter_does_not_import_jupyter():
    import app.execution.notebook_adapter as adapter

    source = Path(adapter.__file__).read_text(encoding="utf-8")
    for forbidden in ("jupyter", "ipykernel", "nbconvert", "nbformat", "notebook.server"):
        assert forbidden not in source


def test_run_job_notebook_kind(workdir: Path):
    job = ExecutionJob.create(
        kind="notebook",
        notebook={"cells": [{"cell_type": "code", "source": "emit('metric', {'n': 3})"}]},
        workdir=str(workdir),
    )
    result = run_job(job)
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.blocks[0].payload["n"] == 3
