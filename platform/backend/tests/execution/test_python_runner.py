from __future__ import annotations

import platform
from pathlib import Path

from app.execution.contracts import (
    EXIT_CRASHED,
    EXIT_FAILED,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    PYTHON_RUNNER_ID,
    ExecutionJob,
)
from app.execution.python_runner import IN_PROCESS_LIBRARY_ONLY, run_callable, run_job, run_source
from app.execution.result_schema import validate_structured_result


def test_in_process_library_flag():
    assert IN_PROCESS_LIBRARY_ONLY is True


def test_success_emit_and_stdout_go_to_diagnostics(workdir: Path):
    source = (
        "emit('table', {'columns': ['id', 'val'], 'rows': [[1, 2.5]]}, title='sample')\n"
        "emit('metric', {'accuracy': 0.94}, title='acc')\n"
        "print('hello-stdout')\n"
    )
    result = run_source(source, workdir=workdir)
    payload = validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.exit_code == EXIT_SUCCESS
    assert result.duration_ms >= 0
    assert result.execution_id
    types = [block.type for block in result.blocks]
    assert "table" in types
    assert "metric" in types
    assert result.diagnostics is not None
    assert "hello-stdout" in result.diagnostics.stdout
    assert all(block.type != "markdown" or "hello-stdout" not in str(block.payload) for block in result.blocks)
    assert payload["diagnostics"]["stdout"].find("hello-stdout") >= 0


def test_last_expression_becomes_metric(workdir: Path):
    result = run_source("1 + 2", workdir=workdir)
    assert result.status == "SUCCESS"
    assert result.blocks[0].type == "metric"
    assert result.blocks[0].payload["value"] == 3


def test_failed_exception(workdir: Path):
    result = run_source("raise ValueError('nope')", workdir=workdir)
    validate_structured_result(result)
    assert result.status == "FAILED"
    assert result.exit_code == EXIT_FAILED
    assert result.diagnostics is not None
    assert "ValueError" in result.diagnostics.stderr
    assert any(block.type == "trace" for block in result.blocks)


def test_syntax_error_is_failed(workdir: Path):
    result = run_source("def (", workdir=workdir)
    validate_structured_result(result)
    assert result.status == "FAILED"


def test_timeout_status(workdir: Path):
    result = run_source("import time\ntime.sleep(5)\n", timeout_sec=0.25, workdir=workdir)
    validate_structured_result(result)
    assert result.status == "TIMEOUT"
    assert result.exit_code == EXIT_TIMEOUT
    assert result.duration_ms < 4000


def test_crashed_system_exit(workdir: Path):
    result = run_source("raise SystemExit(2)", workdir=workdir)
    validate_structured_result(result)
    assert result.status == "CRASHED"
    assert result.exit_code in {2, EXIT_CRASHED}


def test_entrypoint_and_parameters(workdir: Path):
    source = (
        "def run_training_experiment(epochs, lr):\n"
        "    emit('metric', {'epochs': epochs, 'lr': lr}, title='params')\n"
        "    return None\n"
    )
    result = run_source(
        source,
        entrypoint="run_training_experiment",
        parameters={"epochs": 5, "lr": 0.01},
        workdir=workdir,
    )
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.blocks[0].payload["epochs"] == 5
    assert result.blocks[0].payload["lr"] == 0.01


def test_run_callable_return_block(workdir: Path):
    def add(x, y):
        return {"type": "metric", "title": "sum", "payload": {"sum": x + y}}

    result = run_callable(add, parameters={"x": 2, "y": 3}, workdir=workdir)
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.blocks[0].payload["sum"] == 5


def test_reproducibility_metadata(workdir: Path):
    params = {"b": 2, "a": 1}
    first = run_source("x = 1", parameters=params, workdir=workdir)
    second = run_source("x = 1", parameters={"a": 1, "b": 2}, workdir=workdir)
    third = run_source("x = 2", parameters=params, workdir=workdir)
    validate_structured_result(first)
    assert first.reproducibility is not None
    assert first.reproducibility.python_version == platform.python_version()
    assert first.reproducibility.runner_id == PYTHON_RUNNER_ID
    assert first.reproducibility.code_hash == second.reproducibility.code_hash
    assert first.reproducibility.params_hash == second.reproducibility.params_hash
    assert third.reproducibility.code_hash != first.reproducibility.code_hash
    assert len(first.reproducibility.code_hash) == 64
    assert len(first.reproducibility.params_hash) == 64


def test_preserves_execution_id(workdir: Path):
    result = run_source("None", execution_id="exec_fixed", workdir=workdir)
    assert result.execution_id == "exec_fixed"


def test_run_job_python(workdir: Path):
    job = ExecutionJob.create(
        source="emit('metric', {'ok': True}, title='ok')",
        parameters={"n": 1},
        workdir=str(workdir),
        timeout_sec=5,
    )
    result = run_job(job)
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert result.reproducibility is not None
    assert result.reproducibility.runner_id == PYTHON_RUNNER_ID


def test_workdir_is_used_for_writes(workdir: Path, isolated_home: Path):
    result = run_source("open('out.txt', 'w', encoding='utf-8').write('ok')", workdir=workdir)
    assert result.status == "SUCCESS"
    assert (workdir / "out.txt").read_text(encoding="utf-8") == "ok"
    assert not (isolated_home / "out.txt").exists()
