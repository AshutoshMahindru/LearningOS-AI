from __future__ import annotations

import pytest

from app.execution.contracts import (
    BLOCK_TYPES,
    EXECUTION_STATUSES,
    JOB_STATUSES,
    PYTHON_RUNNER_ID,
    ContractError,
    Diagnostics,
    ExecutionJob,
    JobState,
    Reproducibility,
    ResultBlock,
    StructuredResult,
)


def test_structured_result_roundtrip():
    result = StructuredResult(
        execution_id="exec_test_001",
        status="SUCCESS",
        exit_code=0,
        duration_ms=12,
        blocks=(
            ResultBlock(
                type="metric",
                title="accuracy",
                payload={"accuracy": 0.94},
            ),
        ),
        diagnostics=Diagnostics(stdout="ok", stderr=""),
        reproducibility=Reproducibility(
            python_version="3.12.0",
            runner_id=PYTHON_RUNNER_ID,
            code_hash="abc",
            params_hash="def",
        ),
    )
    payload = result.to_dict()
    restored = StructuredResult.from_mapping(payload)
    assert restored.execution_id == "exec_test_001"
    assert restored.status == "SUCCESS"
    assert restored.blocks[0].type == "metric"
    assert restored.reproducibility is not None
    assert restored.reproducibility.runner_id == PYTHON_RUNNER_ID
    assert payload["blocks"][0]["title"] == "accuracy"


def test_invalid_status_rejected():
    with pytest.raises(ContractError) as raised:
        StructuredResult(
            execution_id="e1",
            status="UNSUPPORTED",
            exit_code=0,
            duration_ms=0,
        )
    assert raised.value.path == "status"


def test_invalid_block_type_rejected():
    with pytest.raises(ContractError):
        ResultBlock(type="widget", payload={"x": 1})


def test_negative_duration_rejected():
    with pytest.raises(ContractError) as raised:
        StructuredResult(execution_id="e1", status="SUCCESS", exit_code=0, duration_ms=-1)
    assert raised.value.path == "duration_ms"


def test_job_and_job_state_contracts():
    job = ExecutionJob.create(
        source="emit('metric', {'n': 1})",
        parameters={"epochs": 5},
        timeout_sec=10,
    )
    assert job.kind == "python"
    assert job.runner_id == PYTHON_RUNNER_ID
    assert job.limits.timeout_sec == 10
    payload = job.to_dict()
    assert payload["parameters"]["epochs"] == 5
    state = JobState(job_id=job.job_id, status="PENDING")
    assert state.to_dict()["status"] in JOB_STATUSES
    with pytest.raises(ContractError):
        JobState(job_id=job.job_id, status="SUCCESS")


def test_catalogues_match_wp137():
    assert EXECUTION_STATUSES == ("SUCCESS", "FAILED", "TIMEOUT", "CRASHED")
    assert BLOCK_TYPES == (
        "table",
        "chart",
        "trace",
        "state_diff",
        "diagram",
        "markdown",
        "metric",
        "artifact",
    )


def test_omits_optional_title():
    block = ResultBlock(type="markdown", payload={"text": "hi"})
    assert "title" not in block.to_dict()
