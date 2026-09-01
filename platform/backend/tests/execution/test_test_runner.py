from __future__ import annotations

from pathlib import Path

from app.execution.contracts import PYTHON_RUNNER_ID
from app.execution.python_runner import run_source
from app.execution.result_schema import validate_structured_result
from app.execution.test_runner import AssertionSpec, hash_harness, run_harness


def test_harness_passes_and_records_hash(workdir: Path):
    result = run_source(
        "emit('metric', {'accuracy': 0.95}, title='acc')",
        workdir=workdir,
    )
    assertions = [
        {"path": "status", "equals": "SUCCESS"},
        {"path": "blocks[0].type", "eq": "metric"},
        {"path": "blocks[0].payload.accuracy", "gte": 0.9},
        {"path": "reproducibility.runner_id", "eq": PYTHON_RUNNER_ID},
        AssertionSpec(name="exit", path="exit_code", op="eq", expected=0),
    ]
    report = run_harness(result, assertions, harness_id="accuracy-suite")
    assert report.passed
    assert report.failed_count == 0
    assert report.harness_hash
    assert len(report.harness_hash) == 64
    assert report.execution_id == result.execution_id
    envelope = report.to_structured_result()
    validate_structured_result(envelope)
    assert envelope.reproducibility is not None
    assert envelope.reproducibility.harness_hash == report.harness_hash


def test_harness_hash_is_stable_and_sensitive(workdir: Path):
    assertions = [
        {"path": "status", "equals": "SUCCESS"},
        {"path": "exit_code", "eq": 0},
    ]
    first = hash_harness(assertions, harness_id="suite")
    second = hash_harness(list(assertions), harness_id="suite")
    changed = hash_harness(assertions + [{"path": "status", "equals": "FAILED"}], harness_id="suite")
    other_id = hash_harness(assertions, harness_id="other")
    assert first == second
    assert changed != first
    assert other_id != first


def test_harness_failure_and_callable(workdir: Path):
    result = run_source("emit('metric', {'accuracy': 0.2})", workdir=workdir)

    def accuracy_ok(payload: dict) -> None:
        assert payload["blocks"][0]["payload"]["accuracy"] >= 0.9

    report = run_harness(
        result,
        [
            {"path": "status", "equals": "SUCCESS"},
            {"path": "blocks[0].payload.accuracy", "gte": 0.9},
            accuracy_ok,
        ],
    )
    assert not report.passed
    assert report.failed_count == 2
    failed = [case for case in report.cases if not case.passed]
    assert any("accuracy" in (case.path or case.name) or "0.2" in case.message for case in failed)
    envelope = report.to_structured_result()
    validate_structured_result(envelope)
    assert envelope.status == "FAILED"


def test_exists_op(workdir: Path):
    result = run_source("emit('metric', {'n': 1})", workdir=workdir)
    report = run_harness(
        result,
        [
            {"path": "reproducibility.code_hash", "op": "exists"},
            {"path": "blocks[9].payload", "op": "exists"},
        ],
    )
    assert not report.passed
    assert report.cases[0].passed
    assert not report.cases[1].passed
