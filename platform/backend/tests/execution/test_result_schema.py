from __future__ import annotations

from pathlib import Path

import pytest

from app.execution.contracts import BLOCK_TYPES, EXECUTION_STATUSES, WP137_REQUIRED_FIELDS
from app.execution.result_schema import (
    ResultSchemaError,
    load_result_schema,
    validate_structured_result,
    wp137_schema_path,
)

WP150_SAMPLE = {
    "execution_id": "exec_test_001",
    "status": "SUCCESS",
    "exit_code": 0,
    "duration_ms": 245,
    "blocks": [
        {
            "type": "table",
            "title": "Cleaned Dataset Sample",
            "payload": {"columns": ["id", "val"], "rows": [[1, 2.5], [2, 3.8]]},
        },
        {
            "type": "chart",
            "title": "Loss Curve",
            "payload": {"chart_type": "line", "x": [1, 2, 3], "y": [0.8, 0.4, 0.2]},
        },
        {
            "type": "metric",
            "title": "Validation Accuracy",
            "payload": {"accuracy": 0.94, "f1_score": 0.93},
        },
    ],
}


def test_wp137_is_loaded_from_frozen_architecture():
    path = wp137_schema_path()
    assert path.is_file()
    assert path.name == "WP-137_structured_result_schema.json"
    assert "architecture/learningos-v3/03_technical_architecture" in path.as_posix()
    schema = load_result_schema()
    assert tuple(schema["required"]) == WP137_REQUIRED_FIELDS
    assert tuple(schema["properties"]["status"]["enum"]) == EXECUTION_STATUSES
    assert tuple(schema["properties"]["blocks"]["items"]["properties"]["type"]["enum"]) == BLOCK_TYPES


def test_schema_is_not_copied_into_execution_package():
    exec_dir = Path(__file__).resolve().parents[2] / "app" / "execution"
    assert exec_dir.is_dir()
    assert list(exec_dir.glob("*.json")) == []
    assert not (exec_dir / "WP-137_structured_result_schema.json").exists()


def test_wp150_sample_validates():
    payload = validate_structured_result(WP150_SAMPLE)
    assert payload["execution_id"] == "exec_test_001"


def test_reproducibility_extra_field_is_schema_legal():
    sample = dict(WP150_SAMPLE)
    sample["reproducibility"] = {
        "python_version": "3.12.0",
        "runner_id": "learningos.python_inprocess.v1",
        "code_hash": "aa",
        "params_hash": "bb",
    }
    sample["diagnostics"] = {"stdout": "", "stderr": "", "system_metrics": {"cpu_time_ms": 1.0}}
    validate_structured_result(sample)


def test_missing_required_field_fails():
    sample = dict(WP150_SAMPLE)
    del sample["blocks"]
    with pytest.raises(ResultSchemaError) as raised:
        validate_structured_result(sample)
    assert raised.value.path
    assert raised.value.message
    assert "blocks" in raised.value.path or "blocks" in raised.value.message


def test_invalid_status_fails():
    sample = dict(WP150_SAMPLE)
    sample["status"] = "UNSUPPORTED"
    with pytest.raises(ResultSchemaError) as raised:
        validate_structured_result(sample)
    assert "status" in raised.value.path or "status" in raised.value.message.lower()


def test_invalid_block_type_fails():
    sample = dict(WP150_SAMPLE)
    sample["blocks"] = [{"type": "custom_m01_widget", "payload": {"x": 1}}]
    with pytest.raises(ResultSchemaError) as raised:
        validate_structured_result(sample)
    assert raised.value.message


def test_negative_duration_fails():
    sample = dict(WP150_SAMPLE)
    sample["duration_ms"] = -4
    with pytest.raises(ResultSchemaError):
        validate_structured_result(sample)
