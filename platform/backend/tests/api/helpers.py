from __future__ import annotations


def assert_typed_error(response, status_code: int, code: str) -> dict:
    assert response.status_code == status_code, response.text
    body = response.json()
    assert "error" in body
    error = body["error"]
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error.get("details", {}), dict)
    return error
