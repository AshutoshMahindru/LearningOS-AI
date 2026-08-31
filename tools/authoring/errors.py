"""Authoring-tool errors. Codes stay stable for CLI and tests."""

from __future__ import annotations

from typing import Any


class AuthoringError(ValueError):
    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
