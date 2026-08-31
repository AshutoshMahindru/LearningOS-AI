from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details: dict[str, Any] = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class UnauthorizedError(AppError):
    def __init__(
        self,
        message: str = "Authentication required",
        details: dict[str, Any] | None = None,
        status_code: int = 401,
    ) -> None:
        super().__init__("UNAUTHORIZED", message, status_code, details)


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None) -> None:
        super().__init__("NOT_FOUND", message, 404, details)


class ValidationAppError(AppError):
    def __init__(self, message: str = "Request validation failed", details: dict[str, Any] | None = None) -> None:
        super().__init__("VALIDATION_ERROR", message, 422, details)


class ConflictError(AppError):
    def __init__(self, message: str = "Conflict", details: dict[str, Any] | None = None) -> None:
        super().__init__("CONFLICT", message, 409, details)


class InternalError(AppError):
    def __init__(
        self,
        message: str = "Internal server error",
        details: dict[str, Any] | None = None,
        status_code: int = 500,
    ) -> None:
        super().__init__("INTERNAL", message, status_code, details)


class WorkerUnavailableError(AppError):
    def __init__(self, message: str = "Worker is unavailable", details: dict[str, Any] | None = None) -> None:
        super().__init__("WORKER_UNAVAILABLE", message, 503, details)


class QuarantinedError(AppError):
    def __init__(
        self,
        message: str = "This route is quarantined in G3",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("G3_QUARANTINED", message, 501, details)


class TutorUnavailableError(AppError):
    def __init__(
        self,
        message: str = "Tutor is not available in G3",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("TUTOR_NOT_AVAILABLE", message, 501, details)


class StorageUnavailableError(AppError):
    def __init__(
        self,
        message: str = "Storage is not available",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("STORAGE_UNAVAILABLE", message, 501, details)


class CurriculumUnavailableError(AppError):
    def __init__(
        self,
        message: str = "Curriculum loader is not available",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("CURRICULUM_UNAVAILABLE", message, 501, details)


_STATUS_TO_CODE = {
    400: "VALIDATION_ERROR",
    401: "UNAUTHORIZED",
    403: "UNAUTHORIZED",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    501: "INTERNAL",
    503: "INTERNAL",
    500: "INTERNAL",
}


def error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _json_error(status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code, message, details),
        headers=headers,
    )


def _http_exception_message(exc: StarletteHTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, str) and detail:
        return detail
    if detail is None:
        return exc.phrase if getattr(exc, "phrase", None) else "Request failed"
    return str(detail)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details: dict[str, Any] = {"errors": jsonable_encoder(exc.errors())}
        return _json_error(422, "VALIDATION_ERROR", "Request validation failed", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_TO_CODE.get(exc.status_code, "INTERNAL")
        if exc.status_code == 501:
            detail = str(exc.detail)
            if "TUTOR" in detail.upper():
                code = "TUTOR_NOT_AVAILABLE"
            elif "QUARANTINE" in detail.upper() or "G3_" in detail.upper():
                code = "G3_QUARANTINED"
        return _json_error(exc.status_code, code, _http_exception_message(exc))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, AppError):
            headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
            return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers=headers)
        return _json_error(
            500,
            "INTERNAL",
            "Internal server error",
            {"type": type(exc).__name__},
        )
