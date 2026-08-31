from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class SessionCreateRequest(BaseModel):
    mission_id: str
    learner_id: str


class LearnerCreateRequest(BaseModel):
    username: str
    display_name: Optional[str] = None


class PredictCommitRequest(BaseModel):
    hypothesis: str
    expected_values: Dict[str, Any]


class ExecuteStageRequest(BaseModel):
    code: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class SubmitStageRequest(BaseModel):
    artifacts: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None


class TutorChatRequest(BaseModel):
    session_id: str
    stage_id: str
    role: str
    prompt: str


class ArtifactCreateRequest(BaseModel):
    bytes_b64: str
    media_type: Optional[str] = None
    filename: Optional[str] = None


class CurriculumLoadRequest(BaseModel):
    package_dir: str


class RestoreRequest(BaseModel):
    backup_id: Optional[str] = None
    path: Optional[str] = None
    dest_home: Optional[str] = None

    @model_validator(mode="after")
    def require_backup_target(self) -> "RestoreRequest":
        if not self.backup_id and not self.path:
            raise ValueError("backup_id or path is required")
        return self


class HealthResponse(BaseModel):
    status: str
    version: str
    worker_alive: bool
    database_path: str


class VersionResponse(BaseModel):
    version: str


class AuthBootstrapResponse(BaseModel):
    token: str
    token_type: str = Field(default="bearer")
