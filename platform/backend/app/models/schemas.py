from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class SessionCreateRequest(BaseModel):
    mission_id: str

class PredictCommitRequest(BaseModel):
    hypothesis: str
    expected_values: Dict[str, Any]

class ExecuteStageRequest(BaseModel):
    code: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class SubmitStageRequest(BaseModel):
    artifacts: Optional[List[Dict[str, Any]]] = []
    explanation: Optional[str] = None

class TutorChatRequest(BaseModel):
    session_id: str
    stage_id: str
    role: str
    prompt: str
