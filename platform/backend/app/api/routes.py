from fastapi import APIRouter, HTTPException
import json
import socket
import uuid
import os
from typing import Dict, Any, List
from app.db.database import get_connection
from app.models.schemas import (
    SessionCreateRequest,
    PredictCommitRequest,
    ExecuteStageRequest,
    SubmitStageRequest,
    TutorChatRequest
)

router = APIRouter()

@router.get("/system/health")
async def health_check():
    return {
        "status": "HEALTHY",
        "version": "3.0.0",
        "worker_alive": False, # TODO: integrate with worker
        "database_path": "TBD" # TODO: integrate with db
    }

@router.get("/missions")
async def list_missions():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, phase_id, order_index FROM missions ORDER BY order_index")
        missions = [dict(row) for row in cursor.fetchall()]
        return {"missions": missions}
    finally:
        conn.close()

@router.get("/missions/{mission_id}")
async def get_mission(mission_id: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT spec_json FROM missions WHERE id = ?", (mission_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Mission not found")
        
        return json.loads(row["spec_json"])
    finally:
        conn.close()

@router.post("/sessions")
async def create_session(request: SessionCreateRequest):
    conn = get_connection()
    try:
        session_id = str(uuid.uuid4())
        cursor = conn.cursor()
        
        # Verify mission exists
        cursor.execute("SELECT id FROM missions WHERE id = ?", (request.mission_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Mission not found")
            
        cursor.execute("""
            INSERT INTO mission_sessions (id, learner_id, mission_id, current_stage_id)
            VALUES (?, ?, ?, ?)
        """, (session_id, "learner_default", request.mission_id, "start"))
        
        conn.commit()
        return {"session_id": session_id, "mission_id": request.mission_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.post("/sessions/{session_id}/stages/{stage_id}/enter")
async def enter_stage(session_id: str, stage_id: str):
    # TODO: update DB state
    return {"status": "ENTERED", "stage_id": stage_id}

@router.post("/sessions/{session_id}/stages/{stage_id}/predict")
async def predict_stage(session_id: str, stage_id: str, request: PredictCommitRequest):
    conn = get_connection()
    try:
        attempt_id = str(uuid.uuid4())
        prediction_id = str(uuid.uuid4())
        cursor = conn.cursor()
        
        # Create a dummy stage attempt for now to satisfy foreign keys
        cursor.execute("""
            INSERT INTO stage_attempts (id, session_id, stage_id, stage_type)
            VALUES (?, ?, ?, ?)
        """, (attempt_id, session_id, stage_id, "experiment"))
        
        cursor.execute("""
            INSERT INTO predictions (id, stage_attempt_id, hypothesis_text, expected_values_json, prediction_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (prediction_id, attempt_id, request.hypothesis, json.dumps(request.expected_values), "dummy_hash"))
        
        conn.commit()
        return {"status": "PREDICTION_SEALED", "attempt_id": attempt_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.post("/sessions/{session_id}/stages/{stage_id}/execute")
async def execute_stage(session_id: str, stage_id: str, request: ExecuteStageRequest):
    # Attempt to resolve the latest stage_attempt_id for this stage
    conn = get_connection()
    attempt_id = "unknown"
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM stage_attempts WHERE session_id = ? AND stage_id = ? ORDER BY started_at DESC LIMIT 1", (session_id, stage_id))
        row = cursor.fetchone()
        if row:
            attempt_id = row["id"]
    finally:
        conn.close()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect("/tmp/learningos_worker.sock")
        payload = {
            "jsonrpc": "2.0",
            "method": "execute_task",
            "params": {
                "code": request.code,
                "parameters": request.parameters
            },
            "id": 1
        }
        sock.sendall(json.dumps(payload).encode('utf-8'))
        response_data = sock.recv(8192)
        if not response_data:
            raise HTTPException(status_code=500, detail="Worker daemon returned no data")
            
        res = json.loads(response_data.decode('utf-8'))
        if "error" in res:
            raise HTTPException(status_code=500, detail=res["error"])
            
        result_payload = res.get("result", {})
        
        # Log execution to DB
        conn = get_connection()
        try:
            exec_id = str(uuid.uuid4())
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO executions (id, stage_attempt_id, runner_id, input_code, code_hash, exit_code, duration_ms, structured_result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (exec_id, attempt_id, "python_daemon", request.code, "hash_todo", 0, 100, json.dumps(result_payload)))
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Failed to log execution: {e}")
        finally:
            conn.close()
            
        return {"status": "EXECUTED", "results": result_payload}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Worker daemon is not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        sock.close()

@router.post("/sessions/{session_id}/stages/{stage_id}/submit")
async def submit_stage(session_id: str, stage_id: str, request: SubmitStageRequest):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # 1. Resolve attempt_id and mission_id
        cursor.execute("""
            SELECT sa.id as attempt_id, ms.mission_id
            FROM stage_attempts sa
            JOIN mission_sessions ms ON ms.id = sa.session_id
            WHERE sa.session_id = ? AND sa.stage_id = ?
            ORDER BY sa.started_at DESC LIMIT 1
        """, (session_id, stage_id))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No execution attempt found for this stage")
            
        attempt_id = row["attempt_id"]
        mission_id = row["mission_id"]
        
        # 2. Insert Evidence
        evidence_id = str(uuid.uuid4())
        competency_id = "comp.sys.hypothesis_testing" # Hardcoded for WP-700
        
        cursor.execute("""
            INSERT INTO evidence_items (
                id, learner_id, mission_id, stage_id, stage_attempt_id,
                competency_id, knowledge_node_id, artifact_type, artifact_hash,
                assistance_level, curriculum_sha
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evidence_id, "learner_default", mission_id, stage_id, attempt_id,
            competency_id, "kn.dummy", "metric", "dummy_hash_555",
            "UNASSISTED", "HEAD"
        ))
        
        # 3. Update Competency
        cursor.execute("""
            INSERT INTO competency_mastery (learner_id, competency_id, level, last_evidence_item_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(learner_id, competency_id) DO UPDATE SET
                level = 1,
                last_evidence_item_id = excluded.last_evidence_item_id,
                last_evaluated_at = CURRENT_TIMESTAMP
        """, ("learner_default", competency_id, 1, evidence_id))
        
        conn.commit()
        return {"status": "ASSESSED", "evidence_id": evidence_id, "next_stage_unlocked": True}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.get("/learners/{learner_id}/evidence")
async def get_evidence(learner_id: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, mission_id, stage_id, competency_id, artifact_type, artifact_hash, created_at
            FROM evidence_items
            WHERE learner_id = ?
            ORDER BY created_at DESC
        """, (learner_id,))
        items = [dict(row) for row in cursor.fetchall()]
        return {"evidence": items}
    finally:
        conn.close()

@router.get("/learners/{learner_id}/competencies")
async def get_competencies(learner_id: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT competency_id, level, decay_score, last_evaluated_at
            FROM competency_mastery
            WHERE learner_id = ?
        """, (learner_id,))
        items = [dict(row) for row in cursor.fetchall()]
        return {"competencies": items}
    finally:
        conn.close()

@router.post("/sessions/{session_id}/gates/evaluate")
async def evaluate_gate(session_id: str):
    return {"status": "EVALUATED"}

import random
from openai import AsyncOpenAI

SOCRATIC_SYSTEM_PROMPT = """You are a strictly Socratic tutor for a software engineering pedagogy platform called LearningOS.
YOUR CRITICAL DIRECTIVE: You MUST NOT provide direct answers, code solutions, or fix errors for the user.
Instead, you must guide the user to discover the answer themselves by asking probing, open-ended questions.
If the user provides an error, ask them to trace the stack.
If the user asks "why", force them to articulate their expected hypothesis first.
Your goal is to increase their cognitive load, not reduce it. Do not be overly polite or helpful in a traditional sense. Be analytical and rigorous.
Keep your responses concise, ideally under 3 sentences."""

@router.post("/tutor/chat")
async def tutor_chat(request: TutorChatRequest):
    prompt = request.prompt
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            client = AsyncOpenAI(api_key=api_key)
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SOCRATIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.4
            )
            return {"response": completion.choices[0].message.content}
        except Exception as e:
            print(f"OpenAI API failed: {e}. Falling back to heuristics.")
            # Fall back to heuristic
    
    # Heuristic fallback
    prompt_lower = prompt.lower()
    responses = [
        "What happens if you break this down into smaller steps?",
        "Have you verified the assumptions you're making here?",
        "Trace the logic backwards. Where is the first point of failure?",
        "How would you explain what this code is doing to someone else?",
        "What data types are actually passing through this function?"
    ]
    
    if "error" in prompt_lower or "exception" in prompt_lower or "syntax" in prompt_lower:
        response_text = "What line does the stack trace point to, and what syntax rule governs that line?"
    elif "why" in prompt_lower:
        response_text = "Before asking why it doesn't work, can you articulate exactly what you expected to happen?"
    elif "how" in prompt_lower:
        response_text = "I cannot give you the answer. But what documentation or reference material might describe this mechanism?"
    else:
        response_text = random.choice(responses)
        
    return {"response": response_text}
