import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  createSession,
  enterStage,
  evaluateGates,
  executeStage,
  getMission,
  getSession,
  predictStage,
  submitStage,
} from '../api/client';
import type {
  ExecuteStageRequest,
  ExecuteStageResponse,
  GateEvaluateResponse,
  Mission,
  MissionStage,
  PredictCommitRequest,
  PredictCommitResponse,
  Session,
  StageEnterResponse,
  SubmitStageRequest,
  SubmitStageResponse,
} from '../api/types';
import { useAuth } from '../context/AuthContext';

export function resolveStageId(mission: Mission, currentStageId: string | null | undefined): string | null {
  const stages = mission.stages ?? [];
  if (stages.length === 0) {
    return null;
  }
  if (currentStageId && currentStageId !== 'start') {
    const match = stages.find((stage) => stage.id === currentStageId);
    if (match) {
      return match.id;
    }
  }
  return stages[0]?.id ?? null;
}

export function useMissionPlayer() {
  const { id: pathSessionId } = useParams<{ id?: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { learner } = useAuth();

  const sessionId = pathSessionId || searchParams.get('session');
  const missionQuery = searchParams.get('mission');

  const [session, setSession] = useState<Session | null>(null);
  const [mission, setMission] = useState<Mission | null>(null);
  const [currentStageId, setCurrentStageId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<unknown | null>(null);
  const [actionError, setActionError] = useState<unknown | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [enterResult, setEnterResult] = useState<StageEnterResponse | null>(null);
  const [predictResult, setPredictResult] = useState<PredictCommitResponse | null>(null);
  const [executeResult, setExecuteResult] = useState<ExecuteStageResponse | null>(null);
  const [submitResult, setSubmitResult] = useState<SubmitStageResponse | null>(null);
  const [gateResult, setGateResult] = useState<GateEvaluateResponse | null>(null);

  const resetStageResults = useCallback(() => {
    setEnterResult(null);
    setPredictResult(null);
    setExecuteResult(null);
    setSubmitResult(null);
    setGateResult(null);
    setActionError(null);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setLoadError(null);
      resetStageResults();
      try {
        if (sessionId) {
          const loadedSession = await getSession(sessionId);
          const loadedMission = await getMission(loadedSession.mission_id);
          if (cancelled) {
            return;
          }
          setSession(loadedSession);
          setMission(loadedMission);
          setCurrentStageId(resolveStageId(loadedMission, loadedSession.current_stage_id));
          return;
        }

        if (missionQuery && learner) {
          const created = await createSession({
            mission_id: missionQuery,
            learner_id: learner.id,
          });
          if (cancelled) {
            return;
          }
          void navigate(`/sessions/${created.session_id}`, { replace: true });
          return;
        }

        if (!cancelled) {
          setSession(null);
          setMission(null);
          setCurrentStageId(null);
        }
      } catch (error) {
        if (!cancelled) {
          setSession(null);
          setMission(null);
          setLoadError(error);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [sessionId, missionQuery, learner, navigate, resetStageResults]);

  const currentStage: MissionStage | null = useMemo(() => {
    if (!mission || !currentStageId) {
      return null;
    }
    return (mission.stages ?? []).find((stage) => stage.id === currentStageId) ?? null;
  }, [mission, currentStageId]);

  useEffect(() => {
    if (!session || !currentStage) {
      return;
    }
    let cancelled = false;
    const sessionKey = session.session_id;
    const stageKey = currentStage.id;
    void enterStage(sessionKey, stageKey)
      .then((result) => {
        if (cancelled) {
          return;
        }
        setEnterResult(result);
        if (result.current_stage_id && result.current_stage_id !== stageKey) {
          setCurrentStageId(result.current_stage_id);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setActionError(error);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [session, currentStage]);

  const selectStage = useCallback((stageId: string) => {
    resetStageResults();
    setCurrentStageId(stageId);
  }, [resetStageResults]);

  const runAction = useCallback(async (name: string, work: () => Promise<void>) => {
    setBusy(name);
    setActionError(null);
    try {
      await work();
    } catch (error) {
      setActionError(error);
    } finally {
      setBusy(null);
    }
  }, []);

  const onPredict = useCallback(
    (body: PredictCommitRequest) => {
      if (!session || !currentStage) {
        return;
      }
      void runAction('predict', async () => {
        const result = await predictStage(session.session_id, currentStage.id, body);
        setPredictResult(result);
      });
    },
    [session, currentStage, runAction],
  );

  const onExecute = useCallback(
    (body: ExecuteStageRequest) => {
      if (!session || !currentStage) {
        return;
      }
      void runAction('execute', async () => {
        const result = await executeStage(session.session_id, currentStage.id, body);
        setExecuteResult(result);
      });
    },
    [session, currentStage, runAction],
  );

  const onSubmit = useCallback(
    (body: SubmitStageRequest) => {
      if (!session || !currentStage) {
        return;
      }
      void runAction('submit', async () => {
        const result = await submitStage(session.session_id, currentStage.id, body);
        setSubmitResult(result);
        const nextId = result.next_stage_id || result.current_stage_id;
        if (nextId) {
          setSession((prev) => (prev ? { ...prev, current_stage_id: nextId } : prev));
          setCurrentStageId(nextId);
        }
      });
    },
    [session, currentStage, runAction],
  );

  const onEvaluateGate = useCallback(() => {
    if (!session) {
      return;
    }
    void runAction('gate', async () => {
      const result = await evaluateGates(session.session_id);
      setGateResult(result);
    });
  }, [session, runAction]);

  return {
    sessionId,
    missionQuery,
    session,
    mission,
    currentStage,
    currentStageId,
    loading,
    loadError,
    actionError,
    busy,
    enterResult,
    predictResult,
    executeResult,
    submitResult,
    gateResult,
    selectStage,
    onPredict,
    onExecute,
    onSubmit,
    onEvaluateGate,
  };
}
