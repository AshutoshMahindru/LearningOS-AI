import { useMemo, useState, type FormEvent } from 'react';
import { getMission, getSession, isApiError } from '../api/client';
import type { Mission, MissionStage, Session } from '../api/types';
import { Button } from '../components/Button';
import { LearnerErrorBanner } from '../components/LearnerErrorBanner';
import { Panel } from '../components/Panel';
import { Spinner } from '../components/Spinner';
import { StatusBanner } from '../components/StatusBanner';
import { postTutorChat, type TutorChatResponse } from '../tutor/api';
import { isAssistanceLocked } from '../tutor/policy';
import { TUTOR_ROLES, type TutorRoleId } from '../tutor/roles';

type TranscriptItem = {
  role: 'learner' | 'tutor';
  text: string;
  meta?: string;
};

function readQuery(name: string): string {
  if (typeof window === 'undefined') {
    return '';
  }
  return new URLSearchParams(window.location.search).get(name) ?? '';
}

function resolveStage(mission: Mission | null, stageId: string, session: Session | null): MissionStage | null {
  const stages = mission?.stages ?? [];
  if (stages.length === 0) {
    return null;
  }
  if (stageId) {
    return stages.find((stage) => stage.id === stageId) ?? null;
  }
  if (session?.current_stage_id) {
    return stages.find((stage) => stage.id === session.current_stage_id) ?? stages[0] ?? null;
  }
  return stages[0] ?? null;
}

export function TutorSurface() {
  const [sessionId, setSessionId] = useState(() => readQuery('session'));
  const [stageId, setStageId] = useState(() => readQuery('stage'));
  const [role, setRole] = useState<TutorRoleId>('SOCRATIC');
  const [prompt, setPrompt] = useState('');
  const [session, setSession] = useState<Session | null>(null);
  const [mission, setMission] = useState<Mission | null>(null);
  const [loadError, setLoadError] = useState<unknown | null>(null);
  const [actionError, setActionError] = useState<unknown | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);

  const currentStage = useMemo(
    () => resolveStage(mission, stageId, session),
    [mission, stageId, session],
  );
  const locked = isAssistanceLocked(currentStage?.assistance_policy);

  const handleLoad = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = sessionId.trim();
    if (!trimmed) {
      setSession(null);
      setMission(null);
      setLoadError(null);
      return;
    }
    setLoading(true);
    setLoadError(null);
    try {
      const loadedSession = await getSession(trimmed);
      const loadedMission = await getMission(loadedSession.mission_id);
      setSession(loadedSession);
      setMission(loadedMission);
      if (!stageId.trim()) {
        const resolved = resolveStage(loadedMission, '', loadedSession);
        if (resolved) {
          setStageId(resolved.id);
        }
      }
    } catch (error) {
      setSession(null);
      setMission(null);
      setLoadError(error);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async (event: FormEvent) => {
    event.preventDefault();
    const nextPrompt = prompt.trim();
    if (!nextPrompt || busy) {
      return;
    }
    if (locked || loading) {
      return;
    }
    setBusy(true);
    setActionError(null);
    setUnavailable(false);
    try {
      const result: TutorChatResponse = await postTutorChat({
        session_id: sessionId.trim(),
        stage_id: (stageId || currentStage?.id || '').trim(),
        role,
        prompt: nextPrompt,
      });
      setTranscript((items) => [
        ...items,
        { role: 'learner', text: nextPrompt },
        { role: 'tutor', text: result.reply, meta: `${result.role} · ${result.provider}` },
      ]);
      setPrompt('');
    } catch (error) {
      if (isApiError(error) && error.code === 'TUTOR_NOT_AVAILABLE') {
        setUnavailable(true);
        setActionError(error);
      } else {
        setActionError(error);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6" data-testid="tutor-surface">
      <div>
        <h1 id="tutor-heading" className="text-3xl font-black tracking-tight">
          Tutor
        </h1>
        <p className="mt-2 text-textSecondary">
          Role-based Socratic guidance for the active stage. Assistance policy is enforced by the
          local API. Credentials never enter this surface.
        </p>
      </div>

      <Panel
        title="Session context"
        description="Load a session so the tutor can honor the stage assistance policy."
      >
        <form className="space-y-4" onSubmit={(event) => void handleLoad(event)}>
          <label className="block text-sm font-medium" htmlFor="tutor-session">
            Session id
          </label>
          <input
            id="tutor-session"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-textPrimary"
            value={sessionId}
            onChange={(event) => setSessionId(event.target.value)}
            autoComplete="off"
            spellCheck={false}
          />
          <label className="block text-sm font-medium" htmlFor="tutor-stage">
            Stage id
          </label>
          <input
            id="tutor-stage"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-textPrimary"
            value={stageId}
            onChange={(event) => setStageId(event.target.value)}
            autoComplete="off"
            spellCheck={false}
          />
          <Button type="submit" variant="secondary" disabled={loading || !sessionId.trim()}>
            {loading ? 'Loading…' : 'Load session'}
          </Button>
        </form>
        {loading ? <Spinner label="Loading session" className="mt-4" /> : null}
        {loadError ? (
          <div className="mt-4">
            <LearnerErrorBanner error={loadError} />
          </div>
        ) : null}
        {currentStage ? (
          <p className="mt-4 font-mono text-xs text-textSecondary" data-testid="tutor-stage-policy">
            {currentStage.title} · {currentStage.type} · {currentStage.assistance_policy ?? 'unspecified'}
          </p>
        ) : null}
      </Panel>

      {locked ? (
        <StatusBanner id="tutor-no-ai-lock" tone="warning" title="No-AI lock">
          Assistance is disabled for this stage. The tutor will not be called until the unassisted
          attempt is complete.
        </StatusBanner>
      ) : null}

      {unavailable ? (
        <StatusBanner tone="info" title="Tutor is not available">
          No tutor provider is configured on the local runtime. The surface stays generic and does
          not store provider credentials.
        </StatusBanner>
      ) : null}

      {actionError && !unavailable ? <LearnerErrorBanner error={actionError} /> : null}

      <Panel title="Guidance" description="Choose a tutor role. The tutor asks questions; it does not complete the work.">
        <form className="space-y-4" onSubmit={(event) => void handleSend(event)}>
          <label className="block text-sm font-medium" htmlFor="tutor-role">
            Tutor role
          </label>
          <select
            id="tutor-role"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
            value={role}
            onChange={(event) => setRole(event.target.value as TutorRoleId)}
            disabled={locked}
            aria-describedby={locked ? 'tutor-no-ai-lock' : undefined}
          >
            {TUTOR_ROLES.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
          <label className="block text-sm font-medium" htmlFor="tutor-prompt">
            Prompt
          </label>
          <textarea
            id="tutor-prompt"
            className="min-h-32 w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            disabled={locked}
            aria-describedby={locked ? 'tutor-no-ai-lock' : undefined}
          />
          <Button type="submit" disabled={locked || busy || loading || !prompt.trim() || !sessionId.trim()}>
            {busy ? 'Asking…' : 'Ask tutor'}
          </Button>
        </form>
      </Panel>

      {transcript.length > 0 ? (
        <Panel title="Transcript">
          <ol
            className="space-y-3"
            data-testid="tutor-transcript"
            role="log"
            aria-live="polite"
            aria-relevant="additions"
            aria-label="Tutor transcript"
          >
            {transcript.map((item, index) => (
              <li key={`${item.role}-${index}`} className="rounded-md border border-border bg-bg p-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-textSecondary">
                  {item.role}
                  {item.meta ? ` · ${item.meta}` : ''}
                </p>
                <p className="mt-1 whitespace-pre-wrap">{item.text}</p>
              </li>
            ))}
          </ol>
        </Panel>
      ) : null}
    </div>
  );
}
