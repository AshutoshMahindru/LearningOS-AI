import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getMission, getSession, isApiError } from '../api/client';
import type { Mission, Session } from '../api/types';
import { EmptyState } from '../components/EmptyState';
import { Panel } from '../components/Panel';
import { Spinner } from '../components/Spinner';
import { StatusBanner } from '../components/StatusBanner';

export function SessionSurface() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<Session | null>(null);
  const [mission, setMission] = useState<Mission | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) {
      return;
    }
    let cancelled = false;
    getSession(id)
      .then(async (data) => {
        if (cancelled) {
          return;
        }
        setSession(data);
        try {
          const spec = await getMission(data.mission_id);
          if (!cancelled) {
            setMission(spec);
          }
        } catch {
          if (!cancelled) {
            setMission(null);
          }
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(isApiError(err) ? err.message : 'Failed to load session');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (!id) {
    return (
      <EmptyState title="No session selected" message="Start a session from the dashboard catalog." />
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-black tracking-tight">Session</h1>
        <p className="mt-2 text-textSecondary">
          Generic session status. Stage runtime, workbench, and tutor are not available in G3.
        </p>
      </div>
      {error ? (
        <StatusBanner tone="error" title="Unable to load session">
          {error}
        </StatusBanner>
      ) : null}
      {!session && !error ? <Spinner label="Loading session" /> : null}
      {session ? (
        <Panel title={mission?.title || session.mission_id}>
          <dl className="grid gap-2 font-mono text-sm">
            <div>
              <dt className="text-textSecondary">Session</dt>
              <dd className="break-all">{session.session_id}</dd>
            </div>
            <div>
              <dt className="text-textSecondary">Mission</dt>
              <dd className="break-all">{session.mission_id}</dd>
            </div>
            <div>
              <dt className="text-textSecondary">Learner</dt>
              <dd className="break-all">{session.learner_id}</dd>
            </div>
          </dl>
          <EmptyState
            className="mt-6"
            title="Runtime not available in G3"
            message="Enter, predict, execute, and submit remain quarantined for later work packages."
            action={
              <Link className="text-primary underline" to="/">
                Back to dashboard
              </Link>
            }
          />
        </Panel>
      ) : null}
    </div>
  );
}
