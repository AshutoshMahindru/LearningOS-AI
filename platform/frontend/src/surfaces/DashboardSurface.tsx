import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { createSession, isApiError, listMissions } from '../api/client';
import type { Mission } from '../api/types';
import { Button } from '../components/Button';
import { EmptyState } from '../components/EmptyState';
import { Spinner } from '../components/Spinner';
import { StatusBanner } from '../components/StatusBanner';
import { useAuth } from '../context/AuthContext';

export function DashboardSurface() {
  const { learner } = useAuth();
  const navigate = useNavigate();
  const [missions, setMissions] = useState<Mission[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startingId, setStartingId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listMissions()
      .then((data) => {
        if (!cancelled) {
          setMissions(data.missions);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setMissions([]);
          setError(isApiError(err) ? err.message : 'Failed to load missions');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const startSession = async (missionId: string) => {
    if (!learner) {
      return;
    }
    setStartingId(missionId);
    try {
      const session = await createSession({
        mission_id: missionId,
        learner_id: learner.id,
      });
      void navigate(`/sessions/${session.session_id}`);
    } catch (err) {
      setError(isApiError(err) ? err.message : 'Failed to create session');
    } finally {
      setStartingId(null);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h1 id="catalog-heading" className="text-3xl font-black tracking-tight">
          Catalog
        </h1>
        <p className="mt-2 text-textSecondary">
          Catalog of missions loaded by the local API. Starting a session opens the generic mission player.
        </p>
      </div>

      {missions === null ? <Spinner label="Loading missions" /> : null}

      {error ? (
        <StatusBanner tone="error" title="Unable to load catalog">
          {error}
        </StatusBanner>
      ) : null}

      {missions && missions.length === 0 && !error ? (
        <EmptyState
          title="No missions loaded"
          message="The catalog is empty. Load a curriculum package from Settings when the local API is available."
          action={
            <Button variant="secondary" onClick={() => void navigate('/settings')}>
              Open settings
            </Button>
          }
        />
      ) : null}

      {missions && missions.length > 0 ? (
        <ul className="grid gap-4 md:grid-cols-2" aria-labelledby="catalog-heading">
          {missions.map((mission) => {
            const title = mission.title || mission.id;
            const headingId = `mission-${mission.id}-title`;
            const starting = startingId === mission.id;
            return (
              <li key={mission.id}>
                <article className="panel space-y-4" aria-labelledby={headingId}>
                  <div>
                    <p className="font-mono text-xs text-textSecondary">{mission.id}</p>
                    <h2 id={headingId} className="text-xl font-bold">
                      {title}
                    </h2>
                    {mission.description ? (
                      <p className="mt-2 text-sm text-textSecondary">{mission.description}</p>
                    ) : null}
                  </div>
                  <Button
                    onClick={() => void startSession(mission.id)}
                    disabled={!learner || starting}
                    aria-label={`Start session: ${title}`}
                    aria-busy={starting || undefined}
                  >
                    {starting ? 'Starting…' : 'Start session'}
                  </Button>
                </article>
              </li>
            );
          })}
        </ul>
      ) : null}

      <p className="text-sm text-textSecondary">
        Need stored bytes? Open the <Link to="/artifacts">artifacts surface</Link>.
      </p>
    </div>
  );
}
