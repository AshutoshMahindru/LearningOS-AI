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
        <h1 className="text-3xl font-black tracking-tight">Dashboard</h1>
        <p className="mt-2 text-textSecondary">
          Catalog of missions loaded by the local API. Mission runtime is not part of G3.
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
        <ul className="grid gap-4 md:grid-cols-2">
          {missions.map((mission) => (
            <li key={mission.id} className="panel space-y-4">
              <div>
                <p className="font-mono text-xs text-textSecondary">{mission.id}</p>
                <h2 className="text-xl font-bold">{mission.title || mission.id}</h2>
                {mission.description ? (
                  <p className="mt-2 text-sm text-textSecondary">{mission.description}</p>
                ) : null}
              </div>
              <Button
                onClick={() => void startSession(mission.id)}
                disabled={!learner || startingId === mission.id}
              >
                {startingId === mission.id ? 'Starting…' : 'Start session'}
              </Button>
            </li>
          ))}
        </ul>
      ) : null}

      <p className="text-sm text-textSecondary">
        Need stored bytes? Open the <Link to="/artifacts">artifacts surface</Link>.
      </p>
    </div>
  );
}
