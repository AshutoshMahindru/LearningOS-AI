import { useState, type FormEvent } from 'react';
import { Button } from '../components/Button';
import { Panel } from '../components/Panel';
import { Spinner } from '../components/Spinner';
import { StatusBanner } from '../components/StatusBanner';
import { useAuth } from '../context/AuthContext';
import { isApiError } from '../api/client';

export function LoginSurface() {
  const { status, errorMessage, createLearner, retry } = useAuth();
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = username.trim();
    if (!trimmed) {
      setFormError('Username is required.');
      return;
    }
    setSubmitting(true);
    setFormError(null);
    try {
      await createLearner(trimmed, displayName.trim() || undefined);
    } catch (err) {
      setFormError(isApiError(err) ? err.message : 'Unable to create learner');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main id="main-content" className="flex min-h-screen items-center justify-center p-6" tabIndex={-1}>
      <Panel
        className="w-full max-w-md"
        title="Identify a local learner"
        description="Bootstrap uses the loopback API. Provider credentials never enter the browser."
      >
        {status === 'bootstrapping' ? <Spinner label="Connecting to local API" /> : null}
        {status === 'error' ? (
          <StatusBanner tone="error" title="Local API unavailable" className="mb-4">
            <p>{errorMessage}</p>
            <Button variant="secondary" className="mt-3" onClick={() => void retry()}>
              Retry connection
            </Button>
          </StatusBanner>
        ) : null}
        <form className="space-y-4" onSubmit={(event) => void handleSubmit(event)}>
          <div>
            <label htmlFor="username" className="mb-1 block text-sm font-medium">
              Username
            </label>
            <input
              id="username"
              name="username"
              autoComplete="username"
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              disabled={status !== 'ready' || submitting}
              required
            />
          </div>
          <div>
            <label htmlFor="display-name" className="mb-1 block text-sm font-medium">
              Display name (optional)
            </label>
            <input
              id="display-name"
              name="displayName"
              autoComplete="nickname"
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              disabled={status !== 'ready' || submitting}
            />
          </div>
          {formError ? (
            <StatusBanner tone="error" title="Could not create learner">
              {formError}
            </StatusBanner>
          ) : null}
          <Button type="submit" disabled={status !== 'ready' || submitting || !username.trim()}>
            {submitting ? 'Creating learner…' : 'Continue'}
          </Button>
        </form>
      </Panel>
    </main>
  );
}
