import { useEffect, useState, type FormEvent } from 'react';
import {
  createBackup,
  getConfig,
  getHealth,
  getVersion,
  isApiError,
  listCurriculumPackages,
  loadCurriculumPackage,
  restoreBackup,
} from '../api/client';
import type {
  BackupResponse,
  ConfigResponse,
  CurriculumPackage,
  HealthResponse,
  VersionResponse,
} from '../api/types';
import { Button } from '../components/Button';
import { EmptyState } from '../components/EmptyState';
import { Panel } from '../components/Panel';
import { Spinner } from '../components/Spinner';
import { StatusBanner } from '../components/StatusBanner';

const CONFIG_FIELDS: Array<keyof ConfigResponse> = [
  'data_home',
  'database_path',
  'worker_socket',
  'bind_host',
  'api_prefix',
];

export function SettingsSurface() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [version, setVersion] = useState<VersionResponse | null>(null);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [packages, setPackages] = useState<CurriculumPackage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [packageDir, setPackageDir] = useState('');
  const [restoreValue, setRestoreValue] = useState('');
  const [backup, setBackup] = useState<BackupResponse | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([getHealth(), getVersion(), getConfig(), listCurriculumPackages()]).then(
      (results) => {
        if (cancelled) {
          return;
        }
        const [healthResult, versionResult, configResult, packagesResult] = results;
        const failures: string[] = [];
        if (healthResult.status === 'fulfilled') {
          setHealth(healthResult.value);
        } else {
          failures.push(isApiError(healthResult.reason) ? healthResult.reason.message : 'health failed');
        }
        if (versionResult.status === 'fulfilled') {
          setVersion(versionResult.value);
        } else {
          failures.push(isApiError(versionResult.reason) ? versionResult.reason.message : 'version failed');
        }
        if (configResult.status === 'fulfilled') {
          setConfig(configResult.value);
        } else {
          failures.push(isApiError(configResult.reason) ? configResult.reason.message : 'config failed');
        }
        if (packagesResult.status === 'fulfilled') {
          setPackages(packagesResult.value.packages);
        } else {
          setPackages([]);
        }
        if (failures.length > 0) {
          setError(failures.join('; '));
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const handleLoadPackage = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await loadCurriculumPackage({ package_dir: packageDir.trim() });
      const refreshed = await listCurriculumPackages();
      setPackages(refreshed.packages);
      setNotice('Curriculum package load requested.');
    } catch (err) {
      setError(isApiError(err) ? err.message : 'Failed to load package');
    } finally {
      setBusy(false);
    }
  };

  const handleBackup = async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await createBackup();
      setBackup(result);
      setNotice(`Backup created: ${result.backup_id}`);
    } catch (err) {
      setError(isApiError(err) ? err.message : 'Backup failed');
    } finally {
      setBusy(false);
    }
  };

  const handleRestore = async (event: FormEvent) => {
    event.preventDefault();
    const value = restoreValue.trim();
    if (!value) {
      setError('Provide a backup id or path.');
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const body = value.includes('/') || value.endsWith('.gz') ? { path: value } : { backup_id: value };
      await restoreBackup(body);
      setNotice('Restore requested.');
    } catch (err) {
      setError(isApiError(err) ? err.message : 'Restore failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-black tracking-tight">Settings</h1>
        <p className="mt-2 text-textSecondary">
          Local health, version, and non-secret configuration. Provider credentials are never displayed.
        </p>
      </div>

      {error ? (
        <StatusBanner tone="error" title="Settings request failed">
          {error}
        </StatusBanner>
      ) : null}
      {notice ? (
        <StatusBanner tone="success" title="Update">
          {notice}
        </StatusBanner>
      ) : null}

      <Panel title="Health">
        {health ? (
          <dl className="grid gap-2 text-sm">
            <div>
              <dt className="text-textSecondary">Status</dt>
              <dd className="font-semibold">{health.status}</dd>
            </div>
            <div>
              <dt className="text-textSecondary">Version</dt>
              <dd>{health.version}</dd>
            </div>
            <div>
              <dt className="text-textSecondary">Worker alive</dt>
              <dd>{health.worker_alive ? 'yes' : 'no'}</dd>
            </div>
            <div>
              <dt className="text-textSecondary">Database path</dt>
              <dd className="break-all font-mono">{health.database_path}</dd>
            </div>
          </dl>
        ) : error ? (
          <EmptyState title="Health unavailable" message="The local API did not return a health document." />
        ) : (
          <Spinner label="Loading health" />
        )}
      </Panel>

      <Panel title="Version">
        {version ? (
          <p className="font-mono">{version.version || 'unknown'}</p>
        ) : error ? (
          <EmptyState title="Version unavailable" message="The local API did not return a version document." />
        ) : (
          <Spinner label="Loading version" />
        )}
      </Panel>

      <Panel title="Config" description="Allowlisted fields only.">
        {config ? (
          <dl className="grid gap-2 text-sm">
            {CONFIG_FIELDS.map((field) => (
              <div key={field}>
                <dt className="text-textSecondary">{field}</dt>
                <dd className="break-all font-mono">{config[field]}</dd>
              </div>
            ))}
          </dl>
        ) : error ? (
          <EmptyState title="Config unavailable" message="Non-secret configuration could not be loaded." />
        ) : (
          <Spinner label="Loading config" />
        )}
      </Panel>

      <Panel title="Curriculum packages">
        {packages && packages.length === 0 ? (
          <EmptyState
            title="No packages loaded"
            message="Provide an absolute path to a fixture package directory if the API is ready."
          />
        ) : null}
        {packages && packages.length > 0 ? (
          <ul className="mb-4 space-y-2 font-mono text-sm">
            {packages.map((pkg) => (
              <li key={pkg.id}>
                {pkg.id}
                {pkg.version ? ` @ ${pkg.version}` : ''}
              </li>
            ))}
          </ul>
        ) : null}
        <form className="space-y-3" onSubmit={(event) => void handleLoadPackage(event)}>
          <label htmlFor="package-dir" className="block text-sm font-medium">
            Package directory
          </label>
          <input
            id="package-dir"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm"
            value={packageDir}
            onChange={(event) => setPackageDir(event.target.value)}
          />
          <Button type="submit" variant="secondary" disabled={busy || !packageDir.trim()}>
            Load package
          </Button>
        </form>
      </Panel>

      <Panel title="Backup and restore">
        <div className="flex flex-wrap gap-3">
          <Button onClick={() => void handleBackup()} disabled={busy}>
            Create backup
          </Button>
        </div>
        {backup ? (
          <p className="mt-3 break-all font-mono text-sm">
            {backup.backup_id} — {backup.path}
          </p>
        ) : null}
        <form className="mt-4 space-y-3" onSubmit={(event) => void handleRestore(event)}>
          <label htmlFor="restore-ref" className="block text-sm font-medium">
            Backup id or path
          </label>
          <input
            id="restore-ref"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm"
            value={restoreValue}
            onChange={(event) => setRestoreValue(event.target.value)}
          />
          <Button type="submit" variant="secondary" disabled={busy || !restoreValue.trim()}>
            Restore
          </Button>
        </form>
      </Panel>
    </div>
  );
}
