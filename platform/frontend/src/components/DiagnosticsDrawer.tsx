import { useEffect, useState } from 'react';
import { clearDiagnostics, subscribeDiagnostics, type DiagnosticRecord } from '../api/diagnostics';
import { Button } from './Button';

type DiagnosticsDrawerProps = {
  open: boolean;
  onClose: () => void;
};

function pretty(value: unknown): string {
  if (value == null) {
    return '—';
  }
  if (typeof value === 'string') {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function DiagnosticsDrawer({ open, onClose }: DiagnosticsDrawerProps) {
  const [entries, setEntries] = useState<DiagnosticRecord[]>([]);

  useEffect(() => subscribeDiagnostics(setEntries), []);

  useEffect(() => {
    if (!open) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <div className="diagnostics-root" data-testid="diagnostics-drawer">
      <button type="button" className="diagnostics-backdrop" aria-label="Close diagnostics" onClick={onClose} />
      <aside className="diagnostics-panel" aria-label="Request diagnostics">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold">Diagnostics</h2>
            <p className="text-sm text-textSecondary">
              Local API request and response log. Tokens and credentials are redacted.
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={clearDiagnostics}>
              Clear
            </Button>
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
        {entries.length === 0 ? (
          <p className="text-textSecondary">No requests recorded in this session.</p>
        ) : (
          <ol className="space-y-4">
            {[...entries].reverse().map((entry) => (
              <li key={entry.id} className="rounded-md border border-border bg-bg p-3">
                <p className="font-mono text-sm">
                  {entry.method} {entry.path}
                  {entry.status != null ? ` → ${entry.status}` : ' → network error'}
                </p>
                <p className="text-xs text-textSecondary">{entry.at}</p>
                {entry.error ? (
                  <p className="mt-2 font-mono text-xs text-danger">
                    {entry.error.code}: {entry.error.message}
                  </p>
                ) : null}
                <details className="mt-2">
                  <summary className="cursor-pointer text-sm font-semibold">Request</summary>
                  <pre className="mt-2 overflow-auto text-xs">{pretty(entry.request_body)}</pre>
                </details>
                <details className="mt-2">
                  <summary className="cursor-pointer text-sm font-semibold">Response</summary>
                  <pre className="mt-2 overflow-auto text-xs">{pretty(entry.response_body)}</pre>
                </details>
              </li>
            ))}
          </ol>
        )}
      </aside>
    </div>
  );
}
