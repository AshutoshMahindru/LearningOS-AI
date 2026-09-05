import { useEffect, useRef, useState } from 'react';
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

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function DiagnosticsDrawer({ open, onClose }: DiagnosticsDrawerProps) {
  const [entries, setEntries] = useState<DiagnosticRecord[]>([]);
  const panelRef = useRef<HTMLElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  useEffect(() => subscribeDiagnostics(setEntries), []);

  useEffect(() => {
    if (!open) {
      return;
    }
    previouslyFocused.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusable = (): HTMLElement[] => {
      const root = panelRef.current;
      if (!root) {
        return [];
      }
      return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
        (node) => node.getAttribute('aria-hidden') !== 'true',
      );
    };
    const frame = window.requestAnimationFrame(() => {
      const nodes = focusable();
      (nodes[0] ?? panelRef.current)?.focus();
    });
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== 'Tab') {
        return;
      }
      const nodes = focusable();
      if (nodes.length === 0) {
        event.preventDefault();
        return;
      }
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('keydown', onKey);
      previouslyFocused.current?.focus();
    };
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <div className="diagnostics-root" data-testid="diagnostics-drawer">
      <div className="diagnostics-backdrop" onClick={onClose} />
      <aside
        ref={panelRef}
        id="diagnostics-panel"
        className="diagnostics-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="diagnostics-title"
        tabIndex={-1}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 id="diagnostics-title" className="text-xl font-bold">
              Diagnostics
            </h2>
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
