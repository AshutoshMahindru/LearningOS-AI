export type DiagnosticRecord = {
  id: string;
  at: string;
  method: string;
  path: string;
  status: number | null;
  request_body: unknown;
  response_body: unknown;
  error: { code: string; message: string; details: Record<string, unknown> } | null;
};

type Listener = (entries: DiagnosticRecord[]) => void;

const MAX_ENTRIES = 50;
const BODY_LIMIT = 8000;

const SECRET_KEY = /^(authorization|cookie|set-cookie|token|access_token|refresh_token|password|secret|bearer|token_type)$/i;

let entries: DiagnosticRecord[] = [];
const listeners = new Set<Listener>();
let sequence = 0;

function notify(): void {
  const snapshot = entries;
  listeners.forEach((listener) => {
    listener(snapshot);
  });
}

function truncate(value: string): string {
  if (value.length <= BODY_LIMIT) {
    return value;
  }
  return `${value.slice(0, BODY_LIMIT)}…`;
}

export function redactValue(value: unknown): unknown {
  if (typeof value === 'string') {
    if (/^bearer\s+/i.test(value)) {
      return '[redacted]';
    }
    return truncate(value);
  }
  if (Array.isArray(value)) {
    return value.map((item) => redactValue(item));
  }
  if (value && typeof value === 'object') {
    const output: Record<string, unknown> = {};
    for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
      output[key] = SECRET_KEY.test(key) ? '[redacted]' : redactValue(nested);
    }
    return output;
  }
  return value;
}

export function redactBody(body: unknown): unknown {
  if (body == null) {
    return null;
  }
  if (typeof body === 'string') {
    try {
      return redactValue(JSON.parse(body));
    } catch {
      return truncate(body);
    }
  }
  if (typeof FormData !== 'undefined' && body instanceof FormData) {
    const fields: Record<string, unknown> = {};
    body.forEach((value, key) => {
      if (SECRET_KEY.test(key)) {
        fields[key] = '[redacted]';
        return;
      }
      if (typeof Blob !== 'undefined' && value instanceof Blob) {
        fields[key] = { bytes: value.size, type: value.type || null };
        return;
      }
      fields[key] = redactValue(value);
    });
    return fields;
  }
  return redactValue(body);
}

export function recordDiagnostic(input: Omit<DiagnosticRecord, 'id' | 'at'> & { id?: string; at?: string }): void {
  sequence += 1;
  const record: DiagnosticRecord = {
    id: input.id ?? `diag-${sequence}`,
    at: input.at ?? new Date().toISOString(),
    method: input.method,
    path: input.path,
    status: input.status,
    request_body: redactBody(input.request_body),
    response_body: redactValue(input.response_body),
    error: input.error
      ? {
          code: input.error.code,
          message: input.error.message,
          details: redactValue(input.error.details) as Record<string, unknown>,
        }
      : null,
  };
  entries = [...entries, record].slice(-MAX_ENTRIES);
  notify();
}

export function clearDiagnostics(): void {
  entries = [];
  notify();
}

export function getDiagnostics(): DiagnosticRecord[] {
  return entries;
}

export function subscribeDiagnostics(listener: Listener): () => void {
  listeners.add(listener);
  listener(entries);
  return () => {
    listeners.delete(listener);
  };
}
