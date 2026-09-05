const LOCKED_POLICY = 'NO_AI_REQUIRED';

export function isAssistanceLocked(policy?: string | null): boolean {
  return (policy ?? '').trim().toUpperCase() === LOCKED_POLICY;
}
