import { learnerFacingError } from '../api/learnerError';
import { StatusBanner } from './StatusBanner';

export function LearnerErrorBanner({ error }: { error: unknown }) {
  const facing = learnerFacingError(error);
  return (
    <StatusBanner tone="error" title={facing.title}>
      <p>{facing.message}</p>
      <p className="mt-1 font-mono text-xs">{facing.code}</p>
    </StatusBanner>
  );
}
