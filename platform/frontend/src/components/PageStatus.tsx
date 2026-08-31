import type { ReactNode } from 'react';
import { EmptyState } from './EmptyState';
import { LearnerErrorBanner } from './LearnerErrorBanner';
import { Spinner } from './Spinner';

type PageStatusProps = {
  loading?: boolean;
  loadingLabel?: string;
  error?: unknown | null;
  empty?: { title: string; message: string; action?: ReactNode } | null;
  children?: ReactNode;
};

export function PageStatus({
  loading = false,
  loadingLabel = 'Loading',
  error = null,
  empty = null,
  children,
}: PageStatusProps) {
  if (loading) {
    return <Spinner label={loadingLabel} />;
  }
  if (error) {
    return <LearnerErrorBanner error={error} />;
  }
  if (empty) {
    return <EmptyState title={empty.title} message={empty.message} action={empty.action} />;
  }
  return children;
}
