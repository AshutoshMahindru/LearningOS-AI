import type { ReactNode } from 'react';
import { cn } from '../cn';

type EmptyStateProps = {
  title: string;
  message: string;
  action?: ReactNode;
  className?: string;
};

export function EmptyState({ title, message, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-start gap-3 rounded-lg border border-dashed border-border bg-elevated/40 p-8',
        className,
      )}
      role="region"
      aria-label={title}
      data-testid="empty-state"
    >
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="max-w-2xl text-textSecondary">{message}</p>
      {action}
    </div>
  );
}
