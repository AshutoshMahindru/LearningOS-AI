import type { ReactNode } from 'react';
import { cn } from '../cn';

export type StatusTone = 'info' | 'error' | 'success' | 'warning';

type StatusBannerProps = {
  tone?: StatusTone;
  title: string;
  children?: ReactNode;
  className?: string;
  id?: string;
};

export function StatusBanner({ tone = 'info', title, children, className, id }: StatusBannerProps) {
  const role = tone === 'error' ? 'alert' : 'status';
  return (
    <div
      id={id}
      className={cn('status-banner', className)}
      data-tone={tone}
      role={role}
      data-testid="status-banner"
    >
      <p className="font-semibold">{title}</p>
      {children ? <div className="mt-1 text-sm text-textSecondary">{children}</div> : null}
    </div>
  );
}
