import type { ReactNode } from 'react';
import { cn } from '../cn';

type PanelProps = {
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
};

export function Panel({ title, description, children, className }: PanelProps) {
  return (
    <section className={cn('panel', className)}>
      {title ? <h2 className="text-xl font-bold tracking-tight">{title}</h2> : null}
      {description ? <p className="mt-1 text-textSecondary">{description}</p> : null}
      <div className={title || description ? 'mt-4' : undefined}>{children}</div>
    </section>
  );
}
