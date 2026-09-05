import type { ReactNode } from 'react';
import { cn } from '../cn';

type PanelProps = {
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
};

function headingId(title: string): string {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `panel-${slug || 'section'}`;
}

export function Panel({ title, description, children, className }: PanelProps) {
  const titleId = title ? headingId(title) : undefined;
  const descriptionId = titleId && description ? `${titleId}-description` : undefined;
  return (
    <section className={cn('panel', className)} aria-labelledby={titleId} aria-describedby={descriptionId}>
      {title ? (
        <h2 id={titleId} className="text-xl font-bold tracking-tight">
          {title}
        </h2>
      ) : null}
      {description ? (
        <p id={descriptionId} className="mt-1 text-textSecondary">
          {description}
        </p>
      ) : null}
      <div className={title || description ? 'mt-4' : undefined}>{children}</div>
    </section>
  );
}
