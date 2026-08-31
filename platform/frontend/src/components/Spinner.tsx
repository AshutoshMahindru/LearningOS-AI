import { cn } from '../cn';

type SpinnerProps = {
  label?: string;
  className?: string;
};

export function Spinner({ label = 'Loading', className }: SpinnerProps) {
  return (
    <div className={cn('flex items-center gap-3 text-textSecondary', className)} role="status">
      <span
        className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-border border-t-primary"
        aria-hidden="true"
      />
      <span>{label}</span>
    </div>
  );
}
