import type { ButtonHTMLAttributes } from 'react';
import { cn } from '../cn';

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

const variantClass: Record<ButtonVariant, string> = {
  primary:
    'bg-primary-strong text-slate-950 hover:bg-primary disabled:bg-slate-600 disabled:text-slate-300',
  secondary:
    'bg-elevated text-textPrimary border border-border hover:border-primary disabled:opacity-50',
  danger: 'bg-red-900/70 text-danger border border-danger hover:bg-red-800 disabled:opacity-50',
  ghost: 'bg-transparent text-textPrimary hover:bg-elevated disabled:opacity-50',
};

export function Button({
  variant = 'primary',
  className,
  type = 'button',
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-semibold transition-colors focus-visible:shadow-focus disabled:cursor-not-allowed',
        variantClass[variant],
        className,
      )}
      {...props}
    />
  );
}
