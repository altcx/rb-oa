import type { ReactNode } from 'react';

export function Panel({
  title,
  right,
  children,
  className = '',
  bodyClassName = '',
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`flex min-h-0 min-w-0 flex-col border border-ink-700 bg-ink-900 ${className}`}
    >
      {title !== undefined && (
        <header className="flex shrink-0 items-center justify-between gap-2 border-b border-ink-700 bg-ink-850 px-2 py-1">
          <h2 className="text-[11px] font-semibold tracking-wide text-ink-300 uppercase">
            {title}
          </h2>
          {right}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="num inline-block min-w-[1.4em] rounded border border-ink-600 bg-ink-800 px-1 text-center text-[10px] leading-[1.5] text-ink-200">
      {children}
    </kbd>
  );
}

export function Button({
  children,
  onClick,
  variant = 'default',
  size = 'md',
  disabled,
  title,
  type = 'button',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'default' | 'primary' | 'ghost' | 'danger';
  size?: 'sm' | 'md';
  disabled?: boolean;
  title?: string;
  type?: 'button' | 'submit';
}) {
  const base =
    'inline-flex items-center gap-1.5 rounded border font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40';
  const sizes = size === 'sm' ? 'px-1.5 py-0.5 text-[11px]' : 'px-2 py-1 text-xs';
  const variants: Record<string, string> = {
    default: 'border-ink-600 bg-ink-800 text-ink-100 hover:bg-ink-700',
    primary: 'border-series-1 bg-series-1/20 text-series-1 hover:bg-series-1/30',
    ghost: 'border-transparent bg-transparent text-ink-300 hover:bg-ink-800 hover:text-ink-100',
    danger: 'border-bad/50 bg-bad/10 text-bad hover:bg-bad/20',
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`${base} ${sizes} ${variants[variant]}`}
    >
      {children}
    </button>
  );
}

export function Stat({
  label,
  value,
  tone = 'default',
  sub,
}: {
  label: string;
  value: ReactNode;
  tone?: 'default' | 'good' | 'warn' | 'bad' | 'series-1' | 'series-2' | 'series-3';
  sub?: ReactNode;
}) {
  const tones: Record<string, string> = {
    default: 'text-ink-100',
    good: 'text-good',
    warn: 'text-warn',
    bad: 'text-bad',
    'series-1': 'text-series-1',
    'series-2': 'text-series-2',
    'series-3': 'text-series-3',
  };
  return (
    <div className="min-w-0">
      <div className="text-[10px] tracking-wide text-ink-400 uppercase">{label}</div>
      <div className={`num truncate text-sm leading-tight ${tones[tone]}`}>{value}</div>
      {sub && <div className="truncate text-[10px] text-ink-400">{sub}</div>}
    </div>
  );
}

export function StatusDot({ tone }: { tone: 'good' | 'warn' | 'bad' | 'idle' }) {
  const tones: Record<string, string> = {
    good: 'bg-good',
    warn: 'bg-warn',
    bad: 'bg-bad',
    idle: 'bg-ink-500',
  };
  return <span className={`inline-block size-2 shrink-0 rounded-full ${tones[tone]}`} />;
}

export function Badge({
  children,
  tone = 'default',
}: {
  children: ReactNode;
  tone?: 'default' | 'good' | 'warn' | 'bad' | 'info';
}) {
  const tones: Record<string, string> = {
    default: 'border-ink-600 bg-ink-800 text-ink-300',
    good: 'border-good/40 bg-good/10 text-good',
    warn: 'border-warn/40 bg-warn/10 text-warn',
    bad: 'border-bad/40 bg-bad/10 text-bad',
    info: 'border-series-1/40 bg-series-1/10 text-series-1',
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1 py-px text-[10px] leading-[1.4] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}
