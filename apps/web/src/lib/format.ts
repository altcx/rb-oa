/**
 * Number and time formatting.
 *
 * Rule for this app: never round in a way that hides a difference between two
 * numbers the user is comparing. Money always carries 2 decimals; scores always
 * carry 2 decimals; nothing is abbreviated to "12.5k" anywhere a comparison
 * happens. Axis ticks are the one place a coarser label is allowed, and even
 * there we keep whole dollars rather than a magnitude suffix.
 */

const money2 = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const money0 = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

const int0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

/** Exact money, always 2 decimals: 12480.75 -> "$12,480.75". */
export function formatMoney(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return money2.format(value);
}

/** Whole-dollar money for axis ticks only: 12480.75 -> "$12,481". */
export function formatMoneyAxis(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return money0.format(value);
}

/** Signed money delta, always 2 decimals and always an explicit sign. */
export function formatMoneyDelta(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const sign = value > 0 ? '+' : value < 0 ? '−' : '±';
  return `${sign}${money2.format(Math.abs(value))}`;
}

/** Scores share money's precision so leaderboard rows line up digit for digit. */
export const formatScore = formatMoney;

/** Integer with thousands separators. */
export function formatCount(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return int0.format(value);
}

/** Milliseconds, integer, grouped: 3420 -> "3,420 ms". */
export function formatMs(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return `${int0.format(Math.round(value))} ms`;
}

/** Milliseconds as seconds with 2 decimals: 3420 -> "3.42 s". */
export function formatMsAsSeconds(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return `${(value / 1000).toFixed(2)} s`;
}

/** Seconds with 1 decimal: 74.25 -> "74.2 s". */
export function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return `${value.toFixed(1)} s`;
}

/** Whole seconds for experiment estimates: 75 -> "~75 s". */
export function formatEstimate(seconds: number): string {
  if (!Number.isFinite(seconds)) return '—';
  return `~${int0.format(Math.round(seconds))} s`;
}

/** Ratio as a percentage with 1 decimal: 0.874 -> "87.4%". */
export function formatPercent(ratio: number): string {
  if (!Number.isFinite(ratio)) return '—';
  return `${(ratio * 100).toFixed(1)}%`;
}

/** "h4" style hour tick. */
export function formatHour(hour: number): string {
  return `h${int0.format(hour)}`;
}

/** Wall-clock HH:MM:SS from an ISO timestamp, 24h, local. */
export function formatClock(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/* ------------------------------------------------------------------ */
/* Latency budget                                                      */
/* ------------------------------------------------------------------ */

/** Extraction must land inside 4 seconds to be useful mid-game. */
export const LATENCY_BUDGET_MS = 4000;

export type LatencyBand = 'good' | 'warn' | 'over';

/**
 * good  = comfortably inside budget (<= 62.5%, i.e. 2.5 s)
 * warn  = inside budget but close (<= 4 s)
 * over  = blew the budget
 */
export function latencyBand(ms: number): LatencyBand {
  if (!Number.isFinite(ms)) return 'over';
  if (ms <= LATENCY_BUDGET_MS * 0.625) return 'good';
  if (ms <= LATENCY_BUDGET_MS) return 'warn';
  return 'over';
}

/** Tailwind text colour for a latency band. */
export function latencyColorClass(ms: number): string {
  switch (latencyBand(ms)) {
    case 'good':
      return 'text-emerald-400';
    case 'warn':
      return 'text-amber-400';
    case 'over':
      return 'text-red-400';
  }
}

/* ------------------------------------------------------------------ */
/* Field values                                                        */
/* ------------------------------------------------------------------ */

/**
 * Render an extracted field value without lying about it. Numbers keep every
 * significant digit they arrived with — 12.5 and 12.50 and 1.25 must never
 * collapse into the same string, because telling them apart is the whole job.
 */
export function formatFieldValue(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    return String(value);
  }
  return JSON.stringify(value);
}

/** Pretty JSON for tool-call cards. */
export function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

/** One-line collapsed summary of a JSON payload for a tool card header. */
export function summarizeJson(value: unknown, max = 88): string {
  let s: string;
  if (value === null || value === undefined) s = 'null';
  else if (typeof value === 'string') s = value;
  else if (Array.isArray(value)) s = `${value.length} item${value.length === 1 ? '' : 's'}`;
  else if (typeof value === 'object') {
    const keys = Object.keys(value as object);
    s = keys.length === 0 ? '{}' : `{ ${keys.slice(0, 4).join(', ')}${keys.length > 4 ? ', …' : ''} }`;
  } else s = String(value);
  s = s.replace(/\s+/g, ' ').trim();
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}
