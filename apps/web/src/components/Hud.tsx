import { useStore } from '../state/store';
import { useSession } from '../hooks/useSession';
import {
  LATENCY_BUDGET_MS,
  formatMsAsSeconds,
  latencyBand,
  latencyColorClass,
} from '../lib/format';

/**
 * Monitor 1 overlay: exactly 240x80, always-on-top, click-through except for the
 * capture button. Nothing here may grow — everything else belongs on monitor 2.
 */
export function Hud() {
  const session = useSession();
  const captures = useStore((s) => s.captures);
  const extraction = useStore((s) => s.extraction);
  const last = captures[0];

  const latency = extraction?.ms ?? last?.latency_ms ?? null;
  const band = latency === null ? null : latencyBand(latency);
  const borderTone =
    band === 'over' ? 'border-bad/70' : band === 'warn' ? 'border-warn/70' : 'border-ink-600';

  return (
    <div
      className={`hud-root pointer-events-none flex h-[80px] w-[240px] gap-1.5 border bg-ink-900/95 p-1 ${borderTone}`}
    >
      {/* Last capture thumbnail */}
      <div className="flex w-[74px] shrink-0 flex-col">
        {last ? (
          <img
            src={last.thumb_url}
            alt="Last capture"
            className="h-[42px] w-[74px] border border-ink-700 bg-ink-950 object-cover"
          />
        ) : (
          <div className="flex h-[42px] w-[74px] items-center justify-center border border-ink-700 bg-ink-950 text-[9px] text-ink-500">
            no capture
          </div>
        )}
        <div className="num mt-0.5 truncate text-[9px] text-ink-500">
          {last ? last.id : '—'}
        </div>
      </div>

      <div className="flex min-w-0 flex-1 flex-col justify-between">
        <button
          type="button"
          onClick={() => void session.captureMonitor(1)}
          className="pointer-events-auto flex items-center justify-center gap-1 rounded border border-series-1 bg-series-1/20 py-0.5 text-[11px] font-semibold tracking-wide text-series-1 uppercase hover:bg-series-1/35"
        >
          <span className="inline-block size-1.5 rounded-full bg-series-1" aria-hidden />
          Capture
        </button>

        <div className="num text-[9px] leading-tight text-ink-400">
          hotkey <span className="text-ink-200">Ctrl+Shift+C</span> · region{' '}
          <span className="text-ink-200">Ctrl+Shift+R</span>
        </div>

        <div className="flex items-baseline justify-between gap-1">
          <span
            className={`num text-[15px] leading-none font-semibold ${
              latency === null ? 'text-ink-500' : latencyColorClass(latency)
            }`}
            title={extraction ? `stage: ${extraction.stage}` : 'last extraction latency'}
          >
            {latency === null ? '—' : formatMsAsSeconds(latency)}
          </span>
          <span className="num text-[9px] text-ink-500">
            / {formatMsAsSeconds(LATENCY_BUDGET_MS)} budget
          </span>
        </div>

        {/* Budget bar: a glance tells you whether extraction fits in the turn. */}
        <div className="h-[3px] w-full bg-ink-800">
          <div
            className={`h-full ${
              band === 'over' ? 'bg-bad' : band === 'warn' ? 'bg-warn' : 'bg-good'
            }`}
            style={{
              width: `${Math.min(100, ((latency ?? 0) / LATENCY_BUDGET_MS) * 100).toFixed(1)}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
}
