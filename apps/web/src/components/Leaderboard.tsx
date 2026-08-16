import type { LeaderboardEntry } from '../api/types';
import { formatClock, formatMoneyDelta, formatScore } from '../lib/format';
import { Badge } from './ui';

/**
 * The game scores on the maximum, not the most recent, so the high-water mark
 * is pinned to the left and everything else is shown as a delta against it.
 */
export function Leaderboard({ entries }: { entries: LeaderboardEntry[] }) {
  const high =
    entries.find((e) => e.is_high_water) ??
    entries.reduce<LeaderboardEntry | null>(
      (best, e) => (best === null || e.score > best.score ? e : best),
      null,
    );
  const rest = entries.filter((e) => e.id !== high?.id);

  return (
    <div className="flex h-full min-h-0 items-stretch border-t border-ink-700 bg-ink-900">
      <div className="flex shrink-0 items-center border-r border-ink-700 bg-ink-850 px-2">
        <span className="text-[10px] leading-tight font-semibold tracking-wide text-ink-400 uppercase">
          Leader
          <br />
          board
        </span>
      </div>

      {high && (
        <div className="flex shrink-0 flex-col justify-center border-r border-warn/40 bg-warn/10 px-2 py-1">
          <div className="flex items-center gap-1.5">
            <Badge tone="warn">HIGH WATER</Badge>
            <span className="truncate text-[11px] text-ink-200">{high.label}</span>
          </div>
          <div className="num text-base leading-tight font-semibold text-warn">
            {formatScore(high.score)}
          </div>
          <div className="num truncate text-[10px] text-ink-400">
            {formatClock(high.tested_at)} · {high.config_summary}
          </div>
        </div>
      )}

      <ul className="flex min-w-0 flex-1 items-stretch gap-px overflow-x-auto" data-testid="leaderboard">
        {rest.length === 0 && (
          <li className="flex items-center px-2 text-[11px] text-ink-500">
            No other configurations tested yet.
          </li>
        )}
        {rest.map((e) => (
          <li
            key={e.id}
            className="flex w-52 shrink-0 flex-col justify-center border-r border-ink-800 px-2 py-1"
            title={e.config_summary}
          >
            <div className="truncate text-[11px] text-ink-300">{e.label}</div>
            <div className="flex items-baseline gap-1.5">
              <span className="num text-sm leading-tight text-ink-100">
                {formatScore(e.score)}
              </span>
              {high && e.id !== high.id && (
                <span className="num text-[10px] text-bad">
                  {formatMoneyDelta(e.score - high.score)}
                </span>
              )}
            </div>
            <div className="num truncate text-[10px] text-ink-500">
              {formatClock(e.tested_at)} · {e.config_summary}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
