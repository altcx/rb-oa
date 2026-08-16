import { useState } from 'react';
import type { RulesResponse } from '../api/types';
import { formatEstimate, formatFieldValue } from '../lib/format';
import { Badge } from './ui';

export function RulesPanel({ rules }: { rules: RulesResponse | null }) {
  const [showResolved, setShowResolved] = useState(true);

  if (!rules) {
    return (
      <p className="px-2 py-6 text-center text-[11px] text-ink-500">
        Rules have not been compiled for this session yet.
      </p>
    );
  }

  const resolved = Object.entries(rules.rules);
  const totalSeconds = rules.unresolved.reduce(
    (s, u) => s + u.experiment.estimated_seconds,
    0,
  );

  return (
    <div className="h-full overflow-y-auto">
      <section>
        <header className="sticky top-0 flex items-center justify-between gap-2 border-b border-ink-750 bg-ink-850 px-2 py-1">
          <h3 className="text-[11px] font-semibold tracking-wide text-bad uppercase">
            Unresolved flags
          </h3>
          <span className="num text-[10px] text-ink-400">
            {rules.unresolved.length} · {formatEstimate(totalSeconds)} to pin all down
          </span>
        </header>

        {rules.unresolved.length === 0 && (
          <p className="px-2 py-2 text-[11px] text-good">Every rule flag is pinned down.</p>
        )}

        <ul className="divide-y divide-ink-800">
          {rules.unresolved.map((u) => (
            <li key={u.flag} className="px-2 py-2">
              <div className="flex items-center gap-2">
                <span className="num truncate text-xs text-ink-100">{u.flag}</span>
                <Badge tone="bad">{u.options.length} ways</Badge>
              </div>

              <div className="mt-1 flex flex-wrap gap-1">
                {u.options.map((o) => (
                  <span
                    key={o}
                    className="rounded border border-ink-600 bg-ink-800 px-1 py-px text-[10px] text-ink-300"
                  >
                    {o}
                  </span>
                ))}
              </div>

              <div className="mt-1.5 rounded border border-ink-700 bg-ink-850 p-1.5">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[10px] tracking-wide text-series-1 uppercase">
                    Experiment
                  </span>
                  <span className="num text-[10px] text-warn">
                    {formatEstimate(u.experiment.estimated_seconds)}
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] leading-snug text-ink-100">
                  {u.experiment.instruction}
                </p>
                <dl className="mt-1 space-y-0.5 text-[10px]">
                  <div className="flex gap-1.5">
                    <dt className="w-20 shrink-0 text-ink-500">watch</dt>
                    <dd className="num min-w-0 text-ink-300">{u.experiment.observable}</dd>
                  </div>
                  <div className="flex gap-1.5">
                    <dt className="w-20 shrink-0 text-ink-500">tells apart</dt>
                    <dd className="num min-w-0 text-ink-300">{u.experiment.discriminator}</dd>
                  </div>
                </dl>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-1">
        <button
          type="button"
          aria-expanded={showResolved}
          onClick={() => setShowResolved((s) => !s)}
          className="flex w-full items-center gap-2 border-y border-ink-750 bg-ink-850 px-2 py-1 text-left hover:bg-ink-800"
        >
          <span className="text-ink-500">{showResolved ? '▾' : '▸'}</span>
          <span className="num text-[11px] text-good">{resolved.length} rules resolved</span>
        </button>
        {showResolved && (
          <ul className="divide-y divide-ink-850">
            {resolved.map(([k, v]) => (
              <li key={k} className="flex items-baseline justify-between gap-3 px-2 py-0.5">
                <span className="num truncate text-[11px] text-ink-400">{k}</span>
                <span className="num shrink-0 text-[11px] text-ink-100">
                  {formatFieldValue(v)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
