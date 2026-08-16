import { useState } from 'react';
import { useStore } from '../state/store';
import {
  formatCount,
  formatFieldValue,
  formatMoney,
  formatMoneyDelta,
  formatPercent,
  formatSeconds,
} from '../lib/format';
import { Badge, Button, Stat } from './ui';

/**
 * Never a spinner: while the optimizer runs we show best-so-far beside the LP
 * bound and the remaining gap, so the operator can decide whether more seconds
 * are worth it.
 */
export function OptimizerBar({
  onStart,
  onCancel,
}: {
  onStart: (seconds: number) => void;
  onCancel: (jobId: string) => void;
}) {
  const opt = useStore((s) => s.optimizer);
  const history = useStore((s) => s.optimizerHistory);
  const [seconds, setSeconds] = useState(30);
  const [actionsOpen, setActionsOpen] = useState(false);

  const gap = opt ? opt.bound - opt.best_value : null;
  const gapRatio = opt && opt.bound !== 0 ? (opt.bound - opt.best_value) / opt.bound : null;
  const prev = history.length >= 2 ? history[history.length - 2] : undefined;
  const improvement = opt && prev ? opt.best_value - prev.best_value : null;

  return (
    <div className="shrink-0 border-b border-ink-750 bg-ink-850">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-2 py-1">
        <h2 className="text-[11px] font-semibold tracking-wide text-ink-300 uppercase">
          Optimizer
        </h2>

        {opt ? (
          <>
            <Stat
              label="best so far"
              value={formatMoney(opt.best_value)}
              tone="series-2"
              sub={improvement !== null ? `${formatMoneyDelta(improvement)} last step` : undefined}
            />
            <Stat label="LP bound" value={formatMoney(opt.bound)} />
            <Stat
              label="gap"
              value={gap === null ? '—' : formatMoney(gap)}
              tone={gapRatio !== null && gapRatio < 0.05 ? 'good' : 'warn'}
              sub={gapRatio === null ? undefined : `${formatPercent(gapRatio)} of bound`}
            />
            <Stat label="iterations" value={formatCount(opt.iterations)} />
            <Stat label="elapsed" value={formatSeconds(opt.elapsed_s)} sub={opt.job_id} />

            <div className="ml-auto flex items-center gap-2">
              {opt.actions.length > 0 && (
                <Button size="sm" onClick={() => setActionsOpen((o) => !o)}>
                  {actionsOpen ? 'Hide' : 'Show'} {opt.actions.length} action
                  {opt.actions.length === 1 ? '' : 's'}
                </Button>
              )}
              <Button size="sm" variant="danger" onClick={() => onCancel(opt.job_id)}>
                Cancel
              </Button>
            </div>
          </>
        ) : (
          <>
            <span className="text-[11px] text-ink-500">
              Not running — results stream in as they improve.
            </span>
            <div className="ml-auto flex items-center gap-1.5">
              <label className="text-[10px] text-ink-400" htmlFor="opt-seconds">
                budget
              </label>
              <input
                id="opt-seconds"
                type="number"
                min={5}
                max={600}
                value={seconds}
                onChange={(e) => setSeconds(Number(e.target.value))}
                className="num w-16 rounded border border-ink-600 bg-ink-900 px-1 py-0.5 text-xs text-ink-100"
              />
              <span className="text-[10px] text-ink-400">s</span>
              <Button size="sm" variant="primary" onClick={() => onStart(seconds)}>
                Optimize
              </Button>
            </div>
          </>
        )}
      </div>

      {opt?.top_warning && (
        <div className="flex items-center gap-1.5 border-t border-warn/30 bg-warn/10 px-2 py-0.5">
          <Badge tone="warn">WARNING</Badge>
          <span className="text-[11px] text-warn">{opt.top_warning}</span>
        </div>
      )}

      {opt && actionsOpen && (
        <ul className="divide-y divide-ink-800 border-t border-ink-750">
          {opt.actions.map((a, i) => (
            <li key={`${a.target}-${a.setting}-${i}`} className="flex gap-2 px-2 py-1">
              <span className="num w-28 shrink-0 truncate text-[11px] text-ink-200">
                {a.target}
              </span>
              <span className="num w-24 shrink-0 truncate text-[11px] text-ink-400">
                {a.setting}
              </span>
              <span className="num w-20 shrink-0 truncate text-[11px] text-series-2">
                {formatFieldValue(a.value)}
              </span>
              <span className="min-w-0 flex-1 text-[11px] text-ink-400">{a.reason}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
