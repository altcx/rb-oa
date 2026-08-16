import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { MoneyPoint } from '../api/types';
import { formatHour, formatMoney, formatMoneyAxis, formatMoneyDelta } from '../lib/format';

/** Validated dark-surface categorical slots 1–3. Identity, not rank. */
const SERIES = [
  { key: 'current', name: 'Current config', color: '#3987e5' },
  { key: 'best', name: 'Optimizer best', color: '#d95926' },
  { key: 'observed', name: 'Observed run', color: '#199e70' },
] as const;

type SeriesKey = (typeof SERIES)[number]['key'];

export interface MoneyChartProps {
  current: MoneyPoint[];
  best: MoneyPoint[];
  observed: MoneyPoint[];
  /** LP relaxation upper bound; drawn as a horizontal reference line. */
  bound: number | null;
}

type Row = { hour: number } & Partial<Record<SeriesKey, number>>;

function buildRows(props: MoneyChartProps): Row[] {
  const byHour = new Map<number, Row>();
  const add = (key: SeriesKey, points: MoneyPoint[]) => {
    for (const p of points) {
      const row = byHour.get(p.hour) ?? { hour: p.hour };
      row[key] = p.value;
      byHour.set(p.hour, row);
    }
  };
  add('current', props.current);
  add('best', props.best);
  add('observed', props.observed);
  return [...byHour.values()].sort((a, b) => a.hour - b.hour);
}

function lastValue(points: MoneyPoint[]): number | null {
  const last = points[points.length - 1];
  return last ? last.value : null;
}

interface TooltipShape {
  active?: boolean;
  label?: string | number;
  payload?: Array<{ dataKey?: string | number; value?: number | string | null }>;
}

export function MoneyChart(props: MoneyChartProps) {
  const { bound } = props;
  const rows = useMemo(() => buildRows(props), [props]);

  const lasts: Record<SeriesKey, number | null> = {
    current: lastValue(props.current),
    best: lastValue(props.best),
    observed: lastValue(props.observed),
  };

  const maxValue = rows.reduce((m, r) => {
    for (const s of SERIES) {
      const v = r[s.key];
      if (typeof v === 'number' && v > m) m = v;
    }
    return m;
  }, 0);
  const ceiling = Math.max(maxValue, bound ?? 0);
  const domainTop = ceiling > 0 ? Math.ceil((ceiling * 1.06) / 500) * 500 : 100;

  const hasData = rows.length > 0;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Legend is always present, and doubles as a direct label of the final
          value of each series so identity is never colour-alone. */}
      <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-b border-ink-750 px-2 py-1">
        {SERIES.map((s) => {
          const v = lasts[s.key];
          return (
            <div key={s.key} className="flex items-baseline gap-1.5">
              <span
                className="inline-block h-0.5 w-4 translate-y-[-3px]"
                style={{ background: s.color }}
                aria-hidden
              />
              <span className="text-[10px] text-ink-400">{s.name}</span>
              <span className="num text-xs text-ink-100">
                {v === null ? '—' : formatMoney(v)}
              </span>
              {s.key !== 'current' && v !== null && lasts.current !== null && (
                <span
                  className={`num text-[10px] ${
                    v - lasts.current >= 0 ? 'text-good' : 'text-bad'
                  }`}
                >
                  {formatMoneyDelta(v - lasts.current)}
                </span>
              )}
            </div>
          );
        })}
        {bound !== null && (
          <div className="ml-auto flex items-baseline gap-1.5">
            <span
              className="inline-block h-0 w-4 translate-y-[-3px] border-t border-dashed border-ink-400"
              aria-hidden
            />
            <span className="text-[10px] text-ink-400">LP bound</span>
            <span className="num text-xs text-ink-200">{formatMoney(bound)}</span>
          </div>
        )}
      </div>

      <div className="min-h-0 flex-1">
        {!hasData ? (
          <p className="px-2 py-8 text-center text-[11px] text-ink-500">
            No money curve yet. Start the optimizer or run a simulation to populate this chart.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 10, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="hour"
                type="number"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(h: number) => formatHour(h)}
                tick={{ fill: '#7d8794', fontSize: 10 }}
                tickLine={false}
                allowDecimals={false}
              />
              <YAxis
                domain={[0, domainTop]}
                tickFormatter={(v: number) => formatMoneyAxis(v)}
                tick={{ fill: '#7d8794', fontSize: 10 }}
                tickLine={false}
                width={62}
              />
              {bound !== null && (
                <ReferenceLine
                  y={bound}
                  stroke="#7d8794"
                  strokeDasharray="4 4"
                  ifOverflow="extendDomain"
                  label={{
                    value: `LP bound ${formatMoney(bound)}`,
                    position: 'insideTopRight',
                    fill: '#a7b1bd',
                    fontSize: 10,
                  }}
                />
              )}
              <Tooltip
                cursor={{ stroke: '#4a5563', strokeWidth: 1 }}
                content={(raw) => {
                  const p = raw as unknown as TooltipShape;
                  if (!p.active || !p.payload || p.payload.length === 0) return null;
                  const values = new Map<string, number>();
                  for (const item of p.payload) {
                    if (typeof item.value === 'number' && item.dataKey !== undefined) {
                      values.set(String(item.dataKey), item.value);
                    }
                  }
                  const currentVal = values.get('current');
                  return (
                    <div className="rounded border border-ink-600 bg-ink-900/95 px-2 py-1.5 shadow-lg">
                      <div className="num mb-1 text-[11px] text-ink-300">
                        {typeof p.label === 'number' ? formatHour(p.label) : String(p.label ?? '')}
                      </div>
                      {SERIES.map((s) => {
                        const v = values.get(s.key);
                        return (
                          <div key={s.key} className="flex items-baseline gap-2">
                            <span
                              className="inline-block size-2 shrink-0 rounded-full"
                              style={{ background: s.color }}
                              aria-hidden
                            />
                            <span className="w-24 text-[10px] text-ink-400">{s.name}</span>
                            <span className="num text-[11px] text-ink-100">
                              {v === undefined ? 'no data' : formatMoney(v)}
                            </span>
                            {v !== undefined &&
                              currentVal !== undefined &&
                              s.key !== 'current' && (
                                <span className="num text-[10px] text-ink-400">
                                  {formatMoneyDelta(v - currentVal)}
                                </span>
                              )}
                          </div>
                        );
                      })}
                      {bound !== null && currentVal !== undefined && (
                        <div className="num mt-1 border-t border-ink-750 pt-1 text-[10px] text-ink-400">
                          gap to bound {formatMoneyDelta(bound - currentVal)}
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              {SERIES.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.name}
                  stroke={s.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 2, stroke: '#0b0d10' }}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
