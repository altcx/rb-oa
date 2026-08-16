import { useEffect, useMemo, useState } from 'react';
import { isMockMode, mockForcedByEnv, setMockMode } from '../api/client';
import { useSession } from '../hooks/useSession';
import { useStore } from '../state/store';
import { hourlyToPoints } from '../lib/money';
import { formatCount, formatMs, latencyColorClass } from '../lib/format';
import { Link } from '../router';
import { ChatPanel } from './ChatPanel';
import { Filmstrip } from './Filmstrip';
import { Leaderboard } from './Leaderboard';
import { MoneyChart } from './MoneyChart';
import { OptimizerBar } from './OptimizerBar';
import { RulesPanel } from './RulesPanel';
import { ShortcutLegend, ShortcutStrip } from './ShortcutLegend';
import { StateInspector } from './StateInspector';
import { Badge, Button, StatusDot } from './ui';

type RightTab = 'chat' | 'rules';

function ConnectionPill({ status, attempt }: { status: string; attempt: number }) {
  const tone =
    status === 'open' ? 'good' : status === 'mock' ? 'warn' : status === 'connecting' ? 'warn' : 'bad';
  const label =
    status === 'mock'
      ? 'MOCK REPLAY'
      : status === 'open'
        ? 'LIVE'
        : status === 'connecting'
          ? 'CONNECTING'
          : attempt > 0
            ? `RETRY ${attempt}`
            : 'OFFLINE';
  return (
    <span className="flex items-center gap-1.5">
      <StatusDot tone={tone as 'good' | 'warn' | 'bad'} />
      <span className="num text-[10px] tracking-wide text-ink-300">{label}</span>
    </span>
  );
}

export function Dashboard() {
  const session = useSession();
  const optimizer = useStore((s) => s.optimizer);
  const extraction = useStore((s) => s.extraction);
  const warnings = useStore((s) => s.warnings);
  const dispatch = useStore((s) => s.dispatch);

  const [tab, setTab] = useState<RightTab>('chat');
  const [legendOpen, setLegendOpen] = useState(false);
  const mock = isMockMode();

  const chart = session.chart;

  // The chart endpoint is the source of truth for all three lines. The one
  // exception is the optimizer's line while a job is streaming: non-final
  // progress events carry `money_by_hour` in band, so the best line moves live
  // between refetches. A final event triggers a refetch, and the endpoint wins.
  const currentCurve = useMemo(() => hourlyToPoints(chart?.current), [chart]);
  const observedCurve = useMemo(() => hourlyToPoints(chart?.observed), [chart]);
  const bestCurve = useMemo(() => {
    const streaming = optimizer && !optimizer.final ? hourlyToPoints(optimizer.money_by_hour) : null;
    return streaming ?? hourlyToPoints(chart?.best);
  }, [optimizer, chart]);

  const bound = chart?.bound ?? optimizer?.bound ?? null;

  const tally =
    extraction?.auto_confirmed !== undefined && extraction.disputed !== undefined
      ? { agreed: extraction.auto_confirmed, total: extraction.auto_confirmed + extraction.disputed }
      : null;

  // Global hotkeys. `A` lives in the inspector; everything else is here.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const t = e.target;
      const typing =
        t instanceof HTMLElement &&
        (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'Escape') {
        setLegendOpen(false);
        return;
      }
      if (typing) return;
      if (e.key === '?') {
        e.preventDefault();
        setLegendOpen((o) => !o);
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        void session.captureMonitor(1);
      } else if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        void session.optimize(30);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [session]);

  return (
    <div className="flex h-full flex-col bg-ink-950">
      {/* ---------------- header ---------------- */}
      <header className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-ink-700 bg-ink-900 px-2 py-1">
        <h1 className="text-xs font-semibold tracking-wide text-ink-100 uppercase">
          Puzzle Copilot
        </h1>
        <ConnectionPill status={session.status} attempt={session.reconnectAttempt} />
        <span className="num text-[10px] text-ink-500">{session.sessionId ?? 'no session'}</span>

        {extraction && (
          <span className="num text-[10px] text-ink-300">
            {extraction.stage}{' '}
            <span className={latencyColorClass(extraction.ms)}>{formatMs(extraction.ms)}</span>
          </span>
        )}

        {/* The tally lands with extraction, before the inspector is opened. */}
        {tally && (
          <span className="num text-[10px] text-ink-200" data-testid="extraction-tally">
            <span className="text-good">{formatCount(tally.agreed)}</span> of{' '}
            {formatCount(tally.total)} fields agreed
          </span>
        )}

        <div className="ml-auto flex items-center gap-2">
          <ShortcutStrip />
          <Button size="sm" onClick={() => void session.captureMonitor(1)}>
            Capture
          </Button>
          <label className="flex items-center gap-1 text-[10px] text-ink-400">
            <input
              type="checkbox"
              checked={mock}
              disabled={mockForcedByEnv()}
              onChange={(e) => {
                setMockMode(e.target.checked);
                window.location.reload();
              }}
            />
            mock
          </label>
          {mock && <Badge tone="warn">FIXTURES</Badge>}
          <Link to="/hud" className="text-[10px] text-series-1 hover:underline">
            /hud
          </Link>
          <Link to="/settings" className="text-[10px] text-series-1 hover:underline">
            settings
          </Link>
        </div>
      </header>

      {session.error && (
        <div className="shrink-0 border-b border-bad/40 bg-bad/10 px-2 py-1 text-[11px] text-bad">
          {session.error}
        </div>
      )}

      {/* A 412 from the optimizer is a precondition, not a crash. */}
      {session.calibrationGate && (
        <div
          role="alert"
          className="shrink-0 border-b border-warn/40 bg-warn/10 px-2 py-1 text-[11px] text-warn"
        >
          {session.calibrationGate}
        </div>
      )}

      {warnings.length > 0 && (
        <div className="shrink-0 border-b border-warn/30 bg-warn/10 px-2 py-0.5">
          {warnings.slice(-2).map((w) => (
            <div key={w.id} className="flex items-center gap-2 text-[11px] text-warn">
              <span className="flex-1">{w.text}</span>
              <button
                type="button"
                className="text-ink-400 hover:text-ink-100"
                onClick={() => dispatch({ kind: 'warning/dismiss', id: w.id })}
              >
                dismiss
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ---------------- three panes ---------------- */}
      <div className="grid min-h-0 flex-1 grid-cols-[13rem_minmax(0,1fr)_22rem]">
        <aside className="min-h-0 border-r border-ink-700 bg-ink-900">
          <Filmstrip
            onExtract={(ids) => void session.extract(ids)}
            extracting={session.extracting}
            lastExtract={session.lastExtract}
          />
        </aside>

        <main className="grid min-h-0 grid-rows-[minmax(0,1fr)_17rem]">
          <div className="min-h-0 border-b border-ink-700 bg-ink-900">
            <StateInspector onConfirm={(patch) => void session.confirm(patch)} />
          </div>

          <section className="flex min-h-0 flex-col bg-ink-900">
            <OptimizerBar
              onStart={(s) => void session.optimize(s)}
              onCancel={(id) => session.cancel(id)}
            />
            <div className="min-h-0 flex-1">
              <MoneyChart
                current={currentCurve}
                best={bestCurve}
                observed={observedCurve}
                bound={bound}
              />
            </div>
          </section>
        </main>

        <aside className="flex min-h-0 flex-col border-l border-ink-700 bg-ink-900">
          <div className="flex shrink-0 border-b border-ink-700 bg-ink-850">
            {(['chat', 'rules'] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                aria-pressed={tab === t}
                className={`px-2.5 py-1 text-[11px] font-semibold tracking-wide uppercase ${
                  tab === t
                    ? 'border-b-2 border-b-series-1 text-ink-100'
                    : 'text-ink-400 hover:text-ink-200'
                }`}
              >
                {t}
                {t === 'rules' && session.rules && session.rules.unresolved.length > 0 && (
                  <span className="num ml-1 text-bad">{session.rules.unresolved.length}</span>
                )}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1">
            {tab === 'chat' ? (
              <ChatPanel onSend={(text) => session.chat(text)} />
            ) : (
              <RulesPanel rules={session.rules} />
            )}
          </div>
        </aside>
      </div>

      {/* ---------------- persistent leaderboard ---------------- */}
      <div className="h-16 shrink-0">
        <Leaderboard entries={session.leaderboard} />
      </div>

      {legendOpen && <ShortcutLegend onClose={() => setLegendOpen(false)} />}
    </div>
  );
}
