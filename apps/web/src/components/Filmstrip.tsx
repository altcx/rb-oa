import type { ExtractResponse } from '../api/types';
import { useStore } from '../state/store';
import { formatCount, formatMs, formatClock, latencyColorClass } from '../lib/format';
import { Badge, Button } from './ui';

export function Filmstrip({
  onExtract,
  extracting = false,
  lastExtract = null,
}: {
  onExtract?: (captureIds: string[]) => void;
  extracting?: boolean;
  lastExtract?: ExtractResponse | null;
}) {
  const captures = useStore((s) => s.captures);
  const selected = useStore((s) => s.selectedCaptureId);
  const dispatch = useStore((s) => s.dispatch);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 items-center justify-between border-b border-ink-700 bg-ink-850 px-2 py-1">
        <h2 className="text-[11px] font-semibold tracking-wide text-ink-300 uppercase">
          Captures
        </h2>
        <span className="num text-[10px] text-ink-500">{captures.length}</span>
      </header>

      {/* Extraction runs speculatively the moment a capture lands, so this is
          the recovery path: re-read the selected capture after a failed pass,
          or deliberately re-read a different frame. */}
      {onExtract && selected && (
        <div className="shrink-0 border-b border-ink-750 bg-ink-900 px-1.5 py-1">
          <Button
            size="sm"
            onClick={() => onExtract([selected])}
            disabled={extracting}
            title="Re-run extraction over the selected capture"
          >
            {extracting ? 'Extracting…' : 'Re-extract selected'}
          </Button>
          {lastExtract && !extracting && (
            <div className="mt-1 space-y-0.5" data-testid="extract-result">
              <div className="flex items-baseline gap-1.5">
                <span className={`num text-[11px] ${latencyColorClass(lastExtract.elapsed_ms)}`}>
                  {formatMs(lastExtract.elapsed_ms)}
                </span>
                <span className="text-[9px] text-ink-500">/ 4,000 ms budget</span>
              </div>
              {lastExtract.used_delta && (
                <Badge tone="good">DELTA RE-READ — only what changed</Badge>
              )}
              <div className="num text-[10px] text-ink-400">
                {formatCount(lastExtract.auto_confirmed)} auto-confirmed ·{' '}
                <span className={lastExtract.disputed.length > 0 ? 'text-warn' : 'text-ink-400'}>
                  {formatCount(lastExtract.disputed.length)} disputed
                </span>
                {lastExtract.unresolved.length > 0 && (
                  <>
                    {' · '}
                    <span className="text-bad">
                      {formatCount(lastExtract.unresolved.length)} unresolved
                    </span>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <ul className="min-h-0 flex-1 overflow-y-auto" data-testid="filmstrip">
        {captures.length === 0 && (
          <li className="px-2 py-4 text-center text-[11px] text-ink-500">
            No captures yet. Press the hotkey on monitor 1.
          </li>
        )}
        {captures.map((c) => {
          const isSelected = c.id === selected;
          return (
            <li key={c.id}>
              <button
                type="button"
                onClick={() => dispatch({ kind: 'select-capture', captureId: c.id })}
                aria-current={isSelected}
                className={`flex w-full gap-2 border-l-2 p-1.5 text-left ${
                  isSelected
                    ? 'border-l-series-1 bg-ink-800'
                    : 'border-l-transparent hover:bg-ink-850'
                }`}
              >
                <img
                  src={c.thumb_url}
                  alt={`Capture ${c.id}`}
                  width={84}
                  height={48}
                  className="h-12 w-21 shrink-0 border border-ink-700 bg-ink-950 object-cover"
                />
                <div className="min-w-0 flex-1">
                  <div className="num text-[11px] text-ink-200">{formatClock(c.created_at)}</div>
                  <div className={`num text-[11px] ${latencyColorClass(c.latency_ms)}`}>
                    {formatMs(c.latency_ms)}
                  </div>
                  <div className="mt-0.5 flex flex-wrap gap-1">
                    <Badge>{c.region ? 'region' : `mon ${c.monitor}`}</Badge>
                  </div>
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
