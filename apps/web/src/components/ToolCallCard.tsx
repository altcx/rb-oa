import { useState } from 'react';
import type { ChatToolItem } from '../state/reducer';
import { formatMs, prettyJson, summarizeJson } from '../lib/format';
import { latencyColorClass } from '../lib/format';

function JsonBlock({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="mt-1">
      <div className="text-[10px] tracking-wide text-ink-500 uppercase">{label}</div>
      <pre className="num mt-0.5 max-h-56 overflow-auto rounded border border-ink-750 bg-ink-950 p-1.5 text-[11px] leading-snug whitespace-pre text-ink-200">
        {prettyJson(value)}
      </pre>
    </div>
  );
}

/** Tool calls are collapsed by default: one line in, everything on demand. */
export function ToolCallCard({ item }: { item: ChatToolItem }) {
  const [open, setOpen] = useState(false);
  const pending = item.output === null;

  return (
    <div className="rounded border border-ink-700 bg-ink-850">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-baseline gap-2 px-1.5 py-1 text-left hover:bg-ink-800"
      >
        <span className="text-ink-500">{open ? '▾' : '▸'}</span>
        <span className="num shrink-0 text-[11px] font-semibold text-series-1">{item.name}</span>
        <span className="min-w-0 flex-1 truncate text-[10px] text-ink-400">
          {summarizeJson(item.input)}
        </span>
        {pending ? (
          <span className="shrink-0 text-[10px] text-warn">running…</span>
        ) : (
          <span className={`num shrink-0 text-[10px] ${latencyColorClass(item.ms ?? 0)}`}>
            {item.ms === null ? '' : formatMs(item.ms)}
          </span>
        )}
      </button>

      {open && (
        <div className="border-t border-ink-750 px-1.5 pt-1 pb-1.5">
          <JsonBlock label="input" value={item.input} />
          {pending ? (
            <div className="mt-1 text-[11px] text-ink-500">
              Waiting for result — the input above is final.
            </div>
          ) : (
            <JsonBlock label="output" value={item.output} />
          )}
        </div>
      )}
    </div>
  );
}
