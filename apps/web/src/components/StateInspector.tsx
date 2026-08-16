import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { JsonValue, Verdict } from '../api/types';
import { useStore } from '../state/store';
import {
  contestedVerdicts,
  openDisputes,
  reviewCounts,
  settledVerdicts,
} from '../state/reducer';
import { formatFieldValue } from '../lib/format';
import { CropZoom } from './CropZoom';
import { Badge, Kbd } from './ui';

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || target.isContentEditable;
}

function VoteRow({ model, value, agrees }: { model: string; value: JsonValue; agrees: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="truncate text-[10px] text-ink-400">{model}</span>
      <span
        className={`num shrink-0 text-[11px] ${agrees ? 'text-ink-100' : 'text-warn'}`}
      >
        {formatFieldValue(value)}
      </span>
    </div>
  );
}

function DisputeRow({
  verdict,
  index,
  active,
  onAccept,
  registerRef,
}: {
  verdict: Verdict;
  index: number;
  active: boolean;
  onAccept: (path: string, value: JsonValue) => void;
  registerRef: (index: number, el: HTMLDivElement | null) => void;
}) {
  const captures = useStore((s) => s.captures);
  const votes = Object.entries(verdict.votes);
  const agreeing = votes.filter(([, v]) => formatFieldValue(v) === formatFieldValue(verdict.value));
  const isSplit = verdict.status === 'split';

  return (
    <div
      ref={(el) => registerRef(index, el)}
      role="option"
      aria-selected={active}
      tabIndex={0}
      data-path={verdict.path}
      data-status={verdict.status}
      className={`flex gap-3 border-l-2 px-2 py-2 outline-none ${
        active
          ? 'border-l-series-1 bg-ink-800'
          : 'border-l-transparent bg-ink-900 hover:bg-ink-850'
      }`}
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="num truncate text-xs text-ink-200">{verdict.path}</span>
          <Badge tone={isSplit ? 'bad' : 'warn'}>
            {isSplit ? 'SPLIT' : `MAJORITY ${agreeing.length}/${votes.length}`}
          </Badge>
        </div>

        <div className="mt-1 flex items-baseline gap-2">
          <span className="num text-lg leading-tight font-semibold text-ink-100">
            {formatFieldValue(verdict.value)}
          </span>
          <span className="text-[10px] text-ink-400">
            {isSplit ? 'leading value — no majority' : 'majority value'}
          </span>
          {active && (
            <span className="ml-auto text-[10px] text-ink-400">
              <Kbd>↵</Kbd> accept
            </span>
          )}
        </div>

        <div className="mt-1.5 grid grid-cols-1 gap-x-4 gap-y-0.5 sm:grid-cols-2">
          {votes.map(([model, value]) => (
            <VoteRow
              key={model}
              model={model}
              value={value}
              agrees={formatFieldValue(value) === formatFieldValue(verdict.value)}
            />
          ))}
        </div>

        {verdict.alternatives.length > 0 && (
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] text-ink-500">instead:</span>
            {verdict.alternatives.map((alt, i) => (
              <button
                key={`${verdict.path}-alt-${i}`}
                type="button"
                onClick={() => onAccept(verdict.path, alt)}
                className="num rounded border border-ink-600 bg-ink-800 px-1.5 py-px text-[11px] text-ink-200 hover:border-series-1 hover:text-series-1"
              >
                <span className="mr-1 text-ink-500">{i + 1}</span>
                {formatFieldValue(alt)}
              </button>
            ))}
          </div>
        )}
      </div>

      <CropZoom
        crop={verdict.crop}
        captures={captures}
        label={`Source crop for ${verdict.path}`}
      />
    </div>
  );
}

export function StateInspector({
  onConfirm,
}: {
  onConfirm?: (patch: Record<string, JsonValue>) => void;
}) {
  const state = useStore();
  const dispatch = state.dispatch;

  const disputes = useMemo(() => openDisputes(state), [state]);
  const settled = useMemo(() => settledVerdicts(state), [state]);
  const contested = useMemo(() => contestedVerdicts(state), [state]);
  const counts = useMemo(() => reviewCounts(state), [state]);

  const [activeIndex, setActiveIndex] = useState(0);
  const [agreedOpen, setAgreedOpen] = useState(false);
  const rowRefs = useRef<Array<HTMLDivElement | null>>([]);
  const wantFocus = useRef(false);

  const registerRef = useCallback((index: number, el: HTMLDivElement | null) => {
    rowRefs.current[index] = el;
  }, []);

  // Keep the cursor inside the shrinking list as fields get accepted.
  useEffect(() => {
    if (activeIndex >= disputes.length && disputes.length > 0) {
      setActiveIndex(disputes.length - 1);
    }
  }, [disputes.length, activeIndex]);

  useEffect(() => {
    if (!wantFocus.current) return;
    wantFocus.current = false;
    rowRefs.current[activeIndex]?.focus();
  }, [activeIndex, disputes.length]);

  const accept = useCallback(
    (path: string, value: JsonValue) => {
      dispatch({ kind: 'accept', path, value });
      onConfirm?.({ [path]: value });
    },
    [dispatch, onConfirm],
  );

  const acceptAllMajority = useCallback(() => {
    const patch: Record<string, JsonValue> = {};
    for (const v of disputes) {
      if (v.status === 'majority') patch[v.path] = v.value;
    }
    if (Object.keys(patch).length === 0) return;
    dispatch({ kind: 'accept-all-majority' });
    onConfirm?.(patch);
  }, [disputes, dispatch, onConfirm]);

  const move = useCallback(
    (delta: number) => {
      if (disputes.length === 0) return;
      wantFocus.current = true;
      setActiveIndex((i) => (i + delta + disputes.length) % disputes.length);
    },
    [disputes.length],
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (isTypingTarget(e.target)) return;
      if (e.key === 'Tab') {
        if (disputes.length === 0) return;
        e.preventDefault();
        move(e.shiftKey ? -1 : 1);
        return;
      }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        move(e.key === 'ArrowDown' ? 1 : -1);
        return;
      }
      if (e.key === 'Enter') {
        const v = disputes[activeIndex];
        if (!v) return;
        e.preventDefault();
        wantFocus.current = true;
        accept(v.path, v.value);
        return;
      }
      if (/^[1-9]$/.test(e.key)) {
        const v = disputes[activeIndex];
        if (!v) return;
        const alt = v.alternatives[Number(e.key) - 1];
        if (alt === undefined) return;
        e.preventDefault();
        wantFocus.current = true;
        accept(v.path, alt);
      }
    },
    [accept, activeIndex, disputes, move],
  );

  // `A` is global so it works the moment a state payload lands, before the
  // operator has tabbed into the list.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'a' && e.key !== 'A') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      acceptAllMajority();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [acceptAllMajority]);

  const unresolved = state.statePayload?.unresolved ?? [];

  return (
    <div
      className="flex h-full min-h-0 flex-col"
      onKeyDown={onKeyDown}
      data-testid="state-inspector"
    >
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-ink-700 bg-ink-850 px-2 py-1">
        <div className="flex items-center gap-2">
          <h2 className="text-[11px] font-semibold tracking-wide text-ink-300 uppercase">
            State inspector
          </h2>
          <span className="num text-[11px] text-ink-400" data-testid="review-counter">
            {counts.fields} fields, {counts.disputed} disputed, {counts.remaining} left to review
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-ink-400">
          <span>
            <Kbd>Tab</Kbd> next
          </span>
          <span>
            <Kbd>↵</Kbd> accept
          </span>
          <span>
            <Kbd>A</Kbd> accept all majority
            {counts.majorityRemaining > 0 && (
              <span className="num ml-1 text-warn">({counts.majorityRemaining})</span>
            )}
          </span>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {counts.fields === 0 && (
          <p className="px-2 py-6 text-center text-xs text-ink-500">
            No extraction yet. Capture the board to populate the inspector.
          </p>
        )}

        {/* Disputed fields first — sorted by disagreement, not document order. */}
        {disputes.length > 0 && (
          <div role="listbox" aria-label="Disputed fields" className="divide-y divide-ink-750">
            {disputes.map((v, i) => (
              <DisputeRow
                key={v.path}
                verdict={v}
                index={i}
                active={i === activeIndex}
                onAccept={accept}
                registerRef={registerRef}
              />
            ))}
          </div>
        )}

        {counts.fields > 0 && disputes.length === 0 && (
          <div className="border-l-2 border-l-good bg-good/5 px-2 py-2 text-xs text-good">
            Board verified — every disputed field settled.
          </div>
        )}

        {unresolved.length > 0 && (
          <div className="border-t border-ink-750 px-2 py-1.5">
            <div className="text-[10px] tracking-wide text-bad uppercase">
              Unresolved — no confident reading
            </div>
            <ul className="num mt-0.5 text-[11px] text-ink-300">
              {unresolved.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Everything the jury agreed on collapses to one row. */}
        {counts.fields > 0 && (
          <div className="border-t border-ink-700">
            <button
              type="button"
              aria-expanded={agreedOpen}
              onClick={() => setAgreedOpen((o) => !o)}
              className="flex w-full items-center gap-2 bg-ink-850 px-2 py-1.5 text-left hover:bg-ink-800"
            >
              <span className="text-ink-500">{agreedOpen ? '▾' : '▸'}</span>
              <span className="num text-xs text-good">{counts.agreed} fields agreed</span>
              <span className="text-[10px] text-ink-500">
                {agreedOpen ? 'click to collapse' : 'click to expand'}
              </span>
            </button>
            {agreedOpen && (
              <ul className="divide-y divide-ink-850" data-testid="agreed-list">
                {settled.map((v) => {
                  const wasContested = contested.some((c) => c.path === v.path);
                  const shown = state.accepted[v.path] ?? v.value;
                  return (
                    <li
                      key={v.path}
                      className="flex items-baseline justify-between gap-3 px-2 py-0.5"
                    >
                      <span className="num truncate text-[11px] text-ink-400">{v.path}</span>
                      <span className="num shrink-0 text-[11px] text-ink-100">
                        {formatFieldValue(shown)}
                        {wasContested && (
                          <span className="ml-1 text-[9px] text-series-1">confirmed</span>
                        )}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
