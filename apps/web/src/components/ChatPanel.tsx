import { useEffect, useRef, useState } from 'react';
import { useStore } from '../state/store';
import type { ChatItem } from '../state/reducer';
import { ToolCallCard } from './ToolCallCard';
import { Kbd } from './ui';

/**
 * An unsourced-number alarm. This is a correctness failure in the assistant's
 * answer, not a dismissable toast, so it renders attached to the message that
 * made the claim and names the exact offending tokens.
 */
function ProvenanceAlarm({ numbers, text }: { numbers: string[]; text: string }) {
  return (
    <div
      role="alert"
      data-testid="provenance-warning"
      className="mt-1 rounded border border-warn bg-warn/15 px-1.5 py-1"
    >
      <div className="flex items-center gap-1.5">
        <span className="text-warn" aria-hidden>
          ⚠
        </span>
        <span className="text-[10px] font-semibold tracking-wide text-warn uppercase">
          Unverified numbers
        </span>
      </div>
      <p className="mt-0.5 text-[11px] leading-snug text-warn/90">{text}</p>
      <div className="mt-1 flex flex-wrap gap-1">
        {numbers.map((n, i) => (
          <span
            key={`${n}-${i}`}
            className="num rounded border border-warn/50 bg-ink-950 px-1 py-px text-[11px] text-warn"
          >
            {n}
          </span>
        ))}
      </div>
    </div>
  );
}

function Bubble({ item }: { item: ChatItem }) {
  switch (item.kind) {
    case 'user':
      return (
        <div className="ml-6 rounded border border-series-1/40 bg-series-1/10 px-1.5 py-1 text-xs text-ink-100">
          {item.text}
        </div>
      );
    case 'agent':
      return (
        <div
          className={
            item.provenance.length > 0
              ? 'border-l-2 border-l-warn pl-1.5'
              : undefined
          }
        >
          <div className="text-xs leading-relaxed whitespace-pre-wrap text-ink-200">
            {item.text}
            {!item.complete && (
              <span className="ml-0.5 inline-block h-3 w-1.5 translate-y-px bg-series-1 align-baseline" />
            )}
          </div>
          {item.provenance.map((p, i) => (
            <ProvenanceAlarm key={i} numbers={p.numbers} text={p.text} />
          ))}
        </div>
      );
    case 'provenance':
      return <ProvenanceAlarm numbers={item.numbers} text={item.text} />;
    case 'tool':
      return <ToolCallCard item={item} />;
    case 'notice':
      return (
        <div className="border-l-2 border-l-good pl-1.5 text-[11px] text-good">{item.text}</div>
      );
  }
}

export function ChatPanel({ onSend }: { onSend: (text: string) => void }) {
  const chat = useStore((s) => s.chat);
  const dispatch = useStore((s) => s.dispatch);
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Follow the stream, but never yank the view if the user scrolled up to read.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [chat]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const submit = () => {
    const text = draft.trim();
    if (!text) return;
    dispatch({ kind: 'chat/send', text });
    onSend(text);
    setDraft('');
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-1.5">
        {chat.length === 0 && (
          <p className="py-4 text-center text-[11px] text-ink-500">
            Ask the strategist about the board. Press <Kbd>/</Kbd> to focus.
          </p>
        )}
        {chat.map((item) => (
          <Bubble key={item.id} item={item} />
        ))}
      </div>

      <form
        className="flex shrink-0 gap-1 border-t border-ink-700 bg-ink-850 p-1.5"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') e.currentTarget.blur();
          }}
          placeholder="Ask about the board…"
          aria-label="Message the strategist"
          className="num min-w-0 flex-1 rounded border border-ink-600 bg-ink-900 px-1.5 py-1 text-xs text-ink-100 placeholder:text-ink-500"
        />
        <button
          type="submit"
          className="rounded border border-series-1 bg-series-1/20 px-2 text-xs text-series-1 hover:bg-series-1/30"
        >
          Send
        </button>
      </form>
    </div>
  );
}
