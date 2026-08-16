import { Kbd } from './ui';

export const SHORTCUTS: Array<{ keys: string[]; what: string; where: string }> = [
  { keys: ['Tab'], what: 'next disputed field', where: 'inspector' },
  { keys: ['⇧', 'Tab'], what: 'previous disputed field', where: 'inspector' },
  { keys: ['↵'], what: 'accept the shown value', where: 'inspector' },
  { keys: ['A'], what: 'accept ALL majority values', where: 'anywhere' },
  { keys: ['1', '…', '9'], what: 'accept nth alternative', where: 'inspector' },
  { keys: ['↑', '↓'], what: 'move between disputes', where: 'inspector' },
  { keys: ['C'], what: 'capture monitor 1', where: 'anywhere' },
  { keys: ['O'], what: 'run optimizer', where: 'anywhere' },
  { keys: ['/'], what: 'focus chat input', where: 'anywhere' },
  { keys: ['?'], what: 'toggle this legend', where: 'anywhere' },
  { keys: ['Esc'], what: 'leave the text field', where: 'chat' },
];

/** Always-visible one-line strip. The full table lives behind `?`. */
export function ShortcutStrip() {
  return (
    <div className="flex items-center gap-3 text-[10px] text-ink-400">
      {SHORTCUTS.slice(0, 4).map((s) => (
        <span key={s.what} className="flex items-center gap-1">
          {s.keys.map((k) => (
            <Kbd key={k}>{k}</Kbd>
          ))}
          <span>{s.what}</span>
        </span>
      ))}
      <span className="flex items-center gap-1">
        <Kbd>?</Kbd>
        <span>all</span>
      </span>
    </div>
  );
}

export function ShortcutLegend({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80"
      role="dialog"
      aria-label="Keyboard shortcuts"
      onClick={onClose}
    >
      <div
        className="w-[26rem] border border-ink-600 bg-ink-900 p-3"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-xs font-semibold tracking-wide text-ink-200 uppercase">
            Keyboard map
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-[11px] text-ink-400 hover:text-ink-100"
          >
            close (Esc)
          </button>
        </div>
        <table className="w-full">
          <tbody>
            {SHORTCUTS.map((s) => (
              <tr key={s.what} className="border-t border-ink-800">
                <td className="w-24 py-0.5">
                  <span className="flex gap-1">
                    {s.keys.map((k) => (
                      <Kbd key={k}>{k}</Kbd>
                    ))}
                  </span>
                </td>
                <td className="py-0.5 text-[11px] text-ink-100">{s.what}</td>
                <td className="py-0.5 text-right text-[10px] text-ink-500">{s.where}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
