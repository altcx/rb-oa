# Puzzle Copilot — web UI

The frontend for a two-monitor setup: a timed puzzle game runs on monitor 1 with a
240×80 always-on-top HUD over it, and this app fills monitor 2 with the board
inspector, the strategist chat, the money chart and the leaderboard.

React 19 · Vite 6 · TypeScript (strict) · Tailwind v4 · recharts · zustand.

## Quick start

```bash
cd apps/web
npm install

npm run dev          # talks to the FastAPI backend on :8000 (proxied)
npm run dev:mock     # same as VITE_MOCK=1 npm run dev — no backend needed
npm run test         # vitest
npm run build        # tsc --noEmit && vite build  ->  apps/web/dist
```

`npm run build` emits `apps/web/dist`, which FastAPI serves.

## Mock mode

Mock mode replaces both halves of the contract — the REST client and the
websocket — with fixtures, so the entire UI is demoable and testable with no
server running. It is how this app was developed and verified.

Turn it on either way:

- **`VITE_MOCK=1`** at dev/build time (`npm run dev:mock`). This *forces* mock
  mode and disables the in-UI toggle.
- **The `mock` checkbox** in the dashboard header or on `/settings`. It persists
  to `localStorage["puzzle-copilot.mock"]` and reloads the page.

What the fake socket replays, in order (`src/mock/fakeSocket.ts`):

1. `capture` — a new board screenshot
2. five `extraction_progress` stages (decode → three jury models → reconcile);
   the terminal stage carries `disputed: 3, auto_confirmed: 37`, which the header
   shows as "37 of 40 fields agreed" before the inspector is opened
3. `state` — **40 fields, 3 disputed** (1 three-way `split`, 2 `majority`)
4. `agent_token` streaming prose, character-accurate word-by-word
5. `agent_tool_call` / `agent_tool_result` for `read_rules`, then for
   `simulate_config` — two collapsible cards
6. more streamed tokens, then `agent_done`
7. a `provenance_warning` naming two figures the agent stated that no tool result
   supports — rendered attached to that assistant message
8. a `warning`
9. a `calibration` event; the mock REST layer flips with it, so `GET /chart`
   stops returning `observed: null` and the absent legend entry becomes a line
10. **8 `optimizer_progress` events** with `best_value` climbing toward a fixed LP
    `bound`, a fresh `money_by_hour` curve each time, a `top_warning` appearing
    partway through, and `final: true` on the last one only

Pressing **C** (or the capture button, or the HUD button) in mock mode publishes
through `src/mock/bus.ts` and pushes a *new* capture + re-extraction down the
fake socket, so capture → state really round-trips. Chat messages get a canned
streamed reply. `POST /leaderboard` recomputes the high-water mark.

Fixtures live in `src/mock/fixtures.ts`; the capture "screenshots" are generated
SVG data URIs, so the zoomed source crops in the inspector land on real pixels.

## Routes

| Route | View | Notes |
|---|---|---|
| `/` | Dashboard (monitor 2) | three panes + chart + leaderboard strip |
| `/hud` | HUD overlay (monitor 1) | exactly 240×80, click-through except the button |
| `/settings` | Settings | OpenRouter key, model roles, measured latency |

Routing is a ~50-line hand-rolled router (`src/router.tsx`) — three routes did not
justify a dependency. It resolves **both** `/hud` and `/#/hud`, so if the desktop
shell opens the HUD window at a hash URL, or the static server has no SPA
fallback, the route still lands.

Path-style routes work off `dist`: `services/core/main.py` mounts `/assets` and
serves `index.html` from a `/{full_path:path}` catch-all. The hash forms remain a
zero-config fallback for any other static host.

The HUD is deliberately kept out of the lazy-loaded chunk: `/hud` ships ~225 kB
and never parses recharts, while the Dashboard chunk (~425 kB) loads only on
monitor 2.

## Keyboard map

The board must be verifiable without touching the mouse. Press **`?`** in the app
for this table.

| Key | Action | Scope |
|---|---|---|
| `Tab` | next disputed field | inspector |
| `Shift`+`Tab` | previous disputed field | inspector |
| `↑` / `↓` | move between disputes | inspector |
| `Enter` | accept the shown value for the focused field | inspector |
| `1`…`9` | accept the *n*th alternative instead | inspector |
| **`A`** | **accept ALL majority values at once** | anywhere |
| `C` | capture monitor 1 | anywhere |
| `O` | run the optimizer (30 s budget) | anywhere |
| `/` | focus the chat input | anywhere |
| `?` | toggle the shortcut legend | anywhere |
| `Esc` | leave the text field / close the legend | anywhere |

`A` deliberately settles only fields that *have* a majority. A three-way `split`
has no safe default, so it is left focused for a human — the counter and the
`A (n)` hint both show how many majorities remain.

## Layout & component tree

```
App (router)
├── /hud   → Hud                       240×80: capture button, hotkey reminder,
│                                      last thumbnail, latency vs 4 s budget
├── /settings → SettingsView           key + validation + storage backend notice,
│                                      catalog, role slots, per-role latency
└── /      → Dashboard
            ├── header                 connection pill, live extraction stage,
            │                          ShortcutStrip, mock toggle, route links
            ├── warnings banner        from {type:"warning"}, dismissable
            ├── LEFT   Filmstrip       newest-first thumbnails, time + latency
            ├── CENTER StateInspector  disputes first (split before majority),
            │          ├── DisputeRow  votes, alternatives, CropZoom
            │          └── agreed collapse  "37 fields agreed" → expandable
            │   BELOW  OptimizerBar    best-so-far ‖ LP bound ‖ gap ‖ iterations
            │          MoneyChart      3 lines + LP-bound reference line
            ├── RIGHT  tabs
            │          ├── ChatPanel   streaming tokens + ToolCallCard (collapsed)
            │          └── RulesPanel  resolved rules, unresolved flags, experiments
            └── BOTTOM Leaderboard     high-water pinned left, others as deltas
```

State: one pure reducer (`src/state/reducer.ts`) wrapped in a zustand store
(`src/state/store.ts`). Every websocket event goes through
`applyServerEvent(state, event)` — a `switch` with one branch per `type` in the
contract, exhaustive under `strict`. Selectors (`openDisputes`, `reviewCounts`,
`settledVerdicts`) are pure functions of that state, so the keyboard flow is
testable without a DOM.

## Design decisions worth knowing

- **Disagreement order, not document order.** The inspector sorts `split` above
  `majority` and hides everything unanimous behind one row. A 40-field board is
  three decisions, not forty.
- **No spinner where a partial result exists.** The optimizer panel shows
  best-so-far *beside* the LP bound and the remaining gap in dollars and percent,
  so the operator can judge whether more seconds are worth it. Tool cards show
  their inputs while the result is still pending.
- **Numbers never round away a difference.** Money and scores always carry two
  decimals and are never abbreviated (`$12,480.75`, never `$12.5k`); only axis
  ticks coarsen, and only to whole dollars. Extracted field values are printed
  verbatim, because telling `12.5` from `1.25` is the entire job. See
  `src/lib/format.ts` and its tests.
- **Chart colours** are the validated dark-surface categorical slots 1–3
  (blue / orange / aqua). The legend direct-labels each series with its final
  value and its delta vs the current config, so identity is never colour-alone.
  The LP bound is a neutral dashed reference line, not a fourth series.
- **Same-family jury warning.** `/settings` flags two extractor slots from one
  family: same-family models fail the same way, so a 2–1 majority can be a single
  shared blind spot rather than a real vote.
- **Provenance alarms are not toasts.** `provenance_warning` means the agent
  stated a number no tool result supports. It renders inside the chat pane
  attached to the offending assistant message, marks that message with a warning
  rule, and lists the exact offending tokens. It cannot be dismissed, because the
  answer above it is wrong. Orphan alarms (no assistant message to attach to)
  still render standalone rather than being dropped.
- **The optimizer's calibration gate is a precondition, not a crash.** `POST
  /api/solve/optimize` answers 412 until a recorded run has matched the
  simulator; the UI catches that status specifically and explains it, instead of
  printing an HTTP error.
- **Re-extraction is the recovery path**, not the primary one — extraction starts
  speculatively when a capture lands. The filmstrip's "Re-extract selected"
  action calls `POST /api/extract`, shows `elapsed_ms` against the same 4 s
  budget the HUD colour-codes, and labels a `used_delta` pass as a delta re-read
  so the operator learns to expect it to be fast.

## Tests

```
src/test/format.test.ts      17  number/latency/value formatting
src/test/reducer.test.ts     27  all 11 socket event types + review selectors
src/test/money.test.ts       11  hour-0 indexing, null-is-not-zero
src/test/inspector.test.tsx  13  Tab/Shift-Tab/Enter/A/1-9 flow, agreed collapse
src/test/app.smoke.test.tsx   4  mock-mode render, full replay, re-extract flow
                             ---
                              72
```

## The money chart and `GET /api/sessions/{id}/chart`

All three lines come from the chart endpoint — no more scavenging curves out of
tool results. Two details in this wiring are easy to get wrong and are covered by
tests in `src/test/money.test.ts`:

**The array index is the hour.** `money_by_hour` has length `horizon_hours + 1`;
index 0 is the money on hand *before* hour 1 runs, and index h is the money at
the end of hour h. `hourlyToPoints` in `src/lib/money.ts` maps index → hour
directly. Plotting index 0 as hour 1 would shift every line by an hour and
misreport when the first sale lands — precisely what the reader is here to see.

**`null` is not zero.** A line the session does not have comes back `null` and is
rendered *absent*: no line drawn, and a greyed legend entry saying why —
"Observed run — not recorded yet", "Optimizer best — optimizer has not run". It
is never coerced to `0`, because a flat zero line reads as a real and
catastrophic run. `hourlyToPoints(null)` returns `null`, and an empty array is
also treated as absent (but a genuine `0.0` value is preserved).

The chart is refetched on mount and when a `state` event, a `calibration` event,
or a **final** `optimizer_progress` event arrives. Non-final optimizer ticks are
deliberately *not* a refetch trigger: they carry `money_by_hour` in band, so the
optimizer line animates live between refetches and the endpoint wins once the
job finishes.

## Known gaps

- The HUD's click-through and always-on-top behaviour is a window-manager
  property; this app only guarantees the 240×80 footprint and
  `pointer-events: none` everywhere except the capture button.
- Region capture (`POST /api/captures/region`) is wired in the client and the
  session hook but has no drag-to-select UI yet; the HUD advertises the
  `Ctrl+Shift+R` hotkey the desktop shell is expected to own.
