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
2. five `extraction_progress` stages (decode → three jury models → reconcile)
3. `state` — **40 fields, 3 disputed** (1 three-way `split`, 2 `majority`)
4. `agent_token` streaming prose, character-accurate word-by-word
5. `agent_tool_call` / `agent_tool_result` for `read_rules`, then for
   `simulate_config` — two collapsible cards
6. more streamed tokens, then `agent_done`
7. a `warning`
8. **8 `optimizer_progress` events** with `best_value` climbing toward a fixed LP
   `bound`, a fresh `money_by_hour` curve each time, and a `top_warning`
   appearing partway through

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

> **Backend note:** for path-style routes to work off `dist`, FastAPI needs a
> catch-all that returns `index.html` for unknown non-`/api`, non-`/ws` paths.
> Without it, use the hash forms (`/#/hud`, `/#/settings`) — they need no server
> support.

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

## Tests

```
src/test/format.test.ts      17  number/latency/value formatting
src/test/reducer.test.ts     20  every socket event type + review selectors
src/test/inspector.test.tsx  13  Tab/Shift-Tab/Enter/A/1-9 flow, agreed collapse
src/test/app.smoke.test.tsx   3  mock-mode render of Dashboard + HUD, full replay
```

## Known gaps

- The **"current config"** money line is read from the newest `simulate_config`
  tool result (`output.by_hour`); the **"observed run"** line has no endpoint in
  the contract at all and is fixture-only. Both fall back to fixtures in mock
  mode and render as absent (not zero) against a live backend. A REST endpoint
  returning both baselines would close this.
- The HUD's click-through and always-on-top behaviour is a window-manager
  property; this app only guarantees the 240×80 footprint and
  `pointer-events: none` everywhere except the capture button.
- Region capture (`POST /api/captures/region`) is wired in the client and the
  session hook but has no drag-to-select UI yet; the HUD advertises the
  `Ctrl+Shift+R` hotkey the desktop shell is expected to own.
