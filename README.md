# Puzzle Copilot v2

A local desktop tool. Monitor 1 runs a timed puzzle game. Monitor 2 runs this.
You hit a hotkey, the tool reads the screen into structured state, deterministic
solvers compute the answer, and a chat agent tells you exactly what to change in
the game.

**The tool never touches the game.** Read only. No input automation, no clicking,
no keystroke injection.

## The one design decision everything follows from

```
screenshot -> structured state (LLM) -> verification -> deterministic solver -> LLM narrates
```

A vision model looking at a factory will produce a confident hourly profit figure
that is wrong, and you will not know it is wrong. So the model reads and narrates;
it does not compute. Every number the agent reports traces to a tool result —
enforced in the system prompt (`services/core/agent/system_prompt.md`) and asserted
by `services/core/agent/guard.py`.

That is also why the build order is simulator and enumerator *first*, screenshot
parser second. A verified solver with hand-entered JSON is a usable tool. A
polished capture pipeline feeding a wrong simulator is worse than nothing.

## Two puzzle families

**Factory.** A production graph of Suppliers, Makers and Sellers. Optimize
configuration for profit over a fixed horizon. Rules are mostly known; the eight
that are not are hypothesis flags (`RuleFlags`) pinned by calibration against a
real recorded run.

**Builder.** Assemble a thing from parts with weights and attributes, under a
weight budget and a part cap, such that it clears a sequence of obstacles. The
rules are *not known in advance* and are read off the screen at runtime. Nothing
in `services/solvers/builder/` names a car, a ship or a vehicle.

## Unknown rules are hypotheses, not blockers

Every unresolved rule is a flag with an enumerated option set. The solver runs
under every combination still consistent with observation
(`services/core/rules/resolve.py`):

- **Consensus** — where all interpretations agree, act now and say nothing about
  the ambiguity.
- **Divergence** — report it, grouped by the flag that drives it, and name the
  single cheapest in-game observation that resolves it (an `Experiment`, executable
  in under fifteen seconds).
- **Value of information** — the spread between best and worst interpretation, so
  you can decide whether resolving is worth the clock.

Resolutions persist to `data/rules/`, keyed by **puzzle instance fingerprint**, not
by puzzle family — a rule pinned on one instance must never leak into another.

## Layout

```
services/core/
  main.py             FastAPI + WebSocket, serves the built UI
  session.py          session state, verdicts, leaderboard, persistence
  settings/           keyring-backed key storage, model catalog and roles
  capture/            hotkeys, screen grab, capture store
  extract/            schemas (single source of truth), jury, delta, pipeline
  rules/              rule DSL, compiler, hypothesis resolution, persistence
  agent/              hand-rolled OpenAI-compatible tool loop, tools, prompt
  llm/                provider interface + OpenRouter client
services/solvers/
  factory/            model, sim, bounds (LP), optimize, calibrate
  builder/            model, clip, count, enumerate, minimal, ordered, cpsat
  runtime/worker.py   warm process, anytime job queue
apps/web/             React 19 + Vite + Tailwind + recharts
tests/                golden tests, fixtures, latency budgets
```

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# build the UI once (FastAPI serves apps/web/dist)
cd apps/web && npm install && npm run build && cd ../..

.venv/bin/python -m services.core.main       # http://127.0.0.1:8765
```

Drag the browser window to monitor 2. The HUD lives at `/hud`, settings at
`/settings`. Paste an OpenRouter key in settings; it is validated against
`GET /api/v1/key` before it is accepted and stored in the OS keyring (or a 0600
file, and the UI says which).

Front-end development without a backend:

```bash
cd apps/web && VITE_MOCK=1 npm run dev
```

## Tests

```bash
.venv/bin/python -m pytest -q                      # everything
.venv/bin/python -m pytest tests/golden -q         # the solvers' truth tests
.venv/bin/python -m pytest tests/latency -q -m latency
```

The golden factory tests carry their expected values as hand-computed arithmetic
in comments. Golden test 10 — a machine set one unit above what upstream can
supply drops to **zero**, not to n-1 — is the one that keeps the optimizer honest:
input allocation is a cliff, not a slope.

## Latency budgets (spec section 6)

| Stage | Target |
|---|---|
| Capture | 30 ms |
| Crop and downscale | 40 ms |
| Vision extraction | 1.5 s |
| Human verification | 5–15 s |
| Solve | first answer at 200 ms |
| Narration, time to first token | under 1.5 s |

Extraction and verification are the whole game, which is why the jury exists:
the same crop goes to three vision models from three *different families* (same-family
models correlate their errors), and fields all three agree on never reach you.

`services/core/extract/practice.py` replays recorded captures through the full
pipeline and reports per-stage timing, accuracy, and — the metric that actually
matters — **wrong auto-confirms**. A wrong auto-confirm is worse than a low
agreement rate, and any field a human has to correct after it auto-confirmed is a
P0 extraction bug whose fixture goes into the test set permanently.

## Calibration is a gate

The optimizer refuses to run (HTTP 412) until at least one recorded game run
matches the simulator to the dollar. On mismatch, `calibrate.py` re-simulates
under every combination of the unresolved factory flags and reports which
combinations reproduce the observed series. The override exists, behind an
explicit flag and a loud warning, because the alternative — a simulator that
silently diverges from the game — is the failure mode that wastes a whole run.

There are nine flags, not the eight the spec lists. The ninth,
`full_storage_behavior`, came out of writing the simulator: a machine whose
storage is full either idles or keeps producing into a full buffer and destroys
the output, and the second case burns inputs and production cost for nothing.
The simulator cannot avoid taking a position on it, so it is a flag with an
experiment attached rather than an assumption in a comment. That is the rule the
whole design follows — an unstated rule you had to guess is a flag, not a note.
