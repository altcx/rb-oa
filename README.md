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

## Running it — Windows

**Windows 10/11 is the production platform.** Screen capture needs a real
desktop session with the monitors attached, so this runs on the machine you are
playing on, not in a container and not over a headless SSH session.

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js LTS.

```powershell
winget install --id=astral-sh.uv
winget install --id=OpenJS.NodeJS.LTS
```

Then, from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # venv + deps + UI build
powershell -ExecutionPolicy Bypass -File scripts\run.ps1     # http://127.0.0.1:8765
```

`setup.ps1` is the whole first-run path; `run.ps1` starts the server and blocks
until Ctrl+C. If you would rather type it out:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"

# build the UI once (FastAPI serves apps\web\dist; without it every page 404s)
cd apps\web ; npm install ; npm run build ; cd ..\..

$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m services.core.main       # http://127.0.0.1:8765
```

Python is **3.12**, not 3.13: `ortools` and `scipy` publish cp312 wheels for
`win_amd64`, and on 3.13 pip falls through to building `ortools` from source,
which is not something to discover with a clock running. `pyproject.toml`
enforces the bound.

`uvicorn[standard]` lists `uvloop`, which is POSIX-only. Upstream already gates
it behind `sys_platform != 'win32'`, so pip silently skips it on Windows and
uvicorn uses the stock asyncio loop. Nothing to do; it is not an error.

Drag the browser window to monitor 2. The HUD lives at `/hud`, settings at
`/settings`. Paste an OpenRouter key in settings; it is validated against
`GET /api/v1/key` before it is accepted, then stored in the Windows Credential
Locker via `keyring`. The UI names the backend it actually used.

> **On key storage.** If the Credential Locker is unavailable the tool falls
> back to a file under `data\.secrets\`, and on Windows that file is locked down
> with an explicit ACL (`icacls /inheritance:r /grant:r <you>:F`), read back to
> confirm, and only then written. `os.chmod(0o600)` is meaningless on NTFS — it
> flips the read-only attribute and leaves every other local account able to
> read the file — so if the ACL cannot be verified the tool **refuses to store
> the key at all** and tells you to fix the keyring. It will not claim a
> protection it is not getting. `OPENROUTER_API_KEY` in the environment works as
> a stopgap.

### Two monitors, two DPI scalings

This tool exists to sit on monitor 2 while the game runs on monitor 1, and on
Windows those two panels very commonly run at **different DPI scaling** — a
150% laptop panel next to a 100% external, say. That is not an edge case here,
it is the normal configuration, and it has consequences:

- **Capture coordinates are physical pixels.** `mss` reports the physical
  desktop; the crop rectangles you draw in a UI running at a different scale are
  logical. If the two monitors disagree about scale, a rectangle picked on one
  does not land where you expect on the other.
- **The process must be per-monitor DPI aware.** A process that is not gets
  handed *virtualised* coordinates by Windows — the OS lies to it consistently
  and the capture silently reads the wrong region, or a blurry upscaled one.
  There is no error; the extraction just gets worse.
- **`PUZZLE_COPILOT_DPI_SCALE` is a single global override.** It applies one
  factor to every monitor, which is correct on a uniform setup and wrong on a
  mixed one. If your monitors differ, this is the knob that cannot express it.
- **Practical advice:** set both monitors to the same scaling for a run if you
  can. If you cannot, verify a capture on monitor 1 before the clock starts —
  a crop that is off by a scale factor is obvious in the first screenshot and
  invisible thereafter.
- Windows may also relocate the browser window between monitors on
  resolution/scaling changes, which moves the HUD mid-run. Pin it before you
  start.

## Developing on Linux / macOS

The tests were written on headless Linux and that is still the fastest place to
run them. Everything except the capture layer works there; screen capture needs
`DISPLAY` or `WAYLAND_DISPLAY` and raises `DisplayUnavailableError` when there
is no display, so the pipeline is exercised against recorded captures instead.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cd apps/web && npm install && npm run build && cd ../..
.venv/bin/python -m services.core.main       # http://127.0.0.1:8765
```

Front-end development without a backend:

```bash
cd apps/web && VITE_MOCK=1 npm run dev
```

## Tests

Windows:

```powershell
.venv\Scripts\python.exe scripts\acceptance.py         # milestone-by-milestone report
.venv\Scripts\python.exe scripts\acceptance.py M5 M6   # just these two
.venv\Scripts\python.exe -m pytest -q                  # everything
```

Linux / macOS:

```bash
.venv/bin/python scripts/acceptance.py             # milestone-by-milestone report
.venv/bin/python scripts/acceptance.py M5 M6       # just these two
.venv/bin/python -m pytest -q                      # everything
.venv/bin/python -m pytest tests/golden -q         # the solvers' truth tests
.venv/bin/python -m pytest tests/latency -q -s     # prints every measurement
```

`scripts/acceptance.py` finds the interpreter under either venv layout
(`Scripts\python.exe` or `bin/python`) and falls back to the one running it, so
the same invocation works from an activated venv on either platform. Its output
is pure ASCII and it forces stdout to UTF-8 before printing, because a
`UnicodeEncodeError` from a cp1252 console is not a failure you want in the ten
minutes before a run. Colour is emitted only when stdout is a TTY and ANSI
processing is actually available (`NO_COLOR` disables it).

`tests/test_windows_platform.py` covers the Windows-only branches from Linux by
monkeypatching `os.name` — interpreter discovery, the ACL refusal path, console
reconfiguration, and non-ASCII round-tripping. What it cannot cover is whether
`icacls` really denies a second account, whether `SetConsoleMode` really flips
a legacy conhost, or how `mss`/`pynput` behave on a real dual-monitor desktop.
Those need a Windows machine.

Run `scripts/acceptance.py` in the ten minutes before a timed run. It answers
"is this ready" milestone by milestone, which is the question that matters then,
and the spec's rule applies: do not proceed past a failing one.

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
