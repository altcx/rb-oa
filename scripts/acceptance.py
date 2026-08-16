#!/usr/bin/env python
"""Milestone acceptance report.  Spec section 12.

One command that answers "is this thing actually ready to use", because the
answer matters most in the ten minutes before a timed run, which is exactly when
nobody wants to read a test log.

    .venv/bin/python scripts/acceptance.py            # all milestones
    .venv/bin/python scripts/acceptance.py M5 M6      # just these
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():  # fall back to whatever is running us
    PYTHON = Path(sys.executable)


@dataclass
class Milestone:
    id: str
    title: str
    criterion: str
    args: list[str]


MILESTONES = [
    Milestone(
        "M0",
        "Capture and store",
        "hotkey writes a PNG with correct metadata; DPI-correct at 1x and 2x",
        ["tests/test_capture_store.py"],
    ),
    Milestone(
        "M1",
        "Factory simulator",
        "all ten golden tests pass, state hand-entered",
        ["tests/golden/factory"],
    ),
    Milestone(
        "M2",
        "Builder solver",
        "exact count agrees with brute force; DP and meet-in-middle agree",
        ["tests/golden/builder"],
    ),
    Milestone(
        "M3",
        "Calibration",
        "simulator reproduces a recorded run exactly; every flag pinned",
        ["tests/test_integration.py", "tests/golden/factory/test_sim_properties.py"],
    ),
    Milestone(
        "M4",
        "OpenRouter plumbing",
        "key validation, keyring storage, redaction, structured outputs",
        ["tests/test_keys_redaction.py", "tests/test_client.py"],
    ),
    Milestone(
        "M5",
        "Consensus extraction",
        "high auto-confirm rate and ZERO wrong auto-confirms (the real bar)",
        ["tests/test_jury.py", "tests/test_practice.py", "tests/test_extraction_wiring.py"],
    ),
    Milestone(
        "M6",
        "Latency",
        "capture->state under 4s, first recommendation fast, solver usable at 200ms",
        ["tests/latency"],
    ),
    Milestone(
        "M7",
        "Rule discovery",
        "rule spec + named unknowns; consensus actions under uncertainty",
        ["tests/test_rule_compile.py", "tests/test_resolve.py"],
    ),
    Milestone(
        "M8",
        "Agent",
        "screenshot to recommendation without the user touching JSON",
        ["tests/test_agent_loop.py", "tests/test_server.py", "tests/test_pipeline.py"],
    ),
]

GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def run(m: Milestone) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", "--timeout=300", *m.args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    summary = tail[-1] if tail else "no output"
    return proc.returncode == 0, elapsed, summary


def main(argv: list[str]) -> int:
    wanted = {a.upper() for a in argv} or {m.id for m in MILESTONES}
    todo = [m for m in MILESTONES if m.id in wanted]
    print(f"\n{BOLD}Puzzle Copilot — milestone acceptance{OFF}\n")
    failures = 0
    for m in todo:
        print(f"  {m.id}  {m.title:<24} {DIM}running…{OFF}", end="\r", flush=True)
        ok, elapsed, summary = run(m)
        mark = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
        print(f"  {m.id}  {m.title:<24} {mark}  {elapsed:5.1f}s  {DIM}{summary}{OFF}")
        print(f"       {DIM}{m.criterion}{OFF}")
        failures += 0 if ok else 1
    print()
    if failures:
        print(f"{RED}{failures} milestone(s) failing — do not proceed past a failing one.{OFF}\n")
    else:
        print(f"{GREEN}All milestones passing.{OFF}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
