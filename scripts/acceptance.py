#!/usr/bin/env python
"""Milestone acceptance report.  Spec section 12.

One command that answers "is this thing actually ready to use", because the
answer matters most in the ten minutes before a timed run, which is exactly when
nobody wants to read a test log.

    .venv\\Scripts\\python.exe scripts\\acceptance.py     # Windows, all milestones
    .venv/bin/python scripts/acceptance.py            # Linux/macOS, all milestones
    .venv/bin/python scripts/acceptance.py M5 M6      # just these

Everything printed here is ASCII, and stdout is forced to UTF-8 before the
first ``print``.  This report is read in the ten minutes before a timed run and
a ``UnicodeEncodeError`` from a cp1252 console at that moment would be the
worst possible failure: the tool would look broken when it is fine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Console: encoding first, then colour
# ---------------------------------------------------------------------------


def enable_utf8_console(*streams: Any) -> bool:
    """Force stdout/stderr to UTF-8 with ``errors="replace"``.

    A legacy Windows console is cp1252 and cannot encode an em dash, an arrow
    or a box character; ``print`` raises ``UnicodeEncodeError`` and takes the
    whole report down.  ``errors="replace"`` is the belt to the UTF-8 braces:
    if the reconfigure is refused (a redirected pipe, an already-detached
    stream, Python built without the method) we still must not crash, so every
    step is guarded and the function reports whether it worked.
    """
    targets = streams or (sys.stdout, sys.stderr)
    ok = True
    for stream in targets:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            ok = False
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Already-closed, detached, or a stream that does not own its
            # encoding.  Not fatal -- ASCII output survives cp1252 anyway.
            ok = False
    return ok


def enable_vt_mode() -> bool:
    """Turn on ANSI escape handling for a legacy ``cmd.exe``.

    Windows Terminal and PowerShell 7 do this for us; an older conhost does
    not, and shows the raw ``ESC[32m`` sequences instead of colour.  Non-nt
    platforms are always fine, so they return True without doing anything.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        ok = True
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            if handle in (0, -1, None):
                ok = False
                continue
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ok = False  # redirected to a file/pipe: no console mode to set
                continue
            if not kernel32.SetConsoleMode(handle, mode.value | enable_vt):
                ok = False
        return ok
    except Exception:
        return False


def use_color(stream: Any = None) -> bool:
    """Colour only when someone can actually see it."""
    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    try:
        if not stream.isatty():
            return False  # redirected to a file or piped into a pager
    except Exception:
        return False
    if os.name == "nt":
        return enable_vt_mode()
    return True


# ---------------------------------------------------------------------------
# Interpreter discovery
# ---------------------------------------------------------------------------


#: Resolved at import, deliberately.  ``Path()`` picks its concrete class from
#: ``os.name``, so building it later -- in a test that has monkeypatched
#: ``os.name`` to "nt" -- would try to instantiate a WindowsPath on Linux.
RUNNING_PYTHON = Path(sys.executable)

_WINDOWS_LAYOUTS = (("Scripts", "python.exe"), ("Scripts", "python"), ("bin", "python.exe"))
_POSIX_LAYOUTS = (("bin", "python"), ("bin", "python3"))


def find_python(root: Path | str | None = None) -> Path:
    """The project interpreter, whichever venv layout this platform uses.

    Windows venvs put it at ``.venv\\Scripts\\python.exe``; POSIX venvs at
    ``.venv/bin/python``.  Both layouts are probed on both platforms -- a venv
    copied across, a MSYS/Cygwin Python, or a WSL checkout mounted on the
    Windows side all produce the "wrong" one -- and the platform's own layout is
    tried first.  If neither is there we run the tests with whatever interpreter
    is running us, which is the right answer when the user invoked us from an
    already-activated venv.
    """
    if root is None:
        base = ROOT
    else:
        # Only wrap when we have to: Path() dispatches on os.name, and re-wrapping
        # an existing Path would change its flavour under a monkeypatched os.name.
        base = root if isinstance(root, Path) else Path(root)
    venv = base / ".venv"
    order = (
        (*_WINDOWS_LAYOUTS, *_POSIX_LAYOUTS)
        if os.name == "nt"
        else (*_POSIX_LAYOUTS, *_WINDOWS_LAYOUTS)
    )
    for parts in order:
        candidate = venv.joinpath(*parts)
        if candidate.exists():
            return candidate
    return RUNNING_PYTHON


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

_ANSI = {
    "GREEN": "\033[32m",
    "RED": "\033[31m",
    "DIM": "\033[2m",
    "BOLD": "\033[1m",
    "OFF": "\033[0m",
}
#: Filled in by :func:`main` once we know whether the console can show colour.
GREEN = RED = DIM = BOLD = OFF = ""


def set_color(enabled: bool) -> None:
    """Bind the colour globals, or blank them so a dumb console stays readable."""
    global GREEN, RED, DIM, BOLD, OFF
    GREEN, RED, DIM, BOLD, OFF = (
        (_ANSI["GREEN"], _ANSI["RED"], _ANSI["DIM"], _ANSI["BOLD"], _ANSI["OFF"])
        if enabled
        else ("", "", "", "", "")
    )


def run(m: Milestone, python: Path | None = None) -> tuple[bool, float, str]:
    interpreter = python or find_python()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            str(interpreter),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--timeout=300",
            *m.args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        # A child pytest inherits the console code page, not our reconfigured
        # stdout, so decode its output as UTF-8 and never let a stray byte in a
        # test name raise here.
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    elapsed = time.perf_counter() - t0
    tail = [ln for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
    summary = tail[-1] if tail else "no output"
    return proc.returncode == 0, elapsed, summary


def main(argv: list[str]) -> int:
    enable_utf8_console()
    set_color(use_color())

    python = find_python()
    wanted = {a.upper() for a in argv} or {m.id for m in MILESTONES}
    todo = [m for m in MILESTONES if m.id in wanted]
    print(f"\n{BOLD}Puzzle Copilot - milestone acceptance{OFF}")
    print(f"{DIM}  interpreter: {python}{OFF}\n")
    failures = 0
    for m in todo:
        print(f"  {m.id}  {m.title:<24} {DIM}running...{OFF}", end="\r", flush=True)
        ok, elapsed, summary = run(m, python)
        mark = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
        print(f"  {m.id}  {m.title:<24} {mark}  {elapsed:5.1f}s  {DIM}{summary}{OFF}")
        print(f"       {DIM}{m.criterion}{OFF}")
        failures += 0 if ok else 1
    print()
    if failures:
        print(f"{RED}{failures} milestone(s) failing - do not proceed past a failing one.{OFF}\n")
    else:
        print(f"{GREEN}All milestones passing.{OFF}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
