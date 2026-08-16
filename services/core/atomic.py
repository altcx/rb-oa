"""Durable writes that also work on Windows.

Two differences from POSIX drove this file.

**Encoding.** ``Path.read_text()`` / ``write_text()`` with no ``encoding=``
default to the ANSI code page on Windows, not UTF-8.  A model slug, a machine
label a vision model read off the screen, or a rule quoted from the game's
instructions round-trips wrong — or raises — as soon as it leaves ASCII.  Every
text file this app owns goes through here with an explicit encoding.

**Open handles.** ``os.replace`` onto a path another handle has open succeeds on
POSIX and fails with ``PermissionError`` on Windows; the same is true of
unlinking an open file.  That turns two concurrent thumbnail requests, or a
session save racing a session read, into a hard error on the target platform and
never on the machine the tests run on.  A short bounded retry covers the window,
which is milliseconds wide.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

#: Windows holds a delete/replace lock only for as long as the other handle
#: lives, which here is a single read.  Six attempts over ~0.35s is far more
#: headroom than that needs and still fails fast enough to surface a real bug.
_ATTEMPTS = 6
_BACKOFF_S = 0.02


def _retry_windows_lock(op: Any, *args: Any) -> None:
    delay = _BACKOFF_S
    for attempt in range(_ATTEMPTS):
        try:
            op(*args)
            return
        except PermissionError:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 1.6


def replace(src: Path | str, dst: Path | str) -> None:
    """``os.replace`` that tolerates a reader holding ``dst`` open."""
    _retry_windows_lock(os.replace, str(src), str(dst))


def unlink(path: Path | str, *, missing_ok: bool = True) -> None:
    p = Path(path)
    if missing_ok and not p.exists():
        return
    _retry_windows_lock(p.unlink)


def write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Atomic, UTF-8, and safe against a concurrent reader."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding)
    replace(tmp, p)


def write_json(path: Path | str, payload: Any, *, indent: int | None = 2) -> None:
    write_text(path, json.dumps(payload, indent=indent, default=str, ensure_ascii=False))


def read_text(path: Path | str, *, encoding: str = "utf-8", default: str | None = None) -> str:
    p = Path(path)
    if default is not None and not p.exists():
        return default
    return p.read_text(encoding=encoding)


def append_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding=encoding) as fh:
        fh.write(text)
