"""Where the tool keeps its data.

Anchored to the repository root, not to the current working directory.  On
Windows the app is normally started from a shortcut, a `.ps1` script or the
Start menu, any of which can set a working directory that has nothing to do with
where the code lives — and a CWD-relative ``data/`` silently creates a second,
empty session store next to whatever folder the shortcut happened to point at.
Losing a leaderboard that way costs a run, and the loss is invisible until you
go looking for the high-water mark.

``PUZZLE_COPILOT_DATA`` overrides it, which is what the tests use.
"""

from __future__ import annotations

import os
from pathlib import Path

#: services/core/paths.py -> services/core -> services -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    override = os.environ.get("PUZZLE_COPILOT_DATA")
    return Path(override).expanduser().resolve() if override else REPO_ROOT / "data"


DATA_ROOT = data_root()
SESSION_ROOT = DATA_ROOT / "sessions"
RULES_ROOT = DATA_ROOT / "rules"
