"""Session state: captures, verified state, rules, leaderboard, chat history.

This is the object the agent's tools read and write, and the object the
WebSocket serializes to the UI.  Persisted under ``data/sessions/<id>/`` so a
crashed run does not lose a high-water mark.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DATA_ROOT = Path(os.environ.get("PUZZLE_COPILOT_DATA", "data"))
SESSION_ROOT = DATA_ROOT / "sessions"

PuzzleType = Literal["factory", "builder"]


class LeaderboardEntry(BaseModel):
    """The game scores on the MAXIMUM over every tested factory, not the most
    recent one.  Losing a high-water mark is therefore a data-loss bug."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = ""
    score: float = 0.0
    tested_at: float = Field(default_factory=time.time)
    config_summary: str = ""
    config: dict[str, Any] | None = None
    source: str = "manual"  # "manual" | "optimizer" | "game"
    is_high_water: bool = False


class FieldVerdictView(BaseModel):
    """What the state inspector needs to sort by disagreement (spec 11)."""

    model_config = ConfigDict(extra="ignore")

    path: str
    value: Any = None
    status: Literal["unanimous", "majority", "split", "manual"] = "unanimous"
    votes: dict[str, Any] = Field(default_factory=dict)
    alternatives: list[Any] = Field(default_factory=list)
    confidence: float = 1.0
    crop: dict[str, Any] | None = None
    #: Not derivable from ``status``: a unanimous null on a field the solver
    #: needs is demoted, so it reads "unanimous" while still requiring a human.
    auto_confirmed: bool = True
    #: Why this field looks the way it does — the difference between "one cheap
    #: read, confirm before solving" and "majority value, one keystroke to
    #: accept". This is what tells the user how far to trust a pre-filled value.
    reason: str = ""


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    puzzle_type: PuzzleType = "factory"
    created_at: float = Field(default_factory=time.time)

    #: The confirmed, solver-ready state (FactoryState or BuilderPuzzle dump).
    state: dict[str, Any] | None = None
    #: The last raw extraction merge, before human confirmation.
    pending_state: dict[str, Any] | None = None
    verdicts: list[FieldVerdictView] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    confirmed_at: float | None = None

    config: dict[str, Any] | None = None
    rules: dict[str, Any] = Field(default_factory=dict)
    rule_unknowns: list[dict[str, Any]] = Field(default_factory=list)
    calibrated: bool = False
    calibration: dict[str, Any] | None = None

    leaderboard: list[LeaderboardEntry] = Field(default_factory=list)
    observed_money_by_hour: list[float] | None = None
    optimizer_best: dict[str, Any] | None = None
    bound: dict[str, Any] | None = None

    chat: list[dict[str, Any]] = Field(default_factory=list)
    #: Every tool result this session has produced, used by the numeric
    #: provenance guard (spec 13: the agent must not narrate a number it made up).
    tool_results: list[dict[str, Any]] = Field(default_factory=list)

    latency: dict[str, float] = Field(default_factory=dict)
    cost_spent: float = 0.0

    # -- leaderboard ----------------------------------------------------

    def record_score(self, entry: LeaderboardEntry) -> LeaderboardEntry:
        self.leaderboard.append(entry)
        best = max(self.leaderboard, key=lambda e: e.score)
        for e in self.leaderboard:
            e.is_high_water = e.id == best.id
        return entry

    def high_water(self) -> LeaderboardEntry | None:
        return max(self.leaderboard, key=lambda e: e.score, default=None)

    # -- persistence ----------------------------------------------------

    def dir(self) -> Path:
        return SESSION_ROOT / self.id

    def save(self) -> None:
        d = self.dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "session.json.tmp"
        tmp.write_text(json.dumps(self.model_dump(mode="json"), indent=2))
        os.replace(tmp, d / "session.json")

    @classmethod
    def load(cls, session_id: str) -> "SessionState | None":
        p = SESSION_ROOT / session_id / "session.json"
        if not p.exists():
            return None
        try:
            return cls.model_validate_json(p.read_text())
        except Exception:
            return None


class SessionManager:
    """In-memory registry with write-through persistence."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, puzzle_type: PuzzleType = "factory") -> SessionState:
        s = SessionState(puzzle_type=puzzle_type)
        self._sessions[s.id] = s
        s.save()
        return s

    def get(self, session_id: str) -> SessionState | None:
        s = self._sessions.get(session_id)
        if s is None:
            s = SessionState.load(session_id)
            if s is not None:
                self._sessions[s.id] = s
        return s

    def require(self, session_id: str) -> SessionState:
        s = self.get(session_id)
        if s is None:
            raise KeyError(session_id)
        return s

    def list(self) -> list[SessionState]:
        known = dict(self._sessions)
        if SESSION_ROOT.exists():
            for d in SESSION_ROOT.iterdir():
                if d.is_dir() and d.name not in known:
                    s = SessionState.load(d.name)
                    if s:
                        known[s.id] = s
        return sorted(known.values(), key=lambda s: s.created_at, reverse=True)

    def save(self, session: SessionState) -> None:
        self._sessions[session.id] = session
        session.save()


sessions = SessionManager()
