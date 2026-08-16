"""Glue between the HTTP/WS layer and the modules it drives.

Kept out of ``main.py`` so the server file stays a routing table, and so the
adapters are unit-testable without starting an app.
"""

from __future__ import annotations

import time
from typing import Any

from services.core.session import SessionState, sessions


# ---------------------------------------------------------------------------
# SessionStore: what services/core/agent/tools.py needs from the server
# ---------------------------------------------------------------------------


class ServerSessionStore:
    """Implements the ``SessionStore`` protocol against real sessions on disk."""

    def list_captures(self, session_id: str) -> list[dict[str, Any]]:
        try:
            from services.core.capture.store import default_store

            return [c.model_dump(mode="json") for c in default_store().list_captures(session_id)]
        except Exception:
            return []

    def get_capture(self, session_id: str, capture_id: str) -> dict[str, Any] | None:
        for c in self.list_captures(session_id):
            if c.get("id") == capture_id:
                return c
        return None

    def get_state(self, session_id: str) -> dict[str, Any] | None:
        s = sessions.get(session_id)
        return None if s is None else (s.state or s.pending_state)

    def set_state(self, session_id: str, state: dict[str, Any]) -> None:
        s = sessions.get(session_id)
        if s is None:
            return
        s.state = state
        s.confirmed_at = s.confirmed_at or time.time()
        sessions.save(s)

    def puzzle_type(self, session_id: str) -> str | None:
        s = sessions.get(session_id)
        return None if s is None else s.puzzle_type

    def get_rules(self, session_id: str, puzzle: str) -> dict[str, Any] | None:
        s = sessions.get(session_id)
        return None if s is None else (s.rules or None)

    def set_rules(self, session_id: str, puzzle: str, rules: dict[str, Any]) -> None:
        s = sessions.get(session_id)
        if s is None:
            return
        s.rules = rules
        sessions.save(s)

    def get_config(self, session_id: str) -> dict[str, Any] | None:
        s = sessions.get(session_id)
        return None if s is None else s.config

    def set_config(self, session_id: str, config: dict[str, Any]) -> None:
        s = sessions.get(session_id)
        if s is None:
            return
        s.config = config
        sessions.save(s)


store = ServerSessionStore()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentUnavailable(RuntimeError):
    pass


def _client(cost_cap: float | None = 2.0):
    from services.core.llm.client import OpenRouterClient
    from services.core.settings.keys import load_key

    key = load_key()
    if not key:
        raise AgentUnavailable("no OpenRouter key stored — paste one in settings")
    return OpenRouterClient(key, title="Puzzle Copilot", referer="http://localhost:8765", cost_cap=cost_cap)


def roles():
    from services.core.settings.models import load_settings

    return load_settings().roles


async def build_agent_session(session: SessionState, *, cost_cap: float = 2.0):
    """Assemble the strategist agent for one session.

    Per-session cost cap and max turn count are set here (spec 9.4).
    """
    from services.core.agent.loop import AgentSession
    from services.core.agent.tools import ToolRegistry
    from services.core.llm.protocol import REASONING_PROVIDER

    assignment = roles()
    model = assignment.strategist
    if not model:
        raise AgentUnavailable("no strategist model assigned — pick one in settings")
    registry = ToolRegistry(store, session.id)
    return AgentSession(
        _client(cost_cap),
        model,
        registry,
        cost_cap=cost_cap,
        max_turns=8,
        max_tokens=700,
        provider=REASONING_PROVIDER,
    )


def event_to_ws(event: Any) -> dict[str, Any]:
    """Map an ``AgentEvent`` onto the WebSocket protocol the UI speaks."""
    kind = getattr(event, "kind", "")
    if kind == "token":
        return {"type": "agent_token", "text": event.text}
    if kind == "tool_call_started":
        return {
            "type": "agent_tool_call",
            "id": event.tool_call_id,
            "name": event.tool,
            "input": event.arguments,
        }
    if kind == "tool_call_finished":
        return {
            "type": "agent_tool_result",
            "id": event.tool_call_id,
            "name": event.tool,
            "output": event.result,
            "ms": event.elapsed_ms,
        }
    if kind == "assistant_message":
        return {"type": "agent_message", "text": event.text}
    if kind == "done":
        return {"type": "agent_done", "message": event.text, "stop_reason": event.stop_reason}
    return {"type": "warning", "text": event.text or "agent error"}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_roles():
    """Turn the stored role assignment into the pipeline's ``ExtractRoles``."""
    from services.core.extract.pipeline import ExtractRoles

    a = roles()
    if not a.extractor_jury:
        raise AgentUnavailable("no extractor jury assigned — pick three models in settings")
    return ExtractRoles(jury=list(a.extractor_jury), tie_breaker=a.tie_breaker)


def _load_captures(session_id: str, capture_ids: list[str]):
    from services.core.capture.store import default_store

    cs = default_store()
    return [cs.load(cid, session_id) for cid in capture_ids]


async def run_extraction(
    session_id: str,
    capture_ids: list[str],
    puzzle_type: str | None = None,
    *,
    use_previous: bool = True,
):
    """Extract and fold the result into the session.

    ``use_previous`` turns on delta extraction once a state has been confirmed:
    the previous state plus the new crop, asking only for fields that differ.
    """
    from services.core.extract import pipeline

    s = sessions.require(session_id)
    ptype = puzzle_type or s.puzzle_type
    client = _client()
    try:
        result = await pipeline.extract(
            client,
            extract_roles(),
            _load_captures(session_id, capture_ids),
            ptype,
            previous_state=(s.state if (use_previous and s.confirmed_at) else None),
        )
    finally:
        close = getattr(client, "aclose", None)
        if close:
            await close()
    apply_extraction(s, result)
    return result


def start_speculative_extraction(session_id: str, capture_id: str):
    """Fire extraction the moment a capture lands, before the user asks.

    Returns the in-flight task, or ``None`` when extraction is not configured —
    a missing key must never break the capture path.
    """
    from services.core.extract import pipeline

    s = sessions.require(session_id)
    try:
        client = _client()
        return pipeline.speculate(
            client,
            extract_roles(),
            _load_captures(session_id, [capture_id]),
            s.puzzle_type,
            previous_state=(s.state if s.confirmed_at else None),
        )
    except Exception:
        return None


def apply_extraction(s: SessionState, result: Any) -> None:
    """Fold an ``ExtractionResult`` into the session the UI reads.

    Disputed fields come first, because the inspector sorts by disagreement
    rather than document order.
    """
    from services.core.session import FieldVerdictView

    solved = result.factory_state or result.builder_puzzle
    s.pending_state = solved.model_dump(mode="json") if solved is not None else result.extraction
    s.unresolved = result.needs_review()
    verdicts: list[FieldVerdictView] = []
    for p in result.provenance:
        # Distinct votes only: two models returning the same wrong value is one
        # alternative to consider, not two.
        alts: dict[str, Any] = {}
        for v in p.votes.values():
            if v != p.value:
                alts.setdefault(repr(v), v)
        verdicts.append(
            FieldVerdictView(
                path=p.path,
                value=p.value,
                status=p.status if p.status in {"unanimous", "majority", "split"} else "split",
                votes=p.votes,
                alternatives=list(alts.values()),
                confidence=p.confidence,
                auto_confirmed=p.auto_confirmed,
                reason=p.reason,
                crop=(
                    {
                        "capture_id": p.capture_id,
                        "box": p.box,
                        "tile": p.tile,
                        "crop_id": p.crop_id,
                    }
                    if p.box
                    else None
                ),
            )
        )
    # Anything still needing a human comes first, whatever its status: a
    # unanimous null on a field the solver needs reads as "unanimous" but is
    # exactly the field the user must fill, and sorting on status alone buries
    # it under every settled field on the board.
    order = {"split": 0, "majority": 1, "unanimous": 2}
    verdicts.sort(key=lambda v: (v.auto_confirmed, order.get(v.status, 0), v.path))
    s.verdicts = verdicts
    s.latency = {**s.latency, **result.stage_ms, "extraction_total": result.elapsed_ms}
    sessions.save(s)


# ---------------------------------------------------------------------------
# Settings adapters
# ---------------------------------------------------------------------------


async def catalog_and_roles() -> dict[str, Any]:
    """Catalog + role assignment + measured per-role latency, for the settings UI.

    Falls back to the stored assignment when there is no key yet, so the pane
    still renders instead of erroring.
    """
    from services.core.settings.models import (
        auto_assign,
        latency_report,
        load_settings,
        save_assignment,
    )

    settings = load_settings()
    catalog: list[Any] = []
    error: str | None = None
    try:
        from services.core.settings.models import fetch_catalog

        client = _client()
        try:
            catalog = await fetch_catalog(client)
        finally:
            close = getattr(client, "aclose", None)
            if close:
                await close()
    except Exception as exc:
        error = str(exc)

    assignment = settings.roles
    if catalog and not assignment.strategist:
        assignment = auto_assign(catalog)
        save_assignment(assignment)

    return {
        "catalog": [
            {
                "slug": m.id,
                "family": m.family,
                "vision": m.vision,
                "structured": m.structured_outputs,
                "tools": m.tools,
                "price_in": m.prompt_price,
                "price_out": m.completion_price,
            }
            for m in catalog
        ],
        "roles": assignment.model_dump(mode="json"),
        "latency": {
            role: [s.model_dump(mode="json") for s in stats]
            for role, stats in latency_report().items()
        },
        "error": error,
    }


def save_roles(raw: dict[str, Any]) -> None:
    from services.core.settings.models import RoleAssignment, save_assignment

    save_assignment(RoleAssignment.model_validate(raw))
