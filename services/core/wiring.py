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
            from services.core.capture.store import CaptureStore

            return [c.model_dump(mode="json") for c in CaptureStore().list_captures(session_id)]
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
