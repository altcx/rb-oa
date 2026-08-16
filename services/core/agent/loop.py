"""The tool loop.

Hand-rolled and OpenAI-compatible on purpose: the Anthropic Agent SDK is
native to a different wire format, and the whole point of routing through
OpenRouter is model choice.  The loop itself is small -- send messages plus
tool schemas, take ``tool_calls`` back, dispatch them, append ``role:"tool"``
results, repeat until the model answers with content and no tool calls.

Streaming is used for every assistant turn so time-to-first-token stays low:
on a timed puzzle the user should be reading the first line of the answer while
the rest is still arriving.

Both caps -- max turns and cost -- stop the loop *cleanly*, with a final
message that says the cap was hit.  A silent stop looks like a crash and costs
the user more clock than the cap saved.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from services.core.agent.tools import ToolRegistry
from services.core.llm.protocol import ChatMessage, LLMResponse, ToolCall

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")

DEFAULT_MAX_TOKENS = 700
DEFAULT_MAX_TURNS = 8


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "token",
        "tool_call_started",
        "tool_call_finished",
        "assistant_message",
        "done",
        "error",
    ]
    text: str = ""
    turn: int = 0
    tool: str = ""
    tool_call_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    response: LLMResponse | None = None
    stop_reason: str = ""
    elapsed_ms: float = 0.0


class AgentSession:
    """One conversation: messages, tools, model, and both caps."""

    def __init__(
        self,
        client: Any,
        model: str,
        registry: ToolRegistry,
        *,
        system_prompt_path: Path | str | None = None,
        system_prompt: str | None = None,
        cost_cap: float | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = 0.2,
        provider: dict[str, Any] | None = None,
        models: list[str] | None = None,
        timeout_s: float | None = 90.0,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.cost_cap = cost_cap
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provider = provider
        self.models = models
        self.timeout_s = timeout_s

        path = Path(system_prompt_path) if system_prompt_path else SYSTEM_PROMPT_PATH
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else path.read_text(encoding="utf-8")
        )
        self.messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
        #: every tool payload this conversation produced -- the provenance set
        #: that ``guard.unsourced_numbers`` checks the final text against.
        self.tool_results: list[dict[str, Any]] = []
        self.user_messages: list[str] = []
        self.turns_used = 0

    # -- accounting ------------------------------------------------------

    @property
    def spent(self) -> float:
        spend = getattr(self.client, "spent", None)
        return float(getattr(spend, "cost", 0.0) or 0.0)

    def _cap_hit(self) -> str | None:
        if self.cost_cap is not None and self.spent >= self.cost_cap:
            return (
                f"Stopping: session cost cap reached "
                f"(${self.spent:.4f} of ${self.cost_cap:.4f}). "
                "Raise the cap in settings to continue."
            )
        return None

    # -- the loop --------------------------------------------------------

    async def run(
        self, user_message: str, *, max_tokens: int | None = None
    ) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        self.messages.append(ChatMessage(role="user", content=user_message))
        self.user_messages.append(user_message)
        tools = self.registry.specs()
        budget = max_tokens if max_tokens is not None else self.max_tokens
        final: LLMResponse | None = None

        for turn in range(1, self.max_turns + 1):
            self.turns_used = turn
            capped = self._cap_hit()
            if capped:
                yield AgentEvent(kind="assistant_message", text=capped, turn=turn)
                yield AgentEvent(
                    kind="done",
                    text=capped,
                    turn=turn,
                    stop_reason="cost_cap",
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
                return

            response: LLMResponse | None = None
            error: str | None = None
            try:
                async for event in self.client.stream(
                    model=self.model,
                    messages=self.messages,
                    tools=tools,
                    provider=self.provider,
                    models=self.models,
                    max_tokens=budget,
                    temperature=self.temperature,
                    timeout_s=self.timeout_s,
                ):
                    if event.kind == "token":
                        yield AgentEvent(kind="token", text=event.text, turn=turn)
                    elif event.kind == "error":
                        error = event.text or "stream error"
                        response = event.response
                    elif event.kind == "done":
                        response = event.response
            except Exception as exc:  # includes CostCapExceeded from the client
                error = f"{type(exc).__name__}: {exc}"

            if error or response is None:
                message = error or "no response from the model"
                yield AgentEvent(kind="error", text=message, turn=turn)
                yield AgentEvent(
                    kind="done",
                    text=message,
                    turn=turn,
                    stop_reason="error",
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
                return

            final = response
            self.messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls or None,
                )
            )

            if not response.tool_calls:
                if response.content:
                    yield AgentEvent(
                        kind="assistant_message", text=response.content, turn=turn
                    )
                yield AgentEvent(
                    kind="done",
                    text=response.content or "",
                    turn=turn,
                    response=response,
                    stop_reason="stop",
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
                return

            for call in response.tool_calls:
                async for event in self._run_tool(call, turn):
                    yield event

        message = (
            f"Stopping: hit the {self.max_turns}-turn limit for this message. "
            "Tell me which single tool to run next and I will run it."
        )
        yield AgentEvent(kind="assistant_message", text=message, turn=self.turns_used)
        yield AgentEvent(
            kind="done",
            text=message,
            turn=self.turns_used,
            response=final,
            stop_reason="max_turns",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _run_tool(self, call: ToolCall, turn: int) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            kind="tool_call_started",
            turn=turn,
            tool=call.name,
            tool_call_id=call.id,
            arguments=call.arguments,
        )
        started = time.perf_counter()
        try:
            result = await self.registry.dispatch(call.name, call.arguments)
        except Exception as exc:  # a tool must never take the loop down
            result = {"ok": False, "tool": call.name, "error": f"{type(exc).__name__}: {exc}"}
        elapsed = (time.perf_counter() - started) * 1000.0
        self.tool_results.append(result)
        self.messages.append(
            ChatMessage(
                role="tool",
                content=json.dumps(result, default=str),
                tool_call_id=call.id,
                name=call.name,
            )
        )
        yield AgentEvent(
            kind="tool_call_finished",
            turn=turn,
            tool=call.name,
            tool_call_id=call.id,
            arguments=call.arguments,
            result=result,
            elapsed_ms=elapsed,
        )

    # -- provenance ------------------------------------------------------

    def unsourced_numbers(self, text: str) -> list[str]:
        """Numbers in ``text`` that this conversation cannot justify."""
        from services.core.agent.guard import unsourced_numbers

        return unsourced_numbers(text, self.tool_results, self.user_messages)
