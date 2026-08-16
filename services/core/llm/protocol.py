"""The thin provider interface.  Spec 9.4: keep the provider call behind an
interface so swapping back to a native SDK later is a single file.

Both the extraction pipeline and the agent loop import *only* from here.
``llm/client.py`` is the one implementation (OpenRouter, OpenAI-compatible).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str | None = None
    #: OpenAI-style multimodal content parts; when set it wins over ``content``.
    parts: list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        msg["content"] = self.parts if self.parts is not None else (self.content or "")
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)
    #: wall-clock for the call, filled by the client
    latency_ms: float = 0.0
    #: parsed JSON when a json_schema response_format was requested
    parsed: Any = None
    error: str | None = None


class StreamEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["token", "tool_call", "done", "error"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    response: LLMResponse | None = None


#: Provider routing presets (spec 9.2).
EXTRACTION_PROVIDER: dict[str, Any] = {
    "sort": "latency",
    "require_parameters": True,
    "data_collection": "deny",
}
REASONING_PROVIDER: dict[str, Any] = {"sort": "latency", "data_collection": "deny"}


@runtime_checkable
class LLMClient(Protocol):
    """What extraction and the agent loop are allowed to depend on."""

    async def complete(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        response_format: dict[str, Any] | None = None,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
        models: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
        plugins: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    def stream(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
        models: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[StreamEvent]: ...


def image_part(data_url: str, detail: str = "high") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url, "detail": detail}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}
