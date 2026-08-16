"""OpenRouterClient: request shape, retries, JSON repair, streaming, cost cap.

No network: every call is intercepted by respx.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from services.core.llm.client import CostCapExceeded, OpenRouterClient
from services.core.llm.protocol import ChatMessage, image_part, text_part

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "Thing",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"a": {"type": ["integer", "null"]}},
            "required": ["a"],
            "additionalProperties": False,
        },
    },
}


def _client(**kw) -> OpenRouterClient:
    kw.setdefault("backoff_base_s", 0.001)
    return OpenRouterClient("sk-or-v1-testkeytestkeytestkey", **kw)


def completion(content: str, *, cost: float = 0.002, tool_calls=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "gen-1",
        "model": "vendor/model-x",
        "provider": "VendorInc",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
    }


def sse(chunks: list[dict]) -> str:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# structured outputs
# ---------------------------------------------------------------------------


@respx.mock
async def test_structured_output_request_shape_and_parse():
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion('{"a": 7}'))
    )
    client = _client(referer="http://localhost", title="Puzzle Copilot")
    resp = await client.complete(
        model="vendor/model-x",
        messages=[
            ChatMessage(
                role="user",
                parts=[text_part("read this"), image_part("data:image/png;base64,AAAA")],
            )
        ],
        response_format=SCHEMA,
        models=["vendor/model-x", "other/model-y"],
        timeout_s=5,
    )
    await client.aclose()

    assert resp.error is None
    assert resp.parsed == {"a": 7}
    assert resp.latency_ms > 0
    assert resp.model == "vendor/model-x"

    request = route.calls.last.request
    assert request.headers["Authorization"].startswith("Bearer sk-or-v1-")
    assert request.headers["HTTP-Referer"] == "http://localhost"
    assert request.headers["X-Title"] == "Puzzle Copilot"

    body = json.loads(request.content)
    assert body["response_format"] == SCHEMA
    # schema calls MUST pin require_parameters so routing can't silently drop it
    assert body["provider"]["require_parameters"] is True
    # and enable server-side JSON repair on non-streaming schema calls
    assert {"id": "response-healing"} in body["plugins"]
    assert body["models"] == ["vendor/model-x", "other/model-y"]
    # image goes as an OpenAI-style content part with a data URL
    parts = body["messages"][0]["content"]
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


@respx.mock
async def test_bad_json_is_repaired_on_one_retry():
    respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("here you go: not json at all")),
            httpx.Response(200, json=completion('{"a": 3}')),
        ]
    )
    client = _client()
    resp = await client.complete(
        model="m",
        messages=[ChatMessage(role="user", content="go")],
        response_format=SCHEMA,
    )
    await client.aclose()
    assert resp.parsed == {"a": 3}
    assert resp.error is None
    assert client.spent.turns == 2


@respx.mock
async def test_bad_json_twice_returns_error_not_raise():
    respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("nope")),
            httpx.Response(200, json=completion("still nope")),
        ]
    )
    client = _client()
    resp = await client.complete(
        model="m",
        messages=[ChatMessage(role="user", content="go")],
        response_format=SCHEMA,
    )
    await client.aclose()
    assert resp.parsed is None
    assert resp.error and "json parse failed twice" in resp.error


@respx.mock
async def test_fenced_json_still_parses():
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion('```json\n{"a": 1}\n```'))
    )
    client = _client()
    resp = await client.complete(
        model="m", messages=[ChatMessage(role="user", content="x")], response_format=SCHEMA
    )
    await client.aclose()
    assert resp.parsed == {"a": 1}


# ---------------------------------------------------------------------------
# retries and failure handling
# ---------------------------------------------------------------------------


@respx.mock
async def test_retries_429_then_succeeds():
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": "slow down"}),
            httpx.Response(503, text="upstream down"),
            httpx.Response(200, json=completion("ok")),
        ]
    )
    client = _client()
    resp = await client.complete(model="m", messages=[ChatMessage(role="user", content="x")])
    await client.aclose()
    assert resp.content == "ok"
    assert route.call_count == 3


@respx.mock
async def test_gives_up_after_max_attempts_with_error_response():
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="boom"))
    client = _client()
    resp = await client.complete(model="m", messages=[ChatMessage(role="user", content="x")])
    await client.aclose()
    assert resp.error and "http 500" in resp.error
    assert resp.latency_ms > 0


@respx.mock
async def test_transport_error_becomes_error_response():
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("no route"))
    client = _client()
    resp = await client.complete(model="m", messages=[ChatMessage(role="user", content="x")])
    await client.aclose()
    assert resp.error and "ConnectError" in resp.error


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


@respx.mock
async def test_stream_accumulates_tool_call_fragments_by_index():
    """The classic bug: arguments arrive in fragments, interleaved by index."""
    chunks = [
        {"model": "vendor/model-x", "choices": [{"delta": {"content": "Checking"}}]},
        {"choices": [{"delta": {"content": " state"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "simulate_factory", "arguments": '{"sec'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "get_state", "arguments": '{"session'},
                            }
                        ]
                    }
                }
            ]
        },
        # fragments come back out of order relative to their calls
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 1, "function": {"arguments": '_id": "s1"}'}}]}}
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'onds": 5}'}}]}}
            ]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "cost": 0.004},
        },
    ]
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, text=sse(chunks), headers={"Content-Type": "text/event-stream"}
        )
    )
    client = _client()
    tokens, done = [], None
    async for event in client.stream(
        model="vendor/model-x", messages=[ChatMessage(role="user", content="hi")]
    ):
        if event.kind == "token":
            tokens.append(event.text)
        elif event.kind == "done":
            done = event.response
    await client.aclose()

    assert tokens == ["Checking", " state"]
    assert done is not None
    assert done.content == "Checking state"
    assert [tc.name for tc in done.tool_calls] == ["simulate_factory", "get_state"]
    assert done.tool_calls[0].arguments == {"seconds": 5}
    assert done.tool_calls[1].arguments == {"session_id": "s1"}
    assert done.finish_reason == "tool_calls"
    assert done.latency_ms > 0
    assert client.spent.cost == pytest.approx(0.004)


@respx.mock
async def test_stream_http_error_yields_error_event():
    respx.post(CHAT_URL).mock(return_value=httpx.Response(502, text="bad gateway"))
    client = _client(max_attempts=1)
    kinds = []
    async for event in client.stream(
        model="m", messages=[ChatMessage(role="user", content="hi")]
    ):
        kinds.append(event.kind)
    await client.aclose()
    assert kinds == ["error"]


# ---------------------------------------------------------------------------
# cost cap
# ---------------------------------------------------------------------------


@respx.mock
async def test_cost_cap_raises_once_passed():
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion("ok", cost=0.03)))
    client = _client(cost_cap=0.05)
    await client.complete(model="m", messages=[ChatMessage(role="user", content="1")])
    assert client.spent.cost == pytest.approx(0.03)
    with pytest.raises(CostCapExceeded):
        await client.complete(model="m", messages=[ChatMessage(role="user", content="2")])
    with pytest.raises(CostCapExceeded):
        await client.complete(model="m", messages=[ChatMessage(role="user", content="3")])
    await client.aclose()
    assert client.spent.turns == 2
    assert client.spent.by_model["vendor/model-x"] == pytest.approx(0.06)


@respx.mock
async def test_preferred_max_latency_reaches_provider_prefs():
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion("ok")))
    client = _client(preferred_max_latency=2.5)
    await client.complete(model="m", messages=[ChatMessage(role="user", content="x")])
    await client.aclose()
    body = json.loads(route.calls.last.request.content)
    assert body["provider"]["max_latency_ms"] == 2500


@respx.mock
async def test_get_json_helper():
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    client = _client()
    assert await client.get_json("/models") == {"data": []}
    await client.aclose()
