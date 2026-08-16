"""OpenRouter implementation of :class:`services.core.llm.protocol.LLMClient`.

One file, one provider.  Everything above this line (extraction, rule
compilation, the agent loop) imports only the protocol, so swapping providers
later is a single-file change (spec 9.4).

Design notes that matter:

*   ``response_format`` is passed straight through to OpenRouter.  Whenever it
    is set we *always* pair it with ``provider.require_parameters = true`` so
    OpenRouter only routes to endpoints that actually honour structured
    outputs -- otherwise a silent fallback provider returns prose and the
    extraction jury votes on garbage.
*   Non-streaming schema calls also enable the Response Healing plugin, which
    repairs truncated / malformed JSON server-side and saves us a round trip.
*   A JSON parse failure is *retried once* with an explicit repair message and
    then downgraded to ``LLMResponse(error=...)``.  Never raises: a timed run
    must not die because one model hiccuped.
*   ``stream()`` accumulates tool-call arguments **by index**.  OpenAI-style
    streams emit ``function.arguments`` in fragments spread over many deltas
    and interleaved across parallel tool calls; concatenating them in arrival
    order is the classic bug.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, AsyncIterator, Iterable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from services.core.llm.protocol import (
    ChatMessage,
    LLMResponse,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: Statuses worth retrying: rate limits and transient upstream failures.
RETRY_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Server-side JSON repair for non-streaming structured-output calls.
RESPONSE_HEALING_PLUGIN: dict[str, Any] = {"id": "response-healing"}

_REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON for the requested schema. "
    "Reply again with ONLY the JSON object. No prose, no code fences, no "
    "trailing commentary. Emit null for anything you cannot determine."
)


class CostCapExceeded(RuntimeError):
    """Raised once cumulative spend passes the client's configured cap."""

    def __init__(self, spent: float, cap: float, response: LLMResponse | None = None) -> None:
        super().__init__(f"cost cap exceeded: spent ${spent:.4f} against cap ${cap:.4f}")
        self.spent = spent
        self.cap = cap
        #: the response that tipped it over, when there was one
        self.response = response


class Spend(BaseModel):
    """Cumulative accounting so a session can enforce a budget."""

    model_config = ConfigDict(extra="forbid")

    cost: float = 0.0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, float] = Field(default_factory=dict)

    def add(self, model: str, usage: Usage) -> None:
        self.turns += 1
        self.cost += usage.cost or 0.0
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        if model:
            self.by_model[model] = self.by_model.get(model, 0.0) + (usage.cost or 0.0)


class OpenRouterClient:
    """OpenAI-compatible OpenRouter client on top of ``httpx.AsyncClient``."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        referer: str | None = None,
        title: str | None = None,
        cost_cap: float | None = None,
        preferred_max_latency: float | None = None,
        default_timeout_s: float = 60.0,
        max_attempts: int = 3,
        backoff_base_s: float = 0.5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.title = title
        self.cost_cap = cost_cap
        #: seconds; emitted to OpenRouter provider preferences as max_latency_ms
        self.preferred_max_latency = preferred_max_latency
        self.default_timeout_s = default_timeout_s
        self.max_attempts = max(1, max_attempts)
        self.backoff_base_s = backoff_base_s
        self.spent = Spend()
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=default_timeout_s)

    # -- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- headers / body ----------------------------------------------------

    def headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            h["HTTP-Referer"] = self.referer
        if self.title:
            h["X-Title"] = self.title
        return h

    def _provider_prefs(
        self, provider: dict[str, Any] | None, *, structured: bool
    ) -> dict[str, Any] | None:
        prefs: dict[str, Any] = dict(provider or {})
        if structured:
            # Non-negotiable: only route where the schema is honoured.
            prefs["require_parameters"] = True
        if self.preferred_max_latency is not None:
            prefs.setdefault("max_latency_ms", int(self.preferred_max_latency * 1000))
            prefs.setdefault("sort", "latency")
        return prefs or None

    def _build_body(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage],
        response_format: dict[str, Any] | None = None,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
        models: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        plugins: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.to_wire() for m in messages],
            # ask OpenRouter to attach real dollar cost so `spent` is truthful
            "usage": {"include": True},
        }
        if models:
            # single-provider outage must not stall a timed run
            body["models"] = list(models)
        if response_format:
            body["response_format"] = response_format
        prefs = self._provider_prefs(provider, structured=bool(response_format))
        if prefs:
            body["provider"] = prefs
        if tools:
            body["tools"] = [t.to_wire() for t in tools]
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        effective_plugins = plugins
        if effective_plugins is None and response_format and not stream:
            effective_plugins = [RESPONSE_HEALING_PLUGIN]
        if effective_plugins:
            body["plugins"] = effective_plugins
        if stream:
            body["stream"] = True
        return body

    # -- HTTP with retries -------------------------------------------------

    def _sleep_for(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 10.0)
            except (TypeError, ValueError):
                pass
        base = self.backoff_base_s * (2**attempt)
        return base * (0.5 + random.random())  # jitter

    async def _request(
        self, path: str, body: dict[str, Any] | None, timeout_s: float | None, method: str = "POST"
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                resp = await self._http.request(
                    method, url, headers=self.headers(), json=body, timeout=timeout
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt + 1 >= self.max_attempts:
                    raise
                await asyncio.sleep(self._sleep_for(attempt))
                continue
            if resp.status_code in RETRY_STATUS and attempt + 1 < self.max_attempts:
                await asyncio.sleep(self._sleep_for(attempt, resp.headers.get("Retry-After")))
                continue
            return resp
        if last_exc:
            raise last_exc
        raise httpx.HTTPError("request failed with no response")  # pragma: no cover

    async def get_json(self, path: str, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Plain GET helper -- used by settings/keys.py and settings/models.py."""
        resp = await self._request(path, None, timeout_s, method="GET")
        resp.raise_for_status()
        return resp.json()

    # -- accounting --------------------------------------------------------

    def _check_cap(self, response: LLMResponse | None = None) -> None:
        if self.cost_cap is not None and self.spent.cost > self.cost_cap:
            raise CostCapExceeded(self.spent.cost, self.cost_cap, response)

    def _account(self, resp: LLMResponse) -> None:
        self.spent.add(resp.model, resp.usage)
        self._check_cap(resp)

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _usage_from(raw: dict[str, Any]) -> Usage:
        u = raw.get("usage") or {}
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            cost=float(u.get("cost") or (u.get("total_cost") or 0.0)),
        )

    @classmethod
    def _to_response(cls, raw: dict[str, Any], latency_ms: float) -> LLMResponse:
        choices = raw.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=(tc.get("function") or {}).get("name") or "",
                arguments=_loads_args((tc.get("function") or {}).get("arguments")),
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=raw.get("model") or "",
            provider=raw.get("provider") or "",
            usage=cls._usage_from(raw),
            finish_reason=(choices[0].get("finish_reason") if choices else "") or "",
            raw=raw,
            latency_ms=latency_ms,
        )

    # -- the interface -----------------------------------------------------

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
    ) -> LLMResponse:
        self._check_cap()
        msgs = list(messages)
        started = time.perf_counter()

        async def once(m: list[ChatMessage]) -> LLMResponse:
            body = self._build_body(
                model=model,
                messages=m,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                provider=provider,
                models=models,
                max_tokens=max_tokens,
                temperature=temperature,
                plugins=plugins,
            )
            resp = await self._request("/chat/completions", body, timeout_s)
            elapsed = (time.perf_counter() - started) * 1000.0
            if resp.status_code >= 400:
                return LLMResponse(
                    error=f"http {resp.status_code}: {resp.text[:400]}",
                    latency_ms=elapsed,
                    model=model,
                )
            try:
                raw = resp.json()
            except ValueError as exc:
                return LLMResponse(
                    error=f"non-json response: {exc}", latency_ms=elapsed, model=model
                )
            if isinstance(raw, dict) and raw.get("error") and not raw.get("choices"):
                err = raw["error"]
                msg = err.get("message") if isinstance(err, dict) else str(err)
                return LLMResponse(error=f"provider error: {msg}", latency_ms=elapsed, model=model)
            return self._to_response(raw, elapsed)

        try:
            out = await once(msgs)
        except CostCapExceeded:
            raise
        except Exception as exc:  # network death is a result, not a crash
            return LLMResponse(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model=model,
            )

        if out.error:
            return out
        self._account(out)

        if response_format is None:
            return out

        parsed, perr = _parse_json(out.content)
        if perr is None:
            out.parsed = parsed
            return out

        # ---- one repair attempt, then give up with a structured error ----
        repair = msgs + [
            ChatMessage(role="assistant", content=(out.content or "")[:4000]),
            ChatMessage(role="user", content=_REPAIR_INSTRUCTION),
        ]
        try:
            retry = await once(repair)
        except CostCapExceeded:
            raise
        except Exception as exc:
            out.error = f"json parse failed ({perr}); repair attempt raised {exc}"
            return out
        if retry.error:
            out.error = f"json parse failed ({perr}); repair attempt: {retry.error}"
            return out
        self._account(retry)
        parsed2, perr2 = _parse_json(retry.content)
        if perr2 is None:
            retry.parsed = parsed2
            return retry
        retry.error = f"json parse failed twice: {perr} / {perr2}"
        retry.parsed = None
        return retry

    async def stream(
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
    ) -> AsyncIterator[StreamEvent]:
        """SSE stream.  Yields ``token`` events, then a single ``done``."""
        self._check_cap()
        body = self._build_body(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            provider=provider,
            models=models,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        started = time.perf_counter()
        content_parts: list[str] = []
        # index -> {"id", "name", "arguments"}; ORDER AND INDEX BOTH MATTER
        acc: dict[int, dict[str, str]] = {}
        finish_reason = ""
        resolved_model = model
        provider_name = ""
        usage = Usage()
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s

        try:
            async with self._http.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self.headers(),
                json=body,
                timeout=timeout,
            ) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    yield StreamEvent(
                        kind="error",
                        text=f"http {resp.status_code}: {text[:400]}",
                        response=LLMResponse(
                            error=f"http {resp.status_code}: {text[:400]}",
                            model=model,
                            latency_ms=(time.perf_counter() - started) * 1000.0,
                        ),
                    )
                    return
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    resolved_model = chunk.get("model") or resolved_model
                    provider_name = chunk.get("provider") or provider_name
                    if chunk.get("usage"):
                        usage = self._usage_from(chunk)
                    for choice in chunk.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or choice.get("message") or {}
                        piece = delta.get("content")
                        if isinstance(piece, list):
                            piece = "".join(
                                p.get("text", "") for p in piece if isinstance(p, dict)
                            )
                        if piece:
                            content_parts.append(piece)
                            yield StreamEvent(kind="token", text=piece)
                        for i, tc in enumerate(delta.get("tool_calls") or []):
                            idx = tc.get("index")
                            idx = i if idx is None else int(idx)
                            slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            name = fn.get("name")
                            if name:
                                # providers send the name once, but a few
                                # fragment it too; append only new text
                                if not slot["name"]:
                                    slot["name"] = name
                                elif slot["name"] != name:
                                    slot["name"] += name
                            args = fn.get("arguments")
                            if args:
                                slot["arguments"] += args
        except CostCapExceeded:
            raise
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            yield StreamEvent(
                kind="error",
                text=err,
                response=LLMResponse(
                    error=err,
                    model=model,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )
            return

        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{idx}",
                name=slot["name"],
                arguments=_loads_args(slot["arguments"]),
            )
            for idx, slot in sorted(acc.items())
            if slot["name"]
        ]
        final = LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            model=resolved_model,
            provider=provider_name,
            usage=usage,
            finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        if tool_calls:
            yield StreamEvent(kind="tool_call", tool_calls=tool_calls)
        self._account(final)
        yield StreamEvent(kind="done", response=final)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _loads_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        out = json.loads(raw)
    except ValueError:
        return {"__raw__": raw}
    return out if isinstance(out, dict) else {"__value__": out}


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
    return t.strip()


def _parse_json(content: str | None) -> tuple[Any, str | None]:
    """Best-effort JSON parse.  Returns ``(value, error_or_None)``."""
    if content is None or not content.strip():
        return None, "empty content"
    text = _strip_fences(content)
    try:
        return json.loads(text), None
    except ValueError as exc:
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(text[first : last + 1]), None
            except ValueError:
                pass
        return None, str(exc)
