"""Three-model vision jury (spec 6.2) -- the single biggest latency win.

The same crop goes to three models from three different *families*,
concurrently.  Family diversity is the whole trick: two checkpoints of the same
base model correlate their errors, so they agree on the wrong answer and the
agreement signal tells you nothing.  Three families disagree independently,
which turns agreement into evidence.

What that buys, per field path (not per document):

* three agree      -> auto-confirmed, never shown to the user.  A unanimous
                      *null* counts: three readers agreeing a field is not on
                      the screen is agreement, and the field is absent, not
                      unread.  (The pipeline demotes it if the assembler turns
                      out to need it -- see ``pipeline._needed_paths``.)
* two agree        -> pre-filled with the majority value and flagged: one
                      keystroke to accept.  A *majority* null stays flagged:
                      two readers seeing nothing against one reading a number is
                      a real disagreement about the screen.
* three differ     -> shown empty, with the crop id and field path recorded so
                      the review UI can zoom the source crop to that field.

Everything runs under one hard timeout with ``provider.sort = "latency"``: a
model that misses the deadline simply does not vote, and two votes still
produce a usable result.  Bounded by the slowest survivor, not by the median.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from services.core.extract.schemas import flatten, json_schema_for, unflatten
from services.core.llm.protocol import EXTRACTION_PROVIDER, ChatMessage, LLMClient, text_part

Status = Literal["unanimous", "majority", "split"]


class FieldVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    value: Any = None
    status: Status = "split"
    #: model slug -> the value that model returned for this path
    votes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    alternatives: list[Any] = Field(default_factory=list)
    #: which crop this field was read from, so the UI can zoom to it
    crop_id: str | None = None
    #: True for a unanimous verdict -- including a unanimous null, which means
    #: "every reader agrees this is not on the screen".  Never shown to the user
    #: unless the pipeline finds the assembler needs the field.
    auto_confirmed: bool = False
    #: Why a field needs a human, when it does.
    reason: str = ""


class JuryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merged: dict[str, Any] = Field(default_factory=dict)
    verdicts: list[FieldVerdict] = Field(default_factory=list)
    unanimous_count: int = 0
    majority_count: int = 0
    split_count: int = 0
    auto_confirmed: list[str] = Field(default_factory=list)
    disputed: list[str] = Field(default_factory=list)
    per_model_latency_ms: dict[str, float] = Field(default_factory=dict)
    elapsed_ms: float = 0.0
    errors: dict[str, str] = Field(default_factory=dict)
    #: models that actually returned a parseable document
    voters: list[str] = Field(default_factory=list)
    crop_id: str | None = None
    tie_breaker_used: bool = False

    @property
    def flat(self) -> dict[str, Any]:
        return {v.path: v.value for v in self.verdicts}

    def verdict(self, path: str) -> FieldVerdict:
        for v in self.verdicts:
            if v.path == path:
                return v
        raise KeyError(path)

    def review_rate(self) -> float:
        total = len(self.verdicts)
        return (len(self.disputed) / total) if total else 0.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _vkey(value: Any) -> str:
    """A hashable identity for a vote.  ``1`` and ``1.0`` are the same vote."""
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        f = float(value)
        return json.dumps(int(f) if f.is_integer() else f)
    return json.dumps(value, sort_keys=True, default=str)


def _parse(content: Any, schema_model: type[BaseModel]) -> dict[str, Any]:
    """Validate a model's answer and flatten it into field paths."""
    data = content
    if isinstance(data, str):
        data = json.loads(data)
    if data is None:
        raise ValueError("empty response")
    obj = schema_model.model_validate(data)
    return flatten(obj)


async def _ask(
    client: LLMClient,
    model: str,
    messages: Sequence[ChatMessage],
    response_format: dict[str, Any],
    schema_model: type[BaseModel],
    timeout_s: float,
    max_tokens: int | None,
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    resp = await client.complete(
        model=model,
        messages=list(messages),
        response_format=response_format,
        provider=dict(EXTRACTION_PROVIDER),
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    if getattr(resp, "error", None):
        raise RuntimeError(str(resp.error))
    payload = resp.parsed if getattr(resp, "parsed", None) is not None else resp.content
    flat = _parse(payload, schema_model)
    return flat, (resp.latency_ms or ms)


async def _gather_votes(
    client: LLMClient,
    models: Sequence[str],
    messages: Sequence[ChatMessage],
    schema_model: type[BaseModel],
    *,
    timeout_s: float,
    max_tokens: int | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, str]]:
    response_format = json_schema_for(schema_model)
    tasks: dict[asyncio.Task, str] = {}
    for model in models:
        task = asyncio.ensure_future(
            _ask(client, model, messages, response_format, schema_model, timeout_s, max_tokens)
        )
        tasks[task] = model

    docs: dict[str, dict[str, Any]] = {}
    latency: dict[str, float] = {}
    errors: dict[str, str] = {}
    if not tasks:
        return docs, latency, errors

    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    for task in pending:
        task.cancel()
        errors[tasks[task]] = f"timeout after {timeout_s:.1f}s (no vote)"
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        model = tasks[task]
        try:
            flat, ms = task.result()
        except asyncio.CancelledError:  # pragma: no cover - defensive
            errors[model] = "cancelled"
            continue
        except Exception as exc:
            errors[model] = f"{type(exc).__name__}: {exc}"
            continue
        docs[model] = flat
        latency[model] = ms
    return docs, latency, errors


def _tally(
    path: str,
    votes: dict[str, Any],
    n_models: int,
    crop_id: str | None,
) -> FieldVerdict:
    groups: dict[str, tuple[Any, list[str]]] = {}
    for model, value in votes.items():
        key = _vkey(value)
        if key not in groups:
            groups[key] = (value, [])
        groups[key][1].append(model)

    voters = len(votes)
    ordered = sorted(groups.values(), key=lambda g: (-len(g[1]), _vkey(g[0])))
    winner, backers = ordered[0]
    top = len(backers)
    tied_at_top = sum(1 for _, ms in ordered if len(ms) == top)

    if voters >= 2 and len(ordered) == 1:
        status: Status = "unanimous"
    elif top >= 2 and tied_at_top == 1:
        status = "majority"
    else:
        status = "split"
        winner = None

    agreement = top / voters if voters else 0.0
    coverage = voters / n_models if n_models else 0.0
    confidence = agreement * (0.5 + 0.5 * coverage)
    reason = ""
    auto = status == "unanimous"
    if status == "unanimous" and winner is None:
        # Every reader agrees the field is not on the screen.  That is agreement,
        # so it auto-confirms as *absent* -- asking a human to confirm that a
        # maker has no sale price is exactly the review load this exists to kill.
        # The pipeline demotes this again if the assembler turns out to need the
        # field: absent is fine, silently-defaulted is not.
        reason = "all readers agree this field is not shown"
    elif status == "majority" and winner is None:
        reason = "majority returned null: not legible"
        confidence *= 0.5
    elif status == "majority":
        reason = "majority value, one keystroke to accept"
    elif status == "split":
        reason = "models disagreed" if voters > 1 else "only one model answered"

    return FieldVerdict(
        path=path,
        value=winner,
        status=status,
        votes=dict(votes),
        confidence=round(confidence, 4),
        alternatives=[v for v, _ in ordered if _vkey(v) != _vkey(winner)],
        crop_id=crop_id,
        auto_confirmed=auto,
        reason=reason,
    )


def _tie_break_messages(
    messages: Sequence[ChatMessage], split_paths: Sequence[str]
) -> list[ChatMessage]:
    """Same crop, but tell the tie-breaker which fields are actually in dispute."""
    out = [m.model_copy(deep=True) for m in messages]
    ask = (
        "Three readers disagreed on exactly these fields: "
        + ", ".join(sorted(split_paths))
        + ". Read only those, carefully, from the image. Everything else may be null."
    )
    for msg in reversed(out):
        if msg.role == "user":
            if msg.parts is not None:
                msg.parts = list(msg.parts) + [text_part(ask)]
            else:
                msg.content = f"{msg.content or ''}\n\n{ask}"
            break
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


async def run_jury(
    client: LLMClient,
    models: list[str],
    messages: Sequence[ChatMessage],
    schema_model: type[BaseModel],
    *,
    timeout_s: float = 12.0,
    tie_breaker: str | None = None,
    crop_id: str | None = None,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    max_tokens: int | None = None,
) -> JuryResult:
    """Fire one crop at ``models`` concurrently and vote per field path.

    ``normalizer`` runs on each model's flattened document before voting; the
    topology call uses it to sort edges, so two models that list the same graph
    in a different order still vote on the same paths.
    """
    t0 = time.perf_counter()
    docs, latency, errors = await _gather_votes(
        client,
        models,
        messages,
        schema_model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )
    if normalizer:
        docs = {m: normalizer(flat) for m, flat in docs.items()}

    votes_by_path: dict[str, dict[str, Any]] = {}
    for model in models:  # stable, request-order columns
        for path, value in (docs.get(model) or {}).items():
            votes_by_path.setdefault(path, {})[model] = value

    verdicts = [
        _tally(path, votes, len(models), crop_id)
        for path, votes in votes_by_path.items()
    ]

    tie_breaker_used = False
    split_paths = [v.path for v in verdicts if v.status == "split"]
    if tie_breaker and split_paths:
        tie_breaker_used = True
        try:
            flat, ms = await _ask(
                client,
                tie_breaker,
                _tie_break_messages(messages, split_paths),
                json_schema_for(schema_model),
                schema_model,
                timeout_s,
                max_tokens,
            )
            latency[tie_breaker] = ms
            for v in verdicts:
                if v.status != "split" or v.path not in flat:
                    continue
                value = flat[v.path]
                # promote only if the tie-breaker seconds one of the existing votes
                seconded = [
                    m
                    for m, val in v.votes.items()
                    if m != tie_breaker and _vkey(val) == _vkey(value)
                ]
                v.votes[tie_breaker] = value
                if seconded and value is not None:
                    v.value = value
                    v.status = "majority"
                    v.alternatives = [
                        val
                        for m, val in v.votes.items()
                        if _vkey(val) != _vkey(value) and m != tie_breaker
                    ]
                    v.confidence = round(
                        (len(seconded) + 1) / max(len(v.votes), 1) * 0.75, 4
                    )
                    v.reason = f"tie broken by {tie_breaker}"
                else:
                    v.reason = "three-way split; tie-breaker did not second any reading"
        except Exception as exc:
            errors[tie_breaker] = f"{type(exc).__name__}: {exc}"

    for v in verdicts:
        # A unanimous null auto-confirms as "absent"; a *majority* null does not,
        # because two readers saying "nothing there" against one reading a number
        # is a genuine disagreement about what is on the screen.
        v.auto_confirmed = v.status == "unanimous"

    merged_flat = {v.path: v.value for v in verdicts}
    auto = [v.path for v in verdicts if v.auto_confirmed]
    disputed = [v.path for v in verdicts if not v.auto_confirmed]

    return JuryResult(
        merged=unflatten(merged_flat) if merged_flat else {},
        verdicts=verdicts,
        unanimous_count=sum(1 for v in verdicts if v.status == "unanimous"),
        majority_count=sum(1 for v in verdicts if v.status == "majority"),
        split_count=sum(1 for v in verdicts if v.status == "split"),
        auto_confirmed=auto,
        disputed=disputed,
        per_model_latency_ms={k: round(v, 2) for k, v in latency.items()},
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        errors=errors,
        voters=[m for m in models if m in docs],
        crop_id=crop_id,
        tie_breaker_used=tie_breaker_used,
    )


def sort_edges(flat: dict[str, Any]) -> dict[str, Any]:
    """Normalizer for topology docs: index edges by ``src->dst``, not by position.

    Without this, two models that read the same graph starting from different
    corners vote on different paths and every edge looks disputed.
    """
    edges: dict[int, dict[str, Any]] = {}
    others: dict[str, Any] = {}
    for path, value in flat.items():
        if path.startswith("edges[") and "]." in path:
            idx_s, _, leaf = path[len("edges[") :].partition("].")
            try:
                idx = int(idx_s)
            except ValueError:
                others[path] = value
                continue
            edges.setdefault(idx, {})[leaf] = value
        else:
            others[path] = value
    out = dict(others)
    for edge in sorted(
        edges.values(), key=lambda e: (str(e.get("src")), str(e.get("dst")))
    ):
        src, dst = edge.get("src"), edge.get("dst")
        if src is None and dst is None:
            continue
        key = f"edges[{src}->{dst}]"
        for leaf, value in edge.items():
            out[f"{key}.{leaf}"] = value
    return out


__all__ = [
    "FieldVerdict",
    "JuryResult",
    "run_jury",
    "sort_edges",
]
