"""Delta extraction (spec 6.2).

After the first confirmed state, re-reading the whole board is waste: the user
changed one output setting and the other thirty-nine fields are identical.  So
send the previous state plus the new crop and ask only for what *differs*.

Output tokens drop by roughly 90%, and time-to-last-token drops with them --
which is the number that matters, because the state is not usable until the
last token lands.

The safety property that makes this usable: every returned path is checked
against the flattened previous state, and an unknown path is **rejected and
reported**, never applied.  A model that invents ``machines[M9].output_max``
must not be able to conjure a machine into the state.
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from services.core.extract.schemas import flatten, json_schema_for
from services.core.llm.protocol import EXTRACTION_PROVIDER, ChatMessage, LLMClient

#: What a delta value is allowed to be on the wire.  Deliberately scalar: a
#: whole nested object coming back would defeat the point of the delta.
Scalar = str | float | bool | None


class FieldChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Dotted field path from the previous state, copied exactly."
    )
    value: Scalar = Field(
        description="The new on-screen value. null if the field is no longer legible; never guess."
    )


class DeltaChanges(BaseModel):
    """The reduced schema: a list of changes, not a document."""

    model_config = ConfigDict(extra="forbid")

    changes: list[FieldChange] = Field(
        description="Only fields whose value differs from the previous state. Empty if nothing changed."
    )


class DeltaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merged: dict[str, Any] = Field(default_factory=dict)
    changed: list[str] = Field(default_factory=list)
    #: paths the model returned that do not exist in the previous state
    rejected: list[str] = Field(default_factory=list)
    #: paths returned with the value they already had
    no_ops: list[str] = Field(default_factory=list)
    model: str = ""
    elapsed_ms: float = 0.0
    error: str | None = None


def _coerce(previous: Any, value: Any) -> Any:
    """Keep the previous field's type where the wire had to widen it.

    The wire says ``float`` because JSON Schema has no int-or-float union that
    every provider honours; the state says ``int``.  Convert back rather than
    letting ``3.0`` into a field the solver indexes with.
    """
    if value is None or previous is None:
        return value
    if isinstance(previous, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if isinstance(previous, int) and not isinstance(value, bool):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return value
        return int(f) if float(f).is_integer() else f
    if isinstance(previous, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if isinstance(previous, str) and not isinstance(value, str):
        return str(value)
    return value


_TOKEN = re.compile(r"([^.\[\]]+)|\[([^\]]*)\]")


def _tokens(path: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    for m in _TOKEN.finditer(path):
        out.append((m.group(1), False) if m.group(1) is not None else (m.group(2), True))
    return out


def _set_path(doc: Any, path: str, value: Any) -> None:
    """Write ``value`` at ``path`` *in place*.

    Deliberately not ``unflatten(flatten(doc) | change)``: that round trip loses
    empty lists and empty dicts, and "no mods installed" is not the same fact as
    "mods not read".
    """
    cursor = doc
    parts = _tokens(path)
    for i, (name, is_index) in enumerate(parts):
        last = i == len(parts) - 1
        if is_index:
            if not isinstance(cursor, list):
                raise KeyError(path)
            idx = None
            for j, item in enumerate(cursor):
                if isinstance(item, dict) and (item.get("id") or item.get("name")) == name:
                    idx = j
                    break
            if idx is None:
                idx = int(name)  # raises for a genuinely unknown key
            if last:
                cursor[idx] = value
            else:
                cursor = cursor[idx]
        else:
            if not isinstance(cursor, dict):
                raise KeyError(path)
            if last:
                cursor[name] = value
            else:
                cursor = cursor[name]


def apply_changes(
    previous: dict[str, Any], changes: Sequence[FieldChange | dict[str, Any]]
) -> DeltaResult:
    """Validate and apply ``changes`` onto a copy of ``previous``.

    Pure; no I/O.  This is where unknown paths die.
    """
    prev_flat = flatten(previous)
    merged = copy.deepcopy(previous)
    changed: list[str] = []
    rejected: list[str] = []
    no_ops: list[str] = []
    for raw in changes:
        change = FieldChange.model_validate(raw) if not isinstance(raw, FieldChange) else raw
        if change.path not in prev_flat:
            rejected.append(change.path)
            continue
        old = prev_flat[change.path]
        value = _coerce(old, change.value)
        if value == old and type(value) is type(old):
            no_ops.append(change.path)
            continue
        try:
            _set_path(merged, change.path, value)
        except (KeyError, IndexError, ValueError):
            rejected.append(change.path)
            continue
        changed.append(change.path)
    return DeltaResult(merged=merged, changed=changed, rejected=rejected, no_ops=no_ops)


async def extract_delta_detailed(
    client: LLMClient,
    model: str,
    previous: dict[str, Any],
    messages: Sequence[ChatMessage],
    schema_model: type[BaseModel] | None = None,
    *,
    timeout_s: float = 12.0,
    max_tokens: int | None = 1024,
) -> DeltaResult:
    """One cheap call that returns only what moved.  Full report."""
    t0 = time.perf_counter()
    try:
        resp = await client.complete(
            model=model,
            messages=list(messages),
            response_format=json_schema_for(DeltaChanges, "DeltaChanges"),
            provider=dict(EXTRACTION_PROVIDER),
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        if getattr(resp, "error", None):
            raise RuntimeError(str(resp.error))
        payload = resp.parsed if getattr(resp, "parsed", None) is not None else resp.content
        if isinstance(payload, str):
            payload = json.loads(payload)
        doc = DeltaChanges.model_validate(payload or {"changes": []})
    except Exception as exc:
        # A failed delta is recoverable: keep the previous state and say why.
        return DeltaResult(
            merged=dict(previous),
            model=model,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            error=f"{type(exc).__name__}: {exc}",
        )

    result = apply_changes(previous, doc.changes)
    result.model = model
    result.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if schema_model is not None:
        # Round-trip through the real schema so a delta can never produce a
        # document the assembler would choke on.
        try:
            result.merged = schema_model.model_validate(result.merged).model_dump()
        except Exception as exc:
            result.error = f"delta produced an invalid document, kept previous: {exc}"
            result.merged = dict(previous)
            result.rejected = result.rejected + result.changed
            result.changed = []
    return result


async def extract_delta(
    client: LLMClient,
    model: str,
    previous: dict[str, Any],
    messages: Sequence[ChatMessage],
    schema_model: type[BaseModel] | None = None,
    **kw: Any,
) -> tuple[dict[str, Any], list[str]]:
    """``(merged state, changed paths)``.

    Rejected paths are reported on :class:`DeltaResult`; use
    :func:`extract_delta_detailed` when you need them (the pipeline does, and
    surfaces them as extraction errors).
    """
    result = await extract_delta_detailed(client, model, previous, messages, schema_model, **kw)
    return result.merged, result.changed


__all__ = [
    "FieldChange",
    "DeltaChanges",
    "DeltaResult",
    "apply_changes",
    "extract_delta",
    "extract_delta_detailed",
]
