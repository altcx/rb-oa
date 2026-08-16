"""Numeric provenance guard.

Hard rule 1 of the system prompt is "never state a number you did not receive
from a tool result".  Prompts are not enforcement, so this module checks it:
pull every numeric token out of the assistant's text and confirm each one is
traceable to a tool result in the same conversation (or to something the user
typed themselves).

Deliberately strict.  Small integers are *not* exempt -- "hour 3" and "machine
2" are exactly the kind of confident-sounding index the model invents when it
has not actually read the log.  In practice they are cheap to satisfy: if the
tool result really mentions hour 3, the number is right there in the payload.

Rounding is allowed: a tool result of ``1234.56`` sources ``1,234.6`` and
``1235``, because narrating a rounded figure is not a fabrication.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

#: Numbers, with optional thousands separators and decimals.  Leading sign is
#: intentionally excluded from the token so "-5" and "5" compare equal after a
#: minus that is really a dash in prose; the value comparison handles signs.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numeric_tokens(text: str) -> list[str]:
    """Every numeric token in ``text``, in order, deduplicated, normalized.

    Normalization strips thousands separators only; the decimal form is kept
    because it carries the precision claim.
    """
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER.finditer(text):
        token = match.group(0).replace(",", "")
        token = token.rstrip(".")
        if not token or not token[0].isdigit():
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _haystack_text(
    tool_results: Sequence[Any], user_messages: Sequence[str] | None = None
) -> str:
    chunks: list[str] = []
    for r in tool_results or []:
        if isinstance(r, str):
            chunks.append(r)
        else:
            try:
                chunks.append(json.dumps(r, default=str))
            except (TypeError, ValueError):
                chunks.append(str(r))
    chunks.extend(user_messages or [])
    return "\n".join(chunks)


def sourced_values(
    tool_results: Sequence[Any], user_messages: Sequence[str] | None = None
) -> tuple[set[str], list[float]]:
    """(verbatim tokens, numeric values) available as provenance."""
    text = _haystack_text(tool_results, user_messages)
    tokens = {m.group(0).replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}
    values = [v for v in (_to_float(t) for t in tokens) if v is not None]
    return tokens, values


def is_sourced(token: str, tokens: set[str], values: Iterable[float]) -> bool:
    if token in tokens:
        return True
    value = _to_float(token)
    if value is None:
        return False
    places = _decimals(token)
    for candidate in values:
        if abs(candidate - value) < 1e-9:
            return True
        if round(candidate, places) == value:
            return True
    return False


def unsourced_numbers(
    text: str,
    tool_results: Sequence[Any],
    user_messages: Sequence[str] | None = None,
) -> list[str]:
    """Numeric tokens in ``text`` that no tool result (or user message) supports."""
    tokens_in_text = extract_numeric_tokens(text)
    if not tokens_in_text:
        return []
    known_tokens, known_values = sourced_values(tool_results, user_messages)
    offenders = [t for t in tokens_in_text if not is_sourced(t, known_tokens, known_values)]
    log.debug(
        "numeric provenance: %d tokens, %d unsourced (%s)",
        len(tokens_in_text),
        len(offenders),
        ", ".join(offenders[:10]),
    )
    return offenders


def provenance_report(
    text: str,
    tool_results: Sequence[Any],
    user_messages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Structured version, for the UI's "unverified number" badge."""
    tokens = extract_numeric_tokens(text)
    offenders = unsourced_numbers(text, tool_results, user_messages)
    return {
        "tokens": tokens,
        "unsourced": offenders,
        "sourced": [t for t in tokens if t not in set(offenders)],
        "clean": not offenders,
        "message": (
            "all numbers trace to a tool result"
            if not offenders
            else "unsourced numbers: " + ", ".join(offenders)
        ),
    }
