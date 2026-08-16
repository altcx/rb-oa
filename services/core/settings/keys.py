"""API-key handling: validate once, store in the OS credential store, and make
sure the key can never reach a log line.

Three rules (spec 9.1, 13):

1.  A key that has not been validated against ``GET /api/v1/key`` is never
    stored.  A typo'd key that silently persists costs the user a whole run.
2.  The key lives in the OS credential store via ``keyring``.  Never in the
    repo, never in the browser, never in ``localStorage``.  The fallback is a
    0600 file under ``data/.secrets/`` and the UI is told which one is in use.
3.  Redaction is installed on *every* log sink, and covers exception
    tracebacks -- the path that actually leaks keys in practice, because the
    key is usually an argument to the call that raised.
"""

from __future__ import annotations

import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, Iterable

import httpx
from pydantic import BaseModel, ConfigDict, Field

SERVICE_NAME = "puzzle-copilot"
USERNAME = "openrouter"

KEY_URL = "https://openrouter.ai/api/v1/key"

REPO_ROOT = Path(__file__).resolve().parents[3]
SECRETS_DIR = REPO_ROOT / "data" / ".secrets"
FALLBACK_KEY_PATH = SECRETS_DIR / "openrouter.key"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

REDACTED = "sk-***REDACTED***"

#: Specific first (so the generic pattern never gets a chance to half-match),
#: then a generic OpenAI-style secret shape.
KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"),
    re.compile(r"sk-[A-Za-z0-9\-]{20,}"),
)


def redact(text: str) -> str:
    """Replace anything shaped like an API key with a fixed marker."""
    if not text:
        return text
    for pat in KEY_PATTERNS:
        text = pat.sub(REDACTED, text)
    return text


def _redact_any(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, tuple):
        return tuple(_redact_any(v) for v in value)
    if isinstance(value, list):
        return [_redact_any(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_any(v) for k, v in value.items()}
    return value


class RedactionFilter(logging.Filter):
    """Scrubs keys out of the message, the args and the traceback.

    Filters attached to a *logger* only see records emitted through that
    logger, so :func:`install_redaction_filter` also attaches this to every
    handler -- that is the only placement that catches propagated records.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        elif record.msg is not None and not isinstance(record.msg, (int, float)):
            text = str(record.msg)
            if any(p.search(text) for p in KEY_PATTERNS):
                record.msg = redact(text)
                record.args = None
        if record.args:
            record.args = _redact_any(record.args)
        if getattr(record, "exc_text", None):
            record.exc_text = redact(record.exc_text)
        if record.exc_info:
            # Format it ourselves, scrub it, and hand the formatter a cached
            # exc_text so no downstream handler ever re-derives the raw one.
            try:
                formatted = "".join(traceback.format_exception(*record.exc_info))
            except Exception:  # pragma: no cover - defensive
                formatted = ""
            record.exc_text = redact(record.exc_text or formatted)
            record.exc_info = None
        if getattr(record, "stack_info", None):
            record.stack_info = redact(record.stack_info)
        return True


_FILTER_MARK = "_puzzle_copilot_redaction"


def _attach(target: Any) -> bool:
    if getattr(target, _FILTER_MARK, False):
        return False
    for existing in getattr(target, "filters", []):
        if isinstance(existing, RedactionFilter):
            setattr(target, _FILTER_MARK, True)
            return False
    target.addFilter(RedactionFilter())
    setattr(target, _FILTER_MARK, True)
    return True


def install_redaction_filter() -> int:
    """Attach the redaction filter to every logger and every handler.

    Returns the number of sinks newly filtered.  Idempotent: calling it again
    after a new handler appears (pytest's caplog, a file handler added later)
    picks up the new sink and leaves the old ones alone.
    """
    count = 0
    root = logging.getLogger()
    targets: list[Any] = [root, *root.handlers]
    for name in list(logging.Logger.manager.loggerDict):
        obj = logging.Logger.manager.loggerDict.get(name)
        if isinstance(obj, logging.Logger):
            targets.append(obj)
            targets.extend(obj.handlers)
    for handler in list(getattr(logging, "_handlerList", [])):
        try:
            ref = handler()
        except TypeError:  # pragma: no cover - weakref proxy shapes
            ref = handler
        if isinstance(ref, logging.Handler):
            targets.append(ref)
    for t in targets:
        if _attach(t):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class KeyInfo(BaseModel):
    """What ``GET /api/v1/key`` tells us.  ``valid`` gates storage."""

    model_config = ConfigDict(extra="forbid")

    valid: bool = False
    label: str = ""
    usage: float = 0.0
    limit: float | None = None
    limit_remaining: float | None = None
    is_free_tier: bool = False
    rate_limit: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def remaining(self) -> float | None:
        """Credit left, derived when OpenRouter only reports usage + limit."""
        if self.limit_remaining is not None:
            return self.limit_remaining
        if self.limit is None:
            return None
        return self.limit - self.usage


class KeyValidationError(RuntimeError):
    pass


def _key_info_from(payload: dict[str, Any]) -> KeyInfo:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return KeyInfo(
        valid=True,
        label=str(data.get("label") or ""),
        usage=float(data.get("usage") or 0.0),
        limit=(None if data.get("limit") is None else float(data["limit"])),
        limit_remaining=(
            None if data.get("limit_remaining") is None else float(data["limit_remaining"])
        ),
        is_free_tier=bool(data.get("is_free_tier") or False),
        rate_limit=dict(data.get("rate_limit") or {}),
    )


async def validate_key(
    key: str, *, http_client: httpx.AsyncClient | None = None, timeout_s: float = 15.0
) -> KeyInfo:
    """Single paste field -> credit readout.  Never raises on a bad key."""
    key = (key or "").strip()
    if not key:
        return KeyInfo(valid=False, error="empty key")
    owns = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout_s)
    try:
        resp = await client.get(
            KEY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=timeout_s
        )
        if resp.status_code == 401:
            return KeyInfo(valid=False, error="key rejected (401)")
        if resp.status_code >= 400:
            return KeyInfo(valid=False, error=f"http {resp.status_code}")
        return _key_info_from(resp.json())
    except Exception as exc:  # network problems are not "invalid key"
        return KeyInfo(valid=False, error=redact(f"{type(exc).__name__}: {exc}"))
    finally:
        if owns:
            await client.aclose()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _keyring():
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    if backend is None or isinstance(backend, FailKeyring):
        return None
    if type(backend).__name__ in {"Keyring", "ChainerBackend"} and not getattr(
        backend, "priority", 1
    ):
        return None
    return keyring


def storage_backend() -> str:
    """Human-readable name of where the key lives, for the settings UI."""
    kr = _keyring()
    if kr is not None:
        try:
            return f"keyring:{type(kr.get_keyring()).__name__}"
        except Exception:
            pass
    return f"file:{FALLBACK_KEY_PATH}"


def _write_fallback(key: str) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(SECRETS_DIR, 0o700)
    except OSError:  # pragma: no cover - platform dependent
        pass
    fd = os.open(FALLBACK_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key)
    try:
        os.chmod(FALLBACK_KEY_PATH, 0o600)
    except OSError:  # pragma: no cover
        pass


def store_key(key: str, info: KeyInfo) -> str:
    """Persist a **validated** key.  Returns the backend used.

    ``info`` must come from :func:`validate_key` and must be ``valid``; this is
    the enforcement point for "never accept an unvalidated key".
    """
    if not isinstance(info, KeyInfo) or not info.valid:
        raise KeyValidationError(
            "refusing to store a key that has not been validated against /api/v1/key"
        )
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(SERVICE_NAME, USERNAME, key)
            return storage_backend()
        except Exception:
            pass
    _write_fallback(key)
    return f"file:{FALLBACK_KEY_PATH}"


async def validate_and_store(
    key: str, *, http_client: httpx.AsyncClient | None = None
) -> tuple[KeyInfo, str | None]:
    """The whole paste-field flow.  Returns ``(info, backend_or_None)``."""
    info = await validate_key(key, http_client=http_client)
    if not info.valid:
        return info, None
    return info, store_key(key, info)


def load_key() -> str | None:
    kr = _keyring()
    if kr is not None:
        try:
            value = kr.get_password(SERVICE_NAME, USERNAME)
            if value:
                return value
        except Exception:
            pass
    if FALLBACK_KEY_PATH.exists():
        text = FALLBACK_KEY_PATH.read_text(encoding="utf-8").strip()
        return text or None
    env = os.environ.get("OPENROUTER_API_KEY")
    return env.strip() if env else None


def delete_key() -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE_NAME, USERNAME)
        except Exception:
            pass
    if FALLBACK_KEY_PATH.exists():
        FALLBACK_KEY_PATH.unlink()


def scrub_iterable(lines: Iterable[str]) -> list[str]:
    """Convenience for scrubbing captured output in tests / bug reports."""
    return [redact(line) for line in lines]
