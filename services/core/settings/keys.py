"""API-key handling: validate once, store in the OS credential store, and make
sure the key can never reach a log line.

Three rules (spec 9.1, 13):

1.  A key that has not been validated against ``GET /api/v1/key`` is never
    stored.  A typo'd key that silently persists costs the user a whole run.
2.  The key lives in the OS credential store via ``keyring``.  Never in the
    repo, never in the browser, never in ``localStorage``.  The fallback is a
    file under ``data/.secrets/`` restricted to the current user, and the UI is
    told which one is in use *and how it is protected*.

    "Restricted" means different things on different platforms and the code has
    to mean the one that is true.  On POSIX it is mode ``0600``.  On Windows
    ``os.chmod`` only toggles the read-only attribute -- it does **not** stop
    another local account reading the file -- so the fallback is locked down
    with an explicit ACL (``icacls /inheritance:r /grant:r <user>:F``), the ACL
    is read back to confirm it took, and if it did not the key is **not**
    written at all.  Claiming protection we do not have is worse than refusing
    to store: the user would carry on believing the secret was safe.
3.  Redaction is installed on *every* log sink, and covers exception
    tracebacks -- the path that actually leaks keys in practice, because the
    key is usually an argument to the call that raised.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
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


class KeyStorageError(RuntimeError):
    """Raised when the key cannot be stored *safely*.

    Not "the write failed" -- "the write would have succeeded but the file
    would have been readable by other accounts on this machine".  Refusing is
    the only honest option, because the alternative is a UI that reports a
    protection the filesystem is not enforcing.
    """


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


def _is_windows() -> bool:
    """Single place to ask, so tests can monkeypatch ``os.name`` and mean it."""
    return os.name == "nt"


# -- Windows ACLs -----------------------------------------------------------
#
# ``icacls`` is a real executable in System32, so this never needs a shell.
# CREATE_NO_WINDOW keeps a console from flashing up when the app is started
# from a shortcut rather than a terminal.

#: Principals that must not appear in the fallback file's ACL.  If any of these
#: can read it, the file is not protected no matter what else is in there.
#: ``users:`` carries the colon deliberately -- it matches the ACE
#: ``BUILTIN\Users:(RX)`` but not the ordinary path ``C:\Users\bob\...`` that
#: shares the first line of icacls output with it.
_ACL_FORBIDDEN = (
    "everyone",
    "authenticated users",
    "builtin\\users",
    "users:",
    "interactive",
    "\\administrators",
)


def _windows_principal() -> str:
    """The account to grant.  Domain-qualified when we can, bare name if not."""
    user = os.environ.get("USERNAME") or ""
    if not user:
        try:
            import getpass

            user = getpass.getuser()
        except Exception:  # pragma: no cover - defensive
            user = ""
    domain = os.environ.get("USERDOMAIN") or ""
    if user and domain:
        return f"{domain}\\{user}"
    return user


def _run_icacls(args: list[str]) -> subprocess.CompletedProcess:
    # ``shell=False`` matters: the secrets directory sits under the repo path,
    # which can contain spaces, and a shell would re-split it.  ``errors``
    # matters because icacls prints account names in the console code page and
    # a non-ASCII group name must degrade to a replacement character rather
    # than raise UnicodeDecodeError inside a security check.
    kwargs: dict[str, Any] = dict(
        capture_output=True, text=True, errors="replace", shell=False, timeout=20
    )
    flag = getattr(subprocess, "CREATE_NO_WINDOW", None)
    if flag:
        kwargs["creationflags"] = flag
    return subprocess.run(["icacls", *args], **kwargs)


def _windows_restrict_acl(path: Path, *, container: bool = False) -> bool:
    """Drop inheritance and grant only the current user.  True if icacls said OK."""
    principal = _windows_principal()
    if not principal:
        return False
    grant = f"{principal}:(OI)(CI)F" if container else f"{principal}:F"
    try:
        proc = _run_icacls([str(path), "/inheritance:r", "/grant:r", grant])
    except Exception:  # icacls missing, timeout, permission -- all mean "no"
        return False
    return proc.returncode == 0


def _windows_acl_is_restricted(path: Path) -> bool:
    """Read the ACL back and check nobody but the current user is on it.

    Verification is the point of the exercise.  ``icacls /grant:r`` can return
    0 and still leave an inherited ACE behind if the ``/inheritance:r`` half was
    refused, and an inherited ACE from ``C:\\Users\\Public``-style ancestry is
    exactly the case that leaves the key world-readable.

    The primary test is a **count**: after a successful lockdown there is
    exactly one ACE and it is ours.  Counting rather than blacklisting is what
    makes this work on a localized Windows, where the group we most need to
    exclude is not spelled "Everyone" -- it is "Jeder", "Tout le monde",
    "Todos".  :data:`_ACL_FORBIDDEN` stays as a second line of defence for the
    English case; it is not load-bearing on its own.
    """
    principal = _windows_principal()
    if not principal:
        return False
    try:
        proc = _run_icacls([str(path)])
    except Exception:
        return False
    if proc.returncode != 0:
        return False

    aces: list[str] = []
    for line in (proc.stdout or "").splitlines():
        text = line.strip()
        if not text or text.lower().startswith("successfully processed"):
            continue
        # The first line is "<path> <ACE>"; strip the path off it.  Subsequent
        # ACEs are indented continuation lines with no path.
        ace = text[len(str(path)):].strip() if text.startswith(str(path)) else text
        if ace:
            aces.append(ace)

    if len(aces) != 1:
        # Zero means we could not read it; more than one means somebody else is
        # still on the ACL, whatever their name happens to be in this locale.
        return False

    low = aces[0].lower()
    if "(i)" in low:  # an inherited ACE means /inheritance:r did not take
        return False
    if any(bad in low for bad in _ACL_FORBIDDEN):
        return False
    # An ACE reads ``DOMAIN\name:(F)``; requiring the colon stops a short
    # username matching as a substring of some other principal's name.
    bare = principal.split("\\")[-1].lower()
    return f"{bare}:" in low or f"{principal.lower()}:" in low


def _fallback_protection() -> str:
    """The *actual* protection on the fallback file, never the intended one."""
    if not FALLBACK_KEY_PATH.exists():
        return "not yet created"
    if _is_windows():
        return "ACL-restricted" if _windows_acl_is_restricted(FALLBACK_KEY_PATH) else "UNPROTECTED"
    try:
        mode = FALLBACK_KEY_PATH.stat().st_mode & 0o777
    except OSError:  # pragma: no cover - defensive
        return "unknown"
    return "mode 0600" if mode == 0o600 else f"mode {mode:04o} (UNPROTECTED)"


def _file_backend() -> str:
    return f"file:{FALLBACK_KEY_PATH} ({_fallback_protection()})"


def storage_backend() -> str:
    """Human-readable name of where the key lives, for the settings UI."""
    kr = _keyring()
    if kr is not None:
        try:
            return f"keyring:{type(kr.get_keyring()).__name__}"
        except Exception:
            pass
    return _file_backend()


_UNPROTECTED_MSG = (
    "Refusing to write the OpenRouter key to disk: this machine has no working "
    "keyring backend, and the file fallback could not be restricted with an ACL, "
    "so every other account on this PC could read it. Fix the credential store "
    "instead -- on Windows the Credential Locker backend ships with `keyring`, so "
    "this usually means `keyring` is not installed in the active venv "
    "(`.venv\\Scripts\\python.exe -m pip install keyring`). As a stopgap, set the "
    "OPENROUTER_API_KEY environment variable for the session instead of storing it."
)


def _write_fallback(key: str) -> None:
    """Write the fallback file, or raise rather than write it unprotected."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    if _is_windows():
        _write_fallback_windows(key)
    else:
        _write_fallback_posix(key)


def _write_fallback_posix(key: str) -> None:
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


def _write_fallback_windows(key: str) -> None:
    """Create empty, lock down, verify, *then* write the secret.

    Order matters.  A file created with inherited ACLs is readable from the
    instant it exists, so writing first and restricting second leaves a window
    -- short, but a background indexer or another user's process only needs
    one read.  An empty file leaking is harmless.
    """
    _windows_restrict_acl(SECRETS_DIR, container=True)  # best effort on the dir

    fd = os.open(FALLBACK_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.close(fd)

    applied = _windows_restrict_acl(FALLBACK_KEY_PATH)
    if not (applied and _windows_acl_is_restricted(FALLBACK_KEY_PATH)):
        try:
            os.unlink(FALLBACK_KEY_PATH)
        except OSError:  # pragma: no cover - defensive
            pass
        raise KeyStorageError(_UNPROTECTED_MSG)

    with open(FALLBACK_KEY_PATH, "w", encoding="utf-8") as fh:
        fh.write(key)


def store_key(key: str, info: KeyInfo) -> str:
    """Persist a **validated** key.  Returns the backend used.

    ``info`` must come from :func:`validate_key` and must be ``valid``; this is
    the enforcement point for "never accept an unvalidated key".

    Raises :class:`KeyStorageError` when neither the keyring nor a *properly
    restricted* file is available.  Callers should surface that verbatim: it is
    actionable, and the alternative is a silently readable secret.
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
    return _file_backend()


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
