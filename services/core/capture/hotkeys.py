"""Global hotkeys (spec 6.1).

``pynput`` is imported lazily and never at module import time: the daemon, the
tests and CI all import this module on machines with no input device, and an
ImportError at import time would take the whole server down for a feature that
is optional.

The dispatch logic lives entirely in :class:`HotkeyManager` and is driven by
:meth:`HotkeyManager.simulate`, so the interesting half is testable without a
keyboard.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

Callback = Callable[..., Any]

#: Action name -> default pynput-style accelerator.
DEFAULT_BINDINGS: dict[str, str] = {
    "capture_monitor": "<ctrl>+<alt>+1",
    "capture_region": "<ctrl>+<alt>+2",
    "capture_window": "<ctrl>+<alt>+3",
    "capture_and_ask": "<ctrl>+<alt>+space",
    "accept_all_majority": "<ctrl>+<alt>+a",
}

#: What each action means, for the settings UI and the "press ? for help" row.
ACTION_HELP: dict[str, str] = {
    "capture_monitor": "Grab the puzzle monitor and start speculative extraction.",
    "capture_region": "Drag a region, grab it, extract it.",
    "capture_window": "Grab the focused window (needs a compositor API).",
    "capture_and_ask": "Grab and immediately run the solver, no review stop.",
    "accept_all_majority": "Accept every majority-flagged field at once.",
}


class HotkeysUnavailable(RuntimeError):
    """Raised by :meth:`HotkeyManager.start` when pynput cannot be used."""


def _import_pynput():
    try:
        from pynput import keyboard  # type: ignore

        return keyboard
    except Exception as exc:  # ImportError, or X11 errors on import
        raise HotkeysUnavailable(
            "Global hotkeys need the 'pynput' package and a display. "
            "Install it (pip install pynput) and run on the desktop session; "
            "the HTTP API's /capture endpoints work without it. "
            f"({exc})"
        ) from exc


def available() -> bool:
    """True when global hotkeys could actually be bound here."""
    try:
        _import_pynput()
    except HotkeysUnavailable:
        return False
    from services.core.capture.grab import display_available

    return display_available()


@dataclass
class HotkeyEvent:
    action: str
    binding: str
    source: str = "simulate"
    payload: dict[str, Any] = field(default_factory=dict)


class HotkeyManager:
    """Callback registry + dispatcher for the five puzzle hotkeys.

    ``simulate(name)`` is the same code path a real key press takes; the only
    difference is who calls it.
    """

    def __init__(
        self,
        bindings: dict[str, str] | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.bindings: dict[str, str] = dict(bindings or DEFAULT_BINDINGS)
        self._callbacks: dict[str, list[Callback]] = {}
        self._listener: Any = None
        self._lock = threading.RLock()
        self._loop = loop
        self.history: list[HotkeyEvent] = []
        self.errors: list[tuple[str, str]] = []

    # -- registry -------------------------------------------------------
    @property
    def actions(self) -> list[str]:
        return list(self.bindings)

    def register(self, action: str, callback: Callback) -> Callback:
        """Bind ``callback`` to ``action``.  Several callbacks may share one action."""
        if action not in self.bindings:
            raise KeyError(
                f"unknown hotkey action {action!r}; known actions: {sorted(self.bindings)}"
            )
        with self._lock:
            self._callbacks.setdefault(action, []).append(callback)
        return callback

    def on(self, action: str) -> Callable[[Callback], Callback]:
        """Decorator form of :meth:`register`."""

        def deco(fn: Callback) -> Callback:
            self.register(action, fn)
            return fn

        return deco

    def unregister(self, action: str, callback: Callback | None = None) -> None:
        with self._lock:
            if callback is None:
                self._callbacks.pop(action, None)
            elif action in self._callbacks:
                self._callbacks[action] = [c for c in self._callbacks[action] if c is not callback]

    def callbacks(self, action: str) -> list[Callback]:
        with self._lock:
            return list(self._callbacks.get(action, ()))

    def rebind(self, action: str, accelerator: str) -> None:
        if action not in self.bindings:
            raise KeyError(f"unknown hotkey action {action!r}")
        conflict = [a for a, b in self.bindings.items() if b == accelerator and a != action]
        if conflict:
            raise ValueError(f"{accelerator!r} is already bound to {conflict[0]!r}")
        self.bindings[action] = accelerator

    # -- dispatch -------------------------------------------------------
    def simulate(self, action: str, **payload: Any) -> list[Any]:
        """Fire ``action`` synchronously; returns each callback's return value.

        Coroutine callbacks are scheduled on the manager's loop (if one was
        given) and their task object is returned -- a hotkey handler must never
        block the listener thread.
        """
        if action not in self.bindings:
            raise KeyError(f"unknown hotkey action {action!r}")
        event = HotkeyEvent(action=action, binding=self.bindings[action], payload=dict(payload))
        self.history.append(event)
        results: list[Any] = []
        for cb in self.callbacks(action):
            try:
                out = cb(**payload) if payload else cb()
                if inspect.isawaitable(out):
                    out = self._schedule(out)
                results.append(out)
            except Exception as exc:  # a broken handler must not kill the listener
                log.exception("hotkey %s callback failed", action)
                self.errors.append((action, repr(exc)))
                results.append(exc)
        return results

    async def simulate_async(self, action: str, **payload: Any) -> list[Any]:
        """Like :meth:`simulate`, but awaits coroutine callbacks."""
        out = self.simulate(action, **payload)
        resolved = []
        for item in out:
            if inspect.isawaitable(item):
                resolved.append(await item)
            else:
                resolved.append(item)
        return resolved

    def _schedule(self, awaitable: Any) -> Any:
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        if loop is None:
            return awaitable  # caller awaits it (simulate_async), or drops it
        if loop.is_running() and threading.current_thread() is not threading.main_thread():
            return asyncio.run_coroutine_threadsafe(awaitable, loop)
        return loop.create_task(awaitable)

    # -- real key binding -----------------------------------------------
    def start(self) -> None:
        """Bind the real global hotkeys.  Raises :class:`HotkeysUnavailable`."""
        keyboard = _import_pynput()
        if self._listener is not None:
            return
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
        mapping = {
            accel: (lambda a=action: self.simulate(a, source="hotkey"))
            for action, accel in self.bindings.items()
        }
        try:
            listener = keyboard.GlobalHotKeys(mapping)
            listener.start()
        except Exception as exc:  # pragma: no cover - needs a display
            raise HotkeysUnavailable(f"could not bind global hotkeys: {exc}") from exc
        self._listener = listener

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            finally:
                self._listener = None

    @property
    def running(self) -> bool:
        return self._listener is not None

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "action": a,
                "binding": self.bindings[a],
                "help": ACTION_HELP.get(a, ""),
                "handlers": str(len(self.callbacks(a))),
            }
            for a in self.bindings
        ]

    def __enter__(self) -> "HotkeyManager":
        try:
            self.start()
        except HotkeysUnavailable:
            log.warning("hotkeys unavailable; continuing without them")
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def build_manager(
    handlers: dict[str, Callback] | None = None,
    bindings: dict[str, str] | None = None,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> HotkeyManager:
    mgr = HotkeyManager(bindings, loop=loop)
    for action, cb in (handlers or {}).items():
        mgr.register(action, cb)
    return mgr


def unbound_actions(mgr: HotkeyManager) -> Iterable[str]:
    return (a for a in mgr.actions if not mgr.callbacks(a))


__all__ = [
    "DEFAULT_BINDINGS",
    "ACTION_HELP",
    "HotkeyEvent",
    "HotkeyManager",
    "HotkeysUnavailable",
    "available",
    "build_manager",
    "unbound_actions",
]
