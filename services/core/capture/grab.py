"""Screen capture primitives (spec section 6.1).

Four rules run through this file:

1.  **Nothing here needs a display to import.**  ``mss`` is imported lazily and
    every display-touching entry point raises :class:`DisplayUnavailableError`
    with an actionable message instead of blowing up with an X11 traceback.
    The same holds for the Windows layer: ``ctypes.windll`` is only ever
    touched behind :func:`_windll`, so this module imports on Linux and macOS.
2.  **DPI is recorded, never assumed.**  A capture carries the physical pixel
    rect *and* the scale factor, so a region saved on a 2x monitor replays
    correctly on a 1x monitor (:func:`replay_rect`).  The maths is pure and is
    tested without a display.
3.  **A full screen is never sent to a model.**  :func:`prepare` refuses to
    encode a whole image unless the caller explicitly opts in, and the only
    sanctioned whole-board crop is the deliberately low-res topology overview
    from :func:`board_overview`.
4.  **On Windows, DPI awareness is claimed before the first device context is
    opened.**  See :func:`ensure_dpi_awareness` -- getting this late is a
    *silent* corruption of every capture, not a crash.

Production target is Windows; development and CI happen on headless Linux.
Every Windows-only branch is therefore written against ``ctypes.windll`` looked
up at call time so it can be exercised with a fake (see
``tests/test_windows_capture.py``).  Behaviours that can only be confirmed on
real Windows hardware are marked ``UNVERIFIED ON WINDOWS`` in a comment right
next to the code.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from PIL import Image

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Budgets (spec 6.1).  Exported so tests and the practice harness agree.
# --------------------------------------------------------------------------

#: Anthropic/OpenAI vision models downsample above this; sending more is pure
#: latency with no accuracy gain.
MAX_LONG_EDGE = 1568
#: The topology call only needs the *shape* of the graph, not the numbers.
OVERVIEW_LONG_EDGE = 768

CAPTURE_BUDGET_MS = 30.0
PREPARE_BUDGET_MS = 40.0


class DisplayUnavailableError(RuntimeError):
    """Raised when screen capture is attempted without a usable display."""


class WindowCaptureUnsupported(NotImplementedError):
    """Raised by :func:`grab_window` where no compositor API is available."""


class WindowCaptureError(RuntimeError):
    """Base for per-window capture failures that are the *user's* to fix."""


class WindowNotFoundError(WindowCaptureError):
    """No window matched the requested title / handle."""


class WindowMinimizedError(WindowCaptureError):
    """The target window exists but is minimised, so it has no pixels."""


class FullscreenExclusiveError(RuntimeError):
    """A grab came back blank -- almost always DirectX exclusive fullscreen.

    GDI/BitBlt (which is what ``mss`` uses) cannot read the front buffer of a
    game that owns the display in exclusive fullscreen mode: the copy succeeds
    and returns black, or the previous frame.  There is no way to fix this from
    the capture side; the game has to be told to stop owning the display.
    """


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle.  ``x``/``y`` are the top-left corner."""

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w < 0 or self.h < 0:
            raise ValueError(f"negative extent in {self!r}")

    # -- conversions ----------------------------------------------------
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def as_box(self) -> tuple[int, int, int, int]:
        """PIL-style (left, upper, right, lower)."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_box(cls, box: Sequence[int]) -> "Rect":
        left, upper, right, lower = box
        return cls(int(left), int(upper), int(right - left), int(lower - upper))

    @classmethod
    def coerce(cls, value: "Rect | Sequence[int] | dict") -> "Rect":
        if isinstance(value, Rect):
            return value
        if isinstance(value, dict):
            return cls(int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"]))
        x, y, w, h = value
        return cls(int(x), int(y), int(w), int(h))

    # -- helpers --------------------------------------------------------
    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def long_edge(self) -> int:
        return max(self.w, self.h)

    def clamp(self, width: int, height: int) -> "Rect":
        """Clip against a ``width`` x ``height`` surface."""
        x = max(0, min(int(self.x), width))
        y = max(0, min(int(self.y), height))
        w = max(0, min(int(self.w), width - x))
        h = max(0, min(int(self.h), height - y))
        return Rect(x, y, w, h)

    def pad(self, px: int) -> "Rect":
        return Rect(self.x - px, self.y - px, self.w + 2 * px, self.h + 2 * px)


def scale_rect(rect: "Rect | Sequence[int] | dict", factor: float) -> Rect:
    """Scale a rect by ``factor``, rounding to whole pixels.

    Extents never collapse to zero for a non-empty input: a 1px wide rect stays
    1px wide when shrunk, because a zero-width crop is a crash, not a rect.
    """
    r = Rect.coerce(rect)
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    w = int(round(r.w * factor))
    h = int(round(r.h * factor))
    return Rect(
        int(round(r.x * factor)),
        int(round(r.y * factor)),
        max(1, w) if r.w else 0,
        max(1, h) if r.h else 0,
    )


def to_physical(rect: "Rect | Sequence[int] | dict", scale: float) -> Rect:
    """Logical (point) coordinates -> physical pixels on a ``scale``x monitor."""
    return scale_rect(rect, scale)


def to_logical(rect: "Rect | Sequence[int] | dict", scale: float) -> Rect:
    """Physical pixels -> logical (point) coordinates."""
    return scale_rect(rect, 1.0 / scale)


def replay_rect(
    rect: "Rect | Sequence[int] | dict", saved_scale: float, target_scale: float
) -> Rect:
    """Replay a rect recorded at ``saved_scale`` on a ``target_scale`` monitor.

    This is the whole point of storing the scale factor: a region box drawn on a
    2x retina panel is 2x too large when replayed against a 1x capture.
    """
    return scale_rect(rect, target_scale / saved_scale)


# --------------------------------------------------------------------------
# Windows platform layer
#
# Everything below reaches Win32 through _windll() rather than through a
# module-level ``ctypes.windll``, for two reasons: ``ctypes.windll`` does not
# exist off Windows (this module must import on Linux), and a test can install
# a fake with ``monkeypatch.setattr(ctypes, "windll", fake, raising=False)``.
# --------------------------------------------------------------------------


def _is_windows() -> bool:
    """True on Windows.  Read through ``sys.platform`` at *call* time so tests
    can flip the platform with ``monkeypatch.setattr(sys, "platform", "win32")``."""
    return sys.platform.startswith("win")


def _windll():
    """``ctypes.windll`` or ``None`` -- the single seam the Windows code uses."""
    return getattr(ctypes, "windll", None)


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


# -- DPI awareness ---------------------------------------------------------
#
#: ``DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`` (Windows 10 1703+).
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
#: ``PROCESS_PER_MONITOR_DPI_AWARE`` for shcore's SetProcessDpiAwareness (8.1+).
PROCESS_PER_MONITOR_DPI_AWARE = 2
#: Win32 ``ERROR_ACCESS_DENIED`` -- "awareness was already set for this process".
_ERROR_ACCESS_DENIED = 5
#: ``E_ACCESSDENIED`` as ctypes hands back an HRESULT (default restype c_int).
_E_ACCESSDENIED = -2147024891  # 0x80070005
#: DPI_AWARENESS enum values returned by GetAwarenessFromDpiAwarenessContext.
_AWARENESS_NAMES = {0: "unaware", 1: "system", 2: "per_monitor"}

#: Levels this module can report, best first.  "not_windows" is not a failure.
DPI_LEVELS = ("per_monitor_v2", "per_monitor", "system", "unaware", "not_windows")

_dpi_lock = threading.Lock()
_dpi_level: str | None = None


def _query_dpi_awareness() -> str | None:
    """Ask Windows what this process's awareness actually *is*, or ``None``.

    Only used when a Set* call reports "already configured": some other
    component (a Qt/Tk UI, a manifest, an injected overlay) may have claimed a
    *weaker* level first, and reporting the level we merely attempted would be
    a lie that hides virtualised coordinates on secondary monitors.

    Cannot distinguish per-monitor v1 from v2 -- both report DPI_AWARENESS 2 --
    so it answers "per_monitor" for either.
    """
    windll = _windll()
    if windll is None:
        return None
    try:
        get_ctx = windll.user32.GetThreadDpiAwarenessContext
        get_ctx.restype = ctypes.c_void_p
        ctx = get_ctx()
        from_ctx = windll.user32.GetAwarenessFromDpiAwarenessContext
        from_ctx.argtypes = [ctypes.c_void_p]
        from_ctx.restype = ctypes.c_int
        return _AWARENESS_NAMES.get(int(from_ctx(ctx)))
    except Exception:
        return None


def _last_error() -> int:
    windll = _windll()
    if windll is None:
        return 0
    try:
        # NOTE: the officially robust form is WinDLL(..., use_last_error=True) +
        # ctypes.get_last_error().  We read kernel32 directly because that keeps
        # the whole Windows layer behind the single ctypes.windll seam that the
        # tests fake.  It is read immediately after the failing call, which is
        # the case where the direct read is reliable in practice.
        # UNVERIFIED ON WINDOWS: the exact GetLastError value after a failed
        # SetProcessDpiAwarenessContext has not been observed on real hardware.
        return int(windll.kernel32.GetLastError())
    except Exception:
        return 0


def _apply_dpi_awareness() -> str:
    if not _is_windows():
        return "not_windows"
    windll = _windll()
    if windll is None:  # pragma: no cover - Windows without ctypes.windll
        return "unaware"

    # 1. Per-monitor v2 (Win10 1703+).  The only level where non-client area,
    #    dialogs and DPI-change messages are all handled sanely.
    try:
        fn = windll.user32.SetProcessDpiAwarenessContext
        # argtypes matter: DPI_AWARENESS_CONTEXT is a *handle*.  Left to ctypes'
        # default int conversion, -4 goes across as a 32-bit int and the upper
        # half of the 64-bit register is undefined.
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_bool
        if fn(ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)):
            return "per_monitor_v2"
        if _last_error() == _ERROR_ACCESS_DENIED:
            # Already set process-wide.  That is success, not failure: this API
            # is one-shot, and a second call can only ever fail.
            return _query_dpi_awareness() or "per_monitor_v2"
    except Exception:
        pass  # older Windows: the export does not exist

    # 2. shcore SetProcessDpiAwareness (Windows 8.1+).
    try:
        fn = windll.shcore.SetProcessDpiAwareness
        fn.argtypes = [ctypes.c_int]
        fn.restype = ctypes.c_long
        hr = int(fn(PROCESS_PER_MONITOR_DPI_AWARE))
        if hr == 0:
            return "per_monitor"
        if hr in (_E_ACCESSDENIED, _E_ACCESSDENIED & 0xFFFFFFFF):
            return _query_dpi_awareness() or "per_monitor"
    except Exception:
        pass

    # 3. SetProcessDPIAware (Vista+).  System-wide DPI only: correct on the
    #    primary monitor at login-time DPI and wrong everywhere else.
    try:
        fn = windll.user32.SetProcessDPIAware
        fn.argtypes = []
        fn.restype = ctypes.c_bool
        if fn():
            return "system"
        if _last_error() == _ERROR_ACCESS_DENIED:
            return _query_dpi_awareness() or "system"
    except Exception:
        pass

    return "unaware"


def ensure_dpi_awareness(*, force: bool = False) -> str:
    """Claim per-monitor DPI awareness for this process.  Returns the level.

    **Ordering constraint -- this is the important part.**  Windows freezes a
    process's DPI awareness the first time it is set, and every device context
    opened afterwards inherits the view of the desktop that was current *then*.
    A process that is not DPI-aware is shown a *virtualised* desktop: on a
    2560x1440 monitor at 150%, GDI reports 1707x960 and BitBlt hands back a
    blurry upscaled bitmap.  ``mss`` is GDI/BitBlt, so if this runs after mss
    has opened its device context, then every crop box the user drags, every
    tile rect and every stored fixture is wrong -- and wrong in the worst
    possible way, because it looks almost right.  Nothing raises.

    Hence: this module calls it at *import* time, before ``mss`` is even
    imported (mss is imported lazily inside :func:`_open_sct`), and
    :func:`_open_sct` calls it again as a cheap belt-and-braces.

    Returns one of :data:`DPI_LEVELS`.  Idempotent: the first result is cached
    and returned thereafter (``force=True`` re-runs it, for tests).

    "Already set" is success, not failure.  The call is process-wide and
    one-shot, so a second attempt -- ours or another library's -- reports
    ERROR_ACCESS_DENIED / E_ACCESSDENIED.  That means somebody already made the
    process aware, which is the outcome we wanted.

    Interaction with mss (checked against mss 10.2, ``mss/windows/gdi.py``):
    ``MSS.__init__`` itself calls ``shcore.SetProcessDpiAwareness(2)`` -- only
    per-monitor **v1** -- and then immediately ``GetWindowDC(0)``.  It ignores
    the HRESULT, so our earlier v2 claim makes mss's call a silent no-op rather
    than the other way round; that ordering is the one we want and it is why
    this runs at import.  It also means the bug this fixes is not "mss captures
    at the wrong resolution" -- mss would have rescued itself at the moment the
    first ``MSS()`` was constructed -- but everything that happens *before*
    that first construction: window rects, ``GetDpiForMonitor``, monitor
    enumeration, and any region-picker overlay the UI puts on screen.  Those
    would be virtualised, and a crop box computed from a virtualised rect is
    wrong against a physical-pixel screenshot.
    """
    global _dpi_level
    with _dpi_lock:
        if _dpi_level is not None and not force:
            return _dpi_level
        _dpi_level = _apply_dpi_awareness()
        if _dpi_level in ("unaware", "system"):
            # Loud, because the symptom otherwise is "the crops are subtly off"
            # rather than any kind of error.
            log.warning(
                "DPI awareness is %r: Windows is showing this process a virtualised "
                "desktop, so capture rects on scaled monitors will be wrong. Expected "
                "'per_monitor_v2'. Another component may have claimed awareness first.",
                _dpi_level,
            )
        return _dpi_level


def dpi_awareness_level() -> str | None:
    """The cached level, or ``None`` if :func:`ensure_dpi_awareness` never ran."""
    return _dpi_level


# -- per-monitor DPI -------------------------------------------------------

MONITOR_DEFAULTTONEAREST = 2
MDT_EFFECTIVE_DPI = 0
#: Windows' reference DPI: 96 dpi == 100% scaling.
BASE_DPI = 96.0

SCALE_ENV_VAR = "PUZZLE_COPILOT_DPI_SCALE"


def _env_scale() -> float | None:
    """The manual override, kept for machines (and VMs, and RDP) that lie."""
    raw = os.environ.get(SCALE_ENV_VAR)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _monitor_handle(windll, rect: "Rect | None"):
    """HMONITOR for ``rect`` (nearest monitor), or for the origin if ``None``."""
    # restype MUST be c_void_p: the default is c_int, which truncates a 64-bit
    # HMONITOR to 32 bits and then sign-extends it back into garbage.
    if rect is not None and (rect.w or rect.h):
        r = _RECT(
            int(rect.x),
            int(rect.y),
            int(rect.x + max(1, rect.w)),
            int(rect.y + max(1, rect.h)),
        )
        fn = windll.user32.MonitorFromRect
        fn.restype = ctypes.c_void_p
        return fn(ctypes.pointer(r), MONITOR_DEFAULTTONEAREST)
    fn = windll.user32.MonitorFromPoint
    fn.restype = ctypes.c_void_p
    return fn(_POINT(0, 0), MONITOR_DEFAULTTONEAREST)


def _windows_scale(rect: "Rect | None" = None) -> float | None:
    """Real scale factor for the monitor under ``rect`` (``None`` = origin).

    ``GetDpiForMonitor(MDT_EFFECTIVE_DPI)`` is the per-monitor number, which is
    the whole point: the product's target setup is a game on a 4K/150% panel
    next to the tool on a 1080p/100% panel, and one global scale is wrong for
    that machine by construction.  Falls back to the system DPI.
    """
    windll = _windll()
    if windll is None:
        return None
    try:
        hmon = _monitor_handle(windll, rect)
        dpi_x = ctypes.c_uint(0)
        dpi_y = ctypes.c_uint(0)
        fn = windll.shcore.GetDpiForMonitor
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        fn.restype = ctypes.c_long
        # ctypes.pointer, not byref: byref hands the callee a temporary the test
        # fake cannot write through, and this path has to be testable off Windows.
        hr = int(fn(hmon, MDT_EFFECTIVE_DPI, ctypes.pointer(dpi_x), ctypes.pointer(dpi_y)))
        if hr == 0 and dpi_x.value > 0:
            return dpi_x.value / BASE_DPI
    except Exception:
        pass  # shcore missing (pre-8.1), or the monitor went away mid-call
    try:
        # GetDpiForSystem is 1607+; it is the login-time system DPI, so it is
        # only right for the primary monitor.  Better than inventing 1.0.
        fn = windll.user32.GetDpiForSystem
        fn.restype = ctypes.c_uint
        dpi = int(fn())
        if dpi > 0:
            return dpi / BASE_DPI
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Monitors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorInfo:
    index: int
    bounds: Rect  #: physical pixels
    scale: float = 1.0
    name: str = ""
    is_primary: bool = False

    @property
    def logical_bounds(self) -> Rect:
        return to_logical(self.bounds, self.scale)

    @property
    def physical_size(self) -> tuple[int, int]:
        return (self.bounds.w, self.bounds.h)

    @property
    def logical_size(self) -> tuple[int, int]:
        lb = self.logical_bounds
        return (lb.w, lb.h)


def display_available() -> bool:
    """True when a screen grab could plausibly succeed.

    Cheap and side-effect free: used by the UI to grey out capture buttons and
    by tests to stay headless.
    """
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    if sys.platform == "win32" and not _windows_has_interactive_desktop():
        return False
    try:
        import mss  # noqa: F401
    except Exception:
        return False
    return True


def _windows_has_interactive_desktop() -> bool:
    """Is there a real, connected desktop to capture?

    ``mss`` imports fine under a service account or a disconnected RDP session
    and then hands back black frames or fails at grab time.  Session 0 is the
    non-interactive services session, and ``GetSystemMetrics(SM_REMOTESESSION)``
    plus a zero-sized virtual screen catch the disconnected-console case.  Being
    told "no desktop" beats being handed a black screenshot that looks real.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        session_id = ctypes.c_ulong()
        if kernel32.ProcessIdToSessionId(kernel32.GetCurrentProcessId(), ctypes.byref(session_id)):
            if session_id.value == 0:
                return False  # services session: no interactive desktop, ever
        SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
        if not (user32.GetSystemMetrics(SM_CXVIRTUALSCREEN) and
                user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)):
            return False
    except Exception:
        # An unexpected ctypes failure is not evidence of no desktop; let the
        # grab itself be the judge rather than disabling capture outright.
        return True
    return True


_NO_DISPLAY_MSG = (
    "No display available for screen capture. "
    "On Linux set DISPLAY (X11) or WAYLAND_DISPLAY, and run the tool on the "
    "machine with the monitors attached; headless containers cannot grab a "
    "screen. Use grab.capture_from_image()/the session store to replay recorded "
    "captures instead."
)


def _open_sct():
    if not display_available():
        raise DisplayUnavailableError(_NO_DISPLAY_MSG)
    # Belt and braces: DPI awareness must be claimed before mss opens a device
    # context.  It already ran at import time; this is idempotent and ~free.
    ensure_dpi_awareness()
    import mss  # local import: importing mss is cheap, opening it is not

    try:
        return mss.mss()
    except Exception as exc:  # pragma: no cover - needs a broken display
        raise DisplayUnavailableError(f"{_NO_DISPLAY_MSG} (mss said: {exc})") from exc


def _detect_scale(default: float = 1.0, *, rect: "Rect | None" = None) -> float:
    """DPI scale factor for the monitor under ``rect`` (``None`` = origin).

    Order: explicit env override, then the real per-monitor value from Windows,
    then ``default``.  The override wins on purpose -- RDP sessions, some VM
    guest drivers and a few docking stations report a DPI that is not the one
    the pixels were rendered at, and the user needs a way out.

    ``mss`` itself reports physical pixels only and knows nothing about scale,
    which is why this exists at all.
    """
    env = _env_scale()
    if env is not None:
        return env
    if _is_windows():
        scale = _windows_scale(rect)
        if scale:
            return scale
    return default


def list_monitors() -> list[MonitorInfo]:
    """Every monitor, index 1..n (index 0 is mss's virtual all-monitors rect).

    Each monitor carries *its own* scale.  A single global scale is wrong for
    the exact configuration this product exists for: the game on a 4K panel at
    150% and the tool's window on a 1080p panel at 100%.
    """
    with _open_sct() as sct:
        bounds_list = [
            Rect(int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"]))
            for m in sct.monitors
        ]
    out: list[MonitorInfo] = []
    for i, bounds in enumerate(bounds_list):
        if i == 0:
            # Index 0 is the union of every monitor; a "scale" for it is
            # meaningless, so report the primary's and let callers use 1..n.
            probe = bounds_list[1] if len(bounds_list) > 1 else None
        else:
            probe = bounds
        out.append(
            MonitorInfo(
                index=i,
                bounds=bounds,
                scale=_detect_scale(rect=probe),
                name="all" if i == 0 else f"monitor-{i}",
                is_primary=(i == 1),
            )
        )
    return out


# --------------------------------------------------------------------------
# Blank-frame detection (DirectX exclusive fullscreen)
# --------------------------------------------------------------------------

#: Marker pushed onto ``Capture.warnings``; also the key the UI switches on.
BLANK_FRAME_WARNING = "blank_frame_fullscreen_exclusive"

FULLSCREEN_EXCLUSIVE_MSG = (
    "The captured frame is blank (all black). This almost always means the game "
    "is running in FULLSCREEN EXCLUSIVE mode, which GDI screen capture cannot "
    "read -- it returns a black or stale frame. Fix: in the game's video "
    "settings switch the display mode to BORDERLESS WINDOWED (sometimes called "
    "'windowed fullscreen'); the capture then works with no other change. "
    "If the game is already borderless, check that the puzzle is actually on "
    "the monitor/region being captured and that no DRM-protected or "
    "'capture-excluded' overlay is covering it."
)

#: Channel value at or below which a sampled pixel counts as black.  Not 0:
#: a stale/undefined GDI back buffer often comes back as 1-4 rather than exact
#: zero, and video drivers dither.
BLANK_CHANNEL_THRESHOLD = 8


def looks_blank(
    image: Image.Image,
    *,
    threshold: int = BLANK_CHANNEL_THRESHOLD,
    grid: int = 32,
    min_dark_fraction: float = 0.995,
) -> bool:
    """True when ``image`` is (near-)uniformly black.  Pure, and cheap.

    Samples a ``grid`` x ``grid`` lattice -- 1024 pixels by default, constant
    regardless of image size -- through PIL's ``PixelAccess`` object.  Scanning
    all 8.3M pixels of a 4K frame would cost more than the grab it is checking,
    and would be run on every capture.

    ``min_dark_fraction`` leaves room for a mouse cursor, a hardware overlay or
    a notification toast sitting on top of an otherwise black frame.

    Not a "is this a useful screenshot" test: a legitimately black region (a
    letterboxed crop, an unlit panel) also reads as blank.  That is why the
    result is a warning on the capture and not a hard failure by default.
    """
    w, h = image.width, image.height
    if w <= 0 or h <= 0:
        return True
    px = image.load()
    if px is None:  # pragma: no cover - PIL always returns an accessor here
        return False
    nx = max(2, min(grid, w))
    ny = max(2, min(grid, h))
    total = nx * ny
    needed = total * min_dark_fraction
    # A non-blank frame bails out after a handful of samples; only a genuinely
    # black frame pays for the full lattice (~0.5 ms on 4K, measured on Linux).
    allowed_light = total - needed
    light = 0
    dark = 0
    for gy in range(ny):
        y = min(h - 1, (2 * gy + 1) * h // (2 * ny))
        for gx in range(nx):
            x = min(w - 1, (2 * gx + 1) * w // (2 * nx))
            value = px[x, y]
            if isinstance(value, tuple):
                # Ignore alpha: a black frame arrives as (0,0,0,255) from BitBlt
                # and as (0,0,0,0) from some capture paths.
                level = max(value[:3]) if len(value) >= 3 else max(value)
            else:
                level = value
            if level <= threshold:
                dark += 1
            else:
                light += 1
                if light > allowed_light:
                    return False
    return dark >= needed


def check_not_blank(capture: "Capture") -> "Capture":
    """Raise :class:`FullscreenExclusiveError` if ``capture`` came back black."""
    return capture.raise_if_blank()


# --------------------------------------------------------------------------
# Captures
# --------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Capture:
    """One grabbed image plus everything needed to replay or crop it later."""

    image: Image.Image
    region: Rect
    capture_id: str = field(default_factory=lambda: f"cap_{uuid.uuid4().hex[:12]}")
    session_id: str = "default"
    monitor_index: int = 0
    scale: float = 1.0
    captured_at: str = field(default_factory=_now_iso)
    puzzle_type: str | None = None
    grab_ms: float = 0.0
    #: Sub-regions to fan out over (filled by the region picker or a fixture).
    tiles: list["Tile"] = field(default_factory=list)
    source: str = "screen"
    #: Non-fatal problems found while grabbing, e.g. :data:`BLANK_FRAME_WARNING`.
    #: The store still writes the PNG -- seeing the black frame is how a user
    #: understands what happened -- so the warning travels on the object.
    warnings: list[str] = field(default_factory=list)
    #: Window title / handle when this came from :func:`grab_window`.
    window: str | None = None

    @property
    def physical_size(self) -> tuple[int, int]:
        return (self.image.width, self.image.height)

    @property
    def is_blank(self) -> bool:
        return BLANK_FRAME_WARNING in self.warnings

    def raise_if_blank(self) -> "Capture":
        """Turn a blank-frame warning into :class:`FullscreenExclusiveError`.

        For callers that would rather fail loudly than hand a black frame to an
        extractor and pay for the tokens.
        """
        if self.is_blank:
            raise FullscreenExclusiveError(FULLSCREEN_EXCLUSIVE_MSG)
        return self

    @property
    def logical_region(self) -> Rect:
        return to_logical(self.region, self.scale)

    def crop(self, box: "Rect | Sequence[int] | dict") -> Image.Image:
        r = Rect.coerce(box).clamp(self.image.width, self.image.height)
        return self.image.crop(r.as_box())

    def png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return buf.getvalue()


def capture_from_image(
    image: Image.Image,
    *,
    region: "Rect | Sequence[int] | dict | None" = None,
    scale: float = 1.0,
    monitor_index: int = 0,
    session_id: str = "default",
    puzzle_type: str | None = None,
    capture_id: str | None = None,
    tiles: Sequence["Tile"] | None = None,
    source: str = "file",
    check_blank: bool = True,
) -> Capture:
    """Wrap an in-memory image as a :class:`Capture` (fixtures, replay, tests).

    The blank-frame check runs here too, not only on the live grab path: an
    uploaded or replayed black PNG is exactly as useless to extract from, and
    this is the single funnel every not-grabbed-just-now image passes through.
    """
    rect = Rect.coerce(region) if region is not None else Rect(0, 0, image.width, image.height)
    kwargs = {} if capture_id is None else {"capture_id": capture_id}
    cap = Capture(
        image=image,
        region=rect,
        scale=scale,
        monitor_index=monitor_index,
        session_id=session_id,
        puzzle_type=puzzle_type,
        tiles=list(tiles or []),
        source=source,
        **kwargs,
    )
    if check_blank and looks_blank(image):
        cap.warnings.append(BLANK_FRAME_WARNING)
    return cap


def _grab_box(
    box: dict[str, int],
    monitor_index: int,
    scale: float,
    *,
    source: str = "screen",
    raise_on_blank: bool = False,
) -> Capture:
    t0 = time.perf_counter()
    with _open_sct() as sct:
        shot = sct.grab(box)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    ms = (time.perf_counter() - t0) * 1000.0
    cap = Capture(
        image=image,
        region=Rect(box["left"], box["top"], box["width"], box["height"]),
        monitor_index=monitor_index,
        scale=scale,
        grab_ms=ms,
        source=source,
    )
    # A black frame is the signature of DirectX exclusive fullscreen.  Flag it
    # here, once, on the single path every screen grab goes through.
    if looks_blank(cap.image):
        cap.warnings.append(BLANK_FRAME_WARNING)
        if raise_on_blank:
            raise FullscreenExclusiveError(FULLSCREEN_EXCLUSIVE_MSG)
    return cap


def grab_monitor(
    index: int = 1, *, session_id: str = "default", raise_on_blank: bool = False
) -> Capture:
    """Grab a whole monitor.  ``index`` matches :func:`list_monitors`."""
    with _open_sct() as sct:
        if index < 0 or index >= len(sct.monitors):
            raise ValueError(f"no monitor {index}; have 0..{len(sct.monitors) - 1}")
        mon = dict(sct.monitors[index])
    box = {
        "left": int(mon["left"]),
        "top": int(mon["top"]),
        "width": int(mon["width"]),
        "height": int(mon["height"]),
    }
    cap = _grab_box(
        box,
        index,
        # This monitor's own scale, not a global one.
        _detect_scale(rect=Rect(box["left"], box["top"], box["width"], box["height"])),
        raise_on_blank=raise_on_blank,
    )
    cap.session_id = session_id
    return cap


def grab_region(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    monitor_index: int = 1,
    session_id: str = "default",
    source: str = "screen",
    raise_on_blank: bool = False,
) -> Capture:
    """Grab an arbitrary physical-pixel rect of the virtual desktop."""
    if w <= 0 or h <= 0:
        raise ValueError("region must have positive width and height")
    rect = Rect(int(x), int(y), int(w), int(h))
    cap = _grab_box(
        {"left": rect.x, "top": rect.y, "width": rect.w, "height": rect.h},
        monitor_index,
        # Scale of the monitor the region actually lands on -- on a mixed-DPI
        # desktop the answer differs per region, which is the whole point.
        _detect_scale(rect=rect),
        source=source,
        raise_on_blank=raise_on_blank,
    )
    cap.session_id = session_id
    return cap


# -- per-window capture ----------------------------------------------------

#: ``DWMWA_EXTENDED_FRAME_BOUNDS`` for DwmGetWindowAttribute.
DWMWA_EXTENDED_FRAME_BOUNDS = 9


def _win_text(windll, hwnd: int) -> str:
    length = int(windll.user32.GetWindowTextLengthW(ctypes.c_void_p(hwnd)) or 0)
    buf = ctypes.create_unicode_buffer(max(2, length + 1))
    windll.user32.GetWindowTextW(ctypes.c_void_p(hwnd), buf, len(buf))
    return buf.value


def _enum_windows(windll) -> list[tuple[int, str]]:
    """Every visible top-level window as ``(hwnd, title)``."""
    found: list[tuple[int, str]] = []
    # ctypes.WINFUNCTYPE only exists on Windows; CFUNCTYPE keeps this importable
    # (and fake-callable) on Linux.  On x64 the two are the same convention.
    factory = getattr(ctypes, "WINFUNCTYPE", None) or ctypes.CFUNCTYPE
    proto = factory(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _cb(hwnd, _lparam):
        h = int(hwnd or 0)
        if h and windll.user32.IsWindowVisible(ctypes.c_void_p(h)):
            title = _win_text(windll, h)
            if title:
                found.append((h, title))
        return True

    windll.user32.EnumWindows(proto(_cb), 0)
    return found


def _find_window(title: str | None, window_id: int | None) -> tuple[int, str]:
    """Resolve a window handle.  ``title=None`` means the foreground window."""
    windll = _windll()
    if windll is None:  # pragma: no cover - Windows without ctypes.windll
        raise WindowCaptureUnsupported("ctypes.windll is unavailable on this Python build")

    if window_id:
        hwnd = int(window_id)
        if not windll.user32.IsWindow(ctypes.c_void_p(hwnd)):
            raise WindowNotFoundError(
                f"window handle {hwnd} no longer exists. Re-pick the window "
                "(the game may have been restarted, which changes its handle)."
            )
        return hwnd, _win_text(windll, hwnd)

    if title is None:
        fn = windll.user32.GetForegroundWindow
        fn.restype = ctypes.c_void_p
        hwnd = int(fn() or 0)
        if not hwnd:
            raise WindowNotFoundError(
                "no foreground window to capture. Click the game window first, or "
                "pass a title, or use grab_monitor()/grab_region(). (A locked "
                "workstation or a secure-desktop UAC prompt also looks like this.)"
            )
        return hwnd, _win_text(windll, hwnd)

    fn = windll.user32.FindWindowW
    fn.restype = ctypes.c_void_p
    hwnd = int(fn(None, ctypes.c_wchar_p(title)) or 0)
    if hwnd:
        return hwnd, title

    # Exact match failed -- games append " - DirectX 12", a level name, an FPS
    # counter.  Fall back to a case-insensitive substring scan.
    needle = title.casefold()
    matches = [(h, t) for h, t in _enum_windows(windll) if needle in t.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise WindowNotFoundError(
            f"no visible window matching {title!r}. Check the exact title in the "
            "task bar (the match is case-insensitive and partial), make sure the "
            "game is running and not minimised, or use grab_region() instead."
        )
    listed = ", ".join(repr(t) for _, t in matches[:6])
    raise WindowNotFoundError(
        f"{len(matches)} windows match {title!r}: {listed}. Pass a longer, more "
        "specific title."
    )


def _window_rect(hwnd: int) -> Rect:
    """Physical-pixel rect of ``hwnd``'s *visible* frame.

    ``GetWindowRect`` is the wrong API on any DWM-composited system (Vista+,
    i.e. all of them): it returns the window rect *including* the invisible
    resize border DWM leaves around the frame -- typically ~7px left/right/
    bottom on a default theme, and more on high-DPI.  Capturing that rect grabs
    a few pixels of whatever is behind the game on every edge, on every capture.
    ``DWMWA_EXTENDED_FRAME_BOUNDS`` is the actual painted frame.
    """
    windll = _windll()
    if windll is None:  # pragma: no cover
        raise WindowCaptureUnsupported("ctypes.windll is unavailable on this Python build")

    if windll.user32.IsIconic(ctypes.c_void_p(hwnd)):
        raise WindowMinimizedError(
            "the target window is minimised, so it has no pixels to capture "
            "(Windows does not keep a back buffer for minimised windows). "
            "Restore the window and capture again."
        )

    rect = _RECT()
    try:
        fn = windll.dwmapi.DwmGetWindowAttribute
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        fn.restype = ctypes.c_long
        hr = int(
            fn(
                ctypes.c_void_p(hwnd),
                DWMWA_EXTENDED_FRAME_BOUNDS,
                ctypes.pointer(rect),
                ctypes.sizeof(rect),
            )
        )
    except Exception:
        hr = -1
    if hr != 0:
        # Composition off (a Server Core / RDP session with DWM disabled) or a
        # window DWM does not know about: fall back, accepting the border.
        rect = _RECT()
        windll.user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.pointer(rect))

    r = Rect(
        int(rect.left),
        int(rect.top),
        int(rect.right) - int(rect.left),
        int(rect.bottom) - int(rect.top),
    )
    if r.w <= 0 or r.h <= 0:
        raise WindowMinimizedError(
            f"the target window reports an empty rect {r.as_tuple()} -- it is "
            "minimised, collapsed or being animated. Restore it and try again."
        )
    return r


def grab_window(
    title: str | None = None,
    *,
    window_id: int | None = None,
    session_id: str = "default",
    raise_on_blank: bool = False,
) -> Capture:
    """Grab a single window.  ``title=None`` grabs the foreground window.

    Windows: resolves the handle (``GetForegroundWindow``, ``FindWindowW``, then
    an ``EnumWindows``/``GetWindowTextW`` substring scan), takes the DWM
    extended frame bounds, and then hands the rect to :func:`grab_region` so
    that all the DPI/scale bookkeeping stays on one path.

    Deliberately unimplemented on Linux: without a compositor API (XComposite,
    or a portal on Wayland) the only way to "grab a window" is to guess a rect,
    and a guessed rect silently crops half a machine panel.  A clear failure
    beats a wrong crop.

    UNVERIFIED ON WINDOWS: exercised only against a faked ``ctypes.windll``.
    """
    if _is_windows():
        ensure_dpi_awareness()  # window rects are virtualised without it
        hwnd, found_title = _find_window(title, window_id)
        rect = _window_rect(hwnd)
        cap = grab_region(
            rect.x,
            rect.y,
            rect.w,
            rect.h,
            session_id=session_id,
            source="window",
            raise_on_blank=raise_on_blank,
        )
        cap.window = found_title or (title or "")
        return cap

    if sys.platform.startswith("linux"):
        raise WindowCaptureUnsupported(
            "Per-window capture is not implemented on Linux: X11/Wayland expose no "
            "compositor API here, and guessing the window rect would silently crop "
            "the panel. Use grab_region(x, y, w, h) with a saved region instead "
            f"(requested title={title!r}, window_id={window_id!r})."
        )
    raise WindowCaptureUnsupported(
        f"Per-window capture is not implemented on {sys.platform} yet; "
        "use grab_monitor()/grab_region()."
    )


# --------------------------------------------------------------------------
# Crop + downscale: the hot path
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedImage:
    """A crop, downscaled and encoded, ready to hand to a vision model."""

    data_url: str
    width: int
    height: int
    scale: float  #: applied downscale (1.0 = untouched)
    source_box: Rect
    n_bytes: int
    crop_id: str
    elapsed_ms: float = 0.0

    def as_provenance(self) -> dict[str, object]:
        return {
            "crop_id": self.crop_id,
            "box": self.source_box.as_dict(),
            "scale": self.scale,
            "size": [self.width, self.height],
        }


def _encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def prepare(
    image: Image.Image,
    box: "Rect | Sequence[int] | dict | None" = None,
    *,
    max_long_edge: int = MAX_LONG_EDGE,
    allow_full: bool = False,
) -> PreparedImage:
    """Crop to ``box`` and downscale so the long edge is at most ``max_long_edge``.

    LANCZOS is worth its cost here: nearest/bilinear turn a 6 into an 8 at the
    stroke widths these panels use.  Budget: ~40 ms for a 4K source.
    """
    t0 = time.perf_counter()
    if box is None:
        if not allow_full and max(image.width, image.height) > max_long_edge:
            raise ValueError(
                "refusing to prepare a full screen: pass an explicit box (per-machine "
                "tile) or use board_overview() for the low-res topology crop. "
                "Set allow_full=True only for images that are already a crop."
            )
        rect = Rect(0, 0, image.width, image.height)
        cropped = image
    else:
        rect = Rect.coerce(box).clamp(image.width, image.height)
        if rect.w <= 0 or rect.h <= 0:
            raise ValueError(f"empty crop box {rect!r} against {image.size}")
        cropped = image.crop(rect.as_box())

    long_edge = max(cropped.width, cropped.height)
    scale = 1.0
    if long_edge > max_long_edge:
        scale = max_long_edge / float(long_edge)
        new_size = (
            max(1, int(round(cropped.width * scale))),
            max(1, int(round(cropped.height * scale))),
        )
        cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
    if cropped.mode not in ("RGB", "L"):
        cropped = cropped.convert("RGB")

    raw = _encode_png(cropped)
    b64 = base64.b64encode(raw).decode("ascii")
    ms = (time.perf_counter() - t0) * 1000.0
    return PreparedImage(
        data_url=f"data:image/png;base64,{b64}",
        width=cropped.width,
        height=cropped.height,
        scale=scale,
        source_box=rect,
        n_bytes=len(raw),
        crop_id="crop_" + hashlib.sha256(raw).hexdigest()[:16],
        elapsed_ms=ms,
    )


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """A named sub-region plus the extraction target it feeds."""

    name: str
    box: Rect
    target: str = "factory_machine"
    max_long_edge: int = MAX_LONG_EDGE

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "box": self.box.as_dict(),
            "target": self.target,
            "max_long_edge": self.max_long_edge,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tile":
        return cls(
            name=str(d["name"]),
            box=Rect.coerce(d["box"]),
            target=str(d.get("target", "factory_machine")),
            max_long_edge=int(d.get("max_long_edge", MAX_LONG_EDGE)),
        )


def tile_regions(
    boxes: Iterable["Rect | Sequence[int] | dict"],
    *,
    names: Sequence[str] | None = None,
    target: str = "factory_machine",
    pad: int = 0,
    bounds: tuple[int, int] | None = None,
    max_long_edge: int = MAX_LONG_EDGE,
) -> list[Tile]:
    """Turn a list of per-machine boxes into tiles.

    N small crops read concurrently beat one big crop read serially, and a crop
    that contains exactly one panel cannot bleed a neighbour's numbers into the
    answer.
    """
    rects = [Rect.coerce(b) for b in boxes]
    if pad:
        rects = [r.pad(pad) for r in rects]
    if bounds:
        rects = [r.clamp(bounds[0], bounds[1]) for r in rects]
    out: list[Tile] = []
    for i, r in enumerate(rects):
        name = names[i] if names and i < len(names) else f"tile{i + 1}"
        out.append(Tile(name=name, box=r, target=target, max_long_edge=max_long_edge))
    return out


def board_overview(
    size: tuple[int, int] | Image.Image,
    *,
    box: "Rect | Sequence[int] | dict | None" = None,
    name: str = "topology",
    target: str = "factory_topology",
    max_long_edge: int = OVERVIEW_LONG_EDGE,
) -> Tile:
    """The one whole-board crop that is allowed, at deliberately low resolution.

    It exists to read *connections* (which box points at which box), never
    numbers -- so it is cheap, and it is the only call that sees the whole board.
    """
    if isinstance(size, Image.Image):
        size = (size.width, size.height)
    rect = Rect.coerce(box) if box is not None else Rect(0, 0, int(size[0]), int(size[1]))
    return Tile(name=name, box=rect, target=target, max_long_edge=max_long_edge)


def grid_tiles(
    size: tuple[int, int],
    cols: int,
    rows: int,
    *,
    target: str = "factory_machine",
    overlap: int = 0,
) -> list[Tile]:
    """Fallback tiling when no per-machine boxes are known yet."""
    w, h = size
    tw, th = w // cols, h // rows
    boxes = []
    for r in range(rows):
        for c in range(cols):
            boxes.append(
                Rect(
                    max(0, c * tw - overlap),
                    max(0, r * th - overlap),
                    min(w, tw + 2 * overlap),
                    min(h, th + 2 * overlap),
                )
            )
    names = [f"r{r + 1}c{c + 1}" for r in range(rows) for c in range(cols)]
    return tile_regions(boxes, names=names, target=target, bounds=size)


def crop_tiles(
    image: Image.Image, tiles: Sequence[Tile]
) -> list[tuple[Tile, PreparedImage]]:
    """Prepare every tile.  Pure CPU; the caller decides about threads."""
    return [(t, prepare(image, t.box, max_long_edge=t.max_long_edge)) for t in tiles]


# --------------------------------------------------------------------------
# Import-time side effect, and the only one in this module.
#
# DPI awareness is process-wide, one-shot, and must be claimed before anything
# opens a device context.  ``mss`` is imported lazily inside _open_sct(), which
# cannot run before this line, so importing this module is early enough --
# provided nothing else in the process claimed a weaker level first.  If the
# app ever grows a GUI toolkit that sets awareness on import (Qt, Tk, wx),
# import this module *before* it, or set awareness in the app manifest.
# --------------------------------------------------------------------------
try:
    ensure_dpi_awareness()
except Exception:  # pragma: no cover - must never break `import grab`
    pass


__all__ = [
    "MAX_LONG_EDGE",
    "OVERVIEW_LONG_EDGE",
    "CAPTURE_BUDGET_MS",
    "PREPARE_BUDGET_MS",
    "BLANK_FRAME_WARNING",
    "BLANK_CHANNEL_THRESHOLD",
    "FULLSCREEN_EXCLUSIVE_MSG",
    "DPI_LEVELS",
    "SCALE_ENV_VAR",
    "DisplayUnavailableError",
    "WindowCaptureUnsupported",
    "WindowCaptureError",
    "WindowNotFoundError",
    "WindowMinimizedError",
    "FullscreenExclusiveError",
    "Rect",
    "MonitorInfo",
    "Capture",
    "PreparedImage",
    "Tile",
    "scale_rect",
    "to_physical",
    "to_logical",
    "replay_rect",
    "display_available",
    "ensure_dpi_awareness",
    "dpi_awareness_level",
    "looks_blank",
    "check_not_blank",
    "list_monitors",
    "grab_monitor",
    "grab_region",
    "grab_window",
    "capture_from_image",
    "prepare",
    "tile_regions",
    "board_overview",
    "grid_tiles",
    "crop_tiles",
]
