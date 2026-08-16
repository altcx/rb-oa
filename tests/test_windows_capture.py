"""Windows screen-capture paths, exercised on Linux with a faked Win32 layer.

Production is Windows-only; development and CI are headless Linux.  Every
Windows branch in ``services.core.capture.grab`` therefore reaches the OS
through exactly two seams -- ``sys.platform`` and ``ctypes.windll``, both read
at call time -- and this module fakes both.

What that does and does not buy: these tests prove the *call sequence, argument
marshalling, fallback order and error handling* are what we intended.  They
cannot prove Windows agrees.  Behaviours that need real hardware are listed in
the README under "unverified on Windows"; the fake returns what the Microsoft
documentation says the real API returns.
"""

from __future__ import annotations

import ctypes
import sys
import time
from unittest import mock

import pytest
from PIL import Image

from services.core.capture import grab, hotkeys
from services.core.capture.store import CaptureStore

# --------------------------------------------------------------------------
# A fake ctypes.windll
# --------------------------------------------------------------------------


def fn(result=None, side_effect=None) -> mock.MagicMock:
    """One fake exported function.  Attribute-settable, so production code can
    assign ``.argtypes`` / ``.restype`` on it exactly as it does for real."""
    m = mock.MagicMock()
    if side_effect is not None:
        m.side_effect = side_effect
    else:
        m.return_value = result
    return m


class FakeLib:
    """A fake ``WinDLL``.  A missing export raises ``AttributeError`` -- which
    is precisely how ctypes reports "this API does not exist on this Windows
    version", and is the signal our fallback chain is built on."""

    def __init__(self, **funcs: mock.MagicMock) -> None:
        object.__setattr__(self, "_funcs", dict(funcs))

    def __getattr__(self, name: str) -> mock.MagicMock:
        funcs = object.__getattribute__(self, "_funcs")
        if name in funcs:
            return funcs[name]
        raise AttributeError(f"undefined export {name!r}")


class FakeWinDLL:
    """A fake ``ctypes.windll``.  A missing *library* also raises."""

    def __init__(self, **libs: FakeLib) -> None:
        object.__setattr__(self, "_libs", dict(libs))

    def __getattr__(self, name: str) -> FakeLib:
        libs = object.__getattribute__(self, "_libs")
        if name in libs:
            return libs[name]
        raise AttributeError(f"no library {name!r}")


def as_windows(monkeypatch, windll: FakeWinDLL) -> None:
    """Make the module believe it is on Windows, talking to ``windll``."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)


@pytest.fixture(autouse=True)
def _restore_dpi_cache():
    """``ensure_dpi_awareness`` caches process-wide; do not leak a fake result
    into the rest of the suite."""
    before = grab._dpi_level
    yield
    grab._dpi_level = before


@pytest.fixture(autouse=True)
def _no_env_scale(monkeypatch):
    monkeypatch.delenv(grab.SCALE_ENV_VAR, raising=False)


def kernel32(last_error: int = 0) -> FakeLib:
    return FakeLib(GetLastError=fn(last_error))


# --------------------------------------------------------------------------
# 1. DPI awareness
# --------------------------------------------------------------------------


def test_per_monitor_v2_is_tried_first_and_wins():
    """The good path: one call, to the best API, with a real handle-sized arg."""
    v2 = fn(True)
    shcore_set = fn(0)
    windll = FakeWinDLL(
        user32=FakeLib(SetProcessDpiAwarenessContext=v2, SetProcessDPIAware=fn(True)),
        shcore=FakeLib(SetProcessDpiAwareness=shcore_set),
        kernel32=kernel32(),
    )
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        assert grab.ensure_dpi_awareness(force=True) == "per_monitor_v2"

    assert v2.call_count == 1
    arg = v2.call_args[0][0]
    # DPI_AWARENESS_CONTEXT is a handle: it must go across as a pointer-sized
    # value, not as a 32-bit int that leaves the top half of RCX undefined.
    assert isinstance(arg, ctypes.c_void_p)
    assert arg.value == ctypes.c_void_p(grab.DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2).value
    # The weaker APIs are never touched once v2 succeeds.
    assert shcore_set.call_count == 0


def test_falls_back_to_shcore_when_v2_export_is_missing():
    """Windows 8.1 / early 10: SetProcessDpiAwarenessContext does not exist."""
    shcore_set = fn(0)  # S_OK
    windll = FakeWinDLL(
        user32=FakeLib(SetProcessDPIAware=fn(True)),  # no ...Context export
        shcore=FakeLib(SetProcessDpiAwareness=shcore_set),
        kernel32=kernel32(),
    )
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        assert grab.ensure_dpi_awareness(force=True) == "per_monitor"
    shcore_set.assert_called_once_with(grab.PROCESS_PER_MONITOR_DPI_AWARE)


def test_falls_back_to_system_dpi_when_shcore_is_absent():
    """Windows 7/8: no shcore at all, only the ancient system-wide call."""
    legacy = fn(True)
    windll = FakeWinDLL(user32=FakeLib(SetProcessDPIAware=legacy), kernel32=kernel32())
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        assert grab.ensure_dpi_awareness(force=True) == "system"
    assert legacy.call_count == 1


def test_every_api_failing_reports_unaware_rather_than_raising():
    """Awareness is best-effort: capture must still run, just flagged honestly."""
    windll = FakeWinDLL(
        user32=FakeLib(
            SetProcessDpiAwarenessContext=fn(False),
            SetProcessDPIAware=fn(False),
        ),
        shcore=FakeLib(SetProcessDpiAwareness=fn(-2147024809)),  # E_INVALIDARG
        kernel32=kernel32(last_error=0),
    )
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        assert grab.ensure_dpi_awareness(force=True) == "unaware"


def test_already_set_is_success_and_reports_the_real_level():
    """ERROR_ACCESS_DENIED from the one-shot API means somebody already made
    the process aware.  That is the outcome we wanted, not a failure -- and we
    report what the process *actually* is, not what we asked for."""
    shcore_set = fn(0)
    windll = FakeWinDLL(
        user32=FakeLib(
            SetProcessDpiAwarenessContext=fn(False),
            GetThreadDpiAwarenessContext=fn(0xABCD),
            GetAwarenessFromDpiAwarenessContext=fn(2),  # DPI_AWARENESS_PER_MONITOR
            SetProcessDPIAware=fn(True),
        ),
        shcore=FakeLib(SetProcessDpiAwareness=shcore_set),
        kernel32=kernel32(last_error=grab._ERROR_ACCESS_DENIED),
    )
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        level = grab.ensure_dpi_awareness(force=True)

    assert level == "per_monitor"
    assert level != "unaware"  # the point: not treated as a failure
    assert shcore_set.call_count == 0  # and we stop, instead of retrying weaker APIs


def test_already_set_without_a_query_api_still_counts_as_success():
    """Pre-1607 has no GetAwarenessFromDpiAwarenessContext to ask."""
    windll = FakeWinDLL(
        user32=FakeLib(SetProcessDpiAwarenessContext=fn(False)),
        kernel32=kernel32(last_error=grab._ERROR_ACCESS_DENIED),
    )
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        assert grab.ensure_dpi_awareness(force=True) == "per_monitor_v2"


def test_shcore_e_accessdenied_is_success_too():
    windll = FakeWinDLL(
        user32=FakeLib(SetProcessDPIAware=fn(True)),
        shcore=FakeLib(SetProcessDpiAwareness=fn(grab._E_ACCESSDENIED)),
        kernel32=kernel32(),
    )
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        assert grab.ensure_dpi_awareness(force=True) == "per_monitor"


def test_ensure_dpi_awareness_is_idempotent():
    """Process-wide and one-shot: calling twice must not call Windows twice."""
    v2 = fn(True)
    windll = FakeWinDLL(user32=FakeLib(SetProcessDpiAwarenessContext=v2), kernel32=kernel32())
    with pytest.MonkeyPatch.context() as mp:
        as_windows(mp, windll)
        first = grab.ensure_dpi_awareness(force=True)
        again = grab.ensure_dpi_awareness()
        third = grab.ensure_dpi_awareness()

    assert first == again == third == "per_monitor_v2"
    assert v2.call_count == 1
    assert grab.dpi_awareness_level() == "per_monitor_v2"


def test_non_windows_reports_not_windows_and_touches_nothing():
    assert grab.ensure_dpi_awareness(force=True) == "not_windows"
    assert grab.ensure_dpi_awareness() == "not_windows"


# --------------------------------------------------------------------------
# 2. Per-monitor DPI
# --------------------------------------------------------------------------

#: The configuration this product exists for: game on a 4K/150% panel, tool on
#: a 1080p/100% panel to its right.
MIXED_DPI_MONITORS = [
    {"left": 0, "top": 0, "width": 5760, "height": 2160},  # index 0: the union
    {"left": 0, "top": 0, "width": 3840, "height": 2160},  # 4K @ 150%
    {"left": 3840, "top": 0, "width": 1920, "height": 1080},  # 1080p @ 100%
]

HMON_4K = 0x1001
HMON_1080 = 0x1002
DPI_BY_HMON = {HMON_4K: 144, HMON_1080: 96}


class FakeSct:
    """Just enough of an ``mss`` instance for :func:`grab.list_monitors`."""

    def __init__(self, monitors, shot=None) -> None:
        self.monitors = monitors
        self._shot = shot

    def grab(self, box):
        return self._shot

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def monitor_from_rect(ptr, flags):
    rect = ptr.contents  # ctypes.pointer, so the fake can read it back
    return HMON_1080 if rect.left >= 3840 else HMON_4K


def get_dpi_for_monitor(hmon, mdt, dpi_x, dpi_y):
    assert mdt == grab.MDT_EFFECTIVE_DPI
    dpi_x.contents.value = DPI_BY_HMON[hmon]
    dpi_y.contents.value = DPI_BY_HMON[hmon]
    return 0  # S_OK


def dpi_windll(**overrides) -> FakeWinDLL:
    user32 = {
        "MonitorFromRect": fn(side_effect=monitor_from_rect),
        "MonitorFromPoint": fn(HMON_4K),
        "GetDpiForSystem": fn(96),
        "SetProcessDpiAwarenessContext": fn(True),
    }
    shcore = {"GetDpiForMonitor": fn(side_effect=get_dpi_for_monitor)}
    user32.update(overrides.pop("user32", {}))
    shcore.update(overrides.pop("shcore", {}))
    libs = {"user32": FakeLib(**user32), "kernel32": kernel32()}
    if shcore:
        libs["shcore"] = FakeLib(**shcore)
    libs.update(overrides)
    return FakeWinDLL(**libs)


def test_each_monitor_reports_its_own_scale(monkeypatch):
    windll = dpi_windll()
    monkeypatch.setattr(grab, "_open_sct", lambda: FakeSct(MIXED_DPI_MONITORS))
    as_windows(monkeypatch, windll)

    mons = grab.list_monitors()
    assert [m.scale for m in mons] == [1.5, 1.5, 1.0]
    # ... and the derived geometry follows per monitor, which is the payoff:
    assert mons[1].physical_size == (3840, 2160)
    assert mons[1].logical_size == (2560, 1440)  # 150% panel
    assert mons[2].logical_size == (1920, 1080)  # 100% panel
    assert mons[0].scale == mons[1].scale  # the union rect borrows the primary's


def test_env_var_still_overrides_the_real_dpi(monkeypatch):
    """Kept for RDP sessions, VM guest drivers and docks that report nonsense."""
    windll = dpi_windll()
    monkeypatch.setattr(grab, "_open_sct", lambda: FakeSct(MIXED_DPI_MONITORS))
    as_windows(monkeypatch, windll)
    monkeypatch.setenv(grab.SCALE_ENV_VAR, "2.0")

    assert [m.scale for m in grab.list_monitors()] == [2.0, 2.0, 2.0]
    assert windll.shcore.GetDpiForMonitor.call_count == 0  # not even consulted


def test_shcore_failure_falls_back_to_system_dpi(monkeypatch):
    windll = dpi_windll(
        shcore={"GetDpiForMonitor": fn(-2147024809)},  # E_INVALIDARG
        user32={"GetDpiForSystem": fn(120)},  # 125%
    )
    as_windows(monkeypatch, windll)
    assert grab._detect_scale(rect=grab.Rect(0, 0, 3840, 2160)) == 1.25


def test_missing_shcore_library_falls_back_cleanly(monkeypatch):
    """Windows 7/8 have no shcore.dll at all -- an AttributeError, not a return code."""
    windll = FakeWinDLL(
        user32=FakeLib(
            MonitorFromRect=fn(side_effect=monitor_from_rect),
            MonitorFromPoint=fn(HMON_4K),
            GetDpiForSystem=fn(144),
        ),
        kernel32=kernel32(),
    )
    as_windows(monkeypatch, windll)
    assert grab._detect_scale(rect=grab.Rect(0, 0, 3840, 2160)) == 1.5


def test_every_dpi_api_failing_returns_the_default(monkeypatch):
    windll = FakeWinDLL(user32=FakeLib(), kernel32=kernel32())
    as_windows(monkeypatch, windll)
    assert grab._detect_scale(rect=grab.Rect(0, 0, 1920, 1080)) == 1.0
    assert grab._detect_scale() == 1.0


def test_monitor_handle_is_read_as_a_pointer_not_an_int(monkeypatch):
    """A 64-bit HMONITOR truncated to ctypes' default c_int restype is garbage."""
    windll = dpi_windll()
    as_windows(monkeypatch, windll)
    grab._detect_scale(rect=grab.Rect(0, 0, 3840, 2160))
    assert windll.user32.MonitorFromRect.restype is ctypes.c_void_p


def test_region_scale_follows_the_monitor_the_region_lands_on(monkeypatch):
    """The same rect maths, but the answer differs per monitor on this desktop."""
    as_windows(monkeypatch, dpi_windll())
    on_4k = grab._detect_scale(rect=grab.Rect(100, 100, 800, 600))
    on_1080p = grab._detect_scale(rect=grab.Rect(4000, 100, 800, 600))
    assert (on_4k, on_1080p) == (1.5, 1.0)


# --------------------------------------------------------------------------
# 3. grab_window
# --------------------------------------------------------------------------

HWND_GAME = 0x00420042
GAME_TITLE = "Factory Puzzle - DirectX 12"
# GetWindowRect would report this: DWM's invisible resize border included.
WINDOW_RECT_WITH_BORDER = (93, 100, 1607, 1108)
# DWMWA_EXTENDED_FRAME_BOUNDS: the frame the user can actually see.
EXTENDED_FRAME_BOUNDS = (100, 100, 1600, 1100)


def window_windll(
    *,
    hwnd: int = HWND_GAME,
    titles: dict[int, str] | None = None,
    iconic: int = 0,
    find_result: int = HWND_GAME,
    dwm_hresult: int = 0,
    bounds: tuple[int, int, int, int] = EXTENDED_FRAME_BOUNDS,
) -> FakeWinDLL:
    titles = titles if titles is not None else {HWND_GAME: GAME_TITLE}

    def get_text(h, buf, n):
        buf.value = titles.get(int(getattr(h, "value", h) or 0), "")
        return len(buf.value)

    def text_length(h):
        return len(titles.get(int(getattr(h, "value", h) or 0), ""))

    def enum_windows(callback, lparam):
        for h in titles:
            callback(h, 0)
        return True

    def dwm(h, attr, ptr, size):
        if dwm_hresult == 0:
            left, top, right, bottom = bounds
            ptr.contents.left, ptr.contents.top = left, top
            ptr.contents.right, ptr.contents.bottom = right, bottom
        return dwm_hresult

    def get_window_rect(h, ptr):
        left, top, right, bottom = WINDOW_RECT_WITH_BORDER
        ptr.contents.left, ptr.contents.top = left, top
        ptr.contents.right, ptr.contents.bottom = right, bottom
        return True

    user32 = {
        "GetForegroundWindow": fn(hwnd),
        "FindWindowW": fn(find_result),
        "EnumWindows": fn(side_effect=enum_windows),
        "IsWindow": fn(1),
        "IsWindowVisible": fn(1),
        "IsIconic": fn(iconic),
        "GetWindowTextW": fn(side_effect=get_text),
        "GetWindowTextLengthW": fn(side_effect=text_length),
        "GetWindowRect": fn(side_effect=get_window_rect),
        "MonitorFromRect": fn(side_effect=monitor_from_rect),
        "MonitorFromPoint": fn(HMON_4K),
        "GetDpiForSystem": fn(96),
        "SetProcessDpiAwarenessContext": fn(True),
    }
    return FakeWinDLL(
        user32=FakeLib(**user32),
        shcore=FakeLib(GetDpiForMonitor=fn(side_effect=get_dpi_for_monitor)),
        dwmapi=FakeLib(DwmGetWindowAttribute=fn(side_effect=dwm)),
        kernel32=kernel32(),
    )


@pytest.fixture()
def grabbed_boxes(monkeypatch):
    """Intercept the one function that actually talks to mss, keeping the whole
    grab_window -> grab_region -> _detect_scale path real."""
    boxes: list[dict] = []

    def fake_grab_box(box, monitor_index, scale, *, source="screen", raise_on_blank=False):
        boxes.append(dict(box))
        return grab.Capture(
            image=Image.new("RGB", (box["width"], box["height"]), (40, 50, 60)),
            region=grab.Rect(box["left"], box["top"], box["width"], box["height"]),
            monitor_index=monitor_index,
            scale=scale,
            source=source,
        )

    monkeypatch.setattr(grab, "_grab_box", fake_grab_box)
    return boxes


def test_grab_window_uses_dwm_extended_frame_bounds(monkeypatch, grabbed_boxes):
    """GetWindowRect includes DWM's invisible resize border; capturing it grabs
    a few pixels of whatever is behind the game on every edge, every time."""
    windll = window_windll()
    as_windows(monkeypatch, windll)

    cap = grab.grab_window()

    dwm_call = windll.dwmapi.DwmGetWindowAttribute.call_args
    assert dwm_call[0][1] == grab.DWMWA_EXTENDED_FRAME_BOUNDS == 9
    assert windll.user32.GetWindowRect.call_count == 0  # the naive API, unused
    assert grabbed_boxes == [{"left": 100, "top": 100, "width": 1500, "height": 1000}]
    assert cap.region.as_tuple() == (100, 100, 1500, 1000)
    assert cap.source == "window"
    assert cap.window == GAME_TITLE
    # The window is on the 4K/150% panel, so the capture records that scale.
    assert cap.scale == 1.5
    assert cap.logical_region == grab.Rect(67, 67, 1000, 667)


def test_grab_window_defaults_to_the_foreground_window(monkeypatch, grabbed_boxes):
    windll = window_windll()
    as_windows(monkeypatch, windll)
    grab.grab_window()
    assert windll.user32.GetForegroundWindow.call_count == 1
    assert windll.user32.FindWindowW.call_count == 0


def test_grab_window_matches_a_title(monkeypatch, grabbed_boxes):
    windll = window_windll()
    as_windows(monkeypatch, windll)
    cap = grab.grab_window(GAME_TITLE)
    assert windll.user32.GetForegroundWindow.call_count == 0
    assert cap.region.w == 1500


def test_grab_window_falls_back_to_a_substring_scan(monkeypatch, grabbed_boxes):
    """Games append " - DirectX 12", a level name, an FPS counter."""
    windll = window_windll(find_result=0)  # exact FindWindowW match fails
    as_windows(monkeypatch, windll)
    cap = grab.grab_window("factory puzzle")
    assert windll.user32.EnumWindows.call_count == 1
    assert cap.window == GAME_TITLE


def test_grab_window_raises_a_clear_error_when_not_found(monkeypatch, grabbed_boxes):
    windll = window_windll(find_result=0, titles={0x99: "Notepad"})
    as_windows(monkeypatch, windll)

    with pytest.raises(grab.WindowNotFoundError) as exc:
        grab.grab_window("Factory Puzzle")
    msg = str(exc.value)
    assert "Factory Puzzle" in msg
    assert "grab_region" in msg  # tells the user what to do instead
    assert not grabbed_boxes  # nothing was captured


def test_ambiguous_title_names_the_candidates(monkeypatch, grabbed_boxes):
    windll = window_windll(find_result=0, titles={1: "Puzzle - one", 2: "Puzzle - two"})
    as_windows(monkeypatch, windll)
    with pytest.raises(grab.WindowNotFoundError) as exc:
        grab.grab_window("Puzzle")
    assert "Puzzle - one" in str(exc.value) and "Puzzle - two" in str(exc.value)


def test_grab_window_raises_a_clear_error_when_minimized(monkeypatch, grabbed_boxes):
    windll = window_windll(iconic=1)
    as_windows(monkeypatch, windll)

    with pytest.raises(grab.WindowMinimizedError) as exc:
        grab.grab_window(GAME_TITLE)
    assert "minimised" in str(exc.value).lower()
    assert "restore" in str(exc.value).lower()
    assert windll.dwmapi.DwmGetWindowAttribute.call_count == 0  # asked before grabbing
    assert not grabbed_boxes


def test_no_foreground_window_is_a_clear_error(monkeypatch, grabbed_boxes):
    windll = window_windll(hwnd=0)
    as_windows(monkeypatch, windll)
    with pytest.raises(grab.WindowNotFoundError) as exc:
        grab.grab_window()
    assert "foreground" in str(exc.value)


def test_dwm_failure_falls_back_to_get_window_rect(monkeypatch, grabbed_boxes):
    """Composition disabled (a DWM-less RDP/Server Core session): take the
    border-inclusive rect rather than failing the capture outright."""
    windll = window_windll(dwm_hresult=-2147024891)
    as_windows(monkeypatch, windll)
    grab.grab_window(GAME_TITLE)
    assert windll.user32.GetWindowRect.call_count == 1
    assert grabbed_boxes == [{"left": 93, "top": 100, "width": 1514, "height": 1008}]


def test_zero_size_window_rect_is_reported_as_minimized(monkeypatch, grabbed_boxes):
    windll = window_windll(bounds=(0, 0, 0, 0))
    as_windows(monkeypatch, windll)
    with pytest.raises(grab.WindowMinimizedError):
        grab.grab_window(GAME_TITLE)


def test_grab_window_still_refuses_on_linux():
    """Unchanged: guessing a rect would silently crop half a machine panel."""
    if not sys.platform.startswith("linux"):  # pragma: no cover - CI is Linux
        pytest.skip("Linux-only assertion")
    with pytest.raises(grab.WindowCaptureUnsupported) as exc:
        grab.grab_window("Factory Puzzle")
    assert "grab_region" in str(exc.value)


# --------------------------------------------------------------------------
# 4. Fullscreen-exclusive / blank frames
# --------------------------------------------------------------------------


def scene_image(w: int = 1920, h: int = 1080) -> Image.Image:
    img = Image.new("RGB", (w, h), (12, 14, 18))  # a dark game UI, but not black
    for i in range(0, w, 97):
        img.paste(Image.new("RGB", (40, 40), (200, 180, 60)), (i, (i * 7) % max(1, h - 40)))
    return img


def test_looks_blank_is_true_for_an_all_black_frame():
    assert grab.looks_blank(Image.new("RGB", (1920, 1080), (0, 0, 0))) is True
    # near-black too: a stale GDI buffer is rarely exactly zero
    assert grab.looks_blank(Image.new("RGB", (1920, 1080), (2, 1, 3))) is True
    assert grab.looks_blank(Image.new("RGBA", (800, 600), (0, 0, 0, 255))) is True
    assert grab.looks_blank(Image.new("L", (800, 600), 0)) is True


def test_looks_blank_is_false_for_a_real_screenshot():
    assert grab.looks_blank(scene_image()) is False
    assert grab.looks_blank(Image.new("RGB", (400, 300), (255, 255, 255))) is False
    # a dark-themed panel is not a blank frame
    assert grab.looks_blank(Image.new("RGB", (400, 300), (24, 26, 30))) is False


def test_looks_blank_tolerates_a_cursor_on_a_black_frame():
    img = Image.new("RGB", (1920, 1080), (0, 0, 0))
    img.paste(Image.new("RGB", (16, 24), (255, 255, 255)), (960, 540))
    assert grab.looks_blank(img) is True


@pytest.mark.latency
def test_looks_blank_costs_well_under_a_millisecond_on_4k(capsys):
    black = Image.new("RGB", (3840, 2160), (0, 0, 0))
    scene = scene_image(3840, 2160)
    black.load()
    scene.load()

    def best_ms(img, n=5):
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            grab.looks_blank(img)
            out.append((time.perf_counter() - t0) * 1000.0)
        return min(out)

    black_ms = best_ms(black)
    scene_ms = best_ms(scene)
    with capsys.disabled():
        print(
            f"\n[latency] looks_blank on 4K: black {black_ms:.3f} ms "
            f"(worst case, full 32x32 lattice), real screenshot {scene_ms:.3f} ms "
            f"(early exit) -- vs 8.3M pixels if it scanned the frame"
        )
    assert black_ms < 3.0  # generous ceiling; the number above is the real one
    assert scene_ms < black_ms


def black_shot(w: int = 640, h: int = 480):
    return mock.Mock(size=(w, h), bgra=bytes(w * h * 4))


def test_a_blank_grab_is_flagged_not_swallowed(monkeypatch):
    monkeypatch.setattr(grab, "_open_sct", lambda: FakeSct(MIXED_DPI_MONITORS, black_shot()))
    cap = grab.grab_region(0, 0, 640, 480)
    assert cap.is_blank
    assert grab.BLANK_FRAME_WARNING in cap.warnings
    with pytest.raises(grab.FullscreenExclusiveError) as exc:
        cap.raise_if_blank()
    assert "borderless" in str(exc.value).lower()
    assert "exclusive" in str(exc.value).lower()


def test_raise_on_blank_fails_the_grab_immediately(monkeypatch):
    monkeypatch.setattr(grab, "_open_sct", lambda: FakeSct(MIXED_DPI_MONITORS, black_shot()))
    with pytest.raises(grab.FullscreenExclusiveError):
        grab.grab_region(0, 0, 640, 480, raise_on_blank=True)
    with pytest.raises(grab.FullscreenExclusiveError):
        grab.grab_monitor(1, raise_on_blank=True)


def test_a_normal_grab_carries_no_warning(monkeypatch):
    img = scene_image(320, 240)
    shot = mock.Mock(size=(320, 240), bgra=img.convert("RGBA").tobytes("raw", "BGRA"))
    monkeypatch.setattr(grab, "_open_sct", lambda: FakeSct(MIXED_DPI_MONITORS, shot))
    cap = grab.grab_region(0, 0, 320, 240)
    assert cap.warnings == []
    assert cap.is_blank is False
    assert cap.raise_if_blank() is cap


def test_the_store_still_saves_a_blank_capture(tmp_path, monkeypatch):
    """Never silently drop the evidence: the user has to be able to *see* the
    black frame to believe the diagnosis."""
    monkeypatch.setattr(grab, "_open_sct", lambda: FakeSct(MIXED_DPI_MONITORS, black_shot()))
    cap = grab.grab_region(0, 0, 640, 480, session_id="blank")
    store = CaptureStore(tmp_path)
    meta = store.save(cap, notes={"warnings": cap.warnings})

    assert (tmp_path / "blank" / meta.image).exists()
    assert store.verify(meta.capture_id)
    # NOTE: CaptureMeta has no first-class warnings field (store.py is outside
    # this change), so callers pass them through `notes`.
    assert meta.notes["warnings"] == [grab.BLANK_FRAME_WARNING]


# --------------------------------------------------------------------------
# Hotkeys: available() must be platform-correct, not X11-shaped
# --------------------------------------------------------------------------


def test_hotkeys_available_on_windows_without_a_display(monkeypatch):
    """pynput's Windows backend is a Win32 hook; DISPLAY is an X11 notion and
    demanding one here would disable hotkeys on every production machine."""
    monkeypatch.setattr(hotkeys, "_import_pynput", lambda: mock.Mock())
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    monkeypatch.setattr(sys, "platform", "win32")
    assert hotkeys.available() is True
    monkeypatch.setattr(sys, "platform", "darwin")
    assert hotkeys.available() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert hotkeys.available() is False  # no DISPLAY: correct on Linux


def test_hotkeys_unavailable_without_pynput_on_any_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert hotkeys.available() is False  # pynput is not installed in this venv


def test_uipi_elevation_pitfall_is_documented():
    """A game running as administrator silently eats hotkeys (UIPI). Users hit
    this; the fix has to be written down where the maintainer will find it."""
    doc = hotkeys.__doc__ or ""
    assert "UIPI" in doc
    assert "administrator" in doc.lower()
    assert "elevated" in doc.lower()
