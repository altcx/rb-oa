"""Capture, DPI maths, session store, hotkey dispatch -- all headless.

Nothing here opens a display.  The display-touching entry points are asserted
to fail *clearly* instead, which is the behaviour that matters on a machine
without monitors.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from services.core.capture import hotkeys
from services.core.capture.grab import (
    MAX_LONG_EDGE,
    Capture,
    DisplayUnavailableError,
    MonitorInfo,
    PreparedImage,
    Rect,
    Tile,
    board_overview,
    capture_from_image,
    crop_tiles,
    display_available,
    grab_window,
    grid_tiles,
    list_monitors,
    prepare,
    replay_rect,
    scale_rect,
    tile_regions,
    to_logical,
    to_physical,
)
from services.core.capture.store import CaptureStore, atomic_write_json

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_image(w: int = 400, h: int = 300, seed: int = 0) -> Image.Image:
    img = Image.new("RGB", (w, h), (20, 24, 30))
    for i in range(0, w, 40):
        for j in range(0, h, 40):
            img.paste(
                Image.new("RGB", (20, 20), ((i + seed) % 256, (j * 3) % 256, (i + j) % 256)),
                (i, j),
            )
    return img


def make_capture(**kw) -> Capture:
    image = kw.pop("image", None) or make_image()
    return capture_from_image(image, **kw)


# --------------------------------------------------------------------------
# DPI / scale maths (pure, no display)
# --------------------------------------------------------------------------


def test_scale_rect_is_pure_math():
    r = Rect(10, 20, 100, 50)
    assert scale_rect(r, 2.0) == Rect(20, 40, 200, 100)
    assert scale_rect(r, 0.5) == Rect(5, 10, 50, 25)
    assert scale_rect(r, 1.0) == r


def test_region_saved_at_2x_replays_at_1x():
    """The whole reason the scale factor is stored with the capture."""
    on_retina = Rect(400, 200, 640, 480)  # physical px on a 2x panel
    on_1x = replay_rect(on_retina, saved_scale=2.0, target_scale=1.0)
    assert on_1x == Rect(200, 100, 320, 240)
    # ... and back again, losslessly for even numbers
    assert replay_rect(on_1x, 1.0, 2.0) == on_retina


def test_logical_and_physical_round_trip():
    logical = Rect(50, 60, 300, 200)
    physical = to_physical(logical, 2.0)
    assert physical == Rect(100, 120, 600, 400)
    assert to_logical(physical, 2.0) == logical


def test_monitor_info_reports_both_coordinate_systems():
    mon = MonitorInfo(index=1, bounds=Rect(0, 0, 3840, 2160), scale=2.0)
    assert mon.physical_size == (3840, 2160)
    assert mon.logical_size == (1920, 1080)


def test_rect_clamps_to_the_surface():
    assert Rect(-10, -10, 50, 50).clamp(100, 100) == Rect(0, 0, 50, 50)
    assert Rect(90, 90, 50, 50).clamp(100, 100) == Rect(90, 90, 10, 10)


def test_scale_rect_never_collapses_a_visible_rect():
    assert scale_rect(Rect(0, 0, 3, 3), 0.1).w == 1


# --------------------------------------------------------------------------
# Display guards
# --------------------------------------------------------------------------


def test_headless_display_access_fails_clearly():
    if display_available():  # pragma: no cover - depends on the host
        pytest.skip("this host has a display")
    assert display_available() is False
    with pytest.raises(DisplayUnavailableError) as exc:
        list_monitors()
    assert "display" in str(exc.value).lower()


def test_window_capture_says_why_it_is_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        grab_window("Puzzle Game")
    msg = str(exc.value)
    assert "grab_region" in msg  # tells the caller what to do instead


def test_hotkeys_module_imports_without_pynput():
    assert hotkeys.available() is False  # pynput is not installed here
    mgr = hotkeys.HotkeyManager()
    with pytest.raises(hotkeys.HotkeysUnavailable) as exc:
        mgr.start()
    assert "pynput" in str(exc.value)


# --------------------------------------------------------------------------
# Hotkey registry / dispatch (no input device)
# --------------------------------------------------------------------------


def test_hotkey_registry_dispatches_by_name():
    mgr = hotkeys.HotkeyManager()
    fired: list[str] = []
    mgr.register("capture_monitor", lambda: fired.append("monitor"))

    @mgr.on("accept_all_majority")
    def _accept():
        fired.append("accept")
        return "accepted"

    assert set(mgr.actions) >= {
        "capture_monitor",
        "capture_region",
        "capture_window",
        "capture_and_ask",
        "accept_all_majority",
    }
    mgr.simulate("capture_monitor")
    assert mgr.simulate("accept_all_majority") == ["accepted"]
    assert fired == ["monitor", "accept"]
    mgr.simulate("capture_region")  # no handler: silently fine
    assert [e.action for e in mgr.history] == [
        "capture_monitor",
        "accept_all_majority",
        "capture_region",
    ]
    with pytest.raises(KeyError):
        mgr.simulate("not_a_hotkey")


def test_hotkey_callback_error_does_not_break_the_listener():
    mgr = hotkeys.HotkeyManager()
    mgr.register("capture_and_ask", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    mgr.register("capture_and_ask", lambda: "still ran")
    out = mgr.simulate("capture_and_ask")
    assert isinstance(out[0], RuntimeError) and out[1] == "still ran"
    assert mgr.errors and mgr.errors[0][0] == "capture_and_ask"


async def test_hotkey_async_callback_is_awaited():
    mgr = hotkeys.HotkeyManager()

    async def handler(**kw):
        return "async ok"

    mgr.register("capture_region", handler)
    assert await mgr.simulate_async("capture_region") == ["async ok"]


def test_rebinding_rejects_a_conflicting_accelerator():
    mgr = hotkeys.HotkeyManager()
    with pytest.raises(ValueError):
        mgr.rebind("capture_region", mgr.bindings["capture_monitor"])
    mgr.rebind("capture_region", "<ctrl>+<alt>+9")
    assert mgr.bindings["capture_region"] == "<ctrl>+<alt>+9"


# --------------------------------------------------------------------------
# prepare(): crop + downscale
# --------------------------------------------------------------------------


def test_prepare_crops_and_emits_a_data_url():
    img = make_image(800, 600)
    out = prepare(img, Rect(100, 100, 200, 150))
    assert isinstance(out, PreparedImage)
    assert out.data_url.startswith("data:image/png;base64,")
    assert (out.width, out.height) == (200, 150)
    assert out.scale == 1.0
    assert out.crop_id.startswith("crop_")


def test_prepare_downscales_to_the_long_edge_budget():
    img = make_image(3840, 2160)
    out = prepare(img, Rect(0, 0, 3840, 2160))
    assert max(out.width, out.height) == MAX_LONG_EDGE
    assert out.scale == pytest.approx(MAX_LONG_EDGE / 3840, rel=1e-3)


def test_prepare_refuses_to_send_a_whole_screen():
    img = make_image(3840, 2160)
    with pytest.raises(ValueError) as exc:
        prepare(img, None)
    assert "full screen" in str(exc.value)
    assert prepare(img, None, allow_full=True).width == MAX_LONG_EDGE


@pytest.mark.latency
def test_crop_and_downscale_of_4k_is_within_budget(capsys):
    """Budget: 40 ms.  Ceiling asserted generously; the number is printed."""
    img = make_image(3840, 2160)
    prepare(img, Rect(0, 0, 3840, 2160))  # warm the resampler
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = prepare(img, Rect(200, 100, 3000, 1800))
        samples.append((time.perf_counter() - t0) * 1000.0)
    best = min(samples)
    with capsys.disabled():
        print(
            f"\n[latency] crop+downscale 4K -> {out.width}x{out.height}: "
            f"best {best:.1f} ms, samples {[round(s, 1) for s in samples]} "
            f"(budget 40 ms, ceiling 250 ms)"
        )
    assert best < 250.0


@pytest.mark.latency
def test_per_tile_prepare_is_cheap(capsys):
    img = make_image(3840, 2160)
    tiles = grid_tiles((3840, 2160), 3, 2)
    t0 = time.perf_counter()
    prepared = crop_tiles(img, tiles)
    ms = (time.perf_counter() - t0) * 1000.0
    with capsys.disabled():
        print(f"[latency] 6 tile crops of a 4K frame: {ms:.1f} ms total")
    assert len(prepared) == 6
    assert ms < 250.0


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------


def test_tile_regions_names_and_clamps_boxes():
    tiles = tile_regions(
        [Rect(0, 0, 100, 100), (150, 150, 100, 100)],
        names=["M1", "M2"],
        pad=5,
        bounds=(200, 200),
    )
    assert [t.name for t in tiles] == ["M1", "M2"]
    assert all(t.target == "factory_machine" for t in tiles)
    assert tiles[1].box.as_box()[2] <= 200


def test_board_overview_is_low_res_and_topology_only():
    tile = board_overview((3840, 2160))
    assert tile.target == "factory_topology"
    assert tile.max_long_edge < MAX_LONG_EDGE
    out = prepare(make_image(3840, 2160), tile.box, max_long_edge=tile.max_long_edge)
    assert max(out.width, out.height) == tile.max_long_edge


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


def test_save_list_load_thumbnail_round_trip(tmp_path: Path):
    store = CaptureStore(tmp_path)
    cap = make_capture(
        image=make_image(600, 400),
        region=Rect(100, 50, 600, 400),
        scale=2.0,
        monitor_index=1,
        session_id="s1",
        puzzle_type="factory",
        tiles=[Tile("M1", Rect(0, 0, 100, 100), "factory_machine")],
    )
    meta = store.save(cap)

    assert meta.session_id == "s1"
    assert meta.scale == 2.0
    assert meta.physical_size == [600, 400]
    assert meta.logical_size == [300, 200]  # replayable on a 1x monitor
    assert meta.region == {"x": 100, "y": 50, "w": 600, "h": 400}
    assert meta.logical_region == {"x": 50, "y": 25, "w": 300, "h": 200}
    assert len(meta.sha256) == 64
    assert store.verify(meta.capture_id) is True

    listed = store.list_captures("s1")
    assert [m.capture_id for m in listed] == [meta.capture_id]

    loaded = store.load(meta.capture_id)
    assert loaded.image.size == (600, 400)
    assert loaded.scale == 2.0
    assert [t.name for t in loaded.tiles] == ["M1"]
    assert loaded.capture_id == meta.capture_id

    thumb = store.thumbnail(meta.capture_id, max_px=120)
    assert max(thumb.size) == 120
    assert (tmp_path / "s1" / ".thumbs" / f"{meta.capture_id}.120.png").exists()
    # second call is served from the cache and is identical
    assert store.thumbnail(meta.capture_id, max_px=120).size == thumb.size


def test_list_captures_is_empty_for_an_unknown_session(tmp_path: Path):
    assert CaptureStore(tmp_path).list_captures("nope") == []
    with pytest.raises(KeyError):
        CaptureStore(tmp_path).load("cap_missing")


def test_concurrent_writes_are_atomic(tmp_path: Path):
    """Twenty hotkeys in a row must not leave a torn sidecar behind."""
    store = CaptureStore(tmp_path)
    n = 20
    errors: list[BaseException] = []
    barrier = threading.Barrier(n)

    def writer(i: int) -> None:
        try:
            cap = make_capture(
                image=make_image(200, 160, seed=i), session_id="race", puzzle_type="factory"
            )
            barrier.wait(timeout=10)
            store.save(cap)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors
    metas = store.list_captures("race")
    assert len(metas) == n
    assert len({m.capture_id for m in metas}) == n
    for m in metas:
        assert store.verify(m.capture_id)
        json.loads((tmp_path / "race" / f"{m.capture_id}.json").read_text())
    assert not list((tmp_path / "race").glob(".tmp-*"))


def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    path = tmp_path / "meta.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert list(tmp_path.iterdir()) == [path]
