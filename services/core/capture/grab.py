"""Screen capture primitives (spec section 6.1).

Three rules run through this file:

1.  **Nothing here needs a display to import.**  ``mss`` is imported lazily and
    every display-touching entry point raises :class:`DisplayUnavailableError`
    with an actionable message instead of blowing up with an X11 traceback.
2.  **DPI is recorded, never assumed.**  A capture carries the physical pixel
    rect *and* the scale factor, so a region saved on a 2x monitor replays
    correctly on a 1x monitor (:func:`replay_rect`).  The maths is pure and is
    tested without a display.
3.  **A full screen is never sent to a model.**  :func:`prepare` refuses to
    encode a whole image unless the caller explicitly opts in, and the only
    sanctioned whole-board crop is the deliberately low-res topology overview
    from :func:`board_overview`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from PIL import Image

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
    try:
        import mss  # noqa: F401
    except Exception:
        return False
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
    import mss  # local import: importing mss is cheap, opening it is not

    try:
        return mss.mss()
    except Exception as exc:  # pragma: no cover - needs a broken display
        raise DisplayUnavailableError(f"{_NO_DISPLAY_MSG} (mss said: {exc})") from exc


def _detect_scale(default: float = 1.0) -> float:
    """DPI scale factor, overridable for machines that lie about it.

    ``mss`` reports physical pixels only, so on a HiDPI panel the honest thing
    is to take the factor from the environment rather than invent one.
    """
    raw = os.environ.get("PUZZLE_COPILOT_DPI_SCALE")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default


def list_monitors() -> list[MonitorInfo]:
    """Every monitor, index 1..n (index 0 is mss's virtual all-monitors rect)."""
    with _open_sct() as sct:
        out: list[MonitorInfo] = []
        scale = _detect_scale()
        for i, mon in enumerate(sct.monitors):
            bounds = Rect(int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"]))
            out.append(
                MonitorInfo(
                    index=i,
                    bounds=bounds,
                    scale=scale,
                    name="all" if i == 0 else f"monitor-{i}",
                    is_primary=(i == 1),
                )
            )
        return out


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

    @property
    def physical_size(self) -> tuple[int, int]:
        return (self.image.width, self.image.height)

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
) -> Capture:
    """Wrap an in-memory image as a :class:`Capture` (fixtures, replay, tests)."""
    rect = Rect.coerce(region) if region is not None else Rect(0, 0, image.width, image.height)
    kwargs = {} if capture_id is None else {"capture_id": capture_id}
    return Capture(
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


def _grab_box(box: dict[str, int], monitor_index: int, scale: float) -> Capture:
    t0 = time.perf_counter()
    with _open_sct() as sct:
        shot = sct.grab(box)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    ms = (time.perf_counter() - t0) * 1000.0
    return Capture(
        image=image,
        region=Rect(box["left"], box["top"], box["width"], box["height"]),
        monitor_index=monitor_index,
        scale=scale,
        grab_ms=ms,
        source="screen",
    )


def grab_monitor(index: int = 1, *, session_id: str = "default") -> Capture:
    """Grab a whole monitor.  ``index`` matches :func:`list_monitors`."""
    with _open_sct() as sct:
        if index < 0 or index >= len(sct.monitors):
            raise ValueError(f"no monitor {index}; have 0..{len(sct.monitors) - 1}")
        mon = dict(sct.monitors[index])
    cap = _grab_box(
        {
            "left": int(mon["left"]),
            "top": int(mon["top"]),
            "width": int(mon["width"]),
            "height": int(mon["height"]),
        },
        index,
        _detect_scale(),
    )
    cap.session_id = session_id
    return cap


def grab_region(
    x: int, y: int, w: int, h: int, *, monitor_index: int = 1, session_id: str = "default"
) -> Capture:
    """Grab an arbitrary physical-pixel rect of the virtual desktop."""
    if w <= 0 or h <= 0:
        raise ValueError("region must have positive width and height")
    cap = _grab_box(
        {"left": int(x), "top": int(y), "width": int(w), "height": int(h)},
        monitor_index,
        _detect_scale(),
    )
    cap.session_id = session_id
    return cap


def grab_window(
    title: str | None = None, *, window_id: int | None = None, session_id: str = "default"
) -> Capture:
    """Grab a single window.

    Deliberately unimplemented on Linux: without a compositor API (XComposite,
    or a portal on Wayland) the only way to "grab a window" is to guess a rect,
    and a guessed rect silently crops half a machine panel.  A clear failure
    beats a wrong crop.
    """
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


__all__ = [
    "MAX_LONG_EDGE",
    "OVERVIEW_LONG_EDGE",
    "CAPTURE_BUDGET_MS",
    "PREPARE_BUDGET_MS",
    "DisplayUnavailableError",
    "WindowCaptureUnsupported",
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
