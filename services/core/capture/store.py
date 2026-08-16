"""On-disk session store for captures (spec 6.1).

Layout::

    data/sessions/<session_id>/
        cap_ab12cd34ef56.png       # the pixels
        cap_ab12cd34ef56.json      # the sidecar: everything needed to replay
        .thumbs/cap_ab12cd34ef56.240.png

One sidecar per capture, never a shared index file: two hotkeys pressed a
frame apart must not race for a lock.  Every write goes through a temp file in
the destination directory followed by ``os.replace``, so a reader either sees
the previous file or the complete new one -- never a half-written PNG.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.core.capture.grab import Capture, Rect, Tile, capture_from_image
from services.core.paths import SESSION_ROOT

#: Anchored to the repo root, not the CWD: on Windows the app is usually
#: launched from a shortcut whose working directory is somewhere else entirely,
#: and a relative path would quietly start a second, empty capture store there.
DEFAULT_ROOT = SESSION_ROOT
THUMB_DIRNAME = ".thumbs"


class CaptureMeta(BaseModel):
    """The JSON sidecar.  Fixtures use the same shape, which is the point:
    a real capture dropped into ``tests/fixtures/captures`` replays unchanged."""

    model_config = ConfigDict(extra="allow")

    capture_id: str
    #: Alias of ``capture_id``, kept in sync, because the HTTP layer and the web
    #: client address every resource as ``id``.
    id: str = ""
    session_id: str
    created_at: str
    monitor_index: int = 0
    #: Physical-pixel rect of the grab on the virtual desktop.
    region: dict[str, int] = Field(default_factory=dict)
    #: Same rect in logical (point) coordinates -- replay on a 1x monitor.
    logical_region: dict[str, int] = Field(default_factory=dict)
    scale: float = 1.0
    physical_size: list[int] = Field(default_factory=list)
    logical_size: list[int] = Field(default_factory=list)
    puzzle_type: str | None = None
    sha256: str = ""
    image: str = ""  #: file name, relative to the session directory
    source: str = "screen"
    grab_ms: float = 0.0
    tiles: list[dict[str, Any]] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_id(self) -> "CaptureMeta":
        if self.id != self.capture_id:
            object.__setattr__(self, "id", self.capture_id)
        return self

    @property
    def region_rect(self) -> Rect:
        return Rect.coerce(self.region)

    def tile_objects(self) -> list[Tile]:
        return [Tile.from_dict(t) for t in self.tiles]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=False, default=str).encode())


class CaptureStore:
    """Session-scoped capture store.  Safe for concurrent writers."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    # -- paths ----------------------------------------------------------
    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def sessions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def _meta_path(self, session_id: str, capture_id: str) -> Path:
        return self.session_dir(session_id) / f"{capture_id}.json"

    # -- write ----------------------------------------------------------
    def save(
        self,
        capture: Capture,
        *,
        session_id: str | None = None,
        puzzle_type: str | None = None,
        notes: dict[str, Any] | None = None,
    ) -> CaptureMeta:
        """Persist pixels + sidecar.  Returns the sidecar as written."""
        sid = session_id or capture.session_id or "default"
        sdir = self.session_dir(sid)
        sdir.mkdir(parents=True, exist_ok=True)

        png = capture.png_bytes()
        digest = hashlib.sha256(png).hexdigest()
        image_name = f"{capture.capture_id}.png"
        logical = capture.logical_region
        meta = CaptureMeta(
            capture_id=capture.capture_id,
            session_id=sid,
            created_at=capture.captured_at or _now_iso(),
            monitor_index=capture.monitor_index,
            region=capture.region.as_dict(),
            logical_region=logical.as_dict(),
            scale=capture.scale,
            physical_size=[capture.image.width, capture.image.height],
            logical_size=[logical.w, logical.h],
            puzzle_type=puzzle_type or capture.puzzle_type,
            sha256=digest,
            image=image_name,
            source=capture.source,
            grab_ms=capture.grab_ms,
            tiles=[t.as_dict() for t in capture.tiles],
            notes=notes or {},
        )
        # Pixels first: a sidecar that names a missing PNG is the one
        # inconsistency a reader cannot recover from.
        atomic_write_bytes(sdir / image_name, png)
        atomic_write_json(self._meta_path(sid, capture.capture_id), meta.model_dump())
        return meta

    def save_bytes(
        self,
        data: bytes,
        *,
        session_id: str = "default",
        puzzle_type: str | None = None,
        scale: float = 1.0,
        monitor_index: int = 0,
        notes: dict[str, Any] | None = None,
        tiles: Sequence[Tile] | None = None,
    ) -> CaptureMeta:
        """Store an already-encoded image (upload path, replayed fixture, test).

        Lets the tool be driven on a machine where the screen-grab backend does
        not work at all.  ``tiles`` carries the per-panel regions through, so an
        uploaded board fans out the same way a grabbed one does instead of
        falling back to the default grid.
        """
        with Image.open(io.BytesIO(data)) as im:
            image = im.convert("RGB")
        cap = capture_from_image(
            image,
            scale=scale,
            monitor_index=monitor_index,
            session_id=session_id,
            puzzle_type=puzzle_type,
            source="upload",
            tiles=list(tiles) if tiles else None,
        )
        return self.save(cap, notes=notes)

    # -- read -----------------------------------------------------------
    def path_for(self, capture_id: str, session_id: str | None = None) -> Path:
        """Filesystem path of the stored PNG (for serving it straight out)."""
        meta = self.find(capture_id, session_id)
        return self.session_dir(meta.session_id) / meta.image

    def list_captures(self, session_id: str) -> list[CaptureMeta]:
        sdir = self.session_dir(session_id)
        if not sdir.exists():
            return []
        out: list[CaptureMeta] = []
        for p in sorted(sdir.glob("*.json")):
            try:
                out.append(CaptureMeta.model_validate_json(p.read_bytes()))
            except Exception:
                continue  # a partial file cannot happen via atomic_write; skip alien files
        out.sort(key=lambda m: (m.created_at, m.capture_id))
        return out

    def find(self, capture_id: str, session_id: str | None = None) -> CaptureMeta:
        candidates: Iterable[str] = [session_id] if session_id else self.sessions()
        for sid in candidates:
            p = self._meta_path(sid, capture_id)
            if p.exists():
                return CaptureMeta.model_validate_json(p.read_bytes())
        raise KeyError(f"no capture {capture_id!r} under {self.root}")

    def load_meta(self, capture_id: str, session_id: str | None = None) -> CaptureMeta:
        return self.find(capture_id, session_id)

    def load(self, capture_id: str, session_id: str | None = None) -> Capture:
        """Rehydrate a :class:`Capture` (pixels + geometry + tiles)."""
        meta = self.find(capture_id, session_id)
        path = self.session_dir(meta.session_id) / meta.image
        with Image.open(path) as im:
            image = im.convert("RGB")
        cap = capture_from_image(
            image,
            region=meta.region_rect,
            scale=meta.scale,
            monitor_index=meta.monitor_index,
            session_id=meta.session_id,
            puzzle_type=meta.puzzle_type,
            capture_id=meta.capture_id,
            tiles=meta.tile_objects(),
            source="replay",
        )
        cap.captured_at = meta.created_at
        return cap

    def verify(self, capture_id: str, session_id: str | None = None) -> bool:
        """True when the stored PNG still hashes to the sidecar's digest."""
        meta = self.find(capture_id, session_id)
        path = self.session_dir(meta.session_id) / meta.image
        return hashlib.sha256(path.read_bytes()).hexdigest() == meta.sha256

    def thumbnail(
        self, capture_id: str, max_px: int = 240, session_id: str | None = None
    ) -> Image.Image:
        """A cached, aspect-preserving thumbnail for the review strip."""
        meta = self.find(capture_id, session_id)
        sdir = self.session_dir(meta.session_id)
        tpath = sdir / THUMB_DIRNAME / f"{capture_id}.{max_px}.png"
        if tpath.exists():
            with Image.open(tpath) as im:
                return im.convert("RGB")
        with Image.open(sdir / meta.image) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            thumb.save(buf, format="PNG")
            atomic_write_bytes(tpath, buf.getvalue())
            return thumb

    def thumbnail_bytes(
        self, capture_id: str, max_px: int = 240, session_id: str | None = None
    ) -> bytes:
        """PNG bytes of the thumbnail, ready for an HTTP response."""
        self.thumbnail(capture_id, max_px, session_id)  # ensures the cache file
        meta = self.find(capture_id, session_id)
        return (self.session_dir(meta.session_id) / THUMB_DIRNAME / f"{capture_id}.{max_px}.png").read_bytes()

    def delete(self, capture_id: str, session_id: str | None = None) -> None:
        meta = self.find(capture_id, session_id)
        sdir = self.session_dir(meta.session_id)
        for p in (sdir / meta.image, self._meta_path(meta.session_id, capture_id)):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        for p in (sdir / THUMB_DIRNAME).glob(f"{capture_id}.*.png"):
            p.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Module-level convenience bound to the default root
# --------------------------------------------------------------------------

_default = CaptureStore()


def default_store() -> CaptureStore:
    return _default


def set_default_root(root: Path | str) -> CaptureStore:
    global _default
    _default = CaptureStore(root)
    return _default


def save(capture: Capture, **kw: Any) -> CaptureMeta:
    return _default.save(capture, **kw)


def list_captures(session_id: str) -> list[CaptureMeta]:
    return _default.list_captures(session_id)


def load(capture_id: str, session_id: str | None = None) -> Capture:
    return _default.load(capture_id, session_id)


def thumbnail(capture_id: str, max_px: int = 240, session_id: str | None = None) -> Image.Image:
    return _default.thumbnail(capture_id, max_px, session_id)


__all__ = [
    "CaptureMeta",
    "CaptureStore",
    "DEFAULT_ROOT",
    "atomic_write_bytes",
    "atomic_write_json",
    "default_store",
    "set_default_root",
    "save",
    "list_captures",
    "load",
    "thumbnail",
]
