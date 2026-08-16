"""Data lives next to the code, not next to the shortcut that launched it."""

from __future__ import annotations


from pathlib import Path


def test_data_root_is_anchored_to_the_repo_not_the_cwd(monkeypatch, tmp_path):
    """On Windows the app is launched from a shortcut or a .ps1 whose working
    directory is arbitrary. A CWD-relative data/ would quietly start a second,
    empty session store there and the leaderboard's high-water mark would appear
    to vanish."""
    monkeypatch.delenv("PUZZLE_COPILOT_DATA", raising=False)
    monkeypatch.chdir(tmp_path)

    from services.core import paths

    assert paths.data_root() == paths.REPO_ROOT / "data"
    assert not (tmp_path / "data").exists()


def test_env_override_wins(monkeypatch, tmp_path):
    from services.core import paths

    monkeypatch.setenv("PUZZLE_COPILOT_DATA", str(tmp_path / "elsewhere"))
    assert paths.data_root() == (tmp_path / "elsewhere").resolve()
    monkeypatch.delenv("PUZZLE_COPILOT_DATA")
    assert paths.data_root() == paths.REPO_ROOT / "data"


def test_capture_store_default_is_anchored_too():
    from services.core.capture.store import DEFAULT_ROOT
    from services.core.paths import SESSION_ROOT

    assert Path(DEFAULT_ROOT) == SESSION_ROOT
    assert Path(DEFAULT_ROOT).is_absolute()


def test_atomic_replace_retries_when_a_reader_holds_the_file(monkeypatch, tmp_path):
    """os.replace onto a path another handle has open succeeds on POSIX and
    raises PermissionError on Windows. Two concurrent thumbnail requests hit
    exactly that, and only on the target platform."""
    import os as _os

    from services.core import atomic

    dst = tmp_path / "target.json"
    dst.write_text("old", encoding="utf-8")
    calls = {"n": 0}
    real = _os.replace

    def flaky(src, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "The process cannot access the file")
        real(src, target)

    monkeypatch.setattr(_os, "replace", flaky)
    monkeypatch.setattr(atomic, "_BACKOFF_S", 0.001)
    atomic.write_text(dst, "new")
    assert dst.read_text(encoding="utf-8") == "new"
    assert calls["n"] == 3


def test_atomic_replace_gives_up_rather_than_hanging(monkeypatch, tmp_path):
    import os as _os

    from services.core import atomic

    def always_locked(src, target):
        raise PermissionError(5, "locked")

    monkeypatch.setattr(_os, "replace", always_locked)
    monkeypatch.setattr(atomic, "_BACKOFF_S", 0.001)
    try:
        atomic.write_text(tmp_path / "x.json", "data")
    except PermissionError:
        pass
    else:
        raise AssertionError("a permanently locked file must surface, not hang")


def test_text_round_trips_non_ascii(tmp_path):
    """Windows text mode defaults to the ANSI code page, so a machine label a
    vision model read off the screen would round-trip wrong without an explicit
    encoding."""
    import json as _json

    from services.core import atomic

    payload = {"machine": "Schmelzofen No3 №", "note": "café — 25%"}
    atomic.write_json(tmp_path / "s.json", payload)
    assert _json.loads(atomic.read_text(tmp_path / "s.json")) == payload
