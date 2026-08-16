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
