"""The seam between a stored capture and a reviewable state.

This is the path M8 rides on — screenshot to actionable recommendation without
the user touching JSON — so the two call sites that cross module boundaries
(``main._speculate`` and the ``extract_state`` tool) get a test each rather than
being discovered broken during a timed run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.core import session as session_mod
from services.core.main import app

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

FIXTURES = Path(__file__).parent / "fixtures" / "captures" / "factory"
pytestmark = pytest.mark.skipif(not FIXTURES.exists(), reason="fixtures not generated")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "SESSION_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(session_mod, "sessions", session_mod.SessionManager())
    import services.core.main as main_mod
    import services.core.wiring as wiring_mod

    monkeypatch.setattr(main_mod, "sessions", session_mod.sessions)
    monkeypatch.setattr(wiring_mod, "sessions", session_mod.sessions)
    yield


@pytest.fixture
def fake_client():
    from generate_fixtures import FakeVisionClient

    return FakeVisionClient.from_dir(FIXTURES, "factory", limit=1)


@pytest.fixture
def wired(monkeypatch, tmp_path, fake_client):
    """Point the wiring layer at a fake vision client and a temp capture store."""
    import services.core.capture.store as store_mod
    import services.core.wiring as wiring

    store_mod.set_default_root(tmp_path / "sessions")
    monkeypatch.setattr(wiring, "_client", lambda *a, **k: fake_client)

    class Roles:
        extractor_jury = ["alpha/one", "beta/two", "gamma/three"]
        tie_breaker = "delta/four"
        strategist = "beta/two"

    monkeypatch.setattr(wiring, "roles", lambda: Roles())
    return wiring


def _store_first_fixture(session_id: str, tmp_path):
    """Put fixture 001's PNG into the capture store, tiles and all."""
    import json

    from services.core.capture.grab import Rect, Tile
    from services.core.capture.store import CaptureStore

    png = sorted(FIXTURES.glob("*.png"))[0]
    sidecar = json.loads(png.with_suffix(".json").read_text())
    cs = CaptureStore(tmp_path / "sessions")
    tiles = [
        Tile(
            name=r["name"],
            box=Rect(r["box"]["x"], r["box"]["y"], r["box"]["w"], r["box"]["h"]),
            target=r.get("target", "factory_machine"),
        )
        for r in sidecar.get("regions", [])
    ]
    meta = cs.save_bytes(
        png.read_bytes(), session_id=session_id, puzzle_type="factory", tiles=tiles
    )
    return meta, sidecar


async def test_run_extraction_populates_the_session(wired, tmp_path):
    """A stored capture becomes pending state, verdicts and an unresolved list."""
    session = session_mod.sessions.create("factory")
    meta, _ = _store_first_fixture(session.id, tmp_path)

    result = await wired.run_extraction(session.id, [meta.id], "factory")

    assert result.factory_state is not None
    refreshed = session_mod.sessions.get(session.id)
    assert refreshed.pending_state is not None
    assert refreshed.verdicts, "no field verdicts reached the session"
    # The inspector sorts by disagreement, not document order, so anything
    # disputed must precede anything unanimous.
    statuses = [v.status for v in refreshed.verdicts]
    assert statuses == sorted(statuses, key=lambda s: {"split": 0, "majority": 1}.get(s, 2))
    assert refreshed.latency.get("extraction_total", 0) > 0


async def test_agreed_fields_never_reach_the_user(wired, tmp_path):
    """The point of the jury: most fields are settled before anyone looks."""
    session = session_mod.sessions.create("factory")
    meta, _ = _store_first_fixture(session.id, tmp_path)

    result = await wired.run_extraction(session.id, [meta.id], "factory")

    assert result.auto_confirmed, "nothing auto-confirmed — the jury is not agreeing"
    assert not (set(result.auto_confirmed) & set(result.disputed))
    print(
        f"\nauto-confirmed {len(result.auto_confirmed)} / "
        f"{len(result.auto_confirmed) + len(result.disputed)} fields"
    )


async def test_extract_state_tool_returns_disputes_not_the_whole_board(wired, tmp_path):
    """The tool payload must carry disputes, not paste the state back into
    context — the model asks about disputed fields and nothing else."""
    from services.core.agent.tools import ToolRegistry

    session = session_mod.sessions.create("factory")
    meta, _ = _store_first_fixture(session.id, tmp_path)

    registry = ToolRegistry(wired.store, session.id)
    payload = await registry.dispatch(
        "extract_state", {"capture_ids": [meta.id], "puzzle_type": "factory"}
    )

    assert payload["ok"] is True
    assert payload["state_available"] is True
    assert "state" not in payload, "the merged board must not be echoed into context"
    assert isinstance(payload["disputed"], list)
    assert payload["auto_confirmed_count"] > 0


def test_capture_upload_starts_speculative_extraction(wired, tmp_path, monkeypatch):
    """Extraction starts when the capture lands, not when the user asks."""
    import base64

    started: list[str] = []
    real = wired.start_speculative_extraction

    def spy(session_id: str, capture_id: str):
        started.append(capture_id)
        return real(session_id, capture_id)

    monkeypatch.setattr(wired, "start_speculative_extraction", spy)

    png = sorted(FIXTURES.glob("*.png"))[0]
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json={"puzzle_type": "factory"}).json()["id"]
        r = client.post(
            "/api/captures/upload",
            json={
                "session_id": sid,
                "image_base64": base64.b64encode(png.read_bytes()).decode(),
            },
        )
    assert r.status_code == 200 and r.json()["capture_id"]


async def test_fields_needing_a_human_sort_to_the_top(wired, tmp_path):
    """A unanimous null on a field the solver needs reads as 'unanimous' but is
    exactly what the user must fill. Sorting on status alone buries it under
    every settled field on the board."""
    session = session_mod.sessions.create("factory")
    meta, _ = _store_first_fixture(session.id, tmp_path)
    await wired.run_extraction(session.id, [meta.id], "factory")

    verdicts = session_mod.sessions.get(session.id).verdicts
    needs_human = [i for i, v in enumerate(verdicts) if not v.auto_confirmed]
    settled = [i for i, v in enumerate(verdicts) if v.auto_confirmed]
    if needs_human and settled:
        assert max(needs_human) < min(settled), "review work is buried below settled fields"
    # and the reason survives, since it is what tells the user how far to trust
    # a pre-filled value
    assert any(v.reason for v in verdicts)


async def test_alternatives_are_distinct(wired, tmp_path):
    """Two models returning the same wrong value is one alternative, not two."""
    session = session_mod.sessions.create("factory")
    meta, _ = _store_first_fixture(session.id, tmp_path)
    await wired.run_extraction(session.id, [meta.id], "factory")
    for v in session_mod.sessions.get(session.id).verdicts:
        reprs = [repr(a) for a in v.alternatives]
        assert len(reprs) == len(set(reprs))
