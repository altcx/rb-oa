"""HTTP + WebSocket surface.  Uses FastAPI's TestClient (no live network)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from services.core import session as session_mod
from services.core.main import app


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "SESSION_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(session_mod, "sessions", session_mod.SessionManager())
    import services.core.main as main_mod

    monkeypatch.setattr(main_mod, "sessions", session_mod.sessions)
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_create_and_list_session(client):
    sid = client.post("/api/sessions", json={"puzzle_type": "factory"}).json()["id"]
    listing = client.get("/api/sessions").json()["sessions"]
    assert any(s["id"] == sid for s in listing)


def test_hand_entered_state_is_first_class(client):
    """Spec 2.1: a verified solver with manual data entry is a usable tool."""
    sid = client.post("/api/sessions", json={"puzzle_type": "factory"}).json()["id"]
    state = {"machines": [], "edges": [], "starting_money": 100.0, "horizon_hours": 24}
    assert client.post(f"/api/sessions/{sid}/state", json={"state": state}).json()["ok"]
    got = client.get(f"/api/sessions/{sid}/state").json()
    assert got["state"]["starting_money"] == 100.0
    assert got["confirmed"] is True


def test_optimize_is_gated_on_calibration(client):
    """Spec 7.5: the optimizer does not run until a recorded run matches."""
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/state", json={"state": {"machines": []}})
    r = client.post("/api/solve/optimize", json={"session_id": sid, "seconds": 0.2})
    assert r.status_code == 412
    assert "calibration gate" in r.json()["error"]


def test_leaderboard_keeps_the_high_water_mark(client):
    """Spec 7.1: the score is the maximum over every tested factory."""
    sid = client.post("/api/sessions", json={}).json()["id"]
    for label, score in [("a", 100), ("b", 450), ("c", 220)]:
        client.post(f"/api/sessions/{sid}/leaderboard", json={"label": label, "score": score})
    entries = client.get(f"/api/sessions/{sid}/leaderboard").json()["entries"]
    assert [e["label"] for e in entries] == ["b", "c", "a"]
    high = [e for e in entries if e["is_high_water"]]
    assert len(high) == 1 and high[0]["label"] == "b"
    # a later, worse test must not displace it
    client.post(f"/api/sessions/{sid}/leaderboard", json={"label": "d", "score": 10})
    entries = client.get(f"/api/sessions/{sid}/leaderboard").json()["entries"]
    assert [e["label"] for e in entries if e["is_high_water"]] == ["b"]


def test_session_survives_a_restart(client, tmp_path):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/leaderboard", json={"label": "run1", "score": 999})
    session_mod.sessions._sessions.clear()  # simulate a process restart
    entries = client.get(f"/api/sessions/{sid}/leaderboard").json()["entries"]
    assert entries[0]["score"] == 999


def test_missing_session_is_404(client):
    assert client.get("/api/sessions/nope/state").status_code == 404


def test_websocket_ping_pong(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text())["type"] == "pong"


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True
