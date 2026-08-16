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


def test_correcting_an_auto_confirmed_field_is_recorded_as_a_p0(client, tmp_path, monkeypatch):
    """Spec 13: every human correction to an auto-confirmed field is a P0
    extraction bug, and the capture that caused it must be recoverable."""
    import json as _json

    import services.core.main as main_mod

    logfile = tmp_path / "regressions.jsonl"
    monkeypatch.setattr(main_mod, "REGRESSION_LOG", logfile)

    sid = client.post("/api/sessions", json={"puzzle_type": "factory"}).json()["id"]
    s = session_mod.sessions.require(sid)
    s.pending_state = {"machines": [{"id": "m1", "output_max": 4, "storage_max": 9}]}
    s.verdicts = [
        session_mod.FieldVerdictView(
            path="machines[m1].output_max",
            value=4,
            status="unanimous",
            votes={"a/1": 4, "b/2": 4, "c/3": 4},
            crop={"capture_id": "cap_x", "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
        ),
        session_mod.FieldVerdictView(
            path="machines[m1].storage_max", value=9, status="majority", votes={"a/1": 9, "b/2": 9}
        ),
    ]
    session_mod.sessions.save(s)

    # correcting the majority-flagged field is routine and must NOT be logged;
    # correcting the unanimous one is the bug we care about
    client.post(
        f"/api/sessions/{sid}/state/confirm",
        json={"patch": {"machines[m1].storage_max": 12, "machines[m1].output_max": 6}},
    )

    rows = [_json.loads(line) for line in logfile.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["path"] == "machines[m1].output_max"
    assert rows[0]["jury_value"] == 4 and rows[0]["corrected_value"] == 6
    assert rows[0]["crop"]["capture_id"] == "cap_x"
    assert rows[0]["votes"] == {"a/1": 4, "b/2": 4, "c/3": 4}
