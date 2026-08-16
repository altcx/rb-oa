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


def test_a_blank_capture_warns_and_is_not_extracted(client, tmp_path, monkeypatch):
    """A black frame means the game is in exclusive fullscreen (or excluded from
    capture). It must not arrive looking like an ordinary thumbnail, and it must
    not be fed to the extractor — speculating on it spends three model calls to
    produce confident nonsense."""
    import base64
    import io

    from PIL import Image

    import services.core.main as main_mod

    speculated: list[str] = []

    async def spy(session, capture_id):
        speculated.append(capture_id)

    monkeypatch.setattr(main_mod, "_speculate", spy)

    buf = io.BytesIO()
    Image.new("RGB", (320, 200), (0, 0, 0)).save(buf, format="PNG")
    sid = client.post("/api/sessions", json={"puzzle_type": "factory"}).json()["id"]

    with client.websocket_connect(f"/ws/{sid}") as ws:
        r = client.post(
            "/api/captures/upload",
            json={"session_id": sid, "image_base64": base64.b64encode(buf.getvalue()).decode()},
        )
        assert r.status_code == 200
        assert r.json()["warnings"], "a black frame was stored with no warning"
        kinds = [json.loads(ws.receive_text())["type"] for _ in range(2)]

    assert kinds == ["capture", "warning"]
    assert not speculated, "extraction was started on a blank frame"


def test_pasting_a_key_actually_stores_it(client, monkeypatch, tmp_path):
    """This endpoint had never worked: it called store_key with the wrong arity
    and read an attribute KeyInfo does not have, and a broad except turned both
    into a 503 blaming validation. A bug in the handler must not disguise itself
    as an upstream outage."""
    import services.core.settings.keys as keys_mod

    stored: dict[str, object] = {}

    async def fake_validate(key, **kw):
        return keys_mod.KeyInfo(valid=True, label="test key", limit=10.0, usage=2.5)

    monkeypatch.setattr(keys_mod, "validate_key", fake_validate)
    monkeypatch.setattr(keys_mod, "store_key", lambda k, i: stored.update(key=k, info=i))
    monkeypatch.setattr(keys_mod, "storage_backend", lambda: "keyring:Fake")

    r = client.post("/api/settings/key", json={"key": "sk-or-v1-" + "a" * 32})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["valid"] is True and body["backend"] == "keyring:Fake"
    assert body["remaining_credit"] == 7.5
    assert stored["key"].startswith("sk-or-v1-")


def test_a_refused_unprotected_store_is_not_reported_as_an_outage(client, monkeypatch):
    """On Windows the fallback refuses to write an unprotected key file. That is
    a 500 about storage, not a 503 about OpenRouter being unreachable."""
    import services.core.settings.keys as keys_mod

    async def fake_validate(key, **kw):
        return keys_mod.KeyInfo(valid=True, label="k")

    def refuse(key, info):
        raise keys_mod.KeyStorageError("could not restrict the ACL; refusing to store")

    monkeypatch.setattr(keys_mod, "validate_key", fake_validate)
    monkeypatch.setattr(keys_mod, "store_key", refuse)

    r = client.post("/api/settings/key", json={"key": "sk-or-v1-" + "b" * 32})
    assert r.status_code == 500
    assert "refusing to store" in r.json()["error"]
