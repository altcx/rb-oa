"""End-to-end through the real server, the real warm worker and the real solvers.

No LLM anywhere: state is hand-entered, which is exactly the path the spec calls
first-class (2.1 — a verified solver with manual data entry is a usable tool).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from services.core import session as session_mod
from services.core.main import app

pytest.importorskip("services.solvers.factory.sim")
pytest.importorskip("services.solvers.factory.optimize")
pytest.importorskip("services.solvers.factory.calibrate")


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


def factory_state() -> dict:
    """Supplier -> Maker -> Seller, the smallest graph with a real bottleneck."""
    from services.solvers.factory.model import (
        Edge,
        FactoryState,
        Item,
        Machine,
        MachineKind,
        Recipe,
    )

    state = FactoryState(
        machines=[
            Machine(
                id="sup",
                kind=MachineKind.SUPPLIER,
                recipes=[Recipe(id="ore", output_item="ore", output_qty=1, purchase_cost=2.0)],
                storage_max=20,
                output_max=5,
                production_hours=1,
                y=0,
            ),
            Machine(
                id="mak",
                kind=MachineKind.MAKER,
                recipes=[
                    Recipe(
                        id="ingot",
                        inputs={"ore": 2},
                        output_item="ingot",
                        output_qty=1,
                        production_cost=1.0,
                    )
                ],
                storage_max=20,
                output_max=4,
                production_hours=1,
                y=1,
            ),
            Machine(
                id="sell",
                kind=MachineKind.SELLER,
                recipes=[Recipe(id="s", inputs={"ingot": 1}, sale_price=20.0)],
                storage_max=0,
                output_max=4,
                y=2,
            ),
        ],
        edges=[Edge(src="sup", dst="mak"), Edge(src="mak", dst="sell")],
        items=[Item(id="ore"), Item(id="ingot")],
        starting_money=200.0,
        horizon_hours=12,
    )
    return state.model_dump(mode="json")


def _poll(client: TestClient, job_id: str, timeout_s: float = 60.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["done"]:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


def test_factory_end_to_end(client):
    sid = client.post("/api/sessions", json={"puzzle_type": "factory"}).json()["id"]
    state = factory_state()
    assert client.post(f"/api/sessions/{sid}/state", json={"state": state}).json()["ok"]

    # 1. the bound comes first: it is fast and it names the binding constraint
    bound = client.post("/api/solve/bound", json={"session_id": sid}).json()
    assert bound["ceiling"] > 0
    assert bound["bottleneck_machine"] in {"sup", "mak", "sell", None}

    # 2. simulate the configuration currently on the board
    from services.solvers.factory.model import FactoryConfig, FactoryState

    config = FactoryConfig.from_state(FactoryState.model_validate(state)).model_dump(mode="json")
    sim = client.post("/api/solve/simulate", json={"session_id": sid, "config": config}).json()
    assert len(sim["money_by_hour"]) == state["horizon_hours"] + 1
    assert sim["money_by_hour"][0] == pytest.approx(state["starting_money"])

    # 3. the optimizer is gated until a recorded run matches the simulator
    blocked = client.post("/api/solve/optimize", json={"session_id": sid, "seconds": 0.3})
    assert blocked.status_code == 412

    # 4. calibrate against that same series: it must match to the dollar
    observed = sim["money_by_hour"][1:]
    cal = client.post(
        "/api/solve/calibrate", json={"session_id": sid, "config": config, "observed": observed}
    ).json()
    assert cal["matched"] is True, cal.get("message")
    assert all(h["ok"] for h in cal["per_hour"])

    # 5. now the optimizer runs, and cannot beat the LP ceiling
    job_id = client.post("/api/solve/optimize", json={"session_id": sid, "seconds": 2.0}).json()[
        "job_id"
    ]
    job = _poll(client, job_id)
    assert job["error"] is None, job["error"]
    result = job["result"]
    assert result["best_value"] >= result["baseline_value"] - 1e-6
    assert result["best_value"] <= bound["ceiling"] + 1e-6

    # 6. every optimizer action is a concrete UI action the user can execute
    for action in result["actions"]:
        assert action["target"] in {"sup", "mak", "sell"}
        assert action["setting"] in {"recipe", "output", "mod", "hourly_output"}
        assert action["value"] is not None

    # 7. the score lands on the leaderboard and the high-water mark is pinned
    entries = client.get(f"/api/sessions/{sid}/leaderboard").json()["entries"]
    assert entries and sum(e["is_high_water"] for e in entries) == 1


def test_calibration_mismatch_is_not_matched(client):
    """A simulator that silently diverges from the game is the failure mode the
    gate exists to catch, so a wrong series must not pass it."""
    sid = client.post("/api/sessions", json={"puzzle_type": "factory"}).json()["id"]
    state = factory_state()
    client.post(f"/api/sessions/{sid}/state", json={"state": state})
    from services.solvers.factory.model import FactoryConfig, FactoryState

    config = FactoryConfig.from_state(FactoryState.model_validate(state)).model_dump(mode="json")
    bogus = [999.0] * state["horizon_hours"]
    cal = client.post(
        "/api/solve/calibrate", json={"session_id": sid, "config": config, "observed": bogus}
    ).json()
    assert cal["matched"] is False
    assert client.post("/api/solve/optimize", json={"session_id": sid}).status_code == 412


def test_builder_end_to_end(client):
    pytest.importorskip("services.solvers.builder.solve")
    from services.core.rules.dsl import BuilderRules
    from services.solvers.builder.model import BuilderPuzzle, Obstacle, Part

    puzzle = BuilderPuzzle(
        parts=[
            Part(id="a", weight=3, attributes={"power": 4}),
            Part(id="b", weight=2, attributes={"power": 2, "grip": 3}),
            Part(id="c", weight=4, attributes={"grip": 5}),
            Part(id="d", weight=0, attributes={}),
        ],
        obstacles=[Obstacle(id="hill", order=1, requires={"power": 4, "grip": 3})],
        rules=BuilderRules(
            weight_max=9,
            duplicates_allowed=False,
            obstacle_semantics="threshold",
            aggregation="sum",
            failure_mode="all_must_pass",
            objective="count_valid",
        ),
    )
    sid = client.post("/api/sessions", json={"puzzle_type": "builder"}).json()["id"]
    client.post(f"/api/sessions/{sid}/state", json={"state": puzzle.model_dump(mode="json")})
    job_id = client.post("/api/solve/builder", json={"session_id": sid}).json()["job_id"]
    report = _poll(client, job_id)["result"]
    assert report["total_valid"] > 0
    # part "d" weighs nothing and violates nothing, so it doubles the count
    assert "d" in report["free_parts"]
    assert report["free_multiplier"] == 2
    assert report["binding_obstacle"] == "hill"
