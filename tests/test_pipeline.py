"""Orchestration: fan-out concurrency, speculation, delta, provenance, assembly."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from PIL import Image

from services.core.capture.grab import Rect, Tile, board_overview, capture_from_image
from services.core.extract import prompts
from services.core.extract.delta import DeltaChanges, apply_changes, extract_delta_detailed
from services.core.extract.pipeline import (
    ExtractRoles,
    clear_speculations,
    extract,
    inflight_count,
    speculate,
)
from services.core.extract.schemas import flatten
from services.core.llm.protocol import LLMResponse, Usage
from services.solvers.factory.model import FactoryState

MODELS = ["famA/one", "famB/two", "famC/three"]
ROLES = ExtractRoles(jury=MODELS, topology="famA/one", delta="famA/one")


@pytest.fixture(autouse=True)
def _no_leftover_speculation():
    clear_speculations()
    yield
    clear_speculations()


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


def machine_doc(mid: str, **over: Any) -> dict[str, Any]:
    doc = dict(
        id=mid,
        kind="maker",
        name=f"Maker {mid}",
        recipes=[
            {
                "id": f"{mid}-r1",
                "name": "ore->plate",
                "inputs": {"ore": 2},
                "output_item": "plate",
                "output_qty": 1,
                "sale_price": 0.0,
                "purchase_cost": 0.0,
                "production_cost": 3.0,
            }
        ],
        selected_recipe_id=f"{mid}-r1",
        output_setting=2,
        output_max=6,
        storage_max=20,
        production_hours=1,
        installed_mods=[],
        current_storage={"ore": 1},
        x=10.0,
        y=20.0,
    )
    doc.update(over)
    return doc


class SleepyClient:
    """Every call takes ``delay_ms``; answers are chosen by schema name."""

    def __init__(self, delay_ms: float = 200.0, *, overrides: dict[str, Any] | None = None) -> None:
        self.delay_ms = delay_ms
        self.overrides = overrides or {}
        self.calls: list[tuple[str, str]] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def complete(self, *, model, messages, response_format=None, **kw):
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        hint = _hint_of(messages)
        self.calls.append((model, f"{name}:{hint}"))
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self.delay_ms / 1000.0)
        finally:
            self.concurrent -= 1
        doc = self.overrides.get(name)
        if callable(doc):
            doc = doc(model, hint)
        if doc is None:
            doc = _default_doc(name, hint)
        return LLMResponse(
            content=json.dumps(doc), parsed=doc, model=model, usage=Usage(), latency_ms=self.delay_ms
        )

    def stream(self, **kw):  # pragma: no cover
        raise NotImplementedError


def _hint_of(messages) -> str:
    for msg in messages:
        for part in msg.parts or ():
            if part.get("type") == "text" and "labelled" in part.get("text", ""):
                return part["text"].split("labelled '")[1].split("'")[0]
    return ""


def _default_doc(schema_name: str, hint: str) -> dict[str, Any]:
    if schema_name == "MachinePanelExtraction":
        return machine_doc(hint or "M1")
    if schema_name == "TopologyExtraction":
        return {"edges": [{"src": "M1", "dst": "M2"}, {"src": "M2", "dst": "M3"}]}
    if schema_name == "FactoryHUDExtraction":
        return {"money": 500.0, "horizon_hours": 24, "hour_now": 3}
    if schema_name == "DeltaChanges":
        return {"changes": []}
    return {}


def board(n_machines: int = 6, *, with_hud: bool = True, size=(1200, 800)):
    img = Image.new("RGB", size, (20, 24, 30))
    tiles = [
        Tile(f"M{i + 1}", Rect(10 + (i % 3) * 200, 10 + (i // 3) * 200, 180, 180), "factory_machine")
        for i in range(n_machines)
    ]
    if with_hud:
        tiles.append(Tile("hud", Rect(0, 0, size[0], 60), "factory_hud"))
    tiles.append(board_overview(size))
    return capture_from_image(img, tiles=tiles, puzzle_type="factory", capture_id="cap_test")


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------


@pytest.mark.latency
async def test_six_tiles_run_concurrently_not_serially(capsys):
    """6 tiles x 3 jurors + hud + topology at 200 ms each: bounded by one call."""
    client = SleepyClient(200.0)
    cap = board(6, with_hud=True)
    t0 = time.perf_counter()
    result = await extract(client, ROLES, [cap], "factory")
    ms = (time.perf_counter() - t0) * 1000.0

    n_calls = len(client.calls)
    with capsys.disabled():
        print(
            f"\n[latency] 8 tiles / {n_calls} model calls at 200 ms each: "
            f"{ms:.0f} ms wall (serial would be {n_calls * 200} ms); "
            f"peak concurrency {client.max_concurrent}"
        )
    assert n_calls == 6 * 3 + 3 + 1  # machines, hud jury, one topology call
    assert ms < 600.0
    assert ms < n_calls * 200 / 4  # nowhere near serial
    assert client.max_concurrent >= 10
    assert result.stage_ms["total"] == pytest.approx(result.elapsed_ms, rel=0.05)
    assert len(result.factory_state.machines) == 6


async def test_stage_timings_are_emitted_and_returned():
    seen: list[tuple[str, float]] = []
    result = await extract(
        SleepyClient(5.0), ROLES, [board(2)], "factory", on_stage=lambda s, ms: seen.append((s, ms))
    )
    names = {s for s, _ in seen}
    assert {"prepare", "fanout", "assemble", "total"} <= names
    assert any(s.startswith("tile:") for s in names)
    assert set(result.stage_ms) == names
    assert result.stage_ms["total"] > 0


# --------------------------------------------------------------------------
# assembly + provenance
# --------------------------------------------------------------------------


async def test_assembles_into_factory_state_with_provenance_and_unresolved_fields():
    client = SleepyClient(0.0)
    cap = board(3)
    result = await extract(client, ROLES, [cap], "factory")

    state = result.factory_state
    assert isinstance(state, FactoryState)
    assert [m.id for m in state.machines] == ["M1", "M2", "M3"]
    assert state.starting_money == 500.0 and state.horizon_hours == 24
    assert {(e.src, e.dst) for e in state.edges} == {("M1", "M2"), ("M2", "M3")}
    assert state.machines[0].recipes[0].inputs == {"ore": 2}

    # provenance: which model voted what, from which capture and crop
    prov = result.provenance_for("machines[M2].output_max")
    assert prov is not None
    assert prov.votes == {m: 6 for m in MODELS}
    assert prov.capture_id == "cap_test" and prov.tile == "M2"
    assert prov.crop_id and prov.crop_id.startswith("crop_")
    assert prov.box["w"] == 180 and prov.status == "unanimous"
    assert prov.auto_confirmed is True

    assert "machines[M2].output_max" in result.auto_confirmed
    assert result.auto_confirm_rate() > 0.5
    # topology is a single cheap call, so it is never auto-confirmed
    assert any(p.startswith("topology.") for p in result.disputed)
    assert result.unresolved == []  # nothing the assembler needed was missing


async def test_a_needed_field_nobody_could_read_is_never_silently_defaulted():
    """The one null that must reach a human: unreadable *and* load-bearing."""
    client = SleepyClient(
        0.0,
        overrides={
            "MachinePanelExtraction": lambda model, hint: machine_doc(
                hint, storage_max=None, output_max=None
            ),
            "FactoryHUDExtraction": {"money": None, "horizon_hours": None, "hour_now": None},
        },
    )
    result = await extract(client, ROLES, [board(2)], "factory")

    for path in ("hud.money", "machines[M1].storage_max", "machines[M1].output_max"):
        prov = result.provenance_for(path)
        assert prov.status == "unanimous" and prov.value is None
        assert prov.auto_confirmed is False, f"{path} was silently defaulted"
        assert path not in result.auto_confirmed
        assert path in result.unresolved
        assert "solver needs it" in prov.reason
    assert set(result.needs_review()) >= set(result.unresolved)
    # the solver still got a state, with the gaps named rather than guessed
    assert result.factory_state.machines[0].storage_max == 0


async def test_a_field_the_solver_never_reads_auto_confirms_as_absent():
    """The 90-second verification killer: nobody is asked about fields that
    are not on the screen and that nothing downstream reads."""
    client = SleepyClient(
        0.0,
        overrides={
            "MachinePanelExtraction": lambda model, hint: machine_doc(
                hint, name=None, current_storage=None, installed_mods=[]
            ),
            "FactoryHUDExtraction": {"money": 500.0, "horizon_hours": 24, "hour_now": None},
        },
    )
    result = await extract(client, ROLES, [board(2)], "factory")

    for path in ("hud.hour_now", "machines[M1].name", "machines[M1].current_storage"):
        prov = result.provenance_for(path)
        assert prov.value is None and prov.status == "unanimous"
        assert prov.auto_confirmed is True, f"{path} costs a keystroke for nothing"
        assert path in result.auto_confirmed
        assert path not in result.disputed and path not in result.unresolved
    # an empty list is a reading, not an absence, and it survives the merge
    assert result.provenance_for("machines[M1].installed_mods").value == []
    assert result.unresolved == []


async def test_no_auto_confirmed_field_is_ever_wrong_or_load_bearing_and_null():
    """The two safety properties, asserted together over a whole board."""
    client = SleepyClient(
        0.0,
        overrides={
            "MachinePanelExtraction": lambda model, hint: machine_doc(hint, output_max=None),
        },
    )
    result = await extract(client, ROLES, [board(3)], "factory")
    checked = 0
    for path in result.auto_confirmed:
        prov = result.provenance_for(path)
        # every auto-confirm is backed by unanimous agreement...
        assert prov.status == "unanimous"
        assert len(set(map(repr, prov.votes.values()))) == 1
        # ...and no auto-confirmed null is a field the assembler needed
        assert path not in result.unresolved
        if path.startswith("machines["):
            mid, _, leaf = path[len("machines[") :].partition("].")
            truth = flatten(machine_doc(mid, output_max=None))
            assert leaf in truth, f"{path} is not even a field of the panel"
            assert prov.value == truth[leaf], f"{path} auto-confirmed wrongly"
            checked += 1
    assert checked > 20
    assert "machines[M1].output_max" in result.unresolved
    assert not (set(result.auto_confirmed) & set(result.disputed))


async def test_builder_board_assembles_into_a_builder_puzzle():
    img = Image.new("RGB", (900, 600), (20, 24, 30))
    tiles = [
        Tile("P1", Rect(0, 0, 200, 150), "builder_part"),
        Tile("P2", Rect(200, 0, 200, 150), "builder_part"),
        Tile("obstacles", Rect(400, 0, 300, 300), "builder_obstacles"),
        Tile("rules", Rect(400, 300, 300, 200), "builder_instructions"),
    ]
    cap = capture_from_image(img, tiles=tiles, puzzle_type="builder")

    def part(model, hint):
        return {
            "id": hint,
            "name": hint,
            "weight": 3,
            "qty_available": 1,
            "attributes": {"speed": 4},
            "cost": 5,
        }

    client = SleepyClient(
        0.0,
        overrides={
            "PartExtraction": part,
            "ObstaclesPanelExtraction": {
                "obstacles": [
                    {"id": "O1", "order": 1, "requires": {"speed": 6}, "name": "stage 1"}
                ]
            },
            "BuilderRulesExtraction": {
                "weight_max": 10,
                "slot_max": None,
                "money_max": None,
                "duplicates_allowed": False,
                "attribute_names": ["speed"],
                "obstacle_ordering": "unordered",
                "obstacle_semantics": None,
                "aggregation": "sum",
                "failure_mode": None,
                "objective": "count_valid",
                "quoted_rule_text": ["Total weight may not exceed 10."],
            },
        },
    )
    result = await extract(client, ROLES, [cap], "builder")
    puzzle = result.builder_puzzle
    assert [p.id for p in puzzle.parts] == ["P1", "P2"]
    assert [o.id for o in puzzle.obstacles] == ["O1"]
    assert puzzle.rules.weight_max == 10 and puzzle.rules.aggregation == "sum"
    assert result.factory_state is None


# --------------------------------------------------------------------------
# speculation
# --------------------------------------------------------------------------


async def test_speculation_starts_early_and_extract_awaits_the_same_work(capsys):
    client = SleepyClient(120.0)
    cap = board(2)
    calls_per_pass = 2 * 3 + 3 + 1  # 2 machine juries + hud jury + topology

    task = speculate(client, ROLES, cap, "factory")
    await asyncio.sleep(0.04)  # the user is still reaching for the hotkey
    assert inflight_count() == 1

    t0 = time.perf_counter()
    result = await extract(client, ROLES, [cap], "factory")
    waited_ms = (time.perf_counter() - t0) * 1000.0

    assert result.speculated is True
    assert len(client.calls) == calls_per_pass  # one pass only: no duplicated work
    assert task.done()
    assert (await task).factory_state.machines[0].id == "M1"

    fresh = board(2)
    fresh.capture_id = "cap_cold"
    t0 = time.perf_counter()
    await extract(client, ROLES, [fresh], "factory")
    cold_ms = (time.perf_counter() - t0) * 1000.0
    with capsys.disabled():
        print(
            f"\n[latency] speculated wait {waited_ms:.0f} ms vs cold start {cold_ms:.0f} ms "
            f"({calls_per_pass} calls per pass, no duplication)"
        )
    assert waited_ms < cold_ms
    assert len(client.calls) == 2 * calls_per_pass


async def test_speculating_twice_reuses_the_in_flight_task():
    client = SleepyClient(60.0)
    cap = board(2)
    a = speculate(client, ROLES, cap, "factory")
    b = speculate(client, ROLES, cap, "factory")
    assert a is b
    await a
    assert len(client.calls) == 2 * 3 + 3 + 1


async def test_a_different_capture_is_not_served_from_the_cache():
    client = SleepyClient(0.0)
    first, second = board(1), board(1)
    second.capture_id = "cap_other"
    await speculate(client, ROLES, first, "factory")
    calls_after_first = len(client.calls)
    await extract(client, ROLES, [second], "factory")
    assert len(client.calls) > calls_after_first


# --------------------------------------------------------------------------
# delta
# --------------------------------------------------------------------------


def previous_state() -> dict[str, Any]:
    return {
        "hud": {"money": 500.0, "horizon_hours": 24, "hour_now": 3},
        "machines": [machine_doc("M1"), machine_doc("M2")],
        "topology": {"edges": [{"src": "M1", "dst": "M2"}]},
    }


async def test_delta_applies_known_paths_and_rejects_invented_ones():
    prev = previous_state()
    client = SleepyClient(
        0.0,
        overrides={
            "DeltaChanges": {
                "changes": [
                    {"path": "machines[M1].output_setting", "value": 5},
                    {"path": "hud.money", "value": 640.0},
                    {"path": "machines[M9].output_max", "value": 99},  # invented machine
                    {"path": "machines[M1].nonsense", "value": 1},  # invented field
                ]
            }
        },
    )
    result = await extract(client, ROLES, [board(2)], "factory", previous_state=prev)

    assert result.used_delta is True
    assert len(client.calls) == 1  # one call, not a whole re-read
    assert set(result.changed_paths) == {"machines[M1].output_setting", "hud.money"}
    assert set(result.rejected_paths) == {
        "machines[M9].output_max",
        "machines[M1].nonsense",
    }
    assert "delta:rejected_paths" in result.errors

    flat = flatten(result.extraction)
    assert flat["machines[M1].output_setting"] == 5
    assert isinstance(flat["machines[M1].output_setting"], int)  # not 5.0
    assert flat["hud.money"] == 640.0
    assert "machines[M9].output_max" not in flat
    assert result.factory_state.machine("M1").id == "M1"
    assert [m.id for m in result.factory_state.machines] == ["M1", "M2"]
    # changed fields are the ones the user is asked about; the rest carry over
    assert set(result.disputed) == set(result.changed_paths)
    assert "machines[M2].output_max" in result.auto_confirmed


async def test_delta_no_change_keeps_the_confirmed_state_intact():
    prev = previous_state()
    result = await extract(
        SleepyClient(0.0), ROLES, [board(2)], "factory", previous_state=prev
    )
    assert result.changed_paths == []
    assert flatten(result.extraction) == flatten(prev)


def test_apply_changes_is_pure_and_reports_rejections():
    prev = previous_state()
    out = apply_changes(
        prev,
        DeltaChanges.model_validate(
            {
                "changes": [
                    {"path": "hud.money", "value": 1.0},
                    {"path": "hud.gold", "value": 1.0},
                    {"path": "machines[M2].output_setting", "value": 2},  # already 2
                ]
            }
        ).changes,
    )
    assert out.changed == ["hud.money"]
    assert out.rejected == ["hud.gold"]
    assert out.no_ops == ["machines[M2].output_setting"]
    assert flatten(prev)["hud.money"] == 500.0  # input untouched


async def test_delta_failure_falls_back_to_the_previous_state():
    class Broken(SleepyClient):
        async def complete(self, **kw):
            raise RuntimeError("provider 500")

    prev = previous_state()
    out = await extract_delta_detailed(Broken(), "famA/one", prev, [], None)
    assert out.error and "provider 500" in out.error
    assert flatten(out.merged) == flatten(prev)
    assert out.changed == []


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def test_every_prompt_forbids_guessing_and_scopes_to_the_crop():
    for name in prompts.TARGETS:
        messages = prompts.build_messages(name, "data:image/png;base64,AAAA", hint="M1")
        blob = " ".join(
            [m.content or "" for m in messages]
            + [p.get("text", "") for m in messages for p in (m.parts or ())]
        )
        assert "null" in blob
        assert "never guess" in blob.lower() or "not guess" in blob.lower()
        assert "keystroke" in blob and "run" in blob
        assert "only what is inside this crop" in blob.lower()
        assert len(blob) < 1400, f"{name} prompt is too long: tokens are latency"


def test_prompts_carry_the_image_as_a_data_url_part():
    messages = prompts.factory_machine_prompt("data:image/png;base64,AAAA")
    parts = messages[-1].parts
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
