"""The three-model jury: what gets auto-confirmed, and what reaches the user.

Every test here runs against a scripted fake client -- no network, no display.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from services.core.extract.jury import JuryResult, run_jury, sort_edges
from services.core.extract.prompts import factory_machine_prompt
from services.core.extract.schemas import (
    FactoryExtraction,
    MachinePanelExtraction,
    TopologyExtraction,
    flatten,
)
from services.core.llm.protocol import LLMResponse, Usage

MODELS = ["famA/one", "famB/two", "famC/three"]
TIE = "famD/tiebreak"


class ScriptedClient:
    """Returns a canned document per model.  ``"hang"`` never returns."""

    def __init__(self, answers: dict[str, Any], *, latency_ms: float = 0.0) -> None:
        self.answers = answers
        self.latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, model, messages, response_format=None, provider=None, **kw):
        self.calls.append(
            {"model": model, "provider": provider, "response_format": response_format}
        )
        answer = self.answers.get(model)
        if answer == "hang":
            await asyncio.sleep(3600)
        if isinstance(answer, Exception):
            raise answer
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        return LLMResponse(
            content=json.dumps(answer),
            parsed=answer,
            model=model,
            usage=Usage(),
            latency_ms=self.latency_ms,
        )

    def stream(self, **kw):  # pragma: no cover
        raise NotImplementedError

    @property
    def models_called(self) -> list[str]:
        return [c["model"] for c in self.calls]


def machine(**over: Any) -> dict[str, Any]:
    base = dict(
        id="M1",
        kind="maker",
        name="Maker 1",
        recipes=[],
        selected_recipe_id=None,
        output_setting=3,
        output_max=6,
        storage_max=20,
        production_hours=1,
        installed_mods=[],
        current_storage=None,
        x=100.0,
        y=200.0,
    )
    base.update(over)
    return base


MESSAGES = factory_machine_prompt("data:image/png;base64,AAAA", hint="M1")


async def jury(client, **kw) -> JuryResult:
    return await run_jury(
        client, MODELS, MESSAGES, MachinePanelExtraction, timeout_s=kw.pop("timeout_s", 5.0), **kw
    )


# --------------------------------------------------------------------------


async def test_unanimous_fields_are_auto_confirmed_and_never_shown():
    doc = machine()
    result = await jury(ScriptedClient({m: doc for m in MODELS}))

    v = result.verdict("output_max")
    assert v.status == "unanimous" and v.value == 6 and v.auto_confirmed
    assert v.confidence == pytest.approx(1.0)
    assert set(v.votes) == set(MODELS)
    assert "output_max" in result.auto_confirmed
    assert "output_max" not in result.disputed
    assert result.unanimous_count >= 8
    assert result.split_count == 0
    # the merged document is ready for the assembler
    assert MachinePanelExtraction.model_validate(result.merged).output_max == 6


async def test_extraction_provider_sorts_by_latency_on_every_juror():
    client = ScriptedClient({m: machine() for m in MODELS})
    await jury(client)
    assert len(client.calls) == 3
    for call in client.calls:
        assert call["provider"]["sort"] == "latency"
        assert call["provider"]["require_parameters"] is True
        assert call["response_format"]["json_schema"]["strict"] is True


async def test_two_of_three_becomes_a_flagged_majority_not_an_auto_confirm():
    client = ScriptedClient(
        {
            MODELS[0]: machine(output_max=6),
            MODELS[1]: machine(output_max=6),
            MODELS[2]: machine(output_max=8),
        }
    )
    result = await jury(client)

    v = result.verdict("output_max")
    assert v.status == "majority"
    assert v.value == 6  # pre-filled with the majority value...
    assert v.auto_confirmed is False  # ...but the user still presses one key
    assert v.alternatives == [8]
    assert v.votes[MODELS[2]] == 8
    assert 0.5 < v.confidence < 1.0
    assert "output_max" in result.disputed
    assert "output_max" not in result.auto_confirmed


async def test_three_way_split_is_disputed_and_shown_empty():
    client = ScriptedClient(
        {
            MODELS[0]: machine(output_max=6),
            MODELS[1]: machine(output_max=7),
            MODELS[2]: machine(output_max=8),
        }
    )
    result = await jury(client)

    v = result.verdict("output_max")
    assert v.status == "split"
    assert v.value is None  # shown empty, never a coin flip between three readings
    assert sorted(v.alternatives) == [6, 7, 8]
    assert v.crop_id is None or isinstance(v.crop_id, str)
    assert result.split_count == 1
    assert "output_max" in result.disputed


async def test_the_crop_id_travels_with_every_verdict_for_the_zoom_ui():
    result = await run_jury(
        ScriptedClient({m: machine() for m in MODELS}),
        MODELS,
        MESSAGES,
        MachinePanelExtraction,
        crop_id="crop_deadbeef",
    )
    assert result.crop_id == "crop_deadbeef"
    assert all(v.crop_id == "crop_deadbeef" for v in result.verdicts)


async def test_tie_breaker_runs_only_on_a_split_and_only_for_those_fields():
    agree = ScriptedClient({m: machine() for m in MODELS} | {TIE: machine()})
    result = await jury(agree, tie_breaker=TIE)
    assert TIE not in agree.models_called  # nothing split: no extra call, no extra latency
    assert result.tie_breaker_used is False

    split = ScriptedClient(
        {
            MODELS[0]: machine(output_max=6),
            MODELS[1]: machine(output_max=7),
            MODELS[2]: machine(output_max=8),
            TIE: machine(output_max=7),
        }
    )
    result = await jury(split, tie_breaker=TIE)
    assert TIE in split.models_called
    assert result.tie_breaker_used is True
    v = result.verdict("output_max")
    assert v.status == "majority" and v.value == 7 and v.auto_confirmed is False
    # the tie-breaker was told which fields are in dispute
    ask = split.calls[-1]
    assert ask["model"] == TIE
    assert result.split_count == 0


async def test_tie_breaker_that_seconds_nobody_leaves_the_field_split():
    client = ScriptedClient(
        {
            MODELS[0]: machine(output_max=6),
            MODELS[1]: machine(output_max=7),
            MODELS[2]: machine(output_max=8),
            TIE: machine(output_max=9),
        }
    )
    result = await jury(client, tie_breaker=TIE)
    v = result.verdict("output_max")
    assert v.status == "split" and v.value is None
    assert "did not second" in v.reason


async def test_a_model_that_misses_the_timeout_simply_does_not_vote():
    client = ScriptedClient(
        {MODELS[0]: machine(), MODELS[1]: machine(), MODELS[2]: "hang"}
    )
    result = await run_jury(
        client, MODELS, MESSAGES, MachinePanelExtraction, timeout_s=0.15
    )
    assert result.voters == MODELS[:2]
    assert "timeout" in result.errors[MODELS[2]]
    v = result.verdict("output_max")
    assert v.status == "unanimous" and v.value == 6 and v.auto_confirmed
    # two of three is real agreement, but confidence records the thinner jury
    assert v.confidence < 1.0
    assert result.elapsed_ms < 1000


async def test_a_failing_model_is_recorded_and_the_rest_still_vote():
    client = ScriptedClient(
        {MODELS[0]: machine(), MODELS[1]: machine(), MODELS[2]: RuntimeError("502")}
    )
    result = await jury(client)
    assert "502" in result.errors[MODELS[2]]
    assert result.verdict("output_max").auto_confirmed


async def test_majority_null_is_disputed_never_auto_confirmed():
    """A model saying 'not legible' is information, not a confirmed null."""
    client = ScriptedClient(
        {
            MODELS[0]: machine(storage_max=None),
            MODELS[1]: machine(storage_max=None),
            MODELS[2]: machine(storage_max=20),
        }
    )
    result = await jury(client)
    v = result.verdict("storage_max")
    assert v.status == "majority" and v.value is None
    assert v.auto_confirmed is False
    assert "storage_max" in result.disputed
    assert "not legible" in v.reason


async def test_unanimous_null_is_also_disputed():
    client = ScriptedClient({m: machine(storage_max=None) for m in MODELS})
    result = await jury(client)
    v = result.verdict("storage_max")
    assert v.status == "unanimous" and v.value is None
    assert v.auto_confirmed is False and "storage_max" in result.disputed
    assert v.confidence < 1.0


async def test_voting_is_per_field_path_even_when_machines_are_listed_in_a_different_order():
    a = {
        "hud": None,
        "topology": None,
        "machines": [machine(id="M1", output_max=6), machine(id="M2", output_max=4)],
    }
    b = {
        "hud": None,
        "topology": None,
        "machines": [machine(id="M2", output_max=4), machine(id="M1", output_max=6)],
    }
    c = {
        "hud": None,
        "topology": None,
        "machines": [machine(id="M2", output_max=4), machine(id="M1", output_max=9)],
    }
    result = await run_jury(
        ScriptedClient({MODELS[0]: a, MODELS[1]: b, MODELS[2]: c}),
        MODELS,
        MESSAGES,
        FactoryExtraction,
        timeout_s=5.0,
    )
    # M2 agreed by all three despite sitting in a different list position
    assert result.verdict("machines[M2].output_max").status == "unanimous"
    assert result.verdict("machines[M2].output_max").auto_confirmed
    # M1 is 2-1, and the odd reading is attributed to the right model
    m1 = result.verdict("machines[M1].output_max")
    assert m1.status == "majority" and m1.value == 6
    assert m1.votes[MODELS[2]] == 9


async def test_edge_normalizer_makes_graph_order_irrelevant():
    a = {"edges": [{"src": "M1", "dst": "M2"}, {"src": "M2", "dst": "M3"}]}
    b = {"edges": [{"src": "M2", "dst": "M3"}, {"src": "M1", "dst": "M2"}]}
    result = await run_jury(
        ScriptedClient({MODELS[0]: a, MODELS[1]: b, MODELS[2]: a}),
        MODELS,
        MESSAGES,
        TopologyExtraction,
        normalizer=sort_edges,
        timeout_s=5.0,
    )
    assert result.split_count == 0
    assert all(v.status == "unanimous" for v in result.verdicts)
    edges = TopologyExtraction.model_validate(result.merged).edges
    assert {(e["src"], e["dst"]) for e in edges} == {("M1", "M2"), ("M2", "M3")}


async def test_sort_edges_is_a_pure_reindex():
    flat = flatten({"edges": [{"src": "B", "dst": "C"}, {"src": "A", "dst": "B"}]})
    assert sort_edges(flat) == {
        "edges[A->B].src": "A",
        "edges[A->B].dst": "B",
        "edges[B->C].src": "B",
        "edges[B->C].dst": "C",
    }


async def test_all_models_failing_yields_an_empty_but_valid_result():
    client = ScriptedClient({m: RuntimeError("down") for m in MODELS})
    result = await jury(client)
    assert result.verdicts == [] and result.merged == {}
    assert set(result.errors) == set(MODELS)
    assert result.auto_confirmed == [] and result.voters == []


async def test_typical_board_leaves_only_a_few_fields_for_the_user():
    """Thirty-five of forty fields never reach the user (spec 6.2)."""
    truth = machine(
        selected_recipe_id="r1",
        current_storage={"ore": 4},
        recipes=[
            {
                "id": "r1",
                "name": "ore->plate",
                "inputs": {"ore": 2},
                "output_item": "plate",
                "output_qty": 1,
                "sale_price": 0.0,
                "purchase_cost": 0.0,
                "production_cost": 3.0,
            }
        ],
    )
    sloppy = json.loads(json.dumps(truth))
    sloppy["storage_max"] = 21  # one model misreads one field
    result = await jury(
        ScriptedClient({MODELS[0]: truth, MODELS[1]: truth, MODELS[2]: sloppy})
    )
    total = len(result.verdicts)
    assert total >= 15
    assert len(result.disputed) <= 3
    assert result.review_rate() < 0.25
