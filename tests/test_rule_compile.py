"""Rule compilation: nulls survive as unknowns, every unknown is executable.

No network anywhere in here: the client is a stub that returns a canned payload.
"""

from __future__ import annotations

import json

import pytest

from services.core.llm.protocol import LLMResponse
from services.core.rules import compile as C
from services.core.rules.dsl import (
    BUILDER_FLAG_OPTIONS,
    BUILDER_FLAG_ORDER,
    FACTORY_FLAG_OPTIONS,
    BuilderRules,
    FactoryRuleState,
)


class StubClient:
    """Records the request it was given and replays a canned parse."""

    def __init__(self, payload, *, error: str | None = None):
        self.payload = payload
        self.error = error
        self.kwargs: dict = {}

    async def complete(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            return LLMResponse(error=self.error, latency_ms=11.0, model="stub/model")
        return LLMResponse(
            content=json.dumps(self.payload),
            parsed=self.payload,
            latency_ms=42.0,
            model="stub/model",
        )


CANNED = {
    # stated in the instructions
    "weight_max": 100,
    "slot_max": 5,
    "money_max": None,
    "aggregation": "sum",
    "attribute_names": ["power", "grip"],
    # NOT stated -> null, and null must survive
    "duplicates_allowed": None,
    "obstacle_ordering": None,
    "obstacle_semantics": None,
    "failure_mode": None,
    "objective": None,
    "quoted_rule_text": [
        "Your build may weigh at most 100.",
        "You may fit up to 5 parts.",
        "Stats from all parts are added together.",
    ],
}

IMAGES = ["data:image/png;base64,AAAA"]


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


async def test_nulls_survive_as_unknowns_with_experiments():
    client = StubClient(CANNED)
    rules, unknowns = await C.compile_builder_rules(client, "stub/model", IMAGES)

    assert rules.weight_max == 100
    assert rules.slot_max == 5
    assert rules.aggregation == "sum"
    assert rules.attribute_names == ["power", "grip"]

    # every unstated field stayed None rather than being guessed
    assert rules.obstacle_semantics is None
    assert rules.duplicates_allowed is None
    assert rules.objective is None

    flags = [u.flag for u in unknowns]
    assert flags == [
        f for f in BUILDER_FLAG_ORDER if f not in {"aggregation"}
    ], "unknowns must be reported highest-leverage first"
    assert "aggregation" not in flags

    for u in unknowns:
        assert u.experiment is not None, f"{u.flag} has no experiment"
        exp = u.experiment
        assert exp.flag == u.flag
        assert len(exp.instruction) > 20
        assert exp.observable
        assert exp.estimated_seconds <= 15
        # the experiment must say what you would see for EVERY option
        assert set(exp.discriminator) == {str(o) for o in u.options}
        assert all(v.strip() for v in exp.discriminator.values())


async def test_request_uses_strict_schema_and_pins_require_parameters():
    client = StubClient(CANNED)
    await C.compile_builder_rules(client, "stub/model", IMAGES)
    kwargs = client.kwargs

    rf = kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert kwargs["provider"]["require_parameters"] is True
    # latency sort is pointless for a once-per-puzzle cached call
    assert "sort" not in kwargs["provider"]

    parts = kwargs["messages"][1].parts
    assert parts[-1]["type"] == "image_url"
    system = kwargs["messages"][0].content
    assert "null" in system and "verbatim" in system.lower()


async def test_quotes_are_kept_for_every_resolved_field():
    client = StubClient(CANNED)
    out = await C.compile_builder_rules_detailed(client, "stub/model", IMAGES)
    assert len(out.quotes) == 3
    assert out.latency_ms == 42.0
    assert out.error is None


async def test_model_failure_degrades_to_all_unknown():
    client = StubClient(None, error="http 503")
    rules, unknowns = await C.compile_builder_rules(client, "stub/model", IMAGES)
    assert rules == BuilderRules()
    assert [u.flag for u in unknowns] == list(BUILDER_FLAG_ORDER)


async def test_no_images_is_an_error_not_a_crash():
    out = await C.compile_builder_rules_detailed(StubClient(CANNED), "m", [])
    assert out.error and "no instruction images" in out.error
    assert len(out.unknowns) == len(BUILDER_FLAG_ORDER)


# ---------------------------------------------------------------------------
# the experiment library (pure, no client at all)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", sorted({**BUILDER_FLAG_OPTIONS, **FACTORY_FLAG_OPTIONS}))
def test_every_flag_has_a_hand_written_experiment(flag):
    options = C.ALL_FLAG_OPTIONS[flag]
    exp = C.experiment_for(flag, list(options))
    assert flag in C.EXPERIMENTS, f"{flag} falls back to the generic experiment"
    assert exp.flag == flag
    assert exp.estimated_seconds <= 15
    assert set(exp.discriminator) == {str(o) for o in options}
    # an instruction, not a strategy sentence
    assert len(exp.instruction.split()) >= 8


def test_obstacle_semantics_experiment_is_the_spec_one():
    exp = C.experiment_for("obstacle_semantics")
    assert "obstacle 2" in exp.instruction
    assert "consumed" in exp.discriminator["consumable"]
    assert "spent nothing" in exp.discriminator["threshold"]


def test_experiment_narrows_to_the_supplied_options():
    exp = C.experiment_for("aggregation", ["sum", "min"])
    assert exp.options == ["sum", "min"]
    assert set(exp.discriminator) == {"sum", "min"}


def test_unknown_flag_still_gets_an_executable_experiment():
    exp = C.experiment_for("some_future_flag", ["a", "b"])
    assert exp.flag == "some_future_flag"
    assert set(exp.discriminator) == {"a", "b"}
    assert exp.instruction


def test_factory_unknowns_track_resolved_by():
    state = FactoryRuleState(resolved_by={"priority_metric": "calibration:run-3"})
    unknowns = C.factory_unknowns(state)
    flags = [u.flag for u in unknowns]
    assert "priority_metric" not in flags
    assert "two_hour_consume_timing" in flags
    assert all(u.experiment is not None for u in unknowns)


def test_rules_from_extraction_is_pure():
    rules = C.rules_from_extraction(CANNED)
    assert rules.weight_max == 100
    assert rules.obstacle_semantics is None
    assert C.unknowns_for(rules)[0].flag == BUILDER_FLAG_ORDER[0]
