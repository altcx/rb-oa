"""The tool loop and the numeric-provenance guard, with a fake client.

No network, no solvers: the loop's job is dispatch, message bookkeeping and
cap enforcement, and that is exactly what is asserted here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from services.core.agent.guard import (
    extract_numeric_tokens,
    provenance_report,
    unsourced_numbers,
)
from services.core.agent.loop import AgentSession
from services.core.agent.tools import InMemorySessionStore, ToolRegistry
from services.core.llm.protocol import LLMResponse, StreamEvent, ToolCall


class FakeClient:
    """Replays scripted assistant turns as a stream, and charges for them."""

    def __init__(self, script: list[LLMResponse], *, cost_per_turn: float = 0.0, repeat=False):
        self.script = list(script)
        self.repeat = repeat
        self.cost_per_turn = cost_per_turn
        self.spent = SimpleNamespace(cost=0.0, turns=0)
        self.requests: list[dict] = []

    async def stream(self, **kwargs):
        self.requests.append(kwargs)
        if self.script:
            response = self.script[0] if self.repeat else self.script.pop(0)
        else:
            response = LLMResponse(content="nothing left to do")
        self.spent.cost += self.cost_per_turn
        self.spent.turns += 1
        for word in (response.content or "").split():
            yield StreamEvent(kind="token", text=word + " ")
        yield StreamEvent(kind="done", response=response)


def build_session(script, **kw) -> tuple[AgentSession, InMemorySessionStore, FakeClient]:
    store = InMemorySessionStore()
    store.set_state("default", {"machines": [], "starting_money": 500.0})
    store.set_puzzle_type("default", "factory")
    registry = ToolRegistry(store, "default")
    client = FakeClient(script, **{k: v for k, v in kw.items() if k in {"cost_per_turn", "repeat"}})
    session = AgentSession(
        client,
        "vendor/model-x",
        registry,
        **{k: v for k, v in kw.items() if k not in {"cost_per_turn", "repeat"}},
    )
    return session, store, client


def call(name: str, args: dict, cid: str = "call_1") -> LLMResponse:
    return LLMResponse(content=None, tool_calls=[ToolCall(id=cid, name=name, arguments=args)])


async def drain(session, message: str):
    return [event async for event in session.run(message)]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


async def test_system_prompt_is_loaded_at_construction():
    session, _, _ = build_session([LLMResponse(content="hi")])
    assert session.messages[0].role == "system"
    assert "You are Puzzle Copilot." in session.messages[0].content
    assert "Never state a number you did not receive from a tool result" in (
        session.messages[0].content
    )


async def test_loop_dispatches_a_tool_then_terminates_on_content():
    script = [
        call("get_state", {"session_id": "default"}),
        LLMResponse(content="Set m1 output to 4.", finish_reason="stop"),
    ]
    session, _, client = build_session(script)
    events = await drain(session, "what is the state?")

    kinds = [e.kind for e in events]
    assert "tool_call_started" in kinds
    assert "tool_call_finished" in kinds
    assert kinds[-1] == "done"
    assert events[-1].stop_reason == "stop"

    started = next(e for e in events if e.kind == "tool_call_started")
    finished = next(e for e in events if e.kind == "tool_call_finished")
    assert started.tool == "get_state"
    assert started.arguments == {"session_id": "default"}
    assert finished.result["ok"] is True
    assert finished.result["state"]["starting_money"] == 500.0

    roles = [m.role for m in session.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = session.messages[3]
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.name == "get_state"
    assert json.loads(tool_msg.content)["ok"] is True
    # the second request carried the tool result back to the model
    assert len(client.requests) == 2
    assert any(m.role == "tool" for m in client.requests[1]["messages"])


async def test_tokens_stream_before_the_final_message():
    session, _, _ = build_session([LLMResponse(content="Set m1 output to 4.")])
    events = await drain(session, "go")
    tokens = [e.text for e in events if e.kind == "token"]
    assert tokens[0].strip() == "Set"
    assert "".join(tokens).strip() == "Set m1 output to 4."
    assert events[-2].kind == "assistant_message"


async def test_parallel_tool_calls_all_dispatch():
    script = [
        LLMResponse(
            tool_calls=[
                ToolCall(id="a", name="get_state", arguments={}),
                ToolCall(id="b", name="list_captures", arguments={}),
            ]
        ),
        LLMResponse(content="done"),
    ]
    session, _, _ = build_session(script)
    events = await drain(session, "go")
    finished = [e.tool for e in events if e.kind == "tool_call_finished"]
    assert finished == ["get_state", "list_captures"]
    assert [m.tool_call_id for m in session.messages if m.role == "tool"] == ["a", "b"]


async def test_unknown_tool_returns_a_structured_error_not_a_crash():
    script = [call("does_not_exist", {}), LLMResponse(content="ok")]
    session, _, _ = build_session(script)
    events = await drain(session, "go")
    finished = next(e for e in events if e.kind == "tool_call_finished")
    assert finished.result["ok"] is False
    assert "unknown tool" in finished.result["error"]
    assert events[-1].stop_reason == "stop"


async def test_missing_solver_is_reported_as_unavailable():
    script = [call("optimize_factory", {"seconds": 3}), LLMResponse(content="ok")]
    session, _, _ = build_session(script)
    events = await drain(session, "optimize")
    result = next(e for e in events if e.kind == "tool_call_finished").result
    if not result["ok"]:
        assert "not available" in result["error"] or "unavailable" in result
    assert events[-1].stop_reason == "stop"


# ---------------------------------------------------------------------------
# caps
# ---------------------------------------------------------------------------


async def test_max_turns_is_enforced_and_stops_cleanly():
    session, _, client = build_session(
        [call("get_state", {})], repeat=True, max_turns=3
    )
    events = await drain(session, "loop forever")
    assert events[-1].kind == "done"
    assert events[-1].stop_reason == "max_turns"
    assert "turn limit" in events[-1].text
    assert len([e for e in events if e.kind == "tool_call_finished"]) == 3
    assert len(client.requests) == 3
    assert events[-2].kind == "assistant_message"


async def test_cost_cap_is_enforced_and_stops_cleanly():
    session, _, client = build_session(
        [call("get_state", {})], repeat=True, cost_per_turn=0.05, cost_cap=0.06, max_turns=10
    )
    events = await drain(session, "spend it all")
    assert events[-1].stop_reason == "cost_cap"
    assert "cost cap" in events[-1].text
    # stopped as soon as the cap was passed, not after max_turns
    assert len(client.requests) == 2
    assert client.spent.cost == pytest.approx(0.10)


async def test_max_tokens_defaults_low_and_is_overridable():
    session, _, client = build_session([LLMResponse(content="ok")])
    await drain(session, "go")
    assert client.requests[0]["max_tokens"] == 700

    session2, _, client2 = build_session([LLMResponse(content="ok")], max_tokens=2000)
    await drain(session2, "go")
    assert client2.requests[0]["max_tokens"] == 2000


async def test_stream_error_ends_the_run_without_raising():
    class BrokenClient(FakeClient):
        async def stream(self, **kwargs):
            self.requests.append(kwargs)
            yield StreamEvent(kind="error", text="http 502: bad gateway")

    store = InMemorySessionStore()
    session = AgentSession(BrokenClient([]), "m", ToolRegistry(store))
    events = await drain(session, "go")
    assert events[-1].stop_reason == "error"
    assert "502" in events[-1].text


async def test_tools_are_advertised_to_the_model():
    session, _, client = build_session([LLMResponse(content="ok")])
    await drain(session, "go")
    names = {t.name for t in client.requests[0]["tools"]}
    assert {
        "list_captures",
        "extract_state",
        "get_state",
        "patch_state",
        "get_rules",
        "propose_experiment",
        "factory_upper_bound",
        "simulate_factory",
        "optimize_factory",
        "calibrate_factory",
        "solve_builder",
        "solve_all_interpretations",
        "compare_configs",
    } == names


# ---------------------------------------------------------------------------
# numeric provenance
# ---------------------------------------------------------------------------


def test_extract_numeric_tokens():
    assert extract_numeric_tokens("Set m1 output to 4, profit $1,240.50 by hour 12") == [
        "1",
        "4",
        "1240.50",
        "12",
    ]
    assert extract_numeric_tokens("no numbers here") == []


def test_number_absent_from_every_tool_result_is_flagged():
    tool_results = [
        {"tool": "factory_upper_bound", "ceiling": 980.0, "bottleneck_machine": "m3"}
    ]
    text = "Ceiling is 980. Expect roughly 1450 profit by hour 24."
    offenders = unsourced_numbers(text, tool_results)
    assert "1450" in offenders
    assert "980" not in offenders


def test_number_present_in_a_tool_result_is_not_flagged():
    tool_results = [
        {"tool": "optimize_factory", "best_value": 1450.0, "baseline_value": 1200.0},
        {"tool": "simulate_factory", "money_by_hour": [500.0, 610.25]},
    ]
    text = "Best 1450, up from 1200. End-of-run money 610.25."
    assert unsourced_numbers(text, tool_results) == []
    # ...but an hour index nobody reported is still a claim
    assert unsourced_numbers("Peak at hour 17.", tool_results) == ["17"]


def test_small_indices_are_still_checked():
    assert unsourced_numbers("set machine 7 to output 9", [{"machines": ["m1"]}]) == ["7", "9"]
    assert unsourced_numbers("set machine 7", [{"machine_id": "machine 7"}]) == []


def test_numbers_the_user_typed_are_acceptable_provenance():
    text = "You have 45 seconds left; I will use 45."
    assert unsourced_numbers(text, [], ["I have 45 seconds left"]) == []
    assert unsourced_numbers(text, []) == ["45"]


def test_rounding_a_tool_number_is_not_a_fabrication():
    tool_results = [{"final_money": 1234.56}]
    assert unsourced_numbers("You end on 1235.", tool_results) == []
    assert unsourced_numbers("You end on 1234.6.", tool_results) == []
    assert unsourced_numbers("You end on 1300.", tool_results) == ["1300"]


def test_provenance_report_shape():
    report = provenance_report("profit 300 in 4 hours", [{"profit": 300}])
    assert report["tokens"] == ["300", "4"]
    assert report["unsourced"] == ["4"]
    assert report["clean"] is False


async def test_session_checks_its_own_output():
    script = [call("get_state", {}), LLMResponse(content="You have 500.0 and 9 machines.")]
    session, _, _ = build_session(script)
    await drain(session, "go")
    offenders = session.unsourced_numbers("You have 500.0 and 9 machines.")
    assert "9" in offenders
    assert "500.0" not in offenders


# ---------------------------------------------------------------------------
# tools: session plumbing and trimming
# ---------------------------------------------------------------------------


async def test_list_captures_omits_image_payloads():
    from services.core.agent.tools import ToolRegistry

    store = InMemorySessionStore()
    store.add_capture("default", {"id": "c1", "region": "board", "data_url": "data:image/png;base64,AAAA"})
    result = await ToolRegistry(store).dispatch("list_captures", {})
    assert result["count"] == 1
    assert "data_url" not in result["captures"][0]
    assert result["captures"][0]["region"] == "board"


async def test_patch_state_records_a_user_correction():
    from services.core.agent.tools import ToolRegistry

    store = InMemorySessionStore()
    store.set_state("default", {"machines": [{"id": "m1", "output_max": 3}]})
    registry = ToolRegistry(store)
    result = await registry.dispatch(
        "patch_state",
        {"json_patch": [{"op": "replace", "path": "/machines/0/output_max", "value": 5}]},
    )
    assert result["ok"] is True
    assert store.get_state("default")["machines"][0]["output_max"] == 5


async def test_get_rules_reports_unresolved_flags_with_experiments():
    from services.core.agent.tools import ToolRegistry
    from services.core.rules.dsl import FACTORY_FLAG_ORDER

    registry = ToolRegistry(InMemorySessionStore())
    result = await registry.dispatch("get_rules", {"puzzle": "factory"})
    assert result["unresolved_flags"] == list(FACTORY_FLAG_ORDER)
    assert all(u["experiment"]["instruction"] for u in result["unknowns"])

    proposed = await registry.dispatch(
        "propose_experiment", {"puzzle": "factory", "flag": "overflow_timing"}
    )
    assert proposed["experiment"]["flag"] == "overflow_timing"
    bad = await registry.dispatch("propose_experiment", {"puzzle": "factory", "flag": "nope"})
    assert bad["ok"] is False


def test_long_results_are_trimmed_and_say_so():
    from services.core.agent import tools as T

    trimmed: list[str] = []
    rows = T._trim(list(range(500)), T.MAX_LOG_ROWS, "per_machine_log", trimmed)
    assert len(rows) == T.MAX_LOG_ROWS
    assert trimmed == [f"per_machine_log: showing {T.MAX_LOG_ROWS} of 500"]
    assert T.MAX_ARCHIVE == 5

    untouched: list[str] = []
    assert T._trim([1, 2], 10, "x", untouched) == [1, 2]
    assert untouched == []


def test_every_advertised_tool_has_a_handler_and_a_schema():
    from services.core.agent import tools as T

    registry = T.ToolRegistry(InMemorySessionStore())
    assert set(registry._handlers) == set(T.TOOL_NAMES)
    for schema in registry.schemas():
        fn = schema["function"]
        assert fn["parameters"]["additionalProperties"] is False
        assert fn["description"]
