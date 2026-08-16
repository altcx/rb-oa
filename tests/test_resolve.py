"""Solving under unresolved rules: consensus, divergence attribution, the
64-combination cap, and value of information.

The solvers are stubs — this file is about the uncertainty machinery, not about
whether the builder solver counts correctly.
"""

from __future__ import annotations

import pytest

from services.core.rules import resolve as R
from services.core.rules.dsl import (
    BUILDER_FLAG_OPTIONS,
    FACTORY_FLAG_ORDER,
    Action,
    BuilderRules,
    FactoryRuleState,
)
from services.solvers.builder.model import Build, BuildReport, BuilderPuzzle, Obstacle, Part

PUZZLE = BuilderPuzzle(
    parts=[
        Part(id="p1", weight=2, attributes={"power": 3}),
        Part(id="p2", weight=3, attributes={"power": 4}),
        Part(id="p3", weight=1, attributes={"grip": 2}),
    ],
    obstacles=[Obstacle(id="o1", order=1, requires={"power": 5})],
)

# rules with exactly two unresolved flags: aggregation (3) x duplicates (2) = 6
TWO_UNKNOWN = BuilderRules(
    weight_max=10,
    obstacle_semantics="threshold",
    obstacle_ordering="unordered",
    failure_mode="all_must_pass",
    objective="count_valid",
)

#: what the stub solver reports, keyed by the flag that actually drives it
BY_AGGREGATION = {
    "sum": ({"p1": 1, "p2": 1}, 10.0),
    "min": ({"p1": 1, "p3": 1}, 4.0),
    "max": ({"p1": 1, "p3": 1, "p2": 1}, 7.0),
}


def stub_solver(*, puzzle, objective, budget_s, max_builds):
    """Result depends ONLY on aggregation — duplicates_allowed is irrelevant."""
    counts, total = BY_AGGREGATION[puzzle.rules.aggregation]
    return BuildReport(
        total_valid=int(total),
        best_build=Build(counts=dict(counts), weight=5, obstacles_passed=1),
        binding_obstacle="o1",
        method="stub",
    )


# ---------------------------------------------------------------------------
# consensus / divergence
# ---------------------------------------------------------------------------


def test_consensus_is_what_every_interpretation_agrees_on():
    out = R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 1.0, solve_fn=stub_solver)
    assert out.combinations_tried == 6
    assert not out.truncated
    assert [a.key() for a in out.consensus_actions] == [("p1", "include", "1")]
    # p2/p3 are not consensus: they depend on the reading
    keys = {a.target for a in out.consensus_actions}
    assert keys == {"p1"}


def test_divergence_is_attributed_to_the_flag_that_drives_it():
    out = R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 1.0, solve_fn=stub_solver)
    assert len(out.divergent_actions) == 1
    group = out.divergent_actions[0]
    assert group.flag == "aggregation"  # NOT duplicates_allowed
    assert set(group.by_option) == {"sum", "min", "max"}
    assert [a.target for a in group.by_option["sum"]] == ["p2"]
    assert [a.target for a in group.by_option["min"]] == ["p3"]
    assert sorted(a.target for a in group.by_option["max"]) == ["p2", "p3"]


def test_highest_leverage_unknown_is_the_divergence_driver_with_an_experiment():
    out = R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 1.0, solve_fn=stub_solver)
    unknown = out.highest_leverage_unknown
    assert unknown is not None
    assert unknown.flag == "aggregation"
    assert unknown.experiment is not None
    assert unknown.experiment.flag == "aggregation"
    assert set(unknown.experiment.discriminator) == {"sum", "min", "max"}


def test_value_of_information_is_the_best_worst_spread():
    out = R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 1.0, solve_fn=stub_solver)
    assert out.value_of_information == pytest.approx(10.0 - 4.0)


def test_full_agreement_gives_consensus_and_no_divergence():
    def flat(*, puzzle, objective, budget_s, max_builds):
        return BuildReport(
            total_valid=5, best_build=Build(counts={"p1": 1}, weight=2), method="stub"
        )

    out = R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 1.0, solve_fn=flat)
    assert len(out.consensus_actions) == 1
    assert out.divergent_actions == []
    assert out.value_of_information == 0.0
    assert out.highest_leverage_unknown is None


def test_fully_resolved_rules_produce_exactly_one_interpretation():
    resolved = TWO_UNKNOWN.model_copy(update={"aggregation": "sum", "duplicates_allowed": False})
    out = R.solve_all_interpretations(resolved, PUZZLE, 1.0, solve_fn=stub_solver)
    assert out.combinations_tried == 1
    assert len(out.consensus_actions) == 2
    assert out.divergent_actions == []


# ---------------------------------------------------------------------------
# the 64 cap
# ---------------------------------------------------------------------------


def test_cap_pins_the_highest_leverage_flags_first():
    """All six flags unknown = 288 combinations. Pin the leading ones."""
    empty = BuilderRules(weight_max=10)
    out = R.solve_all_interpretations(empty, PUZZLE, 1.0, solve_fn=stub_solver)
    assert out.truncated
    assert out.combinations_tried <= R.MAX_COMBINATIONS
    # 3*3*2*2*2*4 = 288 -> pin obstacle_semantics -> 96 -> pin aggregation -> 32
    assert out.combinations_tried == 32
    joined = " ".join(out.notes)
    assert "obstacle_semantics" in joined and "aggregation" in joined
    assert "cap" in joined
    # every interpretation records the pinned values too, not just the varied ones
    assignment = out.interpretations[0].assignment
    assert assignment["obstacle_semantics"] == BUILDER_FLAG_OPTIONS["obstacle_semantics"][0]
    assert assignment["aggregation"] == BUILDER_FLAG_OPTIONS["aggregation"][0]


def test_plan_is_a_pure_function():
    options = {f: list(o) for f, o in BUILDER_FLAG_OPTIONS.items()}
    varying, pinned, notes, truncated = R.plan_interpretations(
        list(BUILDER_FLAG_OPTIONS), options, cap=4
    )
    assert truncated
    assert R._product(len(options[f]) for f in varying) <= 4
    assert set(pinned) | set(varying) == set(BUILDER_FLAG_OPTIONS)
    assert notes


# ---------------------------------------------------------------------------
# graceful degradation
# ---------------------------------------------------------------------------


def test_a_solver_exception_is_one_bad_interpretation_not_a_crash():
    calls = {"n": 0}

    def flaky(*, puzzle, objective, budget_s, max_builds):
        calls["n"] += 1
        if puzzle.rules.aggregation == "min":
            raise ValueError("degenerate model")
        return stub_solver(
            puzzle=puzzle, objective=objective, budget_s=budget_s, max_builds=max_builds
        )

    out = R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 1.0, solve_fn=flaky)
    errored = [i for i in out.interpretations if i.error]
    assert len(errored) == 2
    assert all("degenerate model" in (i.error or "") for i in errored)
    assert out.consensus_actions  # the surviving readings still agree on p1
    assert any("did not complete" in n for n in out.notes)


def test_real_solver_import_path_never_crashes():
    """The builder solver is written by another component. Whether it is there
    or not, this must return a structured result rather than raise."""
    out = R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 0.2)
    assert out.combinations_tried == 6
    for interp in out.interpretations:
        assert interp.error is None or isinstance(interp.error, str)
    if all(i.error for i in out.interpretations):
        assert out.consensus_actions == []


def test_budget_is_divided_across_interpretations():
    seen: list[float] = []

    def recording(*, puzzle, objective, budget_s, max_builds):
        seen.append(budget_s)
        return BuildReport(total_valid=1, best_build=Build(counts={"p1": 1}))

    R.solve_all_interpretations(TWO_UNKNOWN, PUZZLE, 3.0, solve_fn=recording)
    assert len(seen) == 6
    assert all(s <= 3.0 / 6 + 1e-9 for s in seen)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


class FakeOptimizeResult:
    def __init__(self, actions, value):
        self.actions = actions
        self.best_value = value
        self.baseline_value = 100.0
        self.top_warning = None
        self.iterations = 3
        self.elapsed_s = 0.01
        self.best = None


def test_factory_interpretations_use_the_same_machinery():
    resolved = {f: "x" for f in FACTORY_FLAG_ORDER if f != "insufficient_funds"}
    flag_state = FactoryRuleState(resolved_by=resolved)

    def fake_optimize(state, seconds, flags, seed_config=None, on_improve=None):
        if flags.insufficient_funds == "skipped":
            return FakeOptimizeResult(
                [
                    Action(target="m1", setting="output", value=4),
                    Action(target="m2", setting="output", value=1),
                ],
                180.0,
            )
        return FakeOptimizeResult(
            [
                Action(target="m1", setting="output", value=4),
                Action(target="m2", setting="output", value=3),
            ],
            150.0,
        )

    out = R.solve_all_interpretations_factory(
        state=object(), config=None, flag_state=flag_state, budget_s=0.5, optimize_fn=fake_optimize
    )
    assert out.combinations_tried == 2
    assert [a.key() for a in out.consensus_actions] == [("m1", "output", "4")]
    assert len(out.divergent_actions) == 1
    assert out.divergent_actions[0].flag == "insufficient_funds"
    assert out.value_of_information == pytest.approx(30.0)
    assert out.highest_leverage_unknown.flag == "insufficient_funds"
    assert out.highest_leverage_unknown.experiment is not None


def test_factory_missing_solvers_report_cleanly():
    out = R.solve_all_interpretations_factory(
        state=object(),
        config=None,
        flag_state=FactoryRuleState(
            resolved_by={f: "x" for f in FACTORY_FLAG_ORDER if f != "mod_stacking"}
        ),
        budget_s=0.1,
    )
    assert out.combinations_tried == 2
    for interp in out.interpretations:
        if interp.error:
            assert "unavailable" in interp.error or "Error" in interp.error


# ---------------------------------------------------------------------------
# rules/store.py — cache keyed by INSTANCE, never by family
# ---------------------------------------------------------------------------


def test_a_resolution_is_not_shared_between_puzzle_instances(tmp_path):
    from services.core.rules import store

    other = BuilderPuzzle(
        parts=[Part(id="q1", weight=9, attributes={"power": 1})],
        obstacles=[Obstacle(id="z1", order=1, requires={"power": 1})],
    )
    assert PUZZLE.fingerprint() != other.fingerprint()

    resolved = TWO_UNKNOWN.model_copy(update={"aggregation": "sum", "duplicates_allowed": True})
    store.save_builder_rules(
        PUZZLE, resolved, {"aggregation": "experiment:aggregation"}, root=tmp_path
    )

    back = store.load_builder_rules(PUZZLE, root=tmp_path)
    assert back is not None and back.aggregation == "sum"

    # the other instance of the SAME family must learn nothing from it
    assert store.load_builder_rules(other, root=tmp_path) is None
    assert store.resolved_for("builder", other.fingerprint(), root=tmp_path) is None
    assert store.resolved_for("builder", PUZZLE.fingerprint(), root=tmp_path) is not None
    assert store.fingerprints("builder", root=tmp_path) == [PUZZLE.fingerprint()]


def test_store_merge_accumulates_and_newer_wins(tmp_path):
    from services.core.rules import store

    fp = PUZZLE.fingerprint()
    store.merge("builder", fp, {"flags": {"aggregation": "sum"}, "notes": ["first"]}, tmp_path)
    merged = store.merge(
        "builder",
        fp,
        {"flags": {"aggregation": "max", "duplicates_allowed": True}, "notes": ["second"]},
        tmp_path,
    )
    assert merged.flags == {"aggregation": "max", "duplicates_allowed": True}
    assert merged.notes == ["first", "second"]
    assert store.forget("builder", fp, tmp_path) is True
    assert store.resolved_for("builder", fp, tmp_path) is None


def test_factory_flags_roundtrip_by_fingerprint(tmp_path):
    from services.core.rules import store
    from services.solvers.factory.model import FactoryState

    state = FactoryState(starting_money=500.0, horizon_hours=24)
    flag_state = FactoryRuleState(resolved_by={"priority_metric": "calibration:run-3"})
    store.save_factory_flags(state, flag_state, tmp_path)

    back = store.load_factory_flags(state, tmp_path)
    assert back is not None
    assert back.resolved_by == {"priority_metric": "calibration:run-3"}
    assert back.unresolved() == [f for f in FACTORY_FLAG_ORDER if f != "priority_metric"]

    different = FactoryState(starting_money=999.0, horizon_hours=24)
    assert store.load_factory_flags(different, tmp_path) is None
