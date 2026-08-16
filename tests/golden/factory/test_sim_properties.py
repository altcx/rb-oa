"""Properties the factory package must hold, as opposed to exact golden values.

The first block is the important one: every rule flag in ``RuleFlags`` is a
hypothesis about a rule the game never states, and a hypothesis that cannot
change any outcome is not a hypothesis -- it is dead code pretending to be
careful.  Each test below builds the smallest board on which its flag matters
and asserts that flipping it moves ``money_by_hour``.
"""

from __future__ import annotations

import time

import pytest

from services.core.rules.dsl import FACTORY_FLAG_ORDER, RuleFlags
from services.solvers.factory.bounds import upper_bound
from services.solvers.factory.calibrate import calibrate, flag_space, gate_ok
from services.solvers.factory.model import FactoryConfig, ModKind, Recipe, WarningKind
from services.solvers.factory.optimize import optimize, validate_config, validation_errors
from services.solvers.factory.sim import FLAG_DEFAULTS, make_flags, simulate
from tests.golden.factory.test_golden import conf, maker, seller, supplier, world

# ---------------------------------------------------------------------------
# each flag must actually change behaviour
# ---------------------------------------------------------------------------


def _flip(flag: str, value):
    return RuleFlags(**{flag: value})


def test_flag_two_hour_consume_timing_changes_money():
    # A clock-marked maker that pays 1 per item: whether it eats (and pays) in
    # its first or its second hour moves the whole money curve by an hour.
    st = world(
        [
            supplier("S", "ore", 1.0, out_max=2, storage=20),
            maker(
                "M",
                [
                    Recipe(
                        id="m",
                        inputs={"ore": 1},
                        output_item="widget",
                        output_qty=1,
                        production_cost=1.0,
                    )
                ],
                out_max=2,
                hours=2,
            ),
            seller("Z", "widget", 5.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=100.0,
        horizon=8,
        items=["ore", "widget"],
    )
    cfg = conf(st, {"S": 2, "M": 2, "Z": 2})
    a = simulate(st, cfg, RuleFlags(two_hour_consume_timing="start_hour_1"))
    b = simulate(st, cfg, RuleFlags(two_hour_consume_timing="start_hour_2"))
    assert a.money_by_hour != b.money_by_hour


def test_flag_supplier_cost_timing_changes_money():
    # The same order, paid on placement or on delivery: one hour of difference.
    st = world(
        [supplier("S", "ore", 2.0, out_max=3), seller("Z", "ore", 3.0)],
        [("S", "Z")],
        money=100.0,
        horizon=6,
        items=["ore"],
    )
    cfg = conf(st, {"S": 2, "Z": 2})
    a = simulate(st, cfg, RuleFlags(supplier_cost_timing="at_order"))
    b = simulate(st, cfg, RuleFlags(supplier_cost_timing="at_delivery"))
    assert a.money_by_hour != b.money_by_hour
    assert a.money_by_hour[1] != b.money_by_hour[1]


def _overflow_world():
    """M makes 3 widgets an hour into a 4-slot buffer; Z wants 6 at a time."""
    return world(
        [
            supplier("S", "ore", 1.0, out_max=3, storage=50),
            maker(
                "M",
                [
                    Recipe(
                        id="m",
                        inputs={"ore": 1},
                        output_item="widget",
                        output_qty=1,
                        production_cost=0.5,
                    )
                ],
                out_max=3,
                storage=4,
            ),
            seller("Z", "widget", 5.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=100.0,
        horizon=10,
        items=["ore", "widget"],
    )


def test_flag_overflow_timing_changes_money():
    # In the hour M deposits 3 on top of 3 with a cap of 4, "after_production"
    # lets Z pull all 6 before the excess is destroyed; "before_pull" clips
    # first, so Z (which needs 6) starves and the 2 spare widgets die.
    st = _overflow_world()
    cfg = conf(st, {"S": 3, "M": 3, "Z": 6})
    a = simulate(st, cfg, RuleFlags(overflow_timing="after_production"))
    b = simulate(st, cfg, RuleFlags(overflow_timing="before_pull"))
    assert a.money_by_hour != b.money_by_hour
    assert a.final_money > b.final_money


def test_flag_output_max_meaning_changes_money():
    # "separate_from_storage" gives the machine one whole output batch of room
    # on top of its storage bar, so it overflows one hour later and gets one
    # more run in before it jams.  Z is set beyond anything M will hold, so the
    # buffer is never drained and the cap is the only thing that matters.
    st = _overflow_world()
    cfg = conf(st, {"S": 3, "M": 3, "Z": 10})
    a = simulate(st, cfg, RuleFlags(output_max_meaning="per_hour_ceiling"))
    b = simulate(st, cfg, RuleFlags(output_max_meaning="separate_from_storage"))
    assert a.money_by_hour != b.money_by_hour


def test_flag_priority_metric_changes_money():
    # A is one hop from the seller but 90 pixels away; B is two hops but sits
    # right next to it.  Ore is scarce enough that only one of them can run.
    st = world(
        [
            supplier("S", "ore", 0.5, out_max=4, storage=4, y=30.0),
            maker(
                "A",
                [Recipe(id="a", inputs={"ore": 1}, output_item="widget", output_qty=1)],
                out_max=4,
                y=10.0,
            ),
            maker(
                "B",
                [Recipe(id="b", inputs={"ore": 1}, output_item="gadget", output_qty=1)],
                out_max=4,
                y=90.0,
            ),
            maker(
                "C",
                [Recipe(id="c", inputs={"gadget": 1}, output_item="widget", output_qty=1)],
                out_max=4,
                y=95.0,
            ),
            seller("Z", "widget", 3.0, y=100.0),
        ],
        [("S", "A"), ("S", "B"), ("A", "Z"), ("B", "C"), ("C", "Z")],
        money=100.0,
        horizon=8,
        items=["ore", "widget", "gadget"],
    )
    cfg = conf(st, {"S": 4, "A": 4, "B": 4, "C": 4, "Z": 4})
    a = simulate(st, cfg, RuleFlags(priority_metric="hops"))
    b = simulate(st, cfg, RuleFlags(priority_metric="pixels"))
    assert a.money_by_hour != b.money_by_hour


def test_flag_priority_recompute_changes_money():
    # Two branches with their own sellers.  Z1 is clock-marked, so on the hours
    # it is mid-sale the recomputed distance sees no live path through it and A
    # loses its place in the queue to B.
    st = world(
        [
            supplier("S", "ore", 0.5, out_max=4, storage=4, y=50.0),
            maker(
                "A",
                [Recipe(id="a", inputs={"ore": 1}, output_item="widget", output_qty=1)],
                out_max=4,
                y=0.0,
            ),
            maker(
                "B",
                [Recipe(id="b", inputs={"ore": 1}, output_item="gem", output_qty=1)],
                out_max=4,
                y=1.0,
            ),
            seller("Z1", "widget", 10.0, y=20.0),
            seller("Z2", "gem", 6.0, y=21.0),
        ],
        [("S", "A"), ("S", "B"), ("A", "Z1"), ("B", "Z2")],
        money=100.0,
        horizon=10,
        items=["ore", "widget", "gem"],
    )
    st.machine("Z1").production_hours = 2
    cfg = conf(st, {"S": 4, "A": 4, "B": 4, "Z1": 4, "Z2": 4})
    a = simulate(st, cfg, RuleFlags(priority_recompute="fixed"))
    b = simulate(st, cfg, RuleFlags(priority_recompute="per_hour"))
    assert a.money_by_hour != b.money_by_hour


def test_flag_mod_stacking_changes_money():
    # base 2.0 per item + two 0.5 increments: 3.0 if they add, 4.5 if they
    # multiply the base by (1 + increment) each.
    st = world(
        [
            supplier("S", "ore", 0.5, out_max=4, storage=20),
            maker(
                "M",
                [
                    Recipe(
                        id="m",
                        inputs={"ore": 2},
                        output_item="widget",
                        output_qty=1,
                        production_cost=2.0,
                    )
                ],
                out_max=4,
                mods=[ModKind.HALF_MATERIALS, ModKind.DOUBLE_STORAGE_MAX],
                mod_cost={ModKind.HALF_MATERIALS: 0.5, ModKind.DOUBLE_STORAGE_MAX: 0.5},
            ),
            seller("Z", "widget", 9.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=100.0,
        horizon=8,
        items=["ore", "widget"],
    )
    cfg = conf(
        st,
        {"S": 4, "M": 2, "Z": 2},
        mods={"M": [ModKind.HALF_MATERIALS, ModKind.DOUBLE_STORAGE_MAX]},
    )
    a = simulate(st, cfg, RuleFlags(mod_stacking="independent"))
    b = simulate(st, cfg, RuleFlags(mod_stacking="multiplicative"))
    assert a.money_by_hour != b.money_by_hour


def test_flag_full_storage_behavior_changes_money():
    # The ninth flag, promoted out of the simulator's assumptions block.
    # M makes 3 widgets an hour into a 4-slot buffer and Z (set to 10) never
    # drains it, so from hour 4 M is jammed.
    #   "idle": M stands down.  It stops eating ore, stops paying, and only the
    #           2-widget overshoot that jammed it was ever destroyed.
    #   "produce_and_waste": M keeps running into a full buffer -- paying 1.5 an
    #           hour to destroy 3 widgets an hour, for nothing at all.
    st = _overflow_world()
    cfg = conf(st, {"S": 3, "M": 3, "Z": 10})
    idle = simulate(st, cfg, make_flags(full_storage_behavior="idle"))
    waste = simulate(st, cfg, make_flags(full_storage_behavior="produce_and_waste"))

    assert idle.money_by_hour != waste.money_by_hour
    # burning inputs and production cost for output that is destroyed is strictly
    # worse, and that economic difference is the whole reason this is a flag
    assert waste.final_money < idle.final_money
    assert waste.total_cost > idle.total_cost

    def wasted(res):
        return sum(w.amount for w in res.warnings if w.kind == WarningKind.OVERFLOW)

    def idled(res):
        return sum(1 for w in res.warnings if w.kind == WarningKind.IDLE_FULL)

    assert wasted(waste) > wasted(idle)
    assert idled(idle) > 0 and idled(waste) == 0


def test_flag_insufficient_funds_changes_money():
    # An order of 4 ore at 3 each costs 12 and there is only 10 in the bank:
    # either the whole order is refused, or 3 ore are bought for 9.
    st = world(
        [supplier("S", "ore", 3.0, out_max=4, storage=20), seller("Z", "ore", 4.0)],
        [("S", "Z")],
        money=10.0,
        horizon=6,
        items=["ore"],
    )
    cfg = conf(st, {"S": 4, "Z": 1})
    a = simulate(st, cfg, RuleFlags(insufficient_funds="skipped"))
    b = simulate(st, cfg, RuleFlags(insufficient_funds="partial"))
    assert a.money_by_hour != b.money_by_hour


def test_every_flag_has_a_test():
    """Guard against a flag being added and quietly ignored here.

    Checked against the whole searched space, not just the DSL's list, so a flag
    promoted inside this package is covered from the moment it exists.
    """
    expected = set(flag_space()) | set(FACTORY_FLAG_ORDER) | set(FLAG_DEFAULTS)
    covered = {name for name in expected if f"test_flag_{name}_changes_money" in globals()}
    assert covered == expected, sorted(expected - covered)


def test_calibration_searches_every_flag_the_simulator_honours():
    """A rule the hour loop branches on but calibration never sweeps would be a
    silent guess: the run would look unexplainable and nobody would know why."""
    space = flag_space()
    for name in FACTORY_FLAG_ORDER:
        assert name in space, f"{name} is in the DSL but not searched"
    for name in FLAG_DEFAULTS:
        assert name in space, f"the simulator branches on {name} but nothing searches it"
    assert len(space) >= 9
    for name, options in space.items():
        assert len(options) >= 2, f"{name} has nothing to choose between"


# ---------------------------------------------------------------------------
# optimizer properties
# ---------------------------------------------------------------------------


def _rich_world():
    return world(
        [
            supplier("S1", "ore", 1.0, out_max=6, storage=30, y=40.0),
            supplier("S2", "clay", 2.0, out_max=4, storage=30, y=50.0),
            maker(
                "M1",
                [
                    Recipe(
                        id="ore_way",
                        inputs={"ore": 2},
                        output_item="widget",
                        output_qty=1,
                        production_cost=0.5,
                    ),
                    Recipe(
                        id="clay_way",
                        inputs={"clay": 1},
                        output_item="widget",
                        output_qty=1,
                        production_cost=1.5,
                    ),
                ],
                out_max=4,
                storage=20,
                hours=2,
                y=20.0,
                mods=list(ModKind),
                mod_cost={
                    ModKind.DOUBLE_OUTPUT_MAX: 0.2,
                    ModKind.HALF_MATERIALS: 0.3,
                    ModKind.ONE_HOUR_PRODUCTION: 0.4,
                    ModKind.DOUBLE_STORAGE_MAX: 0.1,
                },
            ),
            maker(
                "M2",
                [
                    Recipe(
                        id="trinket",
                        inputs={"ore": 3},
                        output_item="trinket",
                        output_qty=1,
                        production_cost=0.2,
                    )
                ],
                out_max=3,
                storage=20,
                y=30.0,
                mods=[ModKind.HALF_MATERIALS],
                mod_cost={ModKind.HALF_MATERIALS: 0.3},
            ),
            seller("Z1", "widget", 9.0, y=0.0),
            seller("Z2", "trinket", 4.0, y=1.0),
        ],
        [("S1", "M1"), ("S1", "M2"), ("S2", "M1"), ("M1", "Z1"), ("M2", "Z2")],
        money=150.0,
        horizon=24,
        items=["ore", "clay", "widget", "trinket"],
    )


@pytest.fixture(scope="module")
def solved():
    st = _rich_world()
    return st, optimize(st, seconds=2.0)


def test_optimizer_never_returns_an_invalid_config(solved):
    st, res = solved
    assert validate_config(st, res.best), validation_errors(st, res.best)
    assert res.archive, "the archive must not be empty"
    for entry in res.archive:
        assert validate_config(st, entry.config), validation_errors(st, entry.config)


def test_archive_holds_distinct_configs(solved):
    _st, res = solved
    keys = [e.config.key() for e in res.archive]
    assert len(keys) == len(set(keys))
    assert len(res.archive) <= 20
    values = [e.value for e in res.archive]
    assert values == sorted(values, reverse=True)


def test_best_value_is_under_the_upper_bound(solved):
    st, res = solved
    ub = upper_bound(st)
    # The LP relaxes fractional throughput, ignores storage and the
    # all-or-nothing allocation rule, and charges only ``depth`` ramp-up hours,
    # so it can only ever be loose in the optimizer's favour.  The tolerance is
    # for float noise in the LP objective, not for slack in the argument.
    assert res.best_value <= ub.ceiling + 1e-6
    assert res.bound == pytest.approx(ub.ceiling)


def test_optimizer_beats_or_matches_the_board_as_found(solved):
    _st, res = solved
    assert res.best_value >= res.baseline_value


def test_optimizer_is_usable_at_200ms():
    st = _rich_world()
    res = optimize(st, seconds=0.2)
    assert res.elapsed_s < 1.0
    assert validate_config(st, res.best)
    assert res.best_value >= res.baseline_value
    assert res.iterations > 0
    # either it found something to change, or it is telling us the board is
    # already at its best -- both are usable answers, neither is an empty one
    assert res.actions or res.best.key() == FactoryConfig.from_state(st).key()
    assert res.bound >= res.best_value


def test_on_improve_is_called_and_carries_a_usable_result():
    st = _rich_world()
    seen = []
    res = optimize(st, seconds=1.0, on_improve=lambda r: seen.append(r))
    assert seen, "anytime contract: on_improve must fire at least once"
    for partial in seen:
        assert validate_config(st, partial.best)
    assert seen[-1].best_value <= res.best_value
    # the stream must end on the answer that is actually returned, including any
    # gain found by the post-search layers
    assert seen[-1].best_value == res.best_value


def test_the_caller_is_never_left_in_silence():
    """Anytime means a steady stream, not one callback and then a spinner.

    A solve that plateaus early must still report "still working, here is what I
    have" -- marked ``improved=False`` so a caller can tell the two apart.
    """
    st = _rich_world()
    stamps: list[tuple[float, bool]] = []
    t0 = time.perf_counter()
    optimize(
        st,
        seconds=1.5,
        on_improve=lambda r: stamps.append((time.perf_counter() - t0, r.improved)),
    )
    assert len(stamps) >= 4, f"only {len(stamps)} callbacks in 1.5 s"
    assert any(not improved for _t, improved in stamps), "no heartbeats, only improvements"
    assert any(improved for _t, improved in stamps), "improvements are not marked"
    gaps = [stamps[i + 1][0] - stamps[i][0] for i in range(len(stamps) - 1)]
    assert max(gaps) < 0.75, f"went quiet for {max(gaps):.2f}s"
    assert stamps[0][0] < 0.25, "the first best-so-far arrived too late"


def test_endgame_schedule_is_valid_and_only_ever_helps(solved):
    st, res = solved
    for action in res.endgame:
        assert action.setting == "output"
        assert action.value == 0
        assert action.target in {m.id for m in st.machines}
    assert validate_config(st, res.best)


def test_one_hour_mod_value_is_measured_for_every_two_hour_machine(solved):
    st, res = solved
    expected = {
        m.id
        for m in st.machines
        if m.production_hours >= 2
        and ModKind.ONE_HOUR_PRODUCTION in (set(m.available_mods) | set(m.installed_mods))
    }
    assert set(res.one_hour_mod_value) == expected


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def _calibration_world():
    return world(
        [
            supplier("S", "ore", 2.0, out_max=3, storage=20),
            maker(
                "M",
                [
                    Recipe(
                        id="m",
                        inputs={"ore": 1},
                        output_item="widget",
                        output_qty=1,
                        production_cost=1.0,
                    )
                ],
                out_max=3,
                hours=2,
            ),
            seller("Z", "widget", 6.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=60.0,
        horizon=10,
        items=["ore", "widget"],
    )


def test_calibrate_matches_when_the_defaults_are_right():
    st = _calibration_world()
    cfg = conf(st, {"S": 2, "M": 2, "Z": 2})
    truth = simulate(st, cfg, RuleFlags())
    res = calibrate(st, cfg, truth.money_by_hour[1:])
    assert res.matched
    assert all(d.ok for d in res.per_hour)
    assert res.resolved_flags == RuleFlags()
    assert gate_ok(res)


def test_calibrate_recovers_a_known_non_default_flag_combination():
    st = _calibration_world()
    cfg = conf(st, {"S": 2, "M": 2, "Z": 2})
    truth_flags = make_flags(
        two_hour_consume_timing="start_hour_2",
        supplier_cost_timing="at_delivery",
        insufficient_funds="partial",
    )
    observed = simulate(st, cfg, truth_flags).money_by_hour[1:]

    res = calibrate(st, cfg, observed)  # started from the defaults
    assert not res.matched  # the defaults do NOT reproduce this run
    exact = [c.flags for c in res.candidates if c.exact]
    assert truth_flags in exact, "the true rule combination must survive the sweep"
    # every surviving hypothesis agrees on the two flags this run can see
    assert res.resolved_flags is not None
    assert res.resolved_flags.two_hour_consume_timing == "start_hour_2"
    assert res.resolved_flags.supplier_cost_timing == "at_delivery"
    assert gate_ok(res)
    assert res.discriminating_hours


def test_calibrate_recovers_the_full_storage_flag():
    """The ninth flag is discriminable from a money series, which is exactly
    why it was promoted out of the assumptions block."""
    st = world(
        [
            supplier("S", "ore", 1.0, out_max=3, storage=50),
            maker(
                "M",
                [
                    Recipe(
                        id="m",
                        inputs={"ore": 1},
                        output_item="widget",
                        output_qty=1,
                        production_cost=0.5,
                    )
                ],
                out_max=3,
                storage=4,
            ),
            seller("Z", "widget", 5.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=100.0,
        horizon=10,
        items=["ore", "widget"],
    )
    cfg = conf(st, {"S": 3, "M": 3, "Z": 10})
    truth = make_flags(full_storage_behavior="produce_and_waste")
    observed = simulate(st, cfg, truth).money_by_hour[1:]

    res = calibrate(st, cfg, observed)
    assert not res.matched  # the default ("idle") does not explain this run
    assert res.resolved_flags is not None
    assert res.resolved_flags.full_storage_behavior == "produce_and_waste"
    assert gate_ok(res)


def test_calibrate_refuses_to_gate_an_unexplainable_run():
    st = _calibration_world()
    cfg = conf(st, {"S": 2, "M": 2, "Z": 2})
    # a series no rule combination can produce (money that only ever goes up)
    observed = [1000.0 + 10 * h for h in range(1, st.horizon_hours + 1)]
    res = calibrate(st, cfg, observed)
    assert not res.matched
    assert res.resolved_flags is None
    assert not gate_ok(res)
    assert "do not run the optimizer" in res.message.lower()
