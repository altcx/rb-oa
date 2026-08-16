"""Ten hand-computed golden runs for the factory simulator.

Every expected number below was worked out by hand from the rules in
``sim.py`` (its hour loop and its ASSUMPTIONS block) BEFORE the code was run.
The arithmetic sits in a comment above each assertion so that when the game
disagrees with us we can see exactly which line of reasoning was wrong.

The one rule that drives most of the arithmetic: a job started in hour ``h``
deposits at the top of hour ``h + production_hours``.  So a supplier ordering in
hour 1 has stock in hour 2, the seller pulls it in hour 2, and the first money
arrives in hour 3.
"""

from __future__ import annotations

import pytest

from services.solvers.factory.model import (
    Edge,
    FactoryConfig,
    FactoryState,
    Item,
    Machine,
    MachineConfig,
    MachineKind,
    ModKind,
    Recipe,
    WarningKind,
)
from services.solvers.factory.sim import simulate

# ---------------------------------------------------------------------------
# tiny fixture constructors
# ---------------------------------------------------------------------------


def supplier(
    mid: str,
    item: str,
    cost: float,
    *,
    out_max: int = 5,
    storage: int = 20,
    y: float = 30.0,
    qty: int = 1,
    hours: int = 1,
) -> Machine:
    return Machine(
        id=mid,
        kind=MachineKind.SUPPLIER,
        storage_max=storage,
        output_max=out_max,
        production_hours=hours,
        y=y,
        recipes=[Recipe(id="buy", output_item=item, output_qty=qty, purchase_cost=cost)],
    )


def maker(
    mid: str,
    recipes: list[Recipe],
    *,
    out_max: int = 5,
    storage: int = 20,
    y: float = 10.0,
    hours: int = 1,
    mods: list[ModKind] | None = None,
    mod_cost: dict[ModKind, float] | None = None,
) -> Machine:
    return Machine(
        id=mid,
        kind=MachineKind.MAKER,
        storage_max=storage,
        output_max=out_max,
        production_hours=hours,
        y=y,
        recipes=recipes,
        available_mods=list(mods or []),
        mod_cost=dict(mod_cost or {}),
    )


def seller(mid: str, item: str, price: float, *, out_max: int = 10, y: float = 0.0) -> Machine:
    return Machine(
        id=mid,
        kind=MachineKind.SELLER,
        storage_max=0,
        output_max=out_max,
        y=y,
        recipes=[Recipe(id="sell", inputs={item: 1}, sale_price=price)],
    )


def world(machines, edges, *, money: float, horizon: int, items: list[str]) -> FactoryState:
    return FactoryState(
        machines=machines,
        edges=[Edge(src=a, dst=b) for a, b in edges],
        items=[Item(id=i) for i in items],
        starting_money=money,
        horizon_hours=horizon,
    )


def conf(
    state: FactoryState,
    settings: dict[str, int],
    recipes: dict[str, str] | None = None,
    mods: dict[str, list[ModKind]] | None = None,
) -> FactoryConfig:
    recipes = recipes or {}
    mods = mods or {}
    return FactoryConfig(
        machines={
            m.id: MachineConfig(
                machine_id=m.id,
                recipe_id=recipes.get(m.id, m.default_recipe_id()),
                output_setting=settings.get(m.id, 0),
                mods=mods.get(m.id, []),
            )
            for m in state.machines
        }
    )


def warns(res, kind: WarningKind, machine_id: str | None = None):
    return [
        w
        for w in res.warnings
        if w.kind == kind and (machine_id is None or w.machine_id == machine_id)
    ]


# ---------------------------------------------------------------------------
# 1. supplier -> seller: the base loop and the ramp-up delay
# ---------------------------------------------------------------------------


def test_01_supplier_seller_base_loop_and_rampup():
    st = world(
        [supplier("S", "ore", 1.0, out_max=5), seller("Z", "ore", 3.0)],
        [("S", "Z")],
        money=100.0,
        horizon=6,
        items=["ore"],
    )
    res = simulate(st, conf(st, {"S": 2, "Z": 2}))

    # h1: Z has no ore -> skipped.  S orders 2 ore @1 = -2.        100 - 2 = 98
    # h2: S deposits 2 ore.  Z pulls both and starts selling.
    #     S orders 2 more = -2.                                     98 - 2 = 96
    # h3: Z's sale matures: 2 x 3 = +6.  S deposits 2, Z pulls,
    #     S orders again -2.                                        96 + 6 - 2 = 100
    # h4..h6: steady state +6 - 2 = +4 per hour.  104, 108, 112.
    assert res.money_by_hour == [100.0, 98.0, 96.0, 100.0, 104.0, 108.0, 112.0]
    assert res.final_money == 112.0
    # first revenue lands in hour 3, not hour 1: two machines deep = 2 hours of ramp
    assert res.money_by_hour[3] > res.money_by_hour[2]
    assert res.total_revenue == 24.0  # 4 selling hours (h3..h6) x 2 items x 3
    assert res.total_cost == 12.0  # 6 order hours x 2 items x 1


# ---------------------------------------------------------------------------
# 2. supplier storage below the seller's requirement: starvation and skip
# ---------------------------------------------------------------------------


def test_02_starved_seller_is_skipped_every_hour():
    st = world(
        [
            supplier("S", "ore", 1.0, out_max=2, storage=2),
            seller("Z", "ore", 3.0),
        ],
        [("S", "Z")],
        money=100.0,
        horizon=6,
        items=["ore"],
    )
    res = simulate(st, conf(st, {"S": 2, "Z": 3}))

    # S can never hold more than 2 ore; Z wants 3 per run and never gets them.
    # h1: Z skipped (nothing upstream).  S orders 2 @1 = -2.       100 - 2 = 98
    # h2: S deposits 2 -> storage 2 = cap.  Z needs 3, has 2 -> skipped,
    #     and takes NOTHING (no partial pull), so S stays full and idles.
    # h3..h6: identical.  Money never moves again.                 98
    assert res.money_by_hour == [100.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0]
    assert res.final_money == 98.0
    assert res.total_revenue == 0.0
    assert len(warns(res, WarningKind.SKIPPED, "Z")) == 6  # every hour
    assert len(warns(res, WarningKind.IDLE_FULL, "S")) == 5  # h2..h6
    # the 2 ore are still sitting there at the horizon, worth nothing
    assert res.ending_storage == {"S": {"ore": 2}}


# ---------------------------------------------------------------------------
# 3. one supplier, two consumers, supply covers exactly one:
#    the nearer machine takes everything and the further one gets ZERO
# ---------------------------------------------------------------------------


def test_03_priority_is_all_or_nothing_not_a_share():
    st = world(
        [
            supplier("S", "ore", 0.5, out_max=4, storage=20, y=30.0),
            # A is one hop from the seller but sits LOW on the screen (y=10)
            maker(
                "A",
                [Recipe(id="a", inputs={"ore": 1}, output_item="widget", output_qty=1)],
                out_max=4,
                y=10.0,
            ),
            # B is two hops from the seller but sits at the very TOP (y=0):
            # if the tie-break (topmost) beat the distance rule, B would win.
            maker(
                "B",
                [Recipe(id="b", inputs={"ore": 1}, output_item="gadget", output_qty=1)],
                out_max=4,
                y=0.0,
            ),
            maker(
                "C",
                [Recipe(id="c", inputs={"gadget": 1}, output_item="widget", output_qty=1)],
                out_max=4,
                y=20.0,
            ),
            seller("Z", "widget", 3.0, y=100.0),
        ],
        [("S", "A"), ("S", "B"), ("A", "Z"), ("B", "C"), ("C", "Z")],
        money=100.0,
        horizon=6,
        items=["ore", "widget", "gadget"],
    )
    res = simulate(st, conf(st, {"S": 4, "A": 4, "B": 4, "C": 4, "Z": 4}))

    # Distance to the seller: Z=0, A=1, C=1, B=2, S=2.  A is served before B.
    # S makes exactly 4 ore an hour; A and B each need 4.
    # h1: nothing anywhere.  S orders 4 @0.5 = -2.                 100 - 2 = 98
    # h2: S deposits 4.  A (nearer) takes all 4.  B needs 4, finds 0 ->
    #     skipped, gets ZERO (not the 0 left over, not a share).
    #     S orders -2.                                              98 - 2 = 96
    # h3: A deposits 4 widgets, Z pulls them; A takes the next 4 ore;
    #     B skipped again.  S orders -2.                            96 - 2 = 94
    # h4: Z's sale matures 4 x 3 = +12, and repeats every hour after.
    #                                                               94 + 12 - 2 = 104
    # h5: +10 -> 114.   h6: +10 -> 124.
    assert res.money_by_hour == [100.0, 98.0, 96.0, 94.0, 104.0, 114.0, 124.0]
    assert res.final_money == 124.0
    assert len(warns(res, WarningKind.SKIPPED, "B")) == 6  # starved every single hour
    assert "B" not in res.ending_storage  # B never produced one gadget
    assert "C" not in res.ending_storage


# ---------------------------------------------------------------------------
# 4. a 2-hour maker mid-chain: deposit timing and alternate-hour idling
# ---------------------------------------------------------------------------


def _two_hour_world() -> FactoryState:
    return world(
        [
            supplier("S", "ore", 1.0, out_max=5, storage=20),
            maker(
                "M",
                [Recipe(id="m", inputs={"ore": 1}, output_item="widget", output_qty=1)],
                out_max=5,
                storage=20,
                hours=2,
                mods=[ModKind.ONE_HOUR_PRODUCTION],
                mod_cost={ModKind.ONE_HOUR_PRODUCTION: 0.5},
            ),
            seller("Z", "widget", 5.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=100.0,
        horizon=8,
        items=["ore", "widget"],
    )


def test_04_two_hour_maker_deposits_late_and_idles_every_other_hour():
    st = _two_hour_world()
    res = simulate(st, conf(st, {"S": 2, "M": 2, "Z": 2}))

    # M is clock-marked: it consumes in hour h and deposits at the top of h+2.
    # h1: nothing to eat.  S orders 2 @1 = -2.                     100 - 2 = 98
    # h2: S deposits 2 ore, M eats them and starts a 2-hour job.
    #     S orders -2.                                              98 - 2 = 96
    # h3: M is busy (nothing deposited, nothing eaten).  S orders -2.    94
    # h4: M deposits 2 widgets; Z pulls them; M eats the next 2 ore.
    #     S orders -2.                                              94 - 2 = 92
    # h5: Z's sale matures 2 x 5 = +10; M is busy again.  S orders -2.  100
    # h6: M deposits 2, Z pulls, M restarts.  S orders -2.               98
    # h7: sale +10, M busy.  S orders -2.                               106
    # h8: M deposits, Z pulls (matures after the horizon).  S orders -2. 104
    assert res.money_by_hour == [100.0, 98.0, 96.0, 94.0, 92.0, 100.0, 98.0, 106.0, 104.0]
    assert res.final_money == 104.0
    # revenue arrives only every OTHER hour: 2 sales x 2 items x 5 = 20
    assert res.total_revenue == 20.0
    # S buys 2 ore in each of h1..h8 but the h8 order never lands, so 7 x 2 = 14
    # ore reach storage.  M eats 2 in each of h2, h4, h6, h8 = 8.
    # 14 - 8 = 6 ore stranded, and the money for all 16 is gone.
    assert res.ending_storage["S"]["ore"] == 6


# ---------------------------------------------------------------------------
# 5. the same machine with the 1-Hour mod: one extra downstream cycle
# ---------------------------------------------------------------------------


def test_05_one_hour_mod_buys_an_extra_cycle():
    st = _two_hour_world()
    base = simulate(st, conf(st, {"S": 2, "M": 2, "Z": 2}))
    res = simulate(
        st,
        conf(st, {"S": 2, "M": 2, "Z": 2}, mods={"M": [ModKind.ONE_HOUR_PRODUCTION]}),
    )

    # With the mod M runs every hour, and pays 0.5 per item on top (2 items
    # per run = 1.0 an hour).
    # h1: S orders -2.                                             100 - 2 = 98
    # h2: S deposits 2, M eats them (-1 mod cost), S orders -2.     98 - 3 = 95
    # h3: M deposits 2 widgets, Z pulls; M eats again -1; S -2.     95 - 3 = 92
    # h4: sale +10, M runs -1, S orders -2.                         92 + 10 - 3 = 99
    # h5..h8: steady +10 - 3 = +7 -> 106, 113, 120, 127.
    assert res.money_by_hour == [100.0, 98.0, 95.0, 92.0, 99.0, 106.0, 113.0, 120.0, 127.0]
    assert res.final_money == 127.0
    # 5 sales (h4..h8) instead of 2, i.e. the whole downstream chain cycles
    # twice as often; worth +23 net of the mod's per-item cost.
    assert res.final_money - base.final_money == 23.0
    assert res.total_revenue == 50.0


# ---------------------------------------------------------------------------
# 6. a maker's storage fills while its consumer is starved: overflow accounting
# ---------------------------------------------------------------------------


def test_06_overflow_destroys_exactly_the_overshoot():
    st = world(
        [
            supplier("S", "ore", 1.0, out_max=3, storage=50),
            maker(
                "M",
                [Recipe(id="m", inputs={"ore": 1}, output_item="widget", output_qty=1)],
                out_max=3,
                storage=4,  # holds 4 widgets; M makes 3 an hour
            ),
            seller("Z", "widget", 5.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=100.0,
        horizon=6,
        items=["ore", "widget"],
    )
    # Z is set to 10, more than M will ever have in stock, so Z never pulls and
    # M's buffer backs up.
    res = simulate(st, conf(st, {"S": 3, "M": 3, "Z": 10}))

    # h1: S orders 3 @1 = -3.                                      100 - 3 = 97
    # h2: S deposits 3 ore; M eats them; S orders -3.               97 - 3 = 94
    # h3: M deposits 3 widgets (0+3 = 3 <= cap 4, no loss); M runs
    #     again (3 < 4); S orders -3.                               94 - 3 = 91
    # h4: M deposits 3 more: 3 + 3 = 6 against a cap of 4 ->
    #     2 widgets destroyed.  M's buffer is full so it idles.
    #     S still orders -3.                                        91 - 3 = 88
    # h5, h6: M stays full and idle, S keeps buying.  85, 82.
    assert res.money_by_hour == [100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0]
    assert res.final_money == 82.0
    overflow = warns(res, WarningKind.OVERFLOW, "M")
    assert sum(w.amount for w in overflow) == 2.0  # exactly the overshoot, 6 - 4
    assert len(warns(res, WarningKind.IDLE_FULL, "M")) == 3  # h4, h5, h6
    assert res.ending_storage["M"]["widget"] == 4  # the cap, all of it stranded


# ---------------------------------------------------------------------------
# 7. a recipe switch changes the input type and the chain re-resolves
# ---------------------------------------------------------------------------


def _two_input_world() -> FactoryState:
    return world(
        [
            supplier("S1", "ore", 1.0, out_max=2, storage=50, y=30.0),
            supplier("S2", "clay", 1.0, out_max=2, storage=50, y=40.0),
            maker(
                "M",
                [
                    Recipe(
                        id="from_ore",
                        inputs={"ore": 2},
                        output_item="widget",
                        output_qty=1,
                        production_cost=0.0,
                    ),
                    Recipe(
                        id="from_clay",
                        inputs={"clay": 2},
                        output_item="widget",
                        output_qty=1,
                        production_cost=3.0,
                    ),
                ],
                out_max=2,
            ),
            seller("Z", "widget", 10.0),
        ],
        [("S1", "M"), ("S2", "M"), ("M", "Z")],
        money=100.0,
        horizon=5,
        items=["ore", "clay", "widget"],
    )


def test_07_recipe_switch_repoints_the_chain():
    st = _two_input_world()
    settings = {"S1": 2, "S2": 2, "M": 1, "Z": 1}
    ore = simulate(st, conf(st, settings, recipes={"M": "from_ore"}))
    clay = simulate(st, conf(st, settings, recipes={"M": "from_clay"}))

    # Both suppliers buy 2 units an hour for 2 money an hour each, whatever M
    # is set to; only M's input type and production cost change.
    # ORE recipe (production_cost 0):
    #   h1: -2 -2                                                  100 - 4 = 96
    #   h2: both deposit 2; M eats 2 ore; suppliers reorder -4      96 - 4 = 92
    #   h3: M deposits 1 widget, Z pulls it; M eats 2 ore; -4       92 - 4 = 88
    #   h4: sale +10, M runs, suppliers -4                          88 + 10 - 4 = 94
    #   h5: +10 - 4                                                 100
    assert ore.money_by_hour == [100.0, 96.0, 92.0, 88.0, 94.0, 100.0]
    assert ore.final_money == 100.0
    # CLAY recipe: identical timing, but each widget costs 3 to make.
    #   h1: -4 -> 96;  h2: -4 -3 -> 89;  h3: -4 -3 -> 82
    #   h4: +10 -4 -3 -> 85;  h5: +10 -4 -3 -> 88
    assert clay.money_by_hour == [100.0, 96.0, 89.0, 82.0, 85.0, 88.0]
    assert clay.final_money == 88.0

    # The chain really re-pointed: under the ore recipe the clay piles up
    # untouched (4 hours of deliveries x 2), and vice versa.
    assert ore.ending_storage.get("S2", {}).get("clay") == 8
    assert "S1" not in ore.ending_storage
    assert clay.ending_storage.get("S1", {}).get("ore") == 8
    assert "S2" not in clay.ending_storage


# ---------------------------------------------------------------------------
# 8. money runs out mid-run
# ---------------------------------------------------------------------------


def test_08_money_exhausted_mid_run():
    st = world(
        [supplier("S", "ore", 3.0, out_max=2, storage=20), seller("Z", "ore", 0.5)],
        [("S", "Z")],
        money=5.0,
        horizon=6,
        items=["ore"],
    )
    res = simulate(st, conf(st, {"S": 1, "Z": 1}))

    # h1: S buys 1 ore @3.                                           5 - 3 = 2
    # h2: S deposits it, Z pulls it.  S wants 3 more and has 2 ->
    #     order refused outright (NO_MONEY), nothing bought.               2
    # h3: the sale matures: 1 x 0.5 = +0.5.  Z now has nothing to sell
    #     (skipped) and S still cannot afford 3.                        2.5
    # h4..h6: frozen at 2.5 -- one item of stock bought the whole run.
    assert res.money_by_hour == [5.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5]
    assert res.final_money == 2.5
    assert len(warns(res, WarningKind.NO_MONEY, "S")) == 5  # h2..h6
    assert res.total_revenue == 0.5


# ---------------------------------------------------------------------------
# 9. the horizon ends with items stranded in storage, worth zero
# ---------------------------------------------------------------------------


def test_09_stranded_stock_is_worth_zero():
    st = world(
        [supplier("S", "ore", 1.0, out_max=3, storage=20), seller("Z", "ore", 10.0)],
        [("S", "Z")],
        money=100.0,
        horizon=4,
        items=["ore"],
    )
    # The seller is switched off: everything bought just sits there.
    res = simulate(st, conf(st, {"S": 2, "Z": 0}))

    # 4 hours x 2 ore x 1 = 8 spent, nothing sold.  100 - 8 = 92.
    # Deliveries land in h2, h3, h4 (the h4 order never arrives) = 6 ore held.
    assert res.money_by_hour == [100.0, 98.0, 96.0, 94.0, 92.0]
    assert res.final_money == 92.0
    assert res.ending_storage == {"S": {"ore": 6}}
    stranded = warns(res, WarningKind.STRANDED, "S")
    assert len(stranded) == 1 and stranded[0].amount == 6.0
    # 6 ore that would have sold for 60 contributed exactly 0
    assert res.final_money == 100.0 - 8.0


# ---------------------------------------------------------------------------
# 10. one unit above what upstream can supply -> output goes to ZERO, not n-1.
#     This is the test that saves the optimizer: allocation is a cliff.
# ---------------------------------------------------------------------------


def test_10_one_unit_too_high_is_a_cliff_not_a_slope():
    st = world(
        [
            # S makes exactly 3 ore an hour and can only hold 3
            supplier("S", "ore", 1.0, out_max=3, storage=3),
            maker(
                "M",
                [Recipe(id="m", inputs={"ore": 1}, output_item="widget", output_qty=1)],
                out_max=4,
                storage=20,
            ),
            seller("Z", "widget", 10.0),
        ],
        [("S", "M"), ("M", "Z")],
        money=100.0,
        horizon=6,
        items=["ore", "widget"],
    )
    ok = simulate(st, conf(st, {"S": 3, "M": 3, "Z": 3}))
    cliff = simulate(st, conf(st, {"S": 3, "M": 4, "Z": 3}))

    # M at 3 (exactly what S supplies):
    #   h1: S orders 3 @1 = -3                                     100 - 3 = 97
    #   h2: S deposits 3, M eats all 3, S reorders -3               97 - 3 = 94
    #   h3: M deposits 3 widgets, Z pulls, M eats 3 more, S -3      94 - 3 = 91
    #   h4: sale 3 x 10 = +30, everything repeats -3                91 + 27 = 118
    #   h5: +27 -> 145.   h6: +27 -> 172.
    assert ok.money_by_hour == [100.0, 97.0, 94.0, 91.0, 118.0, 145.0, 172.0]
    assert ok.final_money == 172.0

    # M at 4 (ONE unit above what S can ever hold):
    #   h1: S orders 3 = -3                                        100 - 3 = 97
    #   h2: S deposits 3 -> storage full at its cap of 3.  M needs 4,
    #       finds 3, and takes NOTHING.  S is full so it stops buying.
    #   h3..h6: deadlock.  Money frozen at 97 forever.
    assert cliff.money_by_hour == [100.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    assert cliff.final_money == 97.0
    # Not "n-1 units": M produced literally nothing.
    assert "M" not in cliff.ending_storage
    assert cliff.total_revenue == 0.0
    assert len(warns(cliff, WarningKind.SKIPPED, "M")) == 6


@pytest.mark.latency
def test_evaluate_fast_latency():
    """A 10-machine, 24-hour evaluation must be far below a millisecond.

    The optimizer budgets 10^5..10^6 evaluations inside a 30 second solve, so
    this is the number that decides whether the search is real.  The ceiling is
    deliberately generous (2 ms) so a loaded CI box cannot make it flaky; the
    measurement itself is printed.
    """
    import time

    from services.solvers.factory.sim import compile_factory, config_to_vector, evaluate_fast

    machines = [supplier("m0", "i0", 1.0, out_max=10, storage=50, y=0.0)]
    edges = []
    for k in range(1, 9):
        machines.append(
            maker(
                f"m{k}",
                [
                    Recipe(
                        id="r",
                        inputs={f"i{k - 1}": 1},
                        output_item=f"i{k}",
                        output_qty=1,
                        production_cost=0.1,
                    )
                ],
                out_max=10,
                storage=50,
                y=float(k),
            )
        )
        edges.append((f"m{k - 1}", f"m{k}"))
    machines.append(seller("m9", "i8", 5.0, y=9.0))
    edges.append(("m8", "m9"))
    st = world(
        machines, edges, money=500.0, horizon=24, items=[f"i{k}" for k in range(9)]
    )
    cf = compile_factory(st)
    vec = config_to_vector(cf, conf(st, {m.id: 4 for m in st.machines}))
    assert evaluate_fast(cf, vec) > 0

    reps = 2000
    for _ in range(200):  # warm the plan cache before timing
        evaluate_fast(cf, vec)
    t0 = time.perf_counter()
    for _ in range(reps):
        evaluate_fast(cf, vec)
    per_call_us = (time.perf_counter() - t0) / reps * 1e6
    print(f"\nevaluate_fast: {per_call_us:.1f} us per 10-machine 24-hour simulation")
    assert per_call_us < 2000.0, f"evaluate_fast took {per_call_us:.0f} us"
