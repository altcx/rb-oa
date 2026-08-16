"""Golden tests for the builder solver package.

Every instance here is tiny enough that the answer is worked out by hand in a
comment above the assertion.  Nothing in this file (or the package it tests)
names a concrete domain: parts have opaque named integer attributes and
obstacles have opaque named integer requirements.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.clip import (
    ClippedPuzzle,
    clip_puzzle,
    resolve_rules,
)
from services.solvers.builder.count import (
    StateSpaceTooLarge,
    count_dp,
    count_meet_in_middle,
)
from services.solvers.builder.cpsat import solve_cpsat
from services.solvers.builder.enumerate import enumerate_builds
from services.solvers.builder.minimal import (
    binding_obstacle,
    free_multiplier,
    free_parts,
    minimal_builds,
)
from services.solvers.builder.model import BuilderPuzzle, Obstacle, Part
from services.solvers.builder.ordered import solve_consumable
from services.solvers.builder.solve import brute_force, solve_builder

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def mk(parts, obstacles, provenance=None, **rule_kwargs) -> BuilderPuzzle:
    rules = BuilderRules(**rule_kwargs)
    return BuilderPuzzle(
        parts=parts, obstacles=obstacles, rules=rules, provenance=provenance or {}
    )


def part(pid: str, weight: int = 0, slots: int = 1, qty: int = 1, cost: int = 0, **attrs):
    return Part(id=pid, weight=weight, slots=slots, qty_available=qty, cost=cost, attributes=attrs)


def obstacle(oid: str, order: int = 0, **requires):
    return Obstacle(id=oid, order=order, requires=requires)


def unclipped(puzzle: BuilderPuzzle, semantics: str = "threshold") -> ClippedPuzzle:
    """A ClippedPuzzle whose caps are so high that no clipping actually happens.

    Used by the clip-safety test as the "before" picture.
    """
    cp = clip_puzzle(puzzle, semantics=semantics)
    raw = np.zeros_like(cp.attrs)
    for i, p in enumerate(puzzle.parts):
        for k, name in enumerate(cp.attr_names):
            raw[i, k] = max(int(p.attr(name)), 0)
    caps = raw.sum(axis=0) + cp.reqs.sum(axis=0) + 1
    return ClippedPuzzle(
        attr_names=list(cp.attr_names),
        caps=caps,
        weights=cp.weights,
        attrs=raw,
        qty=cp.qty,
        slots=cp.slots,
        costs=cp.costs,
        reqs=cp.reqs,
        part_ids=list(cp.part_ids),
        obstacle_ids=list(cp.obstacle_ids),
        obstacle_kinds=list(cp.obstacle_kinds),
        cap_policy=cp.cap_policy,
    )


BASE = dict(
    obstacle_semantics="threshold",
    aggregation="sum",
    duplicates_allowed=False,
    obstacle_ordering="unordered",
    failure_mode="all_must_pass",
    objective="count_valid",
)


# ---------------------------------------------------------------------------
# 1. single obstacle, single attribute
# ---------------------------------------------------------------------------


def test_single_obstacle_single_attribute():
    # Parts contribute alpha 1, 2, 3.  One obstacle needs alpha >= 3.
    # Subset sums: {} 0, {a} 1, {b} 2, {c} 3*, {ab} 3*, {ac} 4*, {bc} 5*, {abc} 6*
    # -> 5 valid builds.
    pz = mk(
        [part("a", weight=1, alpha=1), part("b", weight=2, alpha=2), part("c", weight=3, alpha=3)],
        [obstacle("gate", alpha=3)],
        **BASE,
    )
    cp = clip_puzzle(pz)
    assert count_dp(cp, pz.rules) == 5
    assert enumerate_builds(cp, pz.rules)[0] == 5
    assert count_meet_in_middle(cp, pz.rules) == 5
    assert solve_cpsat(cp, pz.rules)[0] == 5
    assert brute_force(pz).total_valid == 5
    assert solve_builder(pz).total_valid == 5


# ---------------------------------------------------------------------------
# 2. two attributes, sum aggregation matters
# ---------------------------------------------------------------------------


def test_two_attributes_sum_aggregation():
    # a=(2,0)  b=(0,2)  c=(1,1);  obstacle needs alpha>=2 AND beta>=2.
    # {} (0,0)x  {a}(2,0)x  {b}(0,2)x  {c}(1,1)x  {ab}(2,2)*  {ac}(3,1)x
    # {bc}(1,3)x  {abc}(3,3)*  -> 2 valid builds.
    pz = mk(
        [part("a", alpha=2), part("b", beta=2), part("c", alpha=1, beta=1)],
        [obstacle("gate", alpha=2, beta=2)],
        **BASE,
    )
    cp = clip_puzzle(pz)
    assert count_dp(cp, pz.rules) == 2
    assert enumerate_builds(cp, pz.rules)[0] == 2
    assert brute_force(pz).total_valid == 2


# ---------------------------------------------------------------------------
# 3. min aggregation: one part drags the minimum down
# ---------------------------------------------------------------------------


def test_min_aggregation_drags_the_build_down():
    # min aggregation, obstacle needs alpha >= 3.  a=5, b=5, c=1.
    # {} 0x (empty build aggregates to 0 by convention)
    # {a}5*  {b}5*  {c}1x  {ab}5*  {ac}1x  {bc}1x  {abc}1x  -> 3 valid builds.
    rules = dict(BASE)
    rules["aggregation"] = "min"
    pz = mk(
        [part("a", alpha=5), part("b", alpha=5), part("c", alpha=1)],
        [obstacle("gate", alpha=3)],
        **rules,
    )
    cp = clip_puzzle(pz)
    assert count_dp(cp, pz.rules) == 3
    assert enumerate_builds(cp, pz.rules)[0] == 3
    assert count_meet_in_middle(cp, pz.rules) == 3
    assert solve_cpsat(cp, pz.rules)[0] == 3
    assert brute_force(pz).total_valid == 3


# ---------------------------------------------------------------------------
# 4. weight budget binding
# ---------------------------------------------------------------------------


def test_weight_budget_binds():
    # a(w5,alpha3) b(w5,alpha3) c(w1,alpha1); obstacle alpha>=3; weight_max=5.
    # {a} w5 alpha3 *   {b} w5 alpha3 *   every other satisfying subset weighs
    # 6 or more -> 2 valid builds, and the weight slack at the optimum is 0.
    rules = dict(BASE)
    rules["weight_max"] = 5
    pz = mk(
        [part("a", weight=5, alpha=3), part("b", weight=5, alpha=3), part("c", weight=1, alpha=1)],
        [obstacle("gate", alpha=3)],
        **rules,
    )
    cp = clip_puzzle(pz)
    assert count_dp(cp, pz.rules) == 2
    assert brute_force(pz).total_valid == 2
    report = solve_builder(pz)
    assert report.total_valid == 2
    assert report.weight_slack == 0
    assert report.best_build is not None and report.best_build.weight == 5


# ---------------------------------------------------------------------------
# 5. slot cap binding
# ---------------------------------------------------------------------------


def test_slot_cap_binds():
    # Four interchangeable parts each worth alpha 1 and one slot; obstacle needs
    # alpha >= 3 and the slot cap is 3.  Only the four 3-part subsets qualify
    # (the 4-part subset needs 4 slots) -> 4 valid builds.
    rules = dict(BASE)
    rules["slot_max"] = 3
    pz = mk(
        [part(pid, weight=1, slots=1, alpha=1) for pid in "abcd"],
        [obstacle("gate", alpha=3)],
        **rules,
    )
    cp = clip_puzzle(pz)
    assert count_dp(cp, pz.rules) == 4
    assert enumerate_builds(cp, pz.rules)[0] == 4
    assert brute_force(pz).total_valid == 4
    assert solve_builder(pz).total_valid == 4


def test_slot_cap_makes_zero_weight_parts_not_free():
    # A zero-weight, zero-attribute part is inert, but with a slot cap it still
    # competes for slots, so it must not be factored out as free.
    rules = dict(BASE)
    rules["slot_max"] = 2
    pz = mk(
        [part("a", weight=0, slots=1, alpha=3), part("filler", weight=0, slots=1)],
        [obstacle("gate", alpha=3)],
        **rules,
    )
    cp = clip_puzzle(pz)
    assert free_parts(cp, pz.rules) == []
    # {a} and {a, filler} both fit two slots -> 2 valid builds.
    assert brute_force(pz).total_valid == 2
    assert solve_builder(pz).total_valid == 2


# ---------------------------------------------------------------------------
# 6. duplicates_allowed: bounded multiset counting
# ---------------------------------------------------------------------------


def test_duplicates_allowed_multiset_count():
    # a: alpha 1, up to 3 copies.  b: alpha 2, up to 2 copies.  Need alpha >= 3.
    # i copies of a, j of b -> alpha = i + 2j.
    #   i=0: j=2 only            -> 1
    #   i=1: j=1,2               -> 2
    #   i=2: j=1,2               -> 2
    #   i=3: j=0,1,2             -> 3
    # -> 8 valid multisets.
    rules = dict(BASE)
    rules["duplicates_allowed"] = True
    pz = mk(
        [part("a", qty=3, alpha=1), part("b", qty=2, alpha=2)],
        [obstacle("gate", alpha=3)],
        **rules,
    )
    cp = clip_puzzle(pz)
    assert count_dp(cp, pz.rules) == 8
    assert solve_cpsat(cp, pz.rules)[0] == 8
    assert brute_force(pz).total_valid == 8
    assert solve_builder(pz).total_valid == 8
    with pytest.raises(ValueError):
        enumerate_builds(cp, pz.rules)
    with pytest.raises(ValueError):
        count_meet_in_middle(cp, pz.rules)


# ---------------------------------------------------------------------------
# 7. consumable semantics: clears obstacle 1, then fails obstacle 2
# ---------------------------------------------------------------------------


def test_consumable_clears_first_obstacle_then_fails_second():
    # Two obstacles each spend alpha 3.  a supplies 5, b supplies 3.
    # THRESHOLD reading: alpha >= 3 is enough -> {a}, {b}, {ab} = 3 valid.
    # CONSUMABLE reading: total spend is 6, so only {ab} (alpha 8) survives;
    # {a} clears obstacle 1 (5 -> 2 left) and then fails obstacle 2 -> 1 valid.
    parts = [part("a", weight=1, alpha=5), part("b", weight=1, alpha=3)]
    obstacles = [obstacle("first", order=0, alpha=3), obstacle("second", order=1, alpha=3)]

    thr = mk(parts, obstacles, **BASE)
    assert brute_force(thr).total_valid == 3
    assert solve_builder(thr).total_valid == 3

    rules = dict(BASE)
    rules["obstacle_semantics"] = "consumable"
    con = mk(parts, obstacles, **rules)
    cp = clip_puzzle(con, semantics="consumable")
    # the consumable cap is the SUM of requirements, not the max
    assert cp.clipped_at() == {"alpha": 6}
    assert count_dp(cp, con.rules) == 1
    assert brute_force(con).total_valid == 1
    report = solve_consumable(cp, con.rules)
    assert report.total_valid == 1
    assert solve_builder(con).total_valid == 1

    # and {a} alone really does clear exactly one of the two obstacles
    only_a = mk([parts[0]], obstacles, **rules)
    cp_a = clip_puzzle(only_a, semantics="consumable")
    _, builds = enumerate_builds(
        cp_a, only_a.rules.with_flag("failure_mode", "count_passed"), max_builds=10
    )
    got = {tuple(sorted(b.counts)): b.obstacles_passed for b in builds}
    assert got[("a",)] == 1


def test_consumable_order_does_not_matter_for_feasibility():
    # Pure consumption: the binding moment is always after the last obstacle, so
    # reversing the sequence cannot change which builds are valid.
    parts = [part("a", alpha=4), part("b", alpha=3), part("c", alpha=2)]
    rules = dict(BASE)
    rules["obstacle_semantics"] = "consumable"
    forward = mk(
        parts,
        [obstacle("x", order=0, alpha=2), obstacle("y", order=1, alpha=5)],
        **rules,
    )
    backward = mk(
        parts,
        [obstacle("y", order=0, alpha=5), obstacle("x", order=1, alpha=2)],
        **rules,
    )
    assert brute_force(forward).total_valid == brute_force(backward).total_valid
    assert solve_builder(forward).total_valid == solve_builder(backward).total_valid


def test_mixed_semantics_uses_per_obstacle_kinds():
    # "spend" consumes alpha 3, "check" only tests for alpha 3 afterwards, so the
    # requirement is 3 + 3 = 6 even though only one obstacle actually spends.
    parts = [part("a", alpha=5), part("b", alpha=3)]
    obstacles = [obstacle("spend", order=0, alpha=3), obstacle("check", order=1, alpha=3)]
    rules = dict(BASE)
    rules["obstacle_semantics"] = "mixed"
    pz = mk(
        parts,
        obstacles,
        provenance={"obstacle_semantics": {"spend": "consumable", "check": "threshold"}},
        **rules,
    )
    assert brute_force(pz).total_valid == 1  # only {a, b}
    assert solve_builder(pz).total_valid == 1


# ---------------------------------------------------------------------------
# 8. free zero-weight parts give an exact 2**|F| multiplier
# ---------------------------------------------------------------------------


def test_free_parts_multiplier_is_exact():
    # Two real parts each supply alpha 3 against an alpha>=3 obstacle:
    # {a}, {b}, {a,b} = 3 valid cores.  Three inert parts (weight 0, no
    # attributes) can be added or left out freely -> 3 * 2**3 = 24 valid builds.
    parts = [part("a", weight=1, alpha=3), part("b", weight=1, alpha=3)]
    parts += [part(f"f{i}", weight=0, slots=0) for i in range(3)]
    pz = mk(parts, [obstacle("gate", alpha=3)], **BASE)
    cp = clip_puzzle(pz)

    assert sorted(cp.part_ids[i] for i in free_parts(cp, pz.rules)) == ["f0", "f1", "f2"]
    assert free_multiplier(cp, pz.rules) == 8

    report = solve_builder(pz)
    assert report.free_multiplier == 8
    assert sorted(report.free_parts) == ["f0", "f1", "f2"]
    assert report.total_valid == 24
    assert brute_force(pz).total_valid == 24
    assert enumerate_builds(cp, pz.rules)[0] == 24


# ---------------------------------------------------------------------------
# 9. the clip is safe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aggregation", ["sum", "min", "max"])
def test_clipping_does_not_change_the_count(aggregation):
    """Brute force the same instance with and without clipping."""
    rules = dict(BASE)
    rules["aggregation"] = aggregation
    rules["weight_max"] = 14
    parts = [
        part("a", weight=3, alpha=17, beta=2),
        part("b", weight=2, alpha=1, beta=9),
        part("c", weight=4, alpha=6, beta=6),
        part("d", weight=1, alpha=25, beta=0),
        part("e", weight=5, alpha=0, beta=13),
        part("f", weight=2, alpha=4, beta=4),
        part("g", weight=1, alpha=2, beta=1),
        part("h", weight=6, alpha=9, beta=9),
    ]
    obstacles = [obstacle("one", order=0, alpha=4, beta=3), obstacle("two", order=1, alpha=6)]
    pz = mk(parts, obstacles, **rules)

    clipped = clip_puzzle(pz)
    raw = unclipped(pz)
    assert (clipped.caps < raw.caps).any()  # the clip really did something

    truth = brute_force(pz).total_valid
    assert enumerate_builds(raw, pz.rules)[0] == truth
    assert enumerate_builds(clipped, pz.rules)[0] == truth
    assert count_dp(raw, pz.rules) == truth
    assert count_dp(clipped, pz.rules) == truth


def test_consumable_clip_uses_the_sum_of_requirements():
    """The threshold cap would silently drop valid builds under consumption."""
    parts = [part("a", alpha=4), part("b", alpha=4)]
    obstacles = [obstacle(f"o{i}", order=i, alpha=3) for i in range(3)]
    rules = dict(BASE)
    rules["obstacle_semantics"] = "consumable"
    pz = mk(parts, obstacles, **rules)

    good = clip_puzzle(pz, semantics="consumable")
    assert good.clipped_at() == {"alpha": 9}
    # 4 + 4 = 8 < 9, so nothing can pay for all three obstacles.
    assert count_dp(good, pz.rules) == 0
    assert brute_force(pz).total_valid == 0

    # With one more supplier the total reaches 12 >= 9 and exactly one build works.
    pz2 = mk(parts + [part("c", alpha=4)], obstacles, **rules)
    good2 = clip_puzzle(pz2, semantics="consumable")
    assert count_dp(good2, pz2.rules) == 1
    assert brute_force(pz2).total_valid == 1

    # Using the threshold cap (3) under consumable semantics destroys the answer:
    # every part is clipped to 3, the pool can no longer represent the 9 points
    # the sequence actually spends, and the count comes out wrong (here every
    # non-empty subset looks valid).  This is the silent failure the cap policy
    # exists to prevent.
    wrong = clip_puzzle(pz2, semantics="threshold")
    assert wrong.clipped_at() == {"alpha": 3}
    assert count_dp(wrong, pz2.rules) != brute_force(pz2).total_valid


# ---------------------------------------------------------------------------
# 10. minimal builds are a real antichain
# ---------------------------------------------------------------------------


def test_minimal_builds_are_an_antichain_and_supersets_are_valid():
    rules = dict(BASE)
    rules["weight_max"] = 18
    parts = [
        part("a", weight=3, alpha=2, beta=1),
        part("b", weight=2, alpha=1, beta=2),
        part("c", weight=4, alpha=3, beta=0),
        part("d", weight=1, alpha=0, beta=3),
        part("e", weight=5, alpha=4, beta=4),
        part("f", weight=2, alpha=1, beta=1),
        part("g", weight=6, alpha=2, beta=2),
    ]
    pz = mk(parts, [obstacle("gate", alpha=4, beta=4)], **rules)
    cp = clip_puzzle(pz)
    cores = minimal_builds(cp, pz.rules, max_cores=50)
    assert cores

    index = {pid: i for i, pid in enumerate(cp.part_ids)}
    sets = [frozenset(c.core.counts) for c in cores]

    def valid(ids) -> bool:
        counts = np.zeros(cp.n_parts, dtype=np.int64)
        for pid in ids:
            counts[index[pid]] = 1
        weight = int((counts * cp.weights).sum())
        if weight > rules["weight_max"]:
            return False
        agg = np.minimum((counts[:, None] * cp.attrs).sum(axis=0), cp.caps)
        return bool((agg >= cp.reqs.max(axis=0)).all())

    # every core is valid, and stops being valid if any single part is removed
    for s in sets:
        assert valid(s)
        for pid in s:
            assert not valid(s - {pid})

    # antichain: no core contains another
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j:
                assert not a < b

    # monotone requirements: every superset inside the budget is valid too
    for s in sets:
        for pid in cp.part_ids:
            if pid not in s and valid(s | {pid}) is False:
                # only possible when the weight budget is exceeded
                counts = np.zeros(cp.n_parts, dtype=np.int64)
                for q in s | {pid}:
                    counts[index[q]] = 1
                assert int((counts * cp.weights).sum()) > rules["weight_max"]


def test_binding_obstacle_is_the_one_that_kills_the_most():
    # "hard" asks for far more than "easy", so it must be the binding obstacle.
    parts = [part(f"p{i}", weight=1, alpha=1) for i in range(6)]
    pz = mk(
        parts,
        [obstacle("easy", order=0, alpha=1), obstacle("hard", order=1, alpha=5)],
        **BASE,
    )
    cp = clip_puzzle(pz)
    binding, counts = binding_obstacle(cp, pz.rules)
    assert binding == "hard"
    # subsets with fewer than 5 parts fail "hard": 1+6+15+20+15 = 57 of 64
    assert counts == {"easy": 1, "hard": 57}
    assert brute_force(pz).obstacle_elimination == counts


# ---------------------------------------------------------------------------
# rule defaults, guards, and report completeness
# ---------------------------------------------------------------------------


def test_unresolved_rules_fall_back_to_documented_defaults():
    pz = mk([part("a", weight=1, alpha=3)], [obstacle("gate", alpha=3)])
    resolved, warnings = resolve_rules(pz.rules)
    assert resolved.obstacle_semantics == "threshold"
    assert resolved.aggregation == "sum"
    assert resolved.duplicates_allowed is False
    assert resolved.obstacle_ordering == "unordered"
    assert resolved.failure_mode == "all_must_pass"
    assert resolved.objective == "count_valid"
    assert len(warnings) == 6

    report = solve_builder(pz)
    blob = " ".join(report.warnings)
    for flag in (
        "obstacle_semantics",
        "aggregation",
        "duplicates_allowed",
        "obstacle_ordering",
        "failure_mode",
        "objective",
    ):
        assert flag in blob
    assert report.total_valid == 1


def test_report_is_fully_populated():
    rules = dict(BASE)
    rules["weight_max"] = 12
    parts = [
        part("a", weight=3, alpha=2),
        part("b", weight=2, alpha=3),
        part("c", weight=4, alpha=1),
        part("inert", weight=0, slots=0),
    ]
    pz = mk(parts, [obstacle("gate", alpha=4), obstacle("other", alpha=2)], **rules)
    report = solve_builder(pz, max_builds=10)
    assert report.total_valid > 0
    assert report.builds
    assert report.minimal_builds
    assert report.free_parts == ["inert"]
    assert report.free_multiplier == 2
    assert report.binding_obstacle == "gate"
    assert set(report.obstacle_elimination) == {"gate", "other"}
    assert report.weight_slack is not None
    assert report.best_build is not None
    assert report.method
    assert report.elapsed_ms >= 0.0
    assert report.exact is True
    assert report.clipped_at == {"alpha": 4}
    assert report.minimal_builds[0].freely_addable == ["inert"]
    assert report.minimal_builds[0].extensions_count == 2


def test_enumerate_refuses_beyond_its_ceiling():
    parts = [part(f"p{i}", weight=1, alpha=1) for i in range(25)]
    pz = mk(parts, [obstacle("gate", alpha=3)], **BASE)
    cp = clip_puzzle(pz)
    with pytest.raises(ValueError):
        enumerate_builds(cp, pz.rules)


def test_state_space_guard_raises_with_the_projected_size():
    rules = dict(BASE)
    rules["weight_max"] = 100
    pz = mk(
        [part("a", weight=1, alpha=5_000_000)],
        [obstacle("gate", alpha=5_000_000)],
        **rules,
    )
    cp = clip_puzzle(pz)
    with pytest.raises(StateSpaceTooLarge) as exc:
        count_dp(cp, pz.rules)
    assert exc.value.projected == 5_000_001 * 101
    # the dispatcher must survive it by falling back
    report = solve_builder(pz)
    assert report.total_valid == 1
    assert report.method != "count_dp"


def test_count_passed_scores_partial_clearance():
    # alpha 2 clears the first obstacle only; alpha 5 clears both.
    rules = dict(BASE)
    rules["failure_mode"] = "count_passed"
    pz = mk(
        [part("a", weight=1, alpha=2), part("b", weight=1, alpha=3)],
        [obstacle("easy", order=0, alpha=2), obstacle("hard", order=1, alpha=5)],
        **rules,
    )
    cp = clip_puzzle(pz)
    # {} passes nothing; {a} 2 -> 1; {b} 3 -> 1; {ab} 5 -> 2.  Valid = passes >= 1.
    assert count_dp(cp, pz.rules) == 3
    assert brute_force(pz).total_valid == 3
    assert solve_cpsat(cp, pz.rules)[0] == 3
    best = solve_builder(pz, objective="max_passed").best_build
    assert best is not None and best.obstacles_passed == 2


def test_max_aggregation_picks_the_single_best_carrier():
    # max aggregation: one part at alpha 4 satisfies alpha>=4 all on its own.
    rules = dict(BASE)
    rules["aggregation"] = "max"
    pz = mk(
        [part("a", weight=1, alpha=4), part("b", weight=1, alpha=1), part("c", weight=1, alpha=2)],
        [obstacle("gate", alpha=4)],
        **rules,
    )
    cp = clip_puzzle(pz)
    # every subset containing "a" is valid: 2**2 = 4.
    assert count_dp(cp, pz.rules) == 4
    assert enumerate_builds(cp, pz.rules)[0] == 4
    assert count_meet_in_middle(cp, pz.rules) == 4
    assert solve_cpsat(cp, pz.rules)[0] == 4
    assert brute_force(pz).total_valid == 4


def test_money_budget_is_honoured():
    rules = dict(BASE)
    rules["money_max"] = 5
    pz = mk(
        [part("a", cost=3, alpha=2), part("b", cost=3, alpha=2), part("c", cost=5, alpha=4)],
        [obstacle("gate", alpha=4)],
        **rules,
    )
    cp = clip_puzzle(pz)
    # {a,b} costs 6 (over), {c} costs 5 -> only {c} works.
    assert count_dp(cp, pz.rules) == 1
    assert brute_force(pz).total_valid == 1


def test_min_aggregation_never_factors_out_free_parts():
    # Regression: a zero-weight part sitting at the clip ceiling looks neutral
    # for ``min``, but a build made only of such parts aggregates to the ceiling
    # while the empty build aggregates to 0.  Factoring it out would change the
    # answer for exactly that build, so nothing is free under min aggregation.
    rules = dict(BASE)
    rules["aggregation"] = "min"
    pz = mk(
        [part("a", weight=1, alpha=4), part("big", weight=0, slots=0, alpha=9)],
        [obstacle("gate", alpha=3)],
        **rules,
    )
    cp = clip_puzzle(pz)
    assert free_parts(cp, pz.rules) == []
    assert free_multiplier(cp, pz.rules) == 1
    # {a} min 4*, {big} min 3(clipped)*, {a,big} min 3* -> 3 valid, {} is 0.
    assert brute_force(pz).total_valid == 3
    assert solve_builder(pz).total_valid == 3


@pytest.mark.parametrize("failure_mode", ["all_must_pass", "count_passed"])
def test_no_obstacles_corner(failure_mode):
    # With nothing to clear, "all must pass" is vacuously true for every build
    # inside the budget, while "count passed" can never reach its >=1 minimum.
    rules = dict(BASE)
    rules["failure_mode"] = failure_mode
    pz = mk([part("a", weight=1, alpha=1), part("b", weight=1, alpha=1)], [], **rules)
    cp = clip_puzzle(pz)
    expected = 4 if failure_mode == "all_must_pass" else 0
    assert brute_force(pz).total_valid == expected
    assert count_dp(cp, pz.rules) == expected
    assert enumerate_builds(cp, pz.rules)[0] == expected
    assert solve_cpsat(cp, pz.rules)[0] == expected
    assert solve_builder(pz).total_valid == expected


def test_cpsat_parallel_path_agrees_with_the_serial_one():
    parts = [part(f"p{i}", weight=1, alpha=(i % 3) + 1) for i in range(8)]
    pz = mk(parts, [obstacle("gate", alpha=5)], weight_max=6, **BASE)
    cp = clip_puzzle(pz)
    serial, _ = solve_cpsat(cp, pz.rules, max_solutions=5)
    parallel, _ = solve_cpsat(cp, pz.rules, max_solutions=5, workers=2)
    assert serial == parallel == brute_force(pz).total_valid
