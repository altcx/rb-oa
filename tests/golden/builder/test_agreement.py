"""Cross-checks: every method must agree with itertools ground truth.

The randomised instances are generated from a fixed seed, so a failure here is
reproducible.  ``brute_force`` is the referee: it works off the *unclipped*
pydantic model with the semantics written out longhand, so agreeing with it
validates the clip, the doubling recurrence, the DP, meet-in-the-middle and
CP-SAT all at once.
"""

from __future__ import annotations

import random
import time

import pytest

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.clip import clip_puzzle
from services.solvers.builder.count import count_dp, count_meet_in_middle
from services.solvers.builder.cpsat import solve_cpsat
from services.solvers.builder.enumerate import enumerate_builds
from services.solvers.builder.model import BuilderPuzzle, Obstacle, Part
from services.solvers.builder.ordered import solve_consumable
from services.solvers.builder.solve import brute_force, solve_builder

ATTRS = ["alpha", "beta", "gamma"]
N_INSTANCES = 40


def random_instance(rng: random.Random, max_parts: int = 12, **overrides) -> BuilderPuzzle:
    n = rng.randint(1, max_parts)
    n_attrs = rng.randint(1, 3)
    names = ATTRS[:n_attrs]
    parts = [
        Part(
            id=f"p{i}",
            weight=rng.randint(0, 6),
            qty_available=rng.randint(1, 3),
            slots=rng.randint(0, 2),
            cost=rng.randint(0, 4),
            attributes={a: rng.randint(0, 4) for a in names},
        )
        for i in range(n)
    ]
    obstacles = [
        Obstacle(id=f"o{j}", order=j, requires={a: rng.randint(0, 5) for a in names})
        for j in range(rng.randint(1, 3))
    ]
    fields = dict(
        weight_max=rng.choice([None, 10, 16]),
        slot_max=rng.choice([None, 4]),
        money_max=rng.choice([None, 8]),
        duplicates_allowed=rng.choice([True, False]),
        obstacle_ordering=rng.choice(["ordered", "unordered"]),
        obstacle_semantics=rng.choice(["threshold", "consumable", "mixed"]),
        aggregation=rng.choice(["sum", "min", "max"]),
        failure_mode=rng.choice(["all_must_pass", "count_passed"]),
        objective="count_valid",
    )
    fields.update(overrides)
    provenance = {}
    if fields["obstacle_semantics"] == "mixed":
        provenance = {
            "obstacle_semantics": {
                o.id: rng.choice(["threshold", "consumable"]) for o in obstacles
            }
        }
    return BuilderPuzzle(
        parts=parts,
        obstacles=obstacles,
        rules=BuilderRules(**fields),
        provenance=provenance,
    )


def _instances(seed: int, max_parts: int = 12, **overrides) -> list[BuilderPuzzle]:
    rng = random.Random(seed)
    return [random_instance(rng, max_parts, **overrides) for _ in range(N_INSTANCES)]


def describe(pz: BuilderPuzzle) -> str:
    r = pz.rules
    return (
        f"n={len(pz.parts)} m={len(pz.obstacles)} sem={r.obstacle_semantics} "
        f"agg={r.aggregation} dup={r.duplicates_allowed} fm={r.failure_mode} "
        f"ord={r.obstacle_ordering} w={r.weight_max} s={r.slot_max} $={r.money_max}"
    )


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pz", _instances(20240517), ids=lambda p: p.fingerprint())
def test_every_method_agrees_with_brute_force(pz: BuilderPuzzle):
    truth = brute_force(pz).total_valid
    cp = clip_puzzle(pz, semantics=pz.rules.obstacle_semantics)
    rules = pz.rules
    ctx = describe(pz)

    assert count_dp(cp, rules) == truth, f"count_dp: {ctx}"
    assert solve_cpsat(cp, rules)[0] == truth, f"cpsat: {ctx}"
    assert solve_builder(pz).total_valid == truth, f"dispatcher: {ctx}"

    if not rules.duplicates_allowed:
        assert enumerate_builds(cp, rules, max_builds=5)[0] == truth, f"enumerate: {ctx}"
        if (
            rules.slot_max is None
            and rules.money_max is None
            and rules.failure_mode == "all_must_pass"
        ):
            assert count_meet_in_middle(cp, rules) == truth, f"mitm: {ctx}"


@pytest.mark.parametrize(
    "pz",
    _instances(881, max_parts=8, duplicates_allowed=True, obstacle_semantics="threshold"),
    ids=lambda p: p.fingerprint(),
)
def test_duplicates_dp_matches_brute_force(pz: BuilderPuzzle):
    """Bounded multiset counting: DP and CP-SAT against itertools.product."""
    truth = brute_force(pz).total_valid
    cp = clip_puzzle(pz, semantics="threshold")
    assert count_dp(cp, pz.rules) == truth, describe(pz)
    assert solve_cpsat(cp, pz.rules)[0] == truth, describe(pz)
    assert solve_builder(pz).total_valid == truth, describe(pz)


@pytest.mark.parametrize(
    "pz",
    _instances(4242, obstacle_semantics="consumable"),
    ids=lambda p: p.fingerprint(),
)
def test_consumable_matches_brute_force(pz: BuilderPuzzle):
    truth = brute_force(pz).total_valid
    cp = clip_puzzle(pz, semantics="consumable")
    report = solve_consumable(cp, pz.rules)
    assert report.total_valid == truth, describe(pz)
    assert solve_builder(pz).total_valid == truth, describe(pz)


@pytest.mark.parametrize(
    "pz",
    _instances(777, obstacle_semantics="mixed", failure_mode="all_must_pass"),
    ids=lambda p: p.fingerprint(),
)
def test_mixed_semantics_matches_brute_force(pz: BuilderPuzzle):
    truth = brute_force(pz).total_valid
    cp = clip_puzzle(pz, semantics="mixed")
    assert solve_consumable(cp, pz.rules).total_valid == truth, describe(pz)
    assert solve_builder(pz).total_valid == truth, describe(pz)


# ---------------------------------------------------------------------------
# size and latency
# ---------------------------------------------------------------------------


def realistic_instance() -> BuilderPuzzle:
    """30 parts, 5 obstacles, 3 attributes — the shape the app actually sees."""
    rng = random.Random(99)
    names = ATTRS[:3]
    parts = [
        Part(
            id=f"p{i}",
            name=f"part {i}",
            weight=rng.randint(1, 12),
            slots=1,
            attributes={a: rng.randint(0, 6) for a in names},
        )
        for i in range(30)
    ]
    obstacles = [
        Obstacle(id=f"o{j}", order=j, requires={a: rng.randint(2, 9) for a in names})
        for j in range(5)
    ]
    rules = BuilderRules(
        weight_max=60,
        obstacle_semantics="threshold",
        aggregation="sum",
        duplicates_allowed=False,
        obstacle_ordering="unordered",
        failure_mode="all_must_pass",
        objective="count_valid",
    )
    return BuilderPuzzle(parts=parts, obstacles=obstacles, rules=rules)


@pytest.mark.timeout(60)
def test_realistic_instance_finishes_inside_the_budget():
    pz = realistic_instance()
    started = time.perf_counter()
    report = solve_builder(pz, budget_s=10.0)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, f"took {elapsed:.2f}s"
    assert report.exact is True
    assert report.total_valid > 0
    assert report.minimal_builds
    assert report.binding_obstacle is not None
    assert report.best_build is not None
    assert report.weight_slack is not None
    assert report.method == "count_dp"


@pytest.mark.timeout(60)
def test_enumeration_of_four_million_subsets_is_fast():
    """2**22 subsets, built by the doubling recurrence, well under a second."""
    pz = realistic_instance()
    trimmed = BuilderPuzzle(
        parts=pz.parts[:22], obstacles=pz.obstacles, rules=pz.rules, provenance=pz.provenance
    )
    cp = clip_puzzle(trimmed)
    started = time.perf_counter()
    total, builds = enumerate_builds(cp, trimmed.rules, max_builds=5)
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"took {elapsed:.2f}s"
    assert total > 0
    assert len(builds) == 5
    assert count_dp(cp, trimmed.rules) == total


@pytest.mark.timeout(60)
def test_large_part_set_falls_back_without_enumerating():
    """60 parts: far past every enumeration ceiling, still exact via the DP."""
    rng = random.Random(31337)
    parts = [
        Part(
            id=f"p{i}",
            weight=rng.randint(1, 9),
            attributes={a: rng.randint(0, 5) for a in ATTRS[:2]},
        )
        for i in range(60)
    ]
    obstacles = [
        Obstacle(id=f"o{j}", order=j, requires={a: rng.randint(3, 8) for a in ATTRS[:2]})
        for j in range(3)
    ]
    pz = BuilderPuzzle(
        parts=parts,
        obstacles=obstacles,
        rules=BuilderRules(
            weight_max=40,
            obstacle_semantics="threshold",
            aggregation="sum",
            duplicates_allowed=False,
            obstacle_ordering="unordered",
            failure_mode="all_must_pass",
            objective="count_valid",
        ),
    )
    started = time.perf_counter()
    report = solve_builder(pz, budget_s=10.0)
    assert time.perf_counter() - started < 10.0
    assert report.method == "count_dp"
    assert report.total_valid > 0
