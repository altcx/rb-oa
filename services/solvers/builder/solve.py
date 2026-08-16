"""The builder dispatcher — the entry point the rest of the app calls.

METHOD SELECTION
----------------
============================================  ==================================
situation                                     method
============================================  ==================================
semantics consumable / mixed                  ordered.solve_consumable
duplicates_allowed                            count.count_dp
n > 22                                        count.count_dp
n <= 22, objective needs concrete builds      enumerate.enumerate_builds
n <= 22, objective = count_valid              count.count_dp
StateSpaceTooLarge, subset-only, n <= 24      enumerate.enumerate_builds
StateSpaceTooLarge, subset-only, n <= 44,     count.count_meet_in_middle
  weight budget only, all_must_pass
StateSpaceTooLarge otherwise                  cpsat.solve_cpsat
============================================  ==================================

Free-part factoring runs first in every branch: parts that are provably inert
(see ``minimal.free_parts``) are removed from the puzzle the counter sees and the
result is multiplied back by ``free_multiplier``.
"""

from __future__ import annotations

import time
from itertools import combinations, product
from typing import Any, Sequence

import numpy as np

from services.core.rules.dsl import BuilderRules
from services.solvers.builder import minimal as minimal_mod
from services.solvers.builder import ordered as ordered_mod
from services.solvers.builder.clip import (
    COUNT_PASSED_MIN_PASSED,
    RULE_DEFAULTS,
    ClippedPuzzle,
    clip_puzzle,
    make_build,
    resolve_rules,
)
from services.solvers.builder.count import (
    MAX_MITM_PARTS,
    StateSpaceTooLarge,
    count_dp,
    count_meet_in_middle,
)
from services.solvers.builder.cpsat import lightest_build, max_passed_build, solve_cpsat
from services.solvers.builder.enumerate import (
    MAX_ENUMERATE_PARTS,
    best_subset,
    enumerate_builds,
)
from services.solvers.builder.model import Build, BuilderPuzzle, BuildReport

#: Above this many parts we stop trying to enumerate concrete subsets.
ENUMERATE_CEILING = 22

BRUTE_FORCE_CEILING = 16


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def solve_builder(
    puzzle: BuilderPuzzle,
    objective: str | None = None,
    budget_s: float = 10.0,
    max_builds: int = 200,
) -> BuildReport:
    started = time.perf_counter()
    rules, warnings = resolve_rules(puzzle.rules)
    if objective is not None:
        rules = rules.with_flag("objective", objective)
    objective = rules.objective or RULE_DEFAULTS["objective"]

    semantics = rules.obstacle_semantics
    cp = clip_puzzle(puzzle, semantics=semantics)

    if (rules.failure_mode or RULE_DEFAULTS["failure_mode"]) == "count_passed":
        warnings.append(
            "failure_mode='count_passed': a build counts as valid when it clears at "
            f"least {COUNT_PASSED_MIN_PASSED} obstacle(s)"
        )
    if semantics == "mixed":
        listed = puzzle.provenance.get("obstacle_semantics")
        if not isinstance(listed, dict) or not listed:
            warnings.append(
                "obstacle_semantics='mixed' but no per-obstacle kinds were supplied in "
                "puzzle.provenance['obstacle_semantics']; every obstacle assumed 'threshold'"
            )

    if semantics in ("consumable", "mixed"):
        report = ordered_mod.solve_consumable(cp, rules, budget_s=budget_s)
        for w in warnings:
            if w not in report.warnings:
                report.warnings.append(w)
        _enrich(report, cp, rules, objective, max_builds, budget_s, keep_elimination=True)
        report.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return report

    report = BuildReport(clipped_at=cp.clipped_at(), warnings=warnings, exact=True)

    # ---- free-part factoring ------------------------------------------------
    free_idx = minimal_mod.free_parts(cp, rules)
    multiplier = minimal_mod.free_multiplier(cp, rules)
    keep = [i for i in range(cp.n_parts) if i not in set(free_idx)]
    core_cp = cp.restrict(keep) if free_idx else cp
    report.free_parts = [cp.part_ids[i] for i in free_idx]
    report.free_multiplier = multiplier
    if free_idx:
        report.notes.append(
            f"{len(free_idx)} inert part(s) factored out; the count over the remaining "
            f"{core_cp.n_parts} part(s) is multiplied by {multiplier}"
        )
    if rules.slot_max is not None:
        report.notes.append(
            "slot_max is set, so parts occupying slots are never treated as free"
        )

    # ---- counting -----------------------------------------------------------
    count, builds, method, exact = _count(core_cp, rules, objective, budget_s, max_builds)
    report.total_valid = count * multiplier
    report.builds = builds
    report.method = method
    report.exact = exact

    _enrich(report, cp, rules, objective, max_builds, budget_s, keep_elimination=False)
    report.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return report


def _count(
    cp: ClippedPuzzle,
    rules: BuilderRules,
    objective: str,
    budget_s: float,
    max_builds: int,
) -> tuple[int, list[Build], str, bool]:
    subset_only = not rules.duplicates_allowed
    wants_builds = objective in ("enumerate", "min_weight", "max_passed")

    if subset_only and cp.n_parts <= ENUMERATE_CEILING and wants_builds:
        total, builds = enumerate_builds(cp, rules, max_builds=max_builds)
        return total, builds, "enumerate", True

    try:
        total = count_dp(cp, rules)
        builds: list[Build] = []
        if subset_only and cp.n_parts <= ENUMERATE_CEILING and max_builds > 0:
            _, builds = enumerate_builds(cp, rules, max_builds=max_builds)
        return total, builds, "count_dp", True
    except StateSpaceTooLarge:
        pass

    if subset_only and cp.n_parts <= MAX_ENUMERATE_PARTS:
        total, builds = enumerate_builds(cp, rules, max_builds=max_builds)
        return total, builds, "enumerate(fallback)", True

    if (
        subset_only
        and cp.n_parts <= MAX_MITM_PARTS
        and rules.slot_max is None
        and rules.money_max is None
        and (rules.failure_mode or RULE_DEFAULTS["failure_mode"]) == "all_must_pass"
    ):
        return count_meet_in_middle(cp, rules), [], "meet_in_middle", True

    total, builds = solve_cpsat(cp, rules, budget_s=budget_s, max_solutions=max_builds)
    return total, builds, "cpsat", True


def _enrich(
    report: BuildReport,
    cp: ClippedPuzzle,
    rules: BuilderRules,
    objective: str,
    max_builds: int,
    budget_s: float,
    keep_elimination: bool,
) -> None:
    """Fill in the human-facing half of the report."""
    if not report.clipped_at:
        report.clipped_at = cp.clipped_at()

    if not report.free_parts:
        free_idx = minimal_mod.free_parts(cp, rules)
        report.free_parts = [cp.part_ids[i] for i in free_idx]
        report.free_multiplier = minimal_mod.free_multiplier(cp, rules)

    try:
        report.minimal_builds = minimal_mod.minimal_builds(cp, rules, max_cores=50)
    except (ValueError, MemoryError) as exc:  # pragma: no cover - defensive
        report.warnings.append(f"minimal cores unavailable: {exc}")

    if not keep_elimination:
        binding, counts = minimal_mod.binding_obstacle(cp, rules)
        report.binding_obstacle = binding
        report.obstacle_elimination = counts
        report.binding_obstacle_eliminated = counts.get(binding, 0) if binding else 0

    if report.best_build is None:
        report.best_build = _best(cp, rules, objective, budget_s)
    if report.weight_slack is None and rules.weight_max is not None:
        lightest = (
            report.best_build
            if objective != "max_passed"
            else minimal_mod.lightest_valid(cp, rules)
        )
        if lightest is not None:
            report.weight_slack = int(rules.weight_max) - int(lightest.weight)


def _best(
    cp: ClippedPuzzle, rules: BuilderRules, objective: str, budget_s: float
) -> Build | None:
    if objective == "max_passed":
        if cp.n_parts <= MAX_ENUMERATE_PARTS and not rules.duplicates_allowed:
            return best_subset(cp, rules, objective="max_passed")
        return max_passed_build(cp, rules, budget_s=min(budget_s, 5.0))
    if cp.n_parts <= MAX_ENUMERATE_PARTS and not rules.duplicates_allowed:
        return best_subset(cp, rules, objective="min_weight")
    return lightest_build(cp, rules, budget_s=min(budget_s, 5.0))


# ---------------------------------------------------------------------------
# Ground truth for tests
# ---------------------------------------------------------------------------


def brute_force(puzzle: BuilderPuzzle) -> BuildReport:
    """Reference implementation: raw itertools, no clipping, no DP.  n <= 16.

    Deliberately written against the *unclipped* pydantic model with the
    semantics spelled out longhand, so it is an independent check of both the
    clip and the fast solvers.
    """
    started = time.perf_counter()
    rules, warnings = resolve_rules(puzzle.rules)
    parts = list(puzzle.parts)
    if len(parts) > BRUTE_FORCE_CEILING:
        raise ValueError(f"brute_force is limited to {BRUTE_FORCE_CEILING} parts")

    names = list(puzzle.attribute_names())
    obstacles = [
        o for _, o in sorted(enumerate(puzzle.obstacles), key=lambda p: (p[1].order, p[0]))
    ]
    prov = puzzle.provenance.get("obstacle_semantics")
    kinds = [
        (prov.get(o.id, "threshold") if isinstance(prov, dict) else None)
        or ("threshold" if rules.obstacle_semantics == "mixed" else rules.obstacle_semantics)
        for o in obstacles
    ]

    ranges: list[Sequence[int]] = []
    for p in parts:
        top = max(int(p.qty_available), 0)
        ranges.append(range(0, (top if rules.duplicates_allowed else min(top, 1)) + 1))

    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    failure_mode = rules.failure_mode or RULE_DEFAULTS["failure_mode"]

    total = 0
    best: Build | None = None
    best_key: tuple[int, int] | None = None
    builds: list[Build] = []
    elimination = {o.id: 0 for o in obstacles}

    for combo in product(*ranges):
        weight = sum(c * p.weight for c, p in zip(combo, parts))
        slots = sum(c * p.slots for c, p in zip(combo, parts))
        cost = sum(c * p.cost for c, p in zip(combo, parts))
        if rules.weight_max is not None and weight > rules.weight_max:
            continue
        if rules.slot_max is not None and slots > rules.slot_max:
            continue
        if rules.money_max is not None and cost > rules.money_max:
            continue

        agg = _raw_aggregate(combo, parts, names, aggregation)
        for o in obstacles:
            if any(agg[a] < o.requires.get(a, 0) for a in names):
                elimination[o.id] += 1

        passed = _raw_passed(agg, obstacles, names, rules, kinds)
        ok = passed == len(obstacles) if failure_mode == "all_must_pass" else (
            passed >= COUNT_PASSED_MIN_PASSED
        )
        if not ok:
            continue
        total += 1
        build = Build(
            counts={p.id: c for c, p in zip(combo, parts) if c},
            weight=weight,
            slots=slots,
            cost=cost,
            attributes=dict(agg),
            obstacles_passed=passed,
        )
        if len(builds) < 200:
            builds.append(build)
        key = (weight, -passed)
        if best_key is None or key < best_key:
            best_key, best = key, build

    report = BuildReport(
        total_valid=total,
        builds=builds,
        best_build=best,
        method="brute_force",
        exact=True,
        warnings=warnings,
        obstacle_elimination=elimination,
    )
    if elimination:
        binding = max(elimination, key=lambda k: (elimination[k], k))
        report.binding_obstacle = binding
        report.binding_obstacle_eliminated = elimination[binding]
    if best is not None and rules.weight_max is not None:
        report.weight_slack = int(rules.weight_max) - int(best.weight)
    report.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return report


def _raw_aggregate(
    combo: tuple[int, ...], parts: list[Any], names: list[str], aggregation: str
) -> dict[str, int]:
    chosen = [p for c, p in zip(combo, parts) if c]
    if not chosen:
        return {a: 0 for a in names}
    if aggregation == "sum":
        return {a: sum(c * p.attr(a) for c, p in zip(combo, parts)) for a in names}
    if aggregation == "min":
        return {a: min(p.attr(a) for p in chosen) for a in names}
    if aggregation == "max":
        return {a: max(p.attr(a) for p in chosen) for a in names}
    raise ValueError(f"unknown aggregation {aggregation!r}")


def _raw_passed(
    agg: dict[str, int],
    obstacles: list[Any],
    names: list[str],
    rules: BuilderRules,
    kinds: list[str],
) -> int:
    semantics = rules.obstacle_semantics or RULE_DEFAULTS["obstacle_semantics"]
    ordering = rules.obstacle_ordering or RULE_DEFAULTS["obstacle_ordering"]
    if semantics == "threshold":
        return sum(
            1 for o in obstacles if all(agg[a] >= o.requires.get(a, 0) for a in names)
        )
    if semantics == "consumable" and ordering == "unordered":
        best = 0
        m = len(obstacles)
        for k in range(m, 0, -1):
            for comb in combinations(range(m), k):
                if all(
                    agg[a] >= sum(obstacles[i].requires.get(a, 0) for i in comb) for a in names
                ):
                    return k
        return best
    remaining = dict(agg)
    passed = 0
    for i, o in enumerate(obstacles):
        if all(remaining[a] >= o.requires.get(a, 0) for a in names):
            passed += 1
            if semantics == "consumable" or kinds[i] == "consumable":
                for a in names:
                    remaining[a] -= o.requires.get(a, 0)
    return passed


_ = (np, make_build)
