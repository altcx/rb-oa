"""Ordered / consumable obstacle semantics.

When passing an obstacle SPENDS the attribute rather than merely testing it, a
build that clears obstacle 1 can still fail obstacle 3, and the order of the
sequence becomes part of the problem.

THE SHORTCUT, AND WHY IT IS EXACT
---------------------------------
For *pure* consumable semantics with ``failure_mode="all_must_pass"``, order does
NOT matter and no sequence simulation is needed.  Let the build's capability for
attribute *a* be ``C_a`` and let obstacle *i* spend ``r_ia >= 0``.  Every
obstacle must be paid, so after any permutation ``pi`` the pool after step *j* is

    C_a - sum_{t<=j} r_{pi(t),a}

The partial sums are non-decreasing in *j* (requirements are non-negative), so
the minimum of that expression over *j* is attained at ``j = m`` and equals
``C_a - sum_i r_ia`` — a quantity independent of ``pi``.  Feasibility is
therefore exactly ``C_a >= sum_i r_ia`` for every *a*, whatever the order.  That
sum is also the LOWER BOUND used to prune candidates before any simulation, and
it is precisely what ``clip.required_capacity`` returns, so the whole consumable
case collapses into one elementwise ``>=`` test and reuses the ordinary counting
machinery.

The same collapse covers ``"mixed"`` under ``all_must_pass``: walking the
sequence, a consumable obstacle *i* needs ``S_i + r_i`` (where ``S_i`` is the
spend already committed) and a threshold obstacle needs the *remaining* pool to
cover ``r_i``, i.e. also ``S_i + r_i``.  So the requirement is
``max_i (S_i + r_i)`` — again a single vector.  See ``clip.required_capacity``.

WHEN A REAL SEQUENCE DP IS UNAVOIDABLE
--------------------------------------
With ``failure_mode="count_passed"`` a build may skip obstacles it cannot afford.
A skipped obstacle spends nothing, so which obstacles you clear depends on the
order in which you meet them, and the total-spend argument no longer applies.
That case is handled by an explicit layered walk over
``(obstacle_index, remaining_resource_vector)`` — vectorised across candidate
builds in ``clip.passes_count_matrix`` — after pruning with the observation that
a build clearing at least one obstacle must cover the *cheapest* obstacle.
"""

from __future__ import annotations

import time

import numpy as np

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.clip import (
    RULE_DEFAULTS,
    ClippedPuzzle,
    budget_mask,
    counts_from_mask,
    make_build,
    passes_count_matrix,
    required_capacity,
    resolve_rules,
)
from services.solvers.builder.count import StateSpaceTooLarge, count_dp
from services.solvers.builder.enumerate import MAX_ENUMERATE_PARTS, subset_arrays
from services.solvers.builder.model import Build, BuildReport


def sequence_prune_bound(cp: ClippedPuzzle, rules: BuilderRules) -> np.ndarray:
    """Lower bound on the capability any *fully successful* build must have."""
    return required_capacity(cp, rules)


def simulate(cp: ClippedPuzzle, rules: BuilderRules, agg: np.ndarray) -> int:
    """Obstacles cleared by a single capability vector."""
    return int(passes_count_matrix(np.asarray(agg, dtype=np.int64)[None, :], cp, rules)[0])


def solve_consumable(
    cp: ClippedPuzzle, rules: BuilderRules, budget_s: float = 10.0
) -> BuildReport:
    """Count and describe valid builds under consumable / mixed semantics."""
    started = time.perf_counter()
    rules, warnings = resolve_rules(rules)
    semantics = rules.obstacle_semantics
    if semantics not in ("consumable", "mixed"):
        warnings.append(
            f"solve_consumable called with obstacle_semantics={semantics!r}; "
            "treating it as a plain threshold reduction"
        )
    if cp.cap_policy != "consumable":
        warnings.append(
            "clipped puzzle was capped with the threshold policy but semantics are "
            "consumable/mixed; re-clip with semantics='consumable' to avoid losing builds"
        )

    failure_mode = rules.failure_mode or RULE_DEFAULTS["failure_mode"]
    needed = sequence_prune_bound(cp, rules)
    report = BuildReport(
        clipped_at=cp.clipped_at(),
        warnings=list(warnings),
        exact=True,
    )
    report.notes.append(
        "consumable/all_must_pass reduces to a single capability threshold "
        f"({dict(zip(cp.attr_names, needed.tolist()))}); order is provably irrelevant"
    )

    if failure_mode == "all_must_pass":
        # Order-free reduction: reuse the ordinary counting DP.
        report.method = "ordered:threshold-reduction+count_dp"
        try:
            report.total_valid = count_dp(cp, rules)
        except StateSpaceTooLarge as exc:
            report.exact = False
            report.warnings.append(str(exc))
            report.total_valid = 0
        best = _best_build(cp, rules, needed, "min_weight")
    else:
        # Genuine sequence walk.  A build's outcome depends only on its
        # capability vector, so the walk can be applied once per lattice cell at
        # the end of the counting DP rather than once per build: the DP state
        # *is* ``(obstacle-independent capability, budgets)`` and the layered
        # ``(obstacle_index, remaining_resource_vector)`` recursion runs over the
        # lattice inside ``clip.passes_count_matrix``.
        report.method = "ordered:sequence-dp"
        report.notes.append(
            "count_passed: a skipped obstacle spends nothing, so the sequence is walked "
            "explicitly (ordered/mixed) or solved as a max affordable obstacle set "
            "(unordered consumable)"
        )
        try:
            report.total_valid = count_dp(cp, rules)
        except StateSpaceTooLarge as exc:
            report.exact = False
            report.warnings.append(str(exc))
            report.total_valid = 0
        best = None
        if cp.n_parts <= MAX_ENUMERATE_PARTS and not rules.duplicates_allowed:
            _, passed, arrays, idx = _enumerate_passed(cp, rules)
            if idx.size:
                order = np.lexsort((arrays["weight"][idx], -passed))
                top = int(idx[order[0]])
                best = make_build(cp, counts_from_mask(cp.n_parts, top), rules)
        else:
            report.warnings.append(
                "best_build unavailable for count_passed with duplicates or large part sets"
            )

    report.best_build = best
    if best is not None and rules.weight_max is not None:
        report.weight_slack = int(rules.weight_max) - int(best.weight)

    elim = _obstacle_elimination(cp, rules)
    report.obstacle_elimination = elim
    if elim:
        binding = max(elim, key=lambda k: elim[k])
        report.binding_obstacle = binding
        report.binding_obstacle_eliminated = elim[binding]
    return _finish(report, started)


def _finish(report: BuildReport, started: float) -> BuildReport:
    report.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return report


def _enumerate_passed(
    cp: ClippedPuzzle, rules: BuilderRules
) -> tuple[int, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Enumerate subsets, prune, then walk the obstacle sequence for each."""
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    arrays = subset_arrays(cp, aggregation)
    inside = budget_mask(cp, rules, arrays["weight"], arrays["slots"], arrays["cost"])
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return 0, np.zeros(0, dtype=np.int64), arrays, idx

    aggs = arrays["agg"][idx].astype(np.int64)
    # PRUNE: to clear anything at all you must cover the cheapest obstacle.
    if cp.n_obstacles:
        cheapest = cp.reqs.min(axis=0)
        alive = (aggs >= cheapest[None, :]).all(axis=1)
        idx = idx[alive]
        aggs = aggs[alive]
    if idx.size == 0:
        return 0, np.zeros(0, dtype=np.int64), arrays, idx

    passed = passes_count_matrix(aggs, cp, rules)
    keep = passed >= 1
    return int(keep.sum()), passed[keep], arrays, idx[keep]


def _best_build(
    cp: ClippedPuzzle, rules: BuilderRules, needed: np.ndarray, objective: str
) -> Build | None:
    """Lightest build meeting ``needed`` (greedy exact for n<=24, else CP-SAT)."""
    from services.solvers.builder.cpsat import lightest_build

    if cp.n_parts <= MAX_ENUMERATE_PARTS and not rules.duplicates_allowed:
        aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
        arrays = subset_arrays(cp, aggregation)
        inside = budget_mask(cp, rules, arrays["weight"], arrays["slots"], arrays["cost"])
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            return None
        aggs = arrays["agg"][idx].astype(np.int64)
        ok = (aggs >= needed[None, :]).all(axis=1)
        idx = idx[ok]
        if idx.size == 0:
            return None
        pick = int(idx[np.argmin(arrays["weight"][idx])])
        return make_build(cp, counts_from_mask(cp.n_parts, pick), rules)
    return lightest_build(cp, rules)


def _obstacle_elimination(cp: ClippedPuzzle, rules: BuilderRules) -> dict[str, int]:
    """Candidates removed by each obstacle, under consumable accounting.

    An obstacle's cost is measured against the *cumulative* spend it sits behind:
    obstacle *i* eliminates every budget-feasible build whose capability fails
    ``S_i + r_i`` (with ``S_i`` the consumable spend committed before it), which
    is the moment that obstacle actually bites.
    """
    if cp.n_obstacles == 0 or cp.n_parts > MAX_ENUMERATE_PARTS:
        return {}
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    arrays = subset_arrays(cp, aggregation)
    inside = budget_mask(cp, rules, arrays["weight"], arrays["slots"], arrays["cost"])
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return {oid: 0 for oid in cp.obstacle_ids}
    aggs = arrays["agg"][idx].astype(np.int64)

    semantics = rules.obstacle_semantics or RULE_DEFAULTS["obstacle_semantics"]
    spent = np.zeros(cp.n_attrs, dtype=np.int64)
    out: dict[str, int] = {}
    for i, oid in enumerate(cp.obstacle_ids):
        gate = spent + cp.reqs[i]
        out[oid] = int((~(aggs >= gate[None, :]).all(axis=1)).sum())
        if semantics == "consumable" or cp.obstacle_kinds[i] == "consumable":
            spent = spent + cp.reqs[i]
    return out
