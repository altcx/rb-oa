"""Turning a count into something a human can act on.

Forty thousand valid rows is not an answer.  Three things are:

* the ANTICHAIN of minimal valid builds — the cores you actually have to choose
  between, each one a build that stops being valid if you drop any single part;
* the FREE PARTS — parts that change nothing, so the honest report is "these 7
  cores, each extendable by any subset of these 4 free parts";
* the BINDING OBSTACLE — the one that eliminates the most candidates, which is
  usually the thing the user was really asking about.

MONOTONICITY.  With ``sum`` or ``max`` aggregation every requirement is monotone
in the part set, so every superset of a valid build that still fits the budgets
is valid too, and the antichain plus the budgets describes the whole solution
set.  With ``min`` aggregation that is false — an extra part can drag the
minimum down — so the antichain is still computed correctly (by single-part
removal) but supersets are not guaranteed valid; callers are warned.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.clip import (
    RULE_DEFAULTS,
    ClippedPuzzle,
    aggregate_index_matrix,
    budget_mask,
    counts_from_mask,
    make_build,
    resolve_rules,
    valid_mask,
)
from services.solvers.builder.enumerate import MAX_ENUMERATE_PARTS, subset_arrays
from services.solvers.builder.model import Build, MinimalCore

#: How many candidate combinations the large-n minimal-core search will look at
#: before giving up and reporting truncation.
CORE_SEARCH_BUDGET = 3_000_000


def free_parts(cp: ClippedPuzzle, rules: BuilderRules) -> list[int]:
    """Indices of parts that can be included or excluded with no effect at all.

    "Violates nothing" is interpreted strictly, as AGGREGATION-NEUTRAL, because
    that is exactly the condition that makes the ``2**|F|`` multiplier exact: for
    such a part ``f`` and any build ``B``, ``valid(B) == valid(B u {f})`` *and*
    ``valid(B) == valid(B \\ {f})``.  A merely-monotone zero-weight part (one
    that helps) would break the multiplier, since it could turn an invalid build
    valid.

    Concretely a part is free when

    * it weighs nothing, and
    * it costs no slots when ``slot_max`` is set (with a slot cap, parts that
      occupy slots are NOT free — they compete for the cap), and
    * it costs no money when ``money_max`` is set, and
    * it is neutral for the aggregation: all-zero clipped attributes.

    NO FACTORING UNDER ``min``.  A part at the clip ceiling looks neutral for
    ``min`` — ``min(x, cap) = x`` — but it is not, because of the empty-build
    convention: a build made only of such parts aggregates to ``cap`` while the
    empty build aggregates to ``0``.  Dropping them would change the answer for
    exactly that build, so under ``min`` aggregation nothing is reported free.
    """
    rules, _ = resolve_rules(rules)
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    if aggregation == "min":
        return []
    out: list[int] = []
    for i in range(cp.n_parts):
        if int(cp.weights[i]) != 0:
            continue
        if rules.slot_max is not None and int(cp.slots[i]) != 0:
            continue
        if rules.money_max is not None and int(cp.costs[i]) != 0:
            continue
        if bool((cp.attrs[i] == 0).all()):
            out.append(i)
    return out


def free_multiplier(cp: ClippedPuzzle, rules: BuilderRules) -> int:
    """How many ways the free parts can be added on top of any valid build.

    ``2**|F|`` for subsets; with ``duplicates_allowed`` a free part may be taken
    0..qty times, so the multiplier is ``prod(qty_f + 1)``.
    """
    rules, _ = resolve_rules(rules)
    free = free_parts(cp, rules)
    if not rules.duplicates_allowed:
        return 1 << len(free)
    mult = 1
    for i in free:
        mult *= int(cp.qty[i]) + 1
    return mult


def minimal_builds(
    cp: ClippedPuzzle, rules: BuilderRules, max_cores: int = 50
) -> list[MinimalCore]:
    """The antichain of valid builds: valid, but invalid after any single removal."""
    rules, _ = resolve_rules(rules)
    free = free_parts(cp, rules)
    free_ids = [cp.part_ids[i] for i in free]
    keep = [i for i in range(cp.n_parts) if i not in set(free)]
    core_cp = cp.restrict(keep) if free else cp

    if core_cp.n_parts <= MAX_ENUMERATE_PARTS and not rules.duplicates_allowed:
        masks = _antichain_by_enumeration(core_cp, rules, max_cores)
        cores = [counts_from_mask(core_cp.n_parts, m) for m in masks]
    else:
        cores = _antichain_by_size_search(core_cp, rules, max_cores)

    out: list[MinimalCore] = []
    for counts in cores:
        build = make_build(core_cp, counts, rules)
        out.append(
            MinimalCore(
                core=build,
                freely_addable=list(free_ids),
                extensions_count=1 << len(free_ids),
            )
        )
    return out


def _antichain_by_enumeration(
    cp: ClippedPuzzle, rules: BuilderRules, max_cores: int
) -> list[int]:
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    arrays = subset_arrays(cp, aggregation)
    inside = budget_mask(cp, rules, arrays["weight"], arrays["slots"], arrays["cost"])
    aggs = arrays["agg"].astype(np.int64)
    ok = np.zeros(inside.shape, dtype=bool)
    idx = np.flatnonzero(inside)
    if idx.size:
        ok[idx] = valid_mask(aggs[idx], cp, rules)
    valid_idx = np.flatnonzero(ok)
    if valid_idx.size == 0:
        return []

    minimal = np.ones(valid_idx.shape, dtype=bool)
    for bit in range(cp.n_parts):
        has = ((valid_idx >> bit) & 1).astype(bool)
        if not has.any():
            continue
        reduced = valid_idx ^ (1 << bit)
        still_valid = np.zeros(valid_idx.shape, dtype=bool)
        still_valid[has] = ok[reduced[has]]
        minimal &= ~still_valid
    picked = valid_idx[minimal]
    # smallest (then lightest) cores first: those are what a human will look at
    ranked = sorted(
        picked.tolist(), key=lambda m: (bin(int(m)).count("1"), int(arrays["weight"][m]), int(m))
    )
    return ranked[:max_cores]


def _antichain_by_size_search(
    cp: ClippedPuzzle, rules: BuilderRules, max_cores: int
) -> list[np.ndarray]:
    """Increasing-size search for minimal cores when the part set is too big.

    Generating candidates by increasing size means a candidate is non-minimal
    exactly when it contains an already-recorded core, which is a cheap test.
    """
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    n = cp.n_parts
    found: list[int] = []  # bitmask of each recorded core
    seen = 0
    for k in range(0, n + 1):
        if len(found) >= max_cores or seen >= CORE_SEARCH_BUDGET:
            break
        combos = [()] if k == 0 else list(combinations(range(n), k))
        if not combos:
            continue
        if seen + len(combos) > CORE_SEARCH_BUDGET:
            combos = combos[: max(CORE_SEARCH_BUDGET - seen, 0)]
        seen += len(combos)
        if not combos:
            break
        masks = np.array([sum(1 << i for i in c) for c in combos], dtype=np.int64)
        if found:
            fa = np.array(found, dtype=np.int64)
            contains = ((masks[:, None] & fa[None, :]) == fa[None, :]).any(axis=1)
            live = ~contains
        else:
            live = np.ones(masks.shape, dtype=bool)
        if not live.any():
            continue
        live_positions = np.flatnonzero(live)
        if k == 0:
            idx = np.zeros((live_positions.size, 0), dtype=np.int64)
        else:
            idx = np.array([combos[j] for j in live_positions], dtype=np.int64).reshape(-1, k)
        weight = cp.weights[idx].sum(axis=1) if k else np.zeros(idx.shape[0], dtype=np.int64)
        slots = cp.slots[idx].sum(axis=1) if k else np.zeros(idx.shape[0], dtype=np.int64)
        cost = cp.costs[idx].sum(axis=1) if k else np.zeros(idx.shape[0], dtype=np.int64)
        inside = budget_mask(cp, rules, weight, slots, cost)
        aggs = aggregate_index_matrix(cp, idx, aggregation)
        ok = inside & valid_mask(aggs, cp, rules)
        for j in np.flatnonzero(ok):
            found.append(int(masks[live_positions[j]]))
            if len(found) >= max_cores:
                break
    return [counts_from_mask(n, m) for m in found]


def binding_obstacle(cp: ClippedPuzzle, rules: BuilderRules) -> tuple[str | None, dict[str, int]]:
    """The obstacle that eliminates the most candidates, plus every count.

    "Eliminated" = budget-feasible builds whose capability fails that obstacle.
    Candidates are counted independently per obstacle, so the numbers overlap;
    that is the point — the largest number names the real bottleneck.
    """
    rules, _ = resolve_rules(rules)
    if cp.n_obstacles == 0:
        return None, {}
    counts = _elimination_counts(cp, rules)
    if not counts:
        return None, {}
    binding = max(counts, key=lambda k: (counts[k], k))
    return binding, counts


def _elimination_counts(cp: ClippedPuzzle, rules: BuilderRules) -> dict[str, int]:
    from services.solvers.builder.count import StateSpaceTooLarge, count_dp_eliminated

    if cp.n_parts <= MAX_ENUMERATE_PARTS and not rules.duplicates_allowed:
        aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
        arrays = subset_arrays(cp, aggregation)
        inside = budget_mask(cp, rules, arrays["weight"], arrays["slots"], arrays["cost"])
        idx = np.flatnonzero(inside)
        aggs = arrays["agg"][idx].astype(np.int64)
        out: dict[str, int] = {}
        for i, oid in enumerate(cp.obstacle_ids):
            out[oid] = int((~(aggs >= cp.reqs[i][None, :]).all(axis=1)).sum())
        return out
    try:
        return count_dp_eliminated(cp, rules)
    except StateSpaceTooLarge:
        return {}


def lightest_valid(cp: ClippedPuzzle, rules: BuilderRules) -> Build | None:
    """The lightest valid build, or ``None`` when nothing is valid."""
    from services.solvers.builder.cpsat import lightest_build
    from services.solvers.builder.enumerate import best_subset

    rules, _ = resolve_rules(rules)
    if cp.n_parts <= MAX_ENUMERATE_PARTS and not rules.duplicates_allowed:
        return best_subset(cp, rules, objective="min_weight")
    return lightest_build(cp, rules)


def weight_slack(cp: ClippedPuzzle, rules: BuilderRules) -> tuple[int | None, Build | None]:
    """``weight_max`` minus the weight of the lightest valid build."""
    rules, _ = resolve_rules(rules)
    best = lightest_valid(cp, rules)
    if best is None or rules.weight_max is None:
        return None, best
    return int(rules.weight_max) - int(best.weight), best
