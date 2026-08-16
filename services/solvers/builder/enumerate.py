"""Full bitmask enumeration of subsets, in numpy.

NOTE: this module shadows the builtin ``enumerate`` inside the package.  Always
import it absolutely (``from services.solvers.builder.enumerate import ...``);
never ``import enumerate``.

The whole file rests on one trick: subset aggregates are built by a DOUBLING
RECURRENCE rather than by looping over ``2**n`` in Python.  If ``X`` holds the
aggregate of every subset of the first ``i`` parts (in bitmask order), then the
array for ``i+1`` parts is::

    X' = concat(X, fold(X, part_i))

because subset index ``m`` with bit ``i`` clear is the old subset ``m``, and with
bit ``i`` set is the old subset ``m ^ (1<<i)`` plus part ``i``.  Every step is a
single vectorised numpy op, so four million subsets cost a handful of
milliseconds rather than minutes, and we end up with the ACTUAL subsets (their
bitmask index) rather than only a count.
"""

from __future__ import annotations

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
    valid_mask,
)
from services.solvers.builder.model import Build

MAX_ENUMERATE_PARTS = 24


def _dtype_for(limit: int) -> type:
    if limit <= np.iinfo(np.int16).max:
        return np.int16
    if limit <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def subset_arrays(cp: ClippedPuzzle, aggregation: str) -> dict[str, np.ndarray]:
    """Weight / slot / cost / aggregate arrays indexed by subset bitmask.

    Returns arrays of length ``2**n_parts``; index ``m`` describes the subset
    whose bit ``i`` is set iff part ``i`` is used once.
    """
    n = cp.n_parts
    if n > MAX_ENUMERATE_PARTS:
        raise ValueError(
            f"enumerate: {n} parts exceeds the {MAX_ENUMERATE_PARTS}-part ceiling; "
            "use count_dp / meet-in-the-middle / CP-SAT instead"
        )

    def scalar_double(values: np.ndarray) -> np.ndarray:
        total = int(values.sum())
        acc = np.zeros(1, dtype=_dtype_for(max(total, 1)))
        for v in values:
            acc = np.concatenate([acc, acc + v.astype(acc.dtype)])
        return acc

    weight = scalar_double(cp.weights)
    slots = scalar_double(cp.slots)
    cost = scalar_double(cp.costs)

    adt = _dtype_for(int(cp.caps.max()) if cp.n_attrs else 1)
    caps = cp.caps.astype(adt)
    if aggregation == "min":
        # min has no zero; seed with the caps (the identity for min over clipped
        # values) and repair the empty subset afterwards.
        agg = np.tile(caps, (1, 1)).astype(adt)
    else:
        agg = np.zeros((1, cp.n_attrs), dtype=adt)

    for i in range(n):
        row = cp.attrs[i].astype(adt)
        if aggregation == "sum":
            nxt = np.minimum(agg + row[None, :], caps[None, :])
        elif aggregation == "min":
            nxt = np.minimum(agg, row[None, :])
        elif aggregation == "max":
            nxt = np.maximum(agg, row[None, :])
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        agg = np.concatenate([agg, nxt], axis=0)

    # EMPTY BUILD CONVENTION: index 0 is the empty subset and aggregates to 0.
    if cp.n_attrs:
        agg[0, :] = 0

    return {"weight": weight, "slots": slots, "cost": cost, "agg": agg}


def enumerate_valid(
    cp: ClippedPuzzle, rules: BuilderRules
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Bitmask indices of every valid subset, plus the raw subset arrays."""
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    arrays = subset_arrays(cp, aggregation)
    inside = budget_mask(cp, rules, arrays["weight"], arrays["slots"], arrays["cost"])
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return idx, arrays
    needed = required_capacity(cp, rules)
    ok = valid_mask(arrays["agg"][idx].astype(np.int64), cp, rules, needed)
    return idx[ok], arrays


def enumerate_builds(
    cp: ClippedPuzzle, rules: BuilderRules, max_builds: int = 1000
) -> tuple[int, list[Build]]:
    """Exact count of valid subsets plus up to ``max_builds`` concrete builds.

    Raises ``ValueError`` above 24 parts (callers fall back to counting methods)
    and whenever ``duplicates_allowed`` is true — bitmask enumeration cannot
    represent multisets, and silently answering the subset question instead
    would be wrong.
    """
    rules, _ = resolve_rules(rules)
    if rules.duplicates_allowed:
        raise ValueError(
            "enumerate_builds is subset-only; duplicates_allowed=True needs count_dp or CP-SAT"
        )
    if cp.n_parts > MAX_ENUMERATE_PARTS:
        raise ValueError(
            f"enumerate_builds: {cp.n_parts} parts exceeds the "
            f"{MAX_ENUMERATE_PARTS}-part ceiling"
        )

    valid_idx, arrays = enumerate_valid(cp, rules)
    total = int(valid_idx.size)

    builds: list[Build] = []
    if max_builds > 0 and total:
        take = valid_idx[: max_builds]
        aggs = arrays["agg"][take].astype(np.int64)
        passed = passes_count_matrix(aggs, cp, rules)
        for row, mask in enumerate(take.tolist()):
            counts = counts_from_mask(cp.n_parts, int(mask))
            builds.append(
                Build(
                    counts={cp.part_ids[i]: 1 for i in np.flatnonzero(counts)},
                    weight=int(arrays["weight"][mask]),
                    slots=int(arrays["slots"][mask]),
                    cost=int(arrays["cost"][mask]),
                    attributes={
                        name: int(v) for name, v in zip(cp.attr_names, aggs[row])
                    },
                    obstacles_passed=int(passed[row]),
                )
            )
    return total, builds


def best_subset(
    cp: ClippedPuzzle, rules: BuilderRules, objective: str = "min_weight"
) -> Build | None:
    """Lightest valid subset, or the subset clearing the most obstacles."""
    rules, _ = resolve_rules(rules)
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    arrays = subset_arrays(cp, aggregation)
    inside = budget_mask(cp, rules, arrays["weight"], arrays["slots"], arrays["cost"])
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return None
    if objective == "max_passed":
        passed = passes_count_matrix(arrays["agg"][idx].astype(np.int64), cp, rules)
        order = np.lexsort((arrays["weight"][idx], -passed))
        pick = int(idx[order[0]])
    else:
        ok = valid_mask(arrays["agg"][idx].astype(np.int64), cp, rules)
        idx = idx[ok]
        if idx.size == 0:
            return None
        pick = int(idx[np.argmin(arrays["weight"][idx])])
    return make_build(cp, counts_from_mask(cp.n_parts, pick), rules)
