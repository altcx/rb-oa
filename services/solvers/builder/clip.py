"""Attribute clipping — the optimisation everything else stands on.

An attribute value above the highest requirement that will ever be asked of it
is indistinguishable from that requirement: no obstacle can tell 900 from 9 when
the biggest number anyone asks for is 9.  Clipping every part attribute at that
ceiling collapses the reachable state space from "product of raw attribute sums"
to "product of small ceilings", which is what makes counting DPs and
meet-in-the-middle tractable on large part sets.

WHERE THE CEILING COMES FROM (get this wrong and you silently lose builds):

* ``semantics="threshold"`` — each obstacle checks the build's aggregate
  capability and does not deplete it.  The most that is ever asked of attribute
  *a* in one check is ``max over obstacles of req[a]``, so that is the cap.

* ``semantics="consumable"`` — passing an obstacle SPENDS the attribute, so the
  build may need every point of it.  The cap must therefore be
  ``sum over obstacles of req[a]``.  Using the threshold cap here would clip
  away capacity that later obstacles still need and would report valid builds as
  invalid.

* ``semantics="mixed"`` — some obstacles spend, some only check.  The sum cap is
  a safe (if loose) upper bound for that case too, so mixed uses the sum cap.

Safety proof for the *sum* aggregation (the interesting one).  Let ``c`` be the
cap and ``r <= c`` a requirement.  Write ``clip(v) = min(v, c)``.  For a build
with raw values ``v_1..v_m``:

    sum_i clip(v_i)  >=  r    iff    sum_i v_i  >=  r

(=>) each ``clip(v_i) <= v_i`` so the clipped sum never exceeds the raw sum.
(<=) if some ``v_i >= c`` then the clipped sum is already ``>= c >= r``;
otherwise no value was clipped and the two sums are equal.  The same argument
lets us additionally saturate the *running* aggregate at ``c`` after every
addition, because ``min(min(a+b, c)+d, c) = min(a+b+d, c)`` for non-negative
values — capped addition is associative, which is exactly what the DP needs.

For ``min``/``max`` aggregation clipping is safe because ``clip`` is monotone
non-decreasing and idempotent, so it commutes with both: ``min(clip a, clip b) =
clip(min(a, b))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.model import Build, BuilderPuzzle

# ---------------------------------------------------------------------------
# Rule defaults.  ``None`` means unresolved; every default we fall back on is
# recorded in BuildReport.warnings so the caller can see what we assumed.
# ---------------------------------------------------------------------------

RULE_DEFAULTS: dict[str, Any] = {
    "obstacle_semantics": "threshold",
    "aggregation": "sum",
    "duplicates_allowed": False,
    "obstacle_ordering": "unordered",
    "failure_mode": "all_must_pass",
    "objective": "count_valid",
}

#: With ``failure_mode="count_passed"`` a build is not required to clear
#: everything.  We define "valid" as clearing at least this many obstacles — a
#: build that clears nothing is not an answer to anything.  Documented
#: assumption, surfaced as a warning by the dispatcher.
COUNT_PASSED_MIN_PASSED = 1


def resolve_rules(rules: BuilderRules) -> tuple[BuilderRules, list[str]]:
    """Fill unresolved (``None``) rule flags with the documented defaults.

    Returns the resolved copy plus one warning string per assumed default.
    """
    update: dict[str, Any] = {}
    warnings: list[str] = []
    for flag, default in RULE_DEFAULTS.items():
        if getattr(rules, flag, None) is None:
            update[flag] = default
            warnings.append(f"rule {flag!r} unresolved: assumed default {default!r}")
    if not update:
        return rules, warnings
    return rules.model_copy(update=update), warnings


# ---------------------------------------------------------------------------
# The clipped, numpy-backed puzzle
# ---------------------------------------------------------------------------


@dataclass
class ClippedPuzzle:
    """Numpy view of a :class:`BuilderPuzzle` with attributes already clipped."""

    attr_names: list[str]
    caps: np.ndarray  # (n_attrs,) int64 — per-attribute clip ceiling
    weights: np.ndarray  # (n_parts,) int64
    attrs: np.ndarray  # (n_parts, n_attrs) int64, already clipped at caps
    qty: np.ndarray  # (n_parts,) int64 availability
    slots: np.ndarray  # (n_parts,) int64
    costs: np.ndarray  # (n_parts,) int64
    reqs: np.ndarray  # (n_obstacles, n_attrs) int64
    part_ids: list[str]
    obstacle_ids: list[str]
    #: per-obstacle "threshold" / "consumable"; only consulted when the rule
    #: ``obstacle_semantics`` is "mixed".
    obstacle_kinds: list[str] = field(default_factory=list)
    #: which cap policy produced ``caps`` ("threshold" or "consumable").
    cap_policy: str = "threshold"

    @property
    def n_parts(self) -> int:
        return len(self.part_ids)

    @property
    def n_attrs(self) -> int:
        return len(self.attr_names)

    @property
    def n_obstacles(self) -> int:
        return len(self.obstacle_ids)

    def clipped_at(self) -> dict[str, int]:
        return {name: int(c) for name, c in zip(self.attr_names, self.caps)}

    def restrict(self, keep: Sequence[int]) -> "ClippedPuzzle":
        """A copy holding only the parts at the given indices (same obstacles)."""
        idx = np.asarray(list(keep), dtype=np.int64)
        return ClippedPuzzle(
            attr_names=list(self.attr_names),
            caps=self.caps.copy(),
            weights=self.weights[idx].copy(),
            attrs=self.attrs[idx].copy(),
            qty=self.qty[idx].copy(),
            slots=self.slots[idx].copy(),
            costs=self.costs[idx].copy(),
            reqs=self.reqs.copy(),
            part_ids=[self.part_ids[i] for i in idx],
            obstacle_ids=list(self.obstacle_ids),
            obstacle_kinds=list(self.obstacle_kinds),
            cap_policy=self.cap_policy,
        )


def clip_puzzle(puzzle: BuilderPuzzle, semantics: str | None = None) -> ClippedPuzzle:
    """Build the clipped numpy view.

    ``semantics`` selects the cap policy; when omitted it is taken from
    ``puzzle.rules.obstacle_semantics`` (defaulting to "threshold").
    """
    if semantics is None:
        semantics = puzzle.rules.obstacle_semantics or RULE_DEFAULTS["obstacle_semantics"]
    if semantics not in ("threshold", "consumable", "mixed"):
        raise ValueError(f"unknown obstacle_semantics {semantics!r}")

    attr_names = list(puzzle.attribute_names())
    n_attrs = len(attr_names)

    obstacles = sorted(
        enumerate(puzzle.obstacles), key=lambda pair: (pair[1].order, pair[0])
    )
    obstacle_ids = [o.id for _, o in obstacles]
    reqs = np.zeros((len(obstacles), n_attrs), dtype=np.int64)
    for row, (_, obs) in enumerate(obstacles):
        for k, name in enumerate(attr_names):
            reqs[row, k] = int(obs.requires.get(name, 0))
    reqs = np.maximum(reqs, 0)

    # Per-obstacle kind for "mixed".  The model has no per-obstacle semantics
    # field, so we read an optional override off ``puzzle.provenance`` and
    # default anything unlisted to "threshold".
    override = {}
    prov = puzzle.provenance.get("obstacle_semantics")
    if isinstance(prov, dict):
        override = prov
    obstacle_kinds = [
        str(override.get(oid, "threshold" if semantics == "mixed" else semantics))
        for oid in obstacle_ids
    ]

    if reqs.size == 0:
        caps = np.zeros(n_attrs, dtype=np.int64)
    elif semantics == "threshold":
        # No depletion: the largest single ask is all that can be distinguished.
        caps = reqs.max(axis=0)
    else:
        # Depletion (consumable / mixed): the build may have to pay for every
        # obstacle, so the whole sum can matter.
        caps = reqs.sum(axis=0)
    caps = caps.astype(np.int64)

    n_parts = len(puzzle.parts)
    weights = np.zeros(n_parts, dtype=np.int64)
    qty = np.ones(n_parts, dtype=np.int64)
    slots = np.zeros(n_parts, dtype=np.int64)
    costs = np.zeros(n_parts, dtype=np.int64)
    attrs = np.zeros((n_parts, n_attrs), dtype=np.int64)
    for i, part in enumerate(puzzle.parts):
        weights[i] = int(part.weight)
        qty[i] = max(int(part.qty_available), 0)
        slots[i] = int(part.slots)
        costs[i] = int(part.cost)
        for k, name in enumerate(attr_names):
            attrs[i, k] = max(int(part.attr(name)), 0)

    # THE CLIP.  Everything downstream sees only clipped values.
    attrs = np.minimum(attrs, caps[None, :])

    return ClippedPuzzle(
        attr_names=attr_names,
        caps=caps,
        weights=weights,
        attrs=attrs,
        qty=qty,
        slots=slots,
        costs=costs,
        reqs=reqs,
        part_ids=[p.id for p in puzzle.parts],
        obstacle_ids=obstacle_ids,
        obstacle_kinds=obstacle_kinds,
        cap_policy="threshold" if semantics == "threshold" else "consumable",
    )


# ---------------------------------------------------------------------------
# Aggregation and obstacle evaluation
# ---------------------------------------------------------------------------


def identity_vector(cp: ClippedPuzzle, aggregation: str) -> np.ndarray:
    """Neutral element of the aggregation, used to seed running aggregates.

    ``min`` has no natural zero, so we seed with the caps (the largest value any
    clipped attribute can take).  The *empty* build is then corrected to the
    all-zero vector by every caller — see ``EMPTY BUILD`` below.

    EMPTY BUILD CONVENTION: a build with no parts aggregates to 0 on every
    attribute under every aggregation.  It therefore clears only obstacles that
    require nothing.
    """
    if aggregation == "min":
        return cp.caps.copy()
    return np.zeros(cp.n_attrs, dtype=np.int64)


def combine(a: np.ndarray, b: np.ndarray, aggregation: str, caps: np.ndarray) -> np.ndarray:
    """Fold ``b`` into the running aggregate ``a`` (both already clipped)."""
    if aggregation == "sum":
        return np.minimum(a + b, caps)
    if aggregation == "min":
        return np.minimum(a, b)
    if aggregation == "max":
        return np.maximum(a, b)
    raise ValueError(f"unknown aggregation {aggregation!r}")


def aggregate_counts(cp: ClippedPuzzle, counts: np.ndarray, aggregation: str) -> np.ndarray:
    """Aggregate a whole build given a per-part multiplicity vector."""
    counts = np.asarray(counts, dtype=np.int64)
    if not counts.any():
        return np.zeros(cp.n_attrs, dtype=np.int64)
    if aggregation == "sum":
        return np.minimum((counts[:, None] * cp.attrs).sum(axis=0), cp.caps)
    sel = cp.attrs[counts > 0]
    if aggregation == "min":
        return sel.min(axis=0)
    if aggregation == "max":
        return sel.max(axis=0)
    raise ValueError(f"unknown aggregation {aggregation!r}")


def required_capacity(cp: ClippedPuzzle, rules: BuilderRules) -> np.ndarray:
    """The single capability vector a build must reach to clear *everything*.

    This collapses all three semantics into one elementwise ``>=`` test, which is
    why the counting machinery never has to simulate a sequence for
    ``all_must_pass``:

    * threshold  -> ``max over obstacles`` (each check is independent).
    * consumable -> ``sum over obstacles``.  Spending is monotone and every
      obstacle must be paid, so the binding moment is after the last one; the
      prefix sums are increasing, so the total is the maximum ever needed.  This
      is also *why order does not matter* for pure consumable + all_must_pass.
    * mixed      -> walk the sequence keeping ``S`` = spend accumulated by the
      consumable obstacles seen so far.  A consumable obstacle *i* needs
      ``S_i + req_i``; a threshold obstacle *i* needs the remaining pool to be at
      least ``req_i``, i.e. also ``S_i + req_i``.  Both have the same shape, so
      the answer is ``max_i (S_i + req_i)``.
    """
    if cp.n_obstacles == 0:
        return np.zeros(cp.n_attrs, dtype=np.int64)
    semantics = rules.obstacle_semantics or RULE_DEFAULTS["obstacle_semantics"]
    if semantics == "threshold":
        need = cp.reqs.max(axis=0)
    elif semantics == "consumable":
        need = cp.reqs.sum(axis=0)
    else:
        spent = np.zeros(cp.n_attrs, dtype=np.int64)
        need = np.zeros(cp.n_attrs, dtype=np.int64)
        for i in range(cp.n_obstacles):
            need = np.maximum(need, spent + cp.reqs[i])
            if cp.obstacle_kinds[i] == "consumable":
                spent = spent + cp.reqs[i]
    return np.minimum(need.astype(np.int64), cp.caps)


def passes_count_matrix(aggs: np.ndarray, cp: ClippedPuzzle, rules: BuilderRules) -> np.ndarray:
    """How many obstacles each row of ``aggs`` clears.  ``aggs`` is (N, n_attrs).

    Semantics:

    * threshold — every obstacle is checked against the full capability.
    * consumable / mixed, ordered — walk the sequence; an obstacle is cleared iff
      the *remaining* pool covers it, and a cleared consumable obstacle spends
      its requirement.  A failed obstacle spends nothing (you did not pass it).
    * consumable, unordered — you may attempt the obstacles in any order, so the
      answer is the largest set of obstacles whose total spend fits inside the
      capability.  Solved by (Pareto-pruned) subset search.
    """
    aggs = np.asarray(aggs, dtype=np.int64)
    n = aggs.shape[0]
    if cp.n_obstacles == 0:
        return np.zeros(n, dtype=np.int64)

    semantics = rules.obstacle_semantics or RULE_DEFAULTS["obstacle_semantics"]
    ordering = rules.obstacle_ordering or RULE_DEFAULTS["obstacle_ordering"]

    if semantics == "threshold":
        passed = np.zeros(n, dtype=np.int64)
        for i in range(cp.n_obstacles):
            passed += (aggs >= cp.reqs[i][None, :]).all(axis=1)
        return passed

    if semantics == "consumable" and ordering == "unordered":
        return _max_affordable_subset(aggs, cp.reqs)

    # ordered consumable, or mixed (mixed is always evaluated in obstacle order)
    remaining = aggs.copy()
    passed = np.zeros(n, dtype=np.int64)
    for i in range(cp.n_obstacles):
        req = cp.reqs[i][None, :]
        ok = (remaining >= req).all(axis=1)
        passed += ok
        spends = semantics == "consumable" or cp.obstacle_kinds[i] == "consumable"
        if spends:
            remaining = remaining - ok[:, None] * cp.reqs[i][None, :]
    return passed


def _max_affordable_subset(aggs: np.ndarray, reqs: np.ndarray) -> np.ndarray:
    """Largest number of obstacles whose summed requirement fits in each row."""
    m = reqs.shape[0]
    if m > 14:
        raise ValueError(
            f"unordered consumable count_passed needs subset search over {m} obstacles; "
            "cap is 14"
        )
    n = aggs.shape[0]
    best = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    for k in range(m, 0, -1):
        hit = np.zeros(n, dtype=bool)
        for comb in combinations(range(m), k):
            need = reqs[list(comb)].sum(axis=0)
            hit |= (aggs >= need[None, :]).all(axis=1)
            if hit.all():
                break
        newly = hit & ~done
        best[newly] = k
        done |= newly
        if done.all():
            break
    return best


def valid_mask(
    aggs: np.ndarray, cp: ClippedPuzzle, rules: BuilderRules, needed: np.ndarray | None = None
) -> np.ndarray:
    """Boolean validity of each aggregate row, ignoring budgets."""
    failure_mode = rules.failure_mode or RULE_DEFAULTS["failure_mode"]
    if failure_mode == "all_must_pass":
        if needed is None:
            needed = required_capacity(cp, rules)
        return (np.asarray(aggs) >= needed[None, :]).all(axis=1)
    return passes_count_matrix(aggs, cp, rules) >= COUNT_PASSED_MIN_PASSED


def budget_mask(
    cp: ClippedPuzzle,
    rules: BuilderRules,
    weight: np.ndarray,
    slots: np.ndarray,
    cost: np.ndarray,
) -> np.ndarray:
    mask = np.ones(np.asarray(weight).shape, dtype=bool)
    if rules.weight_max is not None:
        mask &= weight <= rules.weight_max
    if rules.slot_max is not None:
        mask &= slots <= rules.slot_max
    if rules.money_max is not None:
        mask &= cost <= rules.money_max
    return mask


# ---------------------------------------------------------------------------
# Build construction
# ---------------------------------------------------------------------------


def make_build(cp: ClippedPuzzle, counts: np.ndarray, rules: BuilderRules) -> Build:
    """Materialise a :class:`Build` from a per-part multiplicity vector.

    ``Build.attributes`` reports the *clipped* aggregate — that is the value the
    obstacles actually see, and it is what every solver in this package reasons
    about.
    """
    counts = np.asarray(counts, dtype=np.int64)
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    agg = aggregate_counts(cp, counts, aggregation)
    passed = int(passes_count_matrix(agg[None, :], cp, rules)[0])
    return Build(
        counts={cp.part_ids[i]: int(counts[i]) for i in np.flatnonzero(counts)},
        weight=int((counts * cp.weights).sum()),
        slots=int((counts * cp.slots).sum()),
        cost=int((counts * cp.costs).sum()),
        attributes={name: int(v) for name, v in zip(cp.attr_names, agg)},
        obstacles_passed=passed,
    )


def counts_from_mask(n_parts: int, mask: int) -> np.ndarray:
    counts = np.zeros(n_parts, dtype=np.int64)
    i = 0
    while mask:
        if mask & 1:
            counts[i] = 1
        mask >>= 1
        i += 1
    return counts


def aggregate_index_matrix(cp: ClippedPuzzle, idx: np.ndarray, aggregation: str) -> np.ndarray:
    """Aggregate for each row of a (C, k) matrix of part indices (subsets)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError("idx must be 2-D")
    if idx.shape[1] == 0:
        return np.zeros((idx.shape[0], cp.n_attrs), dtype=np.int64)
    sub = cp.attrs[idx]  # (C, k, A)
    if aggregation == "sum":
        return np.minimum(sub.sum(axis=1), cp.caps[None, :])
    if aggregation == "min":
        return sub.min(axis=1)
    if aggregation == "max":
        return sub.max(axis=1)
    raise ValueError(f"unknown aggregation {aggregation!r}")
