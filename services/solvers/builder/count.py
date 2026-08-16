"""Counting without enumerating: meet-in-the-middle and the general DP.

``count_dp`` is the default for ``objective="count_valid"``.  It is polynomial in
the *clipped* state space and therefore indifferent to the part count; clipping
(see ``clip.py``) is what keeps that state space small.

``count_meet_in_middle`` is the subset-only n<=44 fallback: split the parts in
half, enumerate each half, sort one half by weight, and answer each query from
the other half with a binary search into a prefix-count array.
"""

from __future__ import annotations

import numpy as np

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.clip import (
    RULE_DEFAULTS,
    ClippedPuzzle,
    required_capacity,
    resolve_rules,
    valid_mask,
)
from services.solvers.builder.enumerate import subset_arrays

#: Beyond this many DP cells we refuse and let the dispatcher fall back.
STATE_SPACE_LIMIT = 20_000_000

MAX_MITM_PARTS = 44


class StateSpaceTooLarge(Exception):
    """Raised when the projected DP grid would be unreasonably large."""

    def __init__(self, projected: int, limit: int = STATE_SPACE_LIMIT) -> None:
        self.projected = int(projected)
        self.limit = int(limit)
        super().__init__(
            f"projected DP state space {self.projected:,} exceeds limit {self.limit:,}"
        )


# ---------------------------------------------------------------------------
# General DP
# ---------------------------------------------------------------------------


def _grid_axes(cp: ClippedPuzzle, rules: BuilderRules) -> tuple[list[str], list[int]]:
    """Active DP axes and their sizes.

    A budget dimension is only paid for when the budget exists — with
    ``slot_max=None`` there is simply no slot axis.  Likewise the weight axis
    disappears when there is no weight budget, because weight then constrains
    nothing that counting cares about.
    """
    names: list[str] = []
    sizes: list[int] = []
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    if aggregation == "min":
        # ``min`` needs to distinguish "no parts yet" (identity = caps) from a
        # build that genuinely reached the caps, so carry a nonempty flag.
        names.append("nonempty")
        sizes.append(2)
    if rules.weight_max is not None:
        names.append("weight")
        sizes.append(int(rules.weight_max) + 1)
    if rules.slot_max is not None:
        names.append("slots")
        sizes.append(int(rules.slot_max) + 1)
    if rules.money_max is not None:
        names.append("cost")
        sizes.append(int(rules.money_max) + 1)
    for k, name in enumerate(cp.attr_names):
        names.append(f"attr:{name}")
        sizes.append(int(cp.caps[k]) + 1)
    return names, sizes


def projected_state_space(cp: ClippedPuzzle, rules: BuilderRules) -> int:
    rules, _ = resolve_rules(rules)
    _, sizes = _grid_axes(cp, rules)
    total = 1
    for s in sizes:
        total *= s
    return total


def _axis_map_apply(arr: np.ndarray, axis: int, mapping: np.ndarray) -> np.ndarray:
    """Move mass along ``axis`` according to ``mapping`` (-1 = drop / infeasible)."""
    out = np.zeros_like(arr)
    keep = mapping >= 0
    if not keep.any():
        return out
    src = np.moveaxis(arr, axis, 0)
    dst = np.moveaxis(out, axis, 0)
    np.add.at(dst, mapping[keep], src[keep])
    return out


def count_dp(cp: ClippedPuzzle, rules: BuilderRules) -> int:
    """Number of valid builds, by DP over (part index, budgets, clipped aggregate).

    The DP value is the NUMBER OF SUBSETS (or bounded multisets, when
    ``duplicates_allowed``) that reach each state.  Parts are folded in one at a
    time; the aggregate axes saturate at their clip ceiling, which is exactly the
    property proved safe in ``clip.py``.
    """
    rules, _ = resolve_rules(rules)
    names, sizes = _grid_axes(cp, rules)

    projected = 1
    for s in sizes:
        projected *= s
    if projected > STATE_SPACE_LIMIT:
        raise StateSpaceTooLarge(projected)

    grid = _run_dp(cp, rules, names, sizes)
    return _sum_valid(grid, names, sizes, cp, rules)


def _mapping_for(
    cp: ClippedPuzzle,
    rules: BuilderRules,
    name: str,
    size: int,
    part: int,
    k_copies: int,
    aggregation: str,
) -> np.ndarray | None:
    """Destination index for every source index along one axis, or None = identity."""
    if name == "nonempty":
        return np.array([1, 1], dtype=np.int64)
    if name in ("weight", "slots", "cost"):
        per = {"weight": cp.weights, "slots": cp.slots, "cost": cp.costs}[name][part]
        delta = int(per) * k_copies
        if delta == 0:
            return None
        src = np.arange(size, dtype=np.int64)
        dst = src + delta
        dst[dst >= size] = -1
        return dst
    # attribute axis
    k = cp.attr_names.index(name.split(":", 1)[1])
    value = int(cp.attrs[part, k])
    src = np.arange(size, dtype=np.int64)
    if aggregation == "sum":
        delta = value * k_copies
        if delta == 0:
            return None
        return np.minimum(src + delta, size - 1)
    if aggregation == "min":
        if value >= size - 1:
            return None
        return np.minimum(src, value)
    if aggregation == "max":
        if value <= 0:
            return None
        return np.maximum(src, value)
    raise ValueError(f"unknown aggregation {aggregation!r}")


def _sum_valid(
    grid: np.ndarray,
    names: list[str],
    sizes: list[int],
    cp: ClippedPuzzle,
    rules: BuilderRules,
) -> int:
    """Sum DP mass over every state whose aggregate satisfies the obstacles.

    Budgets need no test here: a state only exists inside the grid if it fit.
    """
    attr_axis0 = len(names) - cp.n_attrs
    if cp.n_attrs == 0:
        ok = bool(valid_mask(np.zeros((1, 0), dtype=np.int64), cp, rules)[0])
        return int(grid.sum()) if ok else 0

    has_flag = bool(names) and names[0] == "nonempty"
    drop = tuple(i for i in range(attr_axis0) if names[i] != "nonempty")
    keep = grid.sum(axis=drop) if drop else grid

    if has_flag:
        empty_part: np.ndarray | None = keep[0]
        nonempty_part = keep[1]
    else:
        empty_part = None
        nonempty_part = keep

    lattice = np.indices(nonempty_part.shape).reshape(cp.n_attrs, -1).T.astype(np.int64)
    ok_rows = valid_mask(lattice, cp, rules)
    total = int(nonempty_part.reshape(-1)[ok_rows].sum())

    if empty_part is not None:
        # Everything on the flag=0 slice is the empty build, which by convention
        # aggregates to zero regardless of the min-identity we seeded with.
        zero_ok = bool(valid_mask(np.zeros((1, cp.n_attrs), dtype=np.int64), cp, rules)[0])
        if zero_ok:
            total += int(empty_part.sum())
    return total


# ---------------------------------------------------------------------------
# Meet in the middle
# ---------------------------------------------------------------------------


#: Largest (weight x attribute-lattice) key space we will collapse a half onto.
COLLAPSE_LIMIT = 5_000_000


def _collapse_half(
    weights: np.ndarray, aggs: np.ndarray, caps: np.ndarray, weight_max: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse one half onto distinct ``(weight, attr vector)`` keys with counts.

    Subsets that agree on weight and clipped aggregate are interchangeable for
    everything that follows, so carrying them separately is pure waste.  Returns
    ``(weights, aggregates, multiplicities)``; the multiplicities are all 1 when
    the key space is too big to collapse onto.
    """
    n_attrs = aggs.shape[1]
    if weight_max is not None:
        keep = weights <= int(weight_max)
        weights = weights[keep]
        aggs = aggs[keep]
    if weights.size == 0:
        return weights, aggs, np.ones(0, dtype=np.int64)

    lattice = 1
    for c in caps:
        lattice *= int(c) + 1
    n_weights = (int(weight_max) + 1) if weight_max is not None else 1
    if n_attrs == 0 or lattice * n_weights > COLLAPSE_LIMIT:
        return weights, aggs, np.ones(weights.shape, dtype=np.int64)

    attr_key = np.zeros(weights.shape, dtype=np.int64)
    stride = 1
    for k in range(n_attrs):
        attr_key += aggs[:, k] * stride
        stride *= int(caps[k]) + 1
    wkey = weights if weight_max is not None else np.zeros_like(weights)
    key = wkey * lattice + attr_key

    counts = np.bincount(key, minlength=lattice * n_weights)
    nz = np.flatnonzero(counts)
    out_w = nz // lattice if weight_max is not None else np.zeros(nz.shape, dtype=np.int64)
    rest = nz % lattice
    out_a = np.zeros((nz.size, n_attrs), dtype=np.int64)
    for k in range(n_attrs):
        size = int(caps[k]) + 1
        out_a[:, k] = rest % size
        rest = rest // size
    return out_w.astype(np.int64), out_a, counts[nz].astype(np.int64)


def count_meet_in_middle(cp: ClippedPuzzle, rules: BuilderRules) -> int:
    """Subset count by splitting the parts in half (n <= 44).

    Each half is enumerated with the doubling recurrence.  The second half is
    sorted by weight; each first-half entry then contributes the number of
    second-half entries that (a) fit in the remaining weight and (b) supply the
    attribute shortfall.  Requirement (b) turns into a *threshold vector*
    ``t = needed - a`` that takes only a handful of distinct values (attributes
    are clipped), so we group the first half by ``t``, build one prefix-count
    array over the sorted second half per distinct ``t``, and binary-search the
    weight limit into it.

    Each half is first COLLAPSED onto ``(weight, clipped attribute vector)`` with
    multiplicities.  Clipping bounds that key space by ``(weight_max+1) * prod
    (caps+1)``, which is orders of magnitude smaller than ``2**(n/2)``, so the
    per-threshold scan of the sorted half stays cheap.
    """
    rules, _ = resolve_rules(rules)
    if rules.duplicates_allowed:
        raise ValueError("count_meet_in_middle is subset-only; use count_dp for multisets")
    if cp.n_parts > MAX_MITM_PARTS:
        raise ValueError(f"count_meet_in_middle: {cp.n_parts} parts exceeds {MAX_MITM_PARTS}")
    if (rules.failure_mode or RULE_DEFAULTS["failure_mode"]) != "all_must_pass":
        raise ValueError(
            "count_meet_in_middle only implements failure_mode='all_must_pass'"
        )
    if rules.slot_max is not None or rules.money_max is not None:
        raise ValueError(
            "count_meet_in_middle only implements the weight budget; "
            "use count_dp when slot_max / money_max are set"
        )

    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    needed = required_capacity(cp, rules)
    half = cp.n_parts // 2
    left = cp.restrict(range(half))
    right = cp.restrict(range(half, cp.n_parts))

    la = subset_arrays(left, aggregation)
    ra = subset_arrays(right, aggregation)
    lw = la["weight"].astype(np.int64)
    rw = ra["weight"].astype(np.int64)
    lagg = la["agg"].astype(np.int64)
    ragg = ra["agg"].astype(np.int64)

    # ``subset_arrays`` zeroes the empty subset, which is right for a whole
    # build but wrong for a *half* under min-aggregation, where the empty half
    # must stay neutral (= caps).  Repair it and fix the genuinely-empty build at
    # the end.
    if aggregation == "min" and cp.n_attrs:
        lagg[0, :] = cp.caps
        ragg[0, :] = cp.caps

    weight_max = rules.weight_max
    lw, lagg, lcount = _collapse_half(lw, lagg, cp.caps, weight_max)
    rw, ragg, rcount = _collapse_half(rw, ragg, cp.caps, weight_max)

    order = np.argsort(rw, kind="stable")
    rw = rw[order]
    ragg = ragg[order]
    rcount = rcount[order]

    if weight_max is None:
        limits = np.full(lw.shape, int(rw.max()) if rw.size else 0, dtype=np.int64)
        keep_left = np.ones(lw.shape, dtype=bool)
    else:
        limits = int(weight_max) - lw
        keep_left = limits >= 0

    # threshold vector each left entry demands of the right half
    if aggregation == "sum":
        thresholds = np.maximum(needed[None, :] - lagg, 0)
    elif aggregation == "min":
        # min(a, b) >= need  <=>  a >= need and b >= need
        keep_left = keep_left & (lagg >= needed[None, :]).all(axis=1)
        thresholds = np.tile(needed, (lagg.shape[0], 1))
    else:  # max
        # max(a, b) >= need  <=>  a >= need (then b is unconstrained) or b >= need
        thresholds = np.where(lagg >= needed[None, :], 0, needed[None, :])

    idx_left = np.flatnonzero(keep_left)
    if idx_left.size == 0:
        return 0

    total = 0
    # group left entries by their threshold vector
    tk = thresholds[idx_left]
    if cp.n_attrs:
        uniq, inverse = np.unique(tk, axis=0, return_inverse=True)
    else:
        uniq = np.zeros((1, 0), dtype=np.int64)
        inverse = np.zeros(idx_left.size, dtype=np.int64)
    inverse = inverse.reshape(-1)

    for g in range(uniq.shape[0]):
        rows = idx_left[inverse == g]
        if rows.size == 0:
            continue
        thr = uniq[g]
        if cp.n_attrs:
            ok_right = (ragg >= thr[None, :]).all(axis=1)
        else:
            ok_right = np.ones(rw.shape, dtype=bool)
        prefix = np.concatenate([[0], np.cumsum(ok_right * rcount)])
        pos = np.searchsorted(rw, limits[rows], side="right")
        total += int((prefix[pos] * lcount[rows]).sum())

    if aggregation == "min" and cp.n_attrs:
        # The all-empty build was counted with aggregate = caps; by convention it
        # aggregates to 0 instead.
        counted_as_valid = bool((cp.caps >= needed).all())
        truly_valid = bool((needed <= 0).all())
        if counted_as_valid and not truly_valid:
            total -= 1
    return total


# ---------------------------------------------------------------------------
# Shared helpers used by the dispatcher
# ---------------------------------------------------------------------------


def count_dp_eliminated(cp: ClippedPuzzle, rules: BuilderRules) -> dict[str, int]:
    """Per-obstacle elimination counts, computed off the same DP grid.

    "Eliminated by obstacle *o*" = budget-feasible builds whose aggregate fails
    obstacle *o*.  The obstacle eliminating the most candidates is the binding
    one.
    """
    rules, _ = resolve_rules(rules)
    names, sizes = _grid_axes(cp, rules)
    projected = 1
    for s in sizes:
        projected *= s
    if projected > STATE_SPACE_LIMIT:
        raise StateSpaceTooLarge(projected)

    # A DP run with no obstacle constraint gives the distribution over aggregates.
    grid = _run_dp(cp, rules, names, sizes)
    attr_axis0 = len(names) - cp.n_attrs
    has_flag = bool(names) and names[0] == "nonempty"
    drop = tuple(i for i in range(attr_axis0) if names[i] != "nonempty")
    keep = grid.sum(axis=drop) if drop else grid
    if has_flag:
        empty_mass = int(keep[0].sum())
        lattice_mass = keep[1]
    else:
        empty_mass = 0
        lattice_mass = keep

    if cp.n_attrs == 0:
        return {oid: 0 for oid in cp.obstacle_ids}

    lattice = np.indices(lattice_mass.shape).reshape(cp.n_attrs, -1).T.astype(np.int64)
    flat = lattice_mass.reshape(-1)
    out: dict[str, int] = {}
    zero = np.zeros((1, cp.n_attrs), dtype=np.int64)
    for i, oid in enumerate(cp.obstacle_ids):
        fails = ~(lattice >= cp.reqs[i][None, :]).all(axis=1)
        total = int(flat[fails].sum())
        if has_flag:
            if not bool((zero[0] >= cp.reqs[i]).all()):
                total += empty_mass
        out[oid] = total
    return out


def _run_dp(
    cp: ClippedPuzzle, rules: BuilderRules, names: list[str], sizes: list[int]
) -> np.ndarray:
    """The raw DP grid: subset/multiset counts per reachable state."""
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    attr_axis0 = len(names) - cp.n_attrs
    grid = np.zeros(sizes, dtype=np.int64)
    start = [0] * len(sizes)
    if aggregation == "min":
        for k in range(cp.n_attrs):
            start[attr_axis0 + k] = int(cp.caps[k])
    grid[tuple(start)] = 1

    for i in range(cp.n_parts):
        max_copies = int(cp.qty[i]) if rules.duplicates_allowed else 1
        if max_copies <= 0:
            continue
        acc = grid
        for k_copies in range(1, max_copies + 1):
            moved = grid
            feasible = True
            for ax, (name, size) in enumerate(zip(names, sizes)):
                mapping = _mapping_for(cp, rules, name, size, i, k_copies, aggregation)
                if mapping is None:
                    continue
                if not (mapping >= 0).any():
                    feasible = False
                    break
                moved = _axis_map_apply(moved, ax, mapping)
            if not feasible:
                break
            acc = acc + moved
        grid = acc
    return grid


def dp_total_feasible(cp: ClippedPuzzle, rules: BuilderRules) -> int:
    """Number of builds inside every budget, ignoring the obstacles."""
    rules, _ = resolve_rules(rules)
    names, sizes = _grid_axes(cp, rules)
    return int(_run_dp(cp, rules, names, sizes).sum())


def count_passed_supported(rules: BuilderRules) -> bool:
    return (rules.failure_mode or RULE_DEFAULTS["failure_mode"]) == "all_must_pass"
