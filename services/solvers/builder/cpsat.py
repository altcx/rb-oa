"""CP-SAT backend, for shapes the DP cannot express cheaply.

TWO GOTCHAS, both load-bearing:

(1) ``enumerate_all_solutions=True`` FORCES SINGLE-THREADED SEARCH.  CP-SAT
    silently ignores / rejects a multi-worker configuration in enumeration mode,
    so we deliberately DO NOT set ``num_search_workers`` anywhere on the
    enumeration path.  Setting it does not speed enumeration up; at best it is a
    no-op, at worst it makes the run non-deterministic.

(2) The only way to parallelise an enumeration is to SPLIT THE SEARCH SPACE
    YOURSELF: fix the first ``k`` part variables to every combination of their
    values (``2**k`` subproblems in the boolean/subset case) and solve those
    subproblems in separate processes, then add up the counts.  That is
    implemented below behind ``workers > 1`` and is opt-in, because process
    startup dominates on small instances.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from ortools.sat.python import cp_model

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.clip import (
    COUNT_PASSED_MIN_PASSED,
    RULE_DEFAULTS,
    ClippedPuzzle,
    make_build,
    required_capacity,
    resolve_rules,
)
from services.solvers.builder.model import Build


def _domains(cp: ClippedPuzzle, rules: BuilderRules) -> list[int]:
    if rules.duplicates_allowed:
        return [max(int(q), 0) for q in cp.qty]
    return [1 if int(q) > 0 else 0 for q in cp.qty]


def _build_model(
    cp: ClippedPuzzle, rules: BuilderRules, fixed: dict[int, int] | None = None
) -> tuple[cp_model.CpModel, list[cp_model.IntVar]]:
    model = cp_model.CpModel()
    ub = _domains(cp, rules)
    x = [model.NewIntVar(0, ub[i], f"x{i}") for i in range(cp.n_parts)]
    for i, value in (fixed or {}).items():
        model.Add(x[i] == value)

    if rules.weight_max is not None:
        model.Add(sum(int(cp.weights[i]) * x[i] for i in range(cp.n_parts)) <= int(rules.weight_max))
    if rules.slot_max is not None:
        model.Add(sum(int(cp.slots[i]) * x[i] for i in range(cp.n_parts)) <= int(rules.slot_max))
    if rules.money_max is not None:
        model.Add(sum(int(cp.costs[i]) * x[i] for i in range(cp.n_parts)) <= int(rules.money_max))

    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    failure_mode = rules.failure_mode or RULE_DEFAULTS["failure_mode"]

    if failure_mode == "all_must_pass":
        # ``required_capacity`` already folded threshold / consumable / mixed into
        # one vector, so this stays a handful of linear constraints.
        needed = required_capacity(cp, rules)
        _add_capability(model, cp, x, aggregation, needed)
        return model, x

    # ---- count_passed ------------------------------------------------------
    # Every auxiliary introduced below is FUNCTIONALLY DETERMINED by ``x``.  That
    # matters: ``enumerate_all_solutions`` counts full assignments, so a free
    # auxiliary would multiply each build by its own number of witnesses.
    if COUNT_PASSED_MIN_PASSED != 1:  # pragma: no cover - guard on a constant
        raise ValueError("CP-SAT count_passed assumes the >=1 threshold")
    agg, agg_ub = _aggregate_vars(model, cp, rules, x, aggregation)
    passed = _passed_bools(model, cp, rules, agg, agg_ub)
    if passed:
        model.Add(sum(passed) >= COUNT_PASSED_MIN_PASSED)
    else:
        # No obstacles at all: nothing can clear the required minimum, so the
        # model must be infeasible rather than accepting every build.
        z = model.NewBoolVar("no_obstacles")
        model.Add(z == 0)
        model.Add(z == 1)
    return model, x


def _selected_bools(
    model: cp_model.CpModel, cp: ClippedPuzzle, x: list[cp_model.IntVar]
) -> list[cp_model.IntVar]:
    """``u[i] <=> x[i] >= 1`` — determined by x, so it is enumeration-safe."""
    u = []
    for i in range(cp.n_parts):
        b = model.NewBoolVar(f"u{i}")
        model.Add(x[i] >= 1).OnlyEnforceIf(b)
        model.Add(x[i] == 0).OnlyEnforceIf(b.Not())
        u.append(b)
    return u


def _aggregate_vars(
    model: cp_model.CpModel,
    cp: ClippedPuzzle,
    rules: BuilderRules,
    x: list[cp_model.IntVar],
    aggregation: str,
) -> tuple[list[cp_model.IntVar], list[int]]:
    """One IntVar per attribute holding the build's aggregate capability."""
    ub = _domains(cp, rules)
    out: list[cp_model.IntVar] = []
    bounds: list[int] = []
    if aggregation == "sum":
        for k in range(cp.n_attrs):
            hi = max(int(sum(int(cp.attrs[i, k]) * ub[i] for i in range(cp.n_parts))), 0)
            v = model.NewIntVar(0, hi, f"agg{k}")
            model.Add(v == sum(int(cp.attrs[i, k]) * x[i] for i in range(cp.n_parts)))
            out.append(v)
            bounds.append(hi)
        return out, bounds

    u = _selected_bools(model, cp, x)
    any_sel = model.NewBoolVar("any_sel")
    if cp.n_parts:
        model.Add(sum(u) >= 1).OnlyEnforceIf(any_sel)
        model.Add(sum(u) == 0).OnlyEnforceIf(any_sel.Not())
    else:
        model.Add(any_sel == 0)

    for k in range(cp.n_attrs):
        cap = int(cp.caps[k])
        v = model.NewIntVar(0, cap, f"agg{k}")
        bounds.append(cap)
        if not cp.n_parts:
            model.Add(v == 0)
            out.append(v)
            continue
        terms = []
        for i in range(cp.n_parts):
            a = int(cp.attrs[i, k])
            t = model.NewIntVar(0, cap, f"t{i}_{k}")
            if aggregation == "min":
                # unselected parts must not drag the minimum: they contribute cap
                model.Add(t == cap - (cap - a) * u[i])
            else:  # max: unselected parts contribute nothing
                model.Add(t == a * u[i])
            terms.append(t)
        raw = model.NewIntVar(0, cap, f"raw{k}")
        if aggregation == "min":
            model.AddMinEquality(raw, terms)
        else:
            model.AddMaxEquality(raw, terms)
        # EMPTY BUILD CONVENTION: no parts aggregates to 0, not to the identity.
        model.Add(v == raw).OnlyEnforceIf(any_sel)
        model.Add(v == 0).OnlyEnforceIf(any_sel.Not())
        out.append(v)
    return out, bounds


def _passed_bools(
    model: cp_model.CpModel,
    cp: ClippedPuzzle,
    rules: BuilderRules,
    agg: list[cp_model.IntVar],
    agg_ub: list[int],
) -> list[cp_model.IntVar]:
    """One determined bool per obstacle: did this build clear it?

    For unordered consumable semantics the *number* cleared is a max-affordable-
    subset problem whose witness is not determined by ``x``; but the validity test
    is only "clears at least one", and a set of size >= 1 exists exactly when some
    single obstacle is affordable.  So the per-obstacle affordability bools below
    answer the question without introducing a free variable.
    """
    semantics = rules.obstacle_semantics or RULE_DEFAULTS["obstacle_semantics"]
    ordering = rules.obstacle_ordering or RULE_DEFAULTS["obstacle_ordering"]
    walk = semantics in ("mixed",) or (semantics == "consumable" and ordering == "ordered")

    remaining: list[cp_model.IntVar | cp_model.LinearExpr] = list(agg)
    passed: list[cp_model.IntVar] = []
    for o in range(cp.n_obstacles):
        b = model.NewBoolVar(f"pass{o}")
        per_attr = []
        for k in range(cp.n_attrs):
            req = int(cp.reqs[o, k])
            if req <= 0:
                continue
            bk = model.NewBoolVar(f"pass{o}_{k}")
            model.Add(remaining[k] >= req).OnlyEnforceIf(bk)
            model.Add(remaining[k] <= req - 1).OnlyEnforceIf(bk.Not())
            per_attr.append(bk)
        if per_attr:
            model.AddBoolAnd(per_attr).OnlyEnforceIf(b)
            model.AddBoolOr([v.Not() for v in per_attr]).OnlyEnforceIf(b.Not())
        else:
            model.Add(b == 1)
        passed.append(b)
        if walk and (semantics == "consumable" or cp.obstacle_kinds[o] == "consumable"):
            nxt: list[cp_model.IntVar] = []
            for k in range(cp.n_attrs):
                req = int(cp.reqs[o, k])
                # b is 1 only when remaining >= req, so the difference stays >= 0
                v = model.NewIntVar(0, max(agg_ub[k], 0), f"rem{o}_{k}")
                model.Add(v == remaining[k] - req * b)
                nxt.append(v)
            remaining = nxt
    return passed


def _add_capability(
    model: cp_model.CpModel,
    cp: ClippedPuzzle,
    x: list[cp_model.IntVar],
    aggregation: str,
    needed: np.ndarray,
) -> None:
    if aggregation == "sum":
        for k in range(cp.n_attrs):
            if int(needed[k]) > 0:
                model.Add(
                    sum(int(cp.attrs[i, k]) * x[i] for i in range(cp.n_parts)) >= int(needed[k])
                )
        return
    if aggregation == "min":
        # min over the selected parts: any part below the bar cannot be used.
        for k in range(cp.n_attrs):
            if int(needed[k]) <= 0:
                continue
            for i in range(cp.n_parts):
                if int(cp.attrs[i, k]) < int(needed[k]):
                    model.Add(x[i] == 0)
        if (needed > 0).any():
            # the empty build aggregates to zero by convention, so it cannot pass
            model.Add(sum(x) >= 1)
        return
    if aggregation == "max":
        for k in range(cp.n_attrs):
            if int(needed[k]) <= 0:
                continue
            carriers = [i for i in range(cp.n_parts) if int(cp.attrs[i, k]) >= int(needed[k])]
            if not carriers:
                model.Add(sum(x) <= -1)  # infeasible
                return
            model.Add(sum(x[i] for i in carriers) >= 1)
        return
    raise ValueError(f"unknown aggregation {aggregation!r}")


class _Collector(cp_model.CpSolverSolutionCallback):
    def __init__(self, x: list[cp_model.IntVar], limit: int) -> None:
        super().__init__()
        self._x = x
        self._limit = limit
        self.count = 0
        self.solutions: list[list[int]] = []

    def on_solution_callback(self) -> None:  # pragma: no cover - exercised indirectly
        self.count += 1
        if len(self.solutions) < self._limit:
            self.solutions.append([int(self.Value(v)) for v in self._x])


def solve_cpsat(
    cp: ClippedPuzzle,
    rules: BuilderRules,
    budget_s: float = 10.0,
    max_solutions: int = 1000,
    workers: int = 1,
) -> tuple[int, list[Build]]:
    """Count valid builds with CP-SAT, returning up to ``max_solutions`` of them.

    ``workers > 1`` opts into the fix-and-split parallel path described at the
    top of this module.
    """
    rules, _ = resolve_rules(rules)
    if workers > 1:
        return _solve_parallel(cp, rules, budget_s, max_solutions, workers)

    model, x = _build_model(cp, rules)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget_s)
    solver.parameters.enumerate_all_solutions = True
    # GOTCHA (1): no num_search_workers here — enumeration is single-threaded.
    collector = _Collector(x, max_solutions)
    solver.Solve(model, collector)
    builds = [make_build(cp, np.asarray(sol, dtype=np.int64), rules) for sol in collector.solutions]
    return collector.count, builds


def _split_prefix(cp: ClippedPuzzle, rules: BuilderRules, workers: int) -> int:
    """How many leading part variables to fix so we get >= ``workers`` chunks."""
    ub = _domains(cp, rules)
    chunks = 1
    k = 0
    while k < cp.n_parts and chunks < workers:
        chunks *= ub[k] + 1
        k += 1
    return k


def _subproblem_count(payload: tuple) -> tuple[int, list[list[int]]]:
    """Worker entry point: solve one fixed-prefix subproblem."""
    cp, rules, fixed, budget_s, max_solutions = payload
    model, x = _build_model(cp, rules, fixed=dict(fixed))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget_s)
    solver.parameters.enumerate_all_solutions = True
    collector = _Collector(x, max_solutions)
    solver.Solve(model, collector)
    return collector.count, collector.solutions


def _solve_parallel(
    cp: ClippedPuzzle,
    rules: BuilderRules,
    budget_s: float,
    max_solutions: int,
    workers: int,
) -> tuple[int, list[Build]]:
    from concurrent.futures import ProcessPoolExecutor

    k = _split_prefix(cp, rules, workers)
    if k == 0:
        return solve_cpsat(cp, rules, budget_s, max_solutions, workers=1)
    ub = _domains(cp, rules)
    assignments = [
        tuple((i, v) for i, v in enumerate(combo))
        for combo in product(*[range(ub[i] + 1) for i in range(k)])
    ]
    payloads = [(cp, rules, a, budget_s, max_solutions) for a in assignments]
    total = 0
    solutions: list[list[int]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, sols in pool.map(_subproblem_count, payloads):
            total += count
            for s in sols:
                if len(solutions) < max_solutions:
                    solutions.append(s)
    builds = [make_build(cp, np.asarray(s, dtype=np.int64), rules) for s in solutions]
    return total, builds


def lightest_build(cp: ClippedPuzzle, rules: BuilderRules, budget_s: float = 5.0) -> Build | None:
    """Minimum-weight valid build (optimisation, not enumeration).

    This path may use every core: the single-threaded restriction in gotcha (1)
    applies to enumeration only.
    """
    rules, _ = resolve_rules(rules)
    try:
        model, x = _build_model(cp, rules)
    except ValueError:
        return None
    model.Minimize(sum(int(cp.weights[i]) * x[i] for i in range(cp.n_parts)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget_s)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    counts = np.array([int(solver.Value(v)) for v in x], dtype=np.int64)
    return make_build(cp, counts, rules)


def max_passed_build(cp: ClippedPuzzle, rules: BuilderRules, budget_s: float = 5.0) -> Build | None:
    """Build clearing the most obstacles.

    Optimisation, not enumeration, so free auxiliaries are harmless here: the
    unordered-consumable case is modelled as "choose the largest set of obstacles
    whose combined spend fits the capability", which is exactly the semantics of
    ``clip.passes_count_matrix`` for that rule combination.
    """
    rules, _ = resolve_rules(rules)
    aggregation = rules.aggregation or RULE_DEFAULTS["aggregation"]
    semantics = rules.obstacle_semantics or RULE_DEFAULTS["obstacle_semantics"]
    ordering = rules.obstacle_ordering or RULE_DEFAULTS["obstacle_ordering"]
    if cp.n_obstacles == 0:
        return None

    model = cp_model.CpModel()
    ub = _domains(cp, rules)
    x = [model.NewIntVar(0, ub[i], f"x{i}") for i in range(cp.n_parts)]
    if rules.weight_max is not None:
        model.Add(sum(int(cp.weights[i]) * x[i] for i in range(cp.n_parts)) <= int(rules.weight_max))
    if rules.slot_max is not None:
        model.Add(sum(int(cp.slots[i]) * x[i] for i in range(cp.n_parts)) <= int(rules.slot_max))
    if rules.money_max is not None:
        model.Add(sum(int(cp.costs[i]) * x[i] for i in range(cp.n_parts)) <= int(rules.money_max))

    agg, agg_ub = _aggregate_vars(model, cp, rules, x, aggregation)
    if semantics == "consumable" and ordering == "unordered":
        chosen = [model.NewBoolVar(f"take{o}") for o in range(cp.n_obstacles)]
        for k in range(cp.n_attrs):
            model.Add(
                sum(int(cp.reqs[o, k]) * chosen[o] for o in range(cp.n_obstacles)) <= agg[k]
            )
        passed = chosen
    else:
        passed = _passed_bools(model, cp, rules, agg, agg_ub)
    if not passed:
        return None
    model.Maximize(sum(passed))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget_s)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    counts = np.array([int(solver.Value(v)) for v in x], dtype=np.int64)
    return make_build(cp, counts, rules)
