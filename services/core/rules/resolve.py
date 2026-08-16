"""Solve under unresolved rules.

The move that makes ambiguity cheap: do not pick the most likely reading of a
rule and hope.  Solve *every* reading, then report the part of the answer that
does not depend on which reading is true.  In practice most recommendations are
identical across interpretations, so the user gets an unhedged answer, and the
one place the readings disagree becomes a single fifteen-second experiment
instead of a paragraph of caveats.

Four outputs (spec 5.3):

``consensus_actions``
    Actions recommended by *every* interpretation, compared by ``Action.key()``.
    Report these with no hedging at all.
``divergent_actions``
    Grouped by the flag that *drives* the disagreement.  A flag drives it when
    partitioning the interpretations by that flag's value makes the
    recommendations agree inside every partition and differ between partitions.
``highest_leverage_unknown``
    The flag whose resolution collapses the most divergence, with the
    experiment that resolves it.
``value_of_information``
    Spread in the final objective between the best and worst interpretation:
    what the ambiguity is actually worth in game currency.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Iterable, Sequence

from services.core.rules.compile import experiment_for
from services.core.rules.dsl import (
    BUILDER_FLAG_OPTIONS,
    BUILDER_FLAG_ORDER,
    FACTORY_FLAG_OPTIONS,
    FACTORY_FLAG_ORDER,
    Action,
    BuilderRules,
    DivergentGroup,
    FactoryRuleState,
    InterpretationResult,
    RuleFlags,
    UncertainResult,
    Unknown,
)

#: Hard cap on interpretations.  Past this the wall clock, not the maths, is
#: the binding constraint on a timed run.
MAX_COMBINATIONS = 64

#: Never hand a solver less than this, or it returns nothing useful at all.
MIN_SLICE_S = 0.05


# ---------------------------------------------------------------------------
# Planning: which flags vary, which get pinned
# ---------------------------------------------------------------------------


def _product(sizes: Iterable[int]) -> int:
    n = 1
    for s in sizes:
        n *= max(1, s)
    return n


def plan_interpretations(
    unresolved: Sequence[str],
    options_by_flag: dict[str, Sequence[Any]],
    *,
    cap: int = MAX_COMBINATIONS,
) -> tuple[list[str], dict[str, Any], list[str], bool]:
    """Return ``(varying, pinned, notes, truncated)``.

    ``unresolved`` must already be in leverage order.  When the cartesian
    product exceeds ``cap`` we pin the *leading* (highest-leverage) flag to its
    first option and re-enter, because that is the flag the user should be
    resolving with an experiment anyway.
    """
    varying = list(unresolved)
    pinned: dict[str, Any] = {}
    notes: list[str] = []
    truncated = False
    while varying and _product(len(options_by_flag[f]) for f in varying) > cap:
        lead = varying.pop(0)
        value = options_by_flag[lead][0]
        pinned[lead] = value
        truncated = True
        notes.append(
            f"combination cap ({cap}) exceeded: pinned highest-leverage flag "
            f"{lead}={value!r} instead of enumerating it. Resolve {lead} first "
            f"— run its experiment — then re-run for a complete answer."
        )
    return varying, pinned, notes, truncated


def enumerate_assignments(
    varying: Sequence[str],
    options_by_flag: dict[str, Sequence[Any]],
    pinned: dict[str, Any],
) -> list[dict[str, Any]]:
    if not varying:
        return [dict(pinned)]
    combos = itertools.product(*(options_by_flag[f] for f in varying))
    return [{**pinned, **dict(zip(varying, combo))} for combo in combos]


# ---------------------------------------------------------------------------
# Running interpretations inside a budget
# ---------------------------------------------------------------------------

SolveOne = Callable[[dict[str, Any], float], InterpretationResult]


def run_interpretations(
    assignments: Sequence[dict[str, Any]],
    solve_one: SolveOne,
    budget_s: float,
) -> list[InterpretationResult]:
    """Slice the budget evenly; never crash, never overrun by much."""
    n = max(1, len(assignments))
    per = max(MIN_SLICE_S, budget_s / n)
    deadline = time.monotonic() + max(budget_s, MIN_SLICE_S)
    out: list[InterpretationResult] = []
    for i, assignment in enumerate(assignments):
        remaining = deadline - time.monotonic()
        if remaining <= 0 and i > 0:
            out.append(
                InterpretationResult(
                    assignment=assignment,
                    error="skipped: interpretation budget exhausted",
                )
            )
            continue
        slice_s = max(MIN_SLICE_S, min(per, remaining if remaining > 0 else per))
        try:
            out.append(solve_one(assignment, slice_s))
        except Exception as exc:  # a solver blowing up is one bad reading, not a crash
            out.append(
                InterpretationResult(
                    assignment=assignment, error=f"{type(exc).__name__}: {exc}"
                )
            )
    return out


# ---------------------------------------------------------------------------
# Consensus / divergence
# ---------------------------------------------------------------------------


def _keysets(results: Sequence[InterpretationResult]) -> list[set[tuple[str, str, str]]]:
    return [{a.key() for a in r.actions} for r in results if r.error is None]


def _ok(results: Sequence[InterpretationResult]) -> list[InterpretationResult]:
    return [r for r in results if r.error is None]


def _divergence_count(sets: Sequence[set]) -> int:
    if not sets:
        return 0
    union: set = set().union(*sets)
    inter: set = set(sets[0]).intersection(*sets)
    return len(union - inter)


def consensus_and_index(
    results: Sequence[InterpretationResult],
) -> tuple[list[Action], dict[tuple[str, str, str], Action]]:
    """Actions present in every successful interpretation, plus a key index."""
    ok = _ok(results)
    index: dict[tuple[str, str, str], Action] = {}
    for r in ok:
        for a in r.actions:
            index.setdefault(a.key(), a)
    if not ok:
        return [], index
    sets = _keysets(results)
    common = set(sets[0]).intersection(*sets)
    order = [a.key() for a in ok[0].actions]
    return [index[k] for k in order if k in common], index


def attribute_divergence(
    results: Sequence[InterpretationResult],
    flags: Sequence[str],
    consensus_keys: set,
    index: dict[tuple[str, str, str], Action],
) -> tuple[list[DivergentGroup], list[str]]:
    """Group the disagreement by the flag that *causes* it.

    A flag explains the split when partitioning interpretations by that flag's
    value yields identical recommendation sets inside every partition and
    different sets across partitions.  That is a genuine causal statement about
    this puzzle instance, not a correlation: every other flag is held at every
    one of its values inside each partition.
    """
    ok = _ok(results)
    notes: list[str] = []
    if len(ok) < 2:
        return [], notes
    sets = {id(r): {a.key() for a in r.actions} for r in ok}
    if _divergence_count(list(sets.values())) == 0:
        return [], notes

    groups: list[DivergentGroup] = []
    for flag in flags:
        buckets: dict[str, list[set]] = {}
        for r in ok:
            buckets.setdefault(str(r.assignment.get(flag)), []).append(sets[id(r)])
        if len(buckets) < 2:
            continue
        internally_consistent = all(
            all(s == bucket[0] for s in bucket) for bucket in buckets.values()
        )
        distinct = len({frozenset(b[0]) for b in buckets.values()}) > 1
        if internally_consistent and distinct:
            groups.append(
                DivergentGroup(
                    flag=flag,
                    by_option={
                        value: [index[k] for k in sorted(bucket[0] - consensus_keys)]
                        for value, bucket in buckets.items()
                    },
                )
            )
    if groups:
        return groups, notes

    # No single flag explains it cleanly -- report the best partial explanation
    # rather than nothing, and say so.
    best_flag, best_residual = None, None
    for flag in flags:
        buckets: dict[str, list[set]] = {}
        for r in ok:
            buckets.setdefault(str(r.assignment.get(flag)), []).append(sets[id(r)])
        if len(buckets) < 2:
            continue
        residual = sum(_divergence_count(b) for b in buckets.values())
        if best_residual is None or residual < best_residual:
            best_flag, best_residual = flag, residual
    if best_flag is None:
        return [], notes
    buckets = {}
    for r in ok:
        buckets.setdefault(str(r.assignment.get(best_flag)), []).append(sets[id(r)])
    notes.append(
        f"divergence is not explained by a single flag; grouping by {best_flag}, "
        "which explains the most of it"
    )
    groups.append(
        DivergentGroup(
            flag=best_flag,
            by_option={
                value: [
                    index[k]
                    for k in sorted(set().union(*bucket) - consensus_keys)
                ]
                for value, bucket in buckets.items()
            },
        )
    )
    return groups, notes


def highest_leverage(
    results: Sequence[InterpretationResult],
    flags: Sequence[str],
    options_by_flag: dict[str, Sequence[Any]],
) -> Unknown | None:
    """The flag whose resolution collapses the most divergence."""
    ok = _ok(results)
    if not flags:
        return None
    sets = {id(r): {a.key() for a in r.actions} for r in ok}
    overall = _divergence_count(list(sets.values()))
    best_flag, best_gain = None, -1.0
    for flag in flags:
        buckets: dict[str, list[set]] = {}
        for r in ok:
            buckets.setdefault(str(r.assignment.get(flag)), []).append(sets[id(r)])
        if not buckets:
            continue
        residual = sum(_divergence_count(b) for b in buckets.values()) / len(buckets)
        gain = overall - residual
        if gain > best_gain:
            best_flag, best_gain = flag, gain
    if best_flag is None:
        best_flag = flags[0]
    options = list(options_by_flag[best_flag])
    return Unknown(
        flag=best_flag,
        options=options,
        reason=(
            f"resolving {best_flag} collapses the most divergence "
            f"({max(best_gain, 0.0):.1f} of {overall} disputed actions)"
        ),
        experiment=experiment_for(best_flag, options),
    )


def value_of_information(results: Sequence[InterpretationResult]) -> float:
    values = [r.objective for r in _ok(results) if r.objective is not None]
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def assemble(
    results: list[InterpretationResult],
    varying: Sequence[str],
    options_by_flag: dict[str, Sequence[Any]],
    notes: list[str],
    truncated: bool,
) -> UncertainResult:
    consensus, index = consensus_and_index(results)
    consensus_keys = {a.key() for a in consensus}
    groups, group_notes = attribute_divergence(results, varying, consensus_keys, index)
    notes = list(notes) + group_notes
    failed = [r for r in results if r.error]
    if failed:
        notes.append(
            f"{len(failed)} of {len(results)} interpretations did not complete: "
            + "; ".join(sorted({r.error or "" for r in failed}))[:300]
        )
    voi = value_of_information(results)
    leverage = highest_leverage(results, list(varying), options_by_flag)
    if leverage is not None and not groups and voi == 0.0:
        leverage = None
    return UncertainResult(
        interpretations=results,
        consensus_actions=consensus,
        divergent_actions=groups,
        highest_leverage_unknown=leverage,
        value_of_information=voi,
        combinations_tried=len(results),
        truncated=truncated,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _builder_objective(report: Any, rules: BuilderRules) -> float | None:
    objective = getattr(rules, "objective", None)
    best = getattr(report, "best_build", None)
    if objective == "min_weight" and best is not None:
        return float(best.weight)
    if objective == "max_passed" and best is not None:
        return float(best.obstacles_passed)
    total = getattr(report, "total_valid", None)
    return None if total is None else float(total)


def actions_from_build_report(report: Any) -> list[Action]:
    """Turn a :class:`BuildReport` into concrete 'put this part in' actions."""
    build = getattr(report, "best_build", None)
    if build is None:
        minimal = getattr(report, "minimal_builds", None) or []
        if minimal:
            build = minimal[0].core
    if build is None:
        return []
    return [
        Action(
            target=part_id,
            setting="include",
            value=count,
            reason="part of the reported build",
        )
        for part_id, count in sorted(build.counts.items())
    ]


def solve_all_interpretations(
    rules: BuilderRules,
    puzzle: Any,
    budget_s: float = 5.0,
    *,
    solve_fn: Callable[..., Any] | None = None,
    max_builds: int = 50,
) -> UncertainResult:
    """Solve a builder puzzle under every reading of its unresolved rules.

    ``solve_fn`` exists for tests; production lazily imports
    ``services.solvers.builder.solve.solve_builder``.
    """
    unresolved = [f for f in BUILDER_FLAG_ORDER if getattr(rules, f, None) is None]
    options_by_flag = {f: list(BUILDER_FLAG_OPTIONS[f]) for f in BUILDER_FLAG_ORDER}
    varying, pinned, notes, truncated = plan_interpretations(unresolved, options_by_flag)
    assignments = enumerate_assignments(varying, options_by_flag, pinned)

    solver = solve_fn
    if solver is None:
        try:
            from services.solvers.builder.solve import solve_builder as solver  # type: ignore
        except ImportError as exc:
            return UncertainResult(
                interpretations=[
                    InterpretationResult(assignment=a, error=f"solver unavailable: {exc}")
                    for a in assignments
                ],
                combinations_tried=len(assignments),
                truncated=truncated,
                notes=notes
                + [
                    "services.solvers.builder.solve.solve_builder is not importable; "
                    "no interpretation could be solved"
                ],
            )

    def solve_one(assignment: dict[str, Any], slice_s: float) -> InterpretationResult:
        concrete = rules
        for flag, value in assignment.items():
            concrete = concrete.with_flag(flag, value)
        concrete_puzzle = puzzle.model_copy(update={"rules": concrete})
        report = solver(
            puzzle=concrete_puzzle,
            objective=concrete.objective,
            budget_s=slice_s,
            max_builds=max_builds,
        )
        return InterpretationResult(
            assignment=assignment,
            objective=_builder_objective(report, concrete),
            actions=actions_from_build_report(report),
            detail={
                "total_valid": getattr(report, "total_valid", None),
                "binding_obstacle": getattr(report, "binding_obstacle", None),
                "weight_slack": getattr(report, "weight_slack", None),
                "exact": getattr(report, "exact", None),
                "method": getattr(report, "method", ""),
            },
        )

    results = run_interpretations(assignments, solve_one, budget_s)
    return assemble(results, varying, options_by_flag, notes, truncated)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def solve_all_interpretations_factory(
    state: Any,
    config: Any,
    flag_state: Any = None,
    budget_s: float = 5.0,
    *,
    optimize_fn: Callable[..., Any] | None = None,
    simulate_fn: Callable[..., Any] | None = None,
    diff_fn: Callable[..., Any] | None = None,
) -> UncertainResult:
    """Same machinery, applied to the factory's eight unstated rules.

    ``flag_state`` may be a :class:`FactoryRuleState` (its ``resolved_by`` says
    which flags are already pinned), a plain dict of resolved flag values, or
    ``None`` (nothing resolved).
    """
    resolved: dict[str, Any] = {}
    if isinstance(flag_state, FactoryRuleState):
        base = flag_state.flags
        resolved = {f: getattr(base, f) for f in flag_state.resolved_by}
    elif isinstance(flag_state, RuleFlags):
        resolved = {f: getattr(flag_state, f) for f in FACTORY_FLAG_ORDER}
    elif isinstance(flag_state, dict):
        resolved = {k: v for k, v in flag_state.items() if k in FACTORY_FLAG_OPTIONS}

    unresolved = [f for f in FACTORY_FLAG_ORDER if f not in resolved]
    options_by_flag = {f: list(FACTORY_FLAG_OPTIONS[f]) for f in FACTORY_FLAG_ORDER}
    varying, pinned, notes, truncated = plan_interpretations(unresolved, options_by_flag)
    assignments = enumerate_assignments(varying, options_by_flag, pinned)

    optimizer = optimize_fn
    simulator = simulate_fn
    differ = diff_fn
    import_errors: list[str] = []
    if optimizer is None:
        try:
            from services.solvers.factory.optimize import optimize as optimizer  # type: ignore
        except ImportError as exc:
            import_errors.append(f"optimize unavailable: {exc}")
            optimizer = None
    if differ is None:
        try:
            from services.solvers.factory.optimize import config_diff as differ  # type: ignore
        except ImportError:
            differ = None
    if simulator is None:
        try:
            from services.solvers.factory.sim import simulate as simulator  # type: ignore
        except ImportError as exc:
            import_errors.append(f"simulate unavailable: {exc}")
            simulator = None

    if optimizer is None and simulator is None:
        return UncertainResult(
            interpretations=[
                InterpretationResult(assignment=a, error="; ".join(import_errors))
                for a in assignments
            ],
            combinations_tried=len(assignments),
            truncated=truncated,
            notes=notes + import_errors,
        )

    def solve_one(assignment: dict[str, Any], slice_s: float) -> InterpretationResult:
        flags = RuleFlags(**{**resolved, **assignment})
        if optimizer is not None:
            result = optimizer(state, slice_s, flags, seed_config=config)
            actions = list(getattr(result, "actions", None) or [])
            if not actions and differ is not None and config is not None:
                best = getattr(result, "best", None)
                if best is not None:
                    actions = list(differ(config, best) or [])
            return InterpretationResult(
                assignment=assignment,
                objective=_as_float(getattr(result, "best_value", None)),
                actions=actions,
                detail={
                    "baseline_value": _as_float(getattr(result, "baseline_value", None)),
                    "top_warning": getattr(result, "top_warning", None),
                    "iterations": getattr(result, "iterations", None),
                    "elapsed_s": getattr(result, "elapsed_s", None),
                },
            )
        sim = simulator(state, config, flags)
        return InterpretationResult(
            assignment=assignment,
            objective=_as_float(getattr(sim, "final_money", None)),
            actions=[],
            detail={
                "warnings": len(getattr(sim, "warnings", None) or []),
                "note": "optimizer unavailable; simulated the current config only",
            },
        )

    results = run_interpretations(assignments, solve_one, budget_s)
    return assemble(results, varying, options_by_flag, notes + import_errors, truncated)


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
