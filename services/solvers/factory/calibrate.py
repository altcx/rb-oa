"""Calibration: pin the eight unstated rules against a real recorded run.

The optimizer is only as trustworthy as the simulator, and the simulator has
eight hypotheses baked into it (``RuleFlags``).  ``calibrate`` replays a real
observed money series through the same config and, when the defaults do not
reproduce it, re-simulates under all 2^8 = 256 flag combinations and reports
which of them do.

Nothing here guesses.  If several combinations reproduce the observation, only
the flags they *all* agree on are pinned; the rest stay at their default and
``message`` says so.  ``gate_ok`` is the switch the agent must respect before
running the optimizer at all.
"""

from __future__ import annotations

import itertools

from pydantic import BaseModel, ConfigDict, Field

from services.core.rules.dsl import FACTORY_FLAG_OPTIONS, FACTORY_FLAG_ORDER, RuleFlags
from services.solvers.factory.model import FactoryConfig, FactoryState
from services.solvers.factory.sim import compile_factory, money_series

TOLERANCE = 1e-6
#: how many near-miss combinations to keep alongside every exact match
NEAR_MISS_KEPT = 10


class HourDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int
    simulated: float
    observed: float
    delta: float
    ok: bool


class FlagCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flags: RuleFlags
    max_abs_error: float
    exact: bool
    first_divergence_hour: int | None = None


class CalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched: bool
    per_hour: list[HourDiff] = Field(default_factory=list)
    resolved_flags: RuleFlags | None = None
    candidates: list[FlagCandidate] = Field(default_factory=list)
    #: hours where the 256 hypotheses disagree -- the only hours worth watching
    discriminating_hours: list[int] = Field(default_factory=list)
    message: str = ""


def _align(series: list[float], observed: list[float]) -> tuple[list[float], list[int]]:
    """Line the simulated series up with what the game showed.

    ``money_by_hour`` is ``horizon + 1`` long (index 0 = starting money).  A
    game-reported series of ``horizon`` values is the tail of it; a series of
    ``horizon + 1`` values includes the starting money.  Anything else is
    compared over the overlap, oldest-first.
    """
    if len(observed) == len(series):
        return series, list(range(0, len(series)))
    if len(observed) == len(series) - 1:
        return series[1:], list(range(1, len(series)))
    k = min(len(observed), len(series) - 1)
    return series[1 : 1 + k], list(range(1, 1 + k))


def calibrate(
    state: FactoryState,
    config: FactoryConfig,
    observed: list[float],
    flags: RuleFlags | None = None,
) -> CalibrationResult:
    start_flags = flags or RuleFlags()
    cf = compile_factory(state, start_flags)
    series = money_series(cf, config)
    aligned, hours = _align(series, observed)

    per_hour = [
        HourDiff(
            hour=hours[i],
            simulated=aligned[i],
            observed=float(observed[i]),
            delta=aligned[i] - float(observed[i]),
            ok=abs(aligned[i] - float(observed[i])) <= TOLERANCE,
        )
        for i in range(min(len(aligned), len(observed)))
    ]
    matched = bool(per_hour) and all(d.ok for d in per_hour)
    partial_note = ""
    if len(observed) < len(series) - 1:
        partial_note = (
            f"  Only {len(per_hour)} of the {len(series) - 1} hours were supplied, so "
            f"anything the later hours would have shown is still untested."
        )

    if matched:
        return CalibrationResult(
            matched=True,
            per_hour=per_hour,
            resolved_flags=start_flags,
            candidates=[
                FlagCandidate(flags=start_flags, max_abs_error=0.0, exact=True)
            ],
            discriminating_hours=[],
            message=(
                "The simulator reproduces the recorded run to the dollar under the "
                "current rule hypotheses.  The optimizer is cleared to run."
                + partial_note
            ),
        )

    if not per_hour:
        return CalibrationResult(
            matched=False,
            per_hour=[],
            message="No observations were supplied, so nothing could be checked.",
        )

    # ---- sweep every combination of the eight flags -----------------------
    option_lists = [FACTORY_FLAG_OPTIONS[name] for name in FACTORY_FLAG_ORDER]
    candidates: list[FlagCandidate] = []
    all_series: list[list[float]] = []
    for combo in itertools.product(*option_lists):
        trial = RuleFlags(**dict(zip(FACTORY_FLAG_ORDER, combo, strict=True)))
        trial_series = money_series(compile_factory(state, trial), config)
        t_aligned, _t_hours = _align(trial_series, observed)
        all_series.append(t_aligned)
        worst = 0.0
        first_bad: int | None = None
        for i in range(min(len(t_aligned), len(observed))):
            err = abs(t_aligned[i] - float(observed[i]))
            if err > worst:
                worst = err
            if err > TOLERANCE and first_bad is None:
                first_bad = hours[i]
        candidates.append(
            FlagCandidate(
                flags=trial,
                max_abs_error=worst,
                exact=worst <= TOLERANCE,
                first_divergence_hour=first_bad,
            )
        )

    candidates.sort(key=lambda c: (not c.exact, c.max_abs_error, c.first_divergence_hour or 0))
    exact = [c for c in candidates if c.exact]

    # hours where the hypotheses actually disagree: the only hours a follow-up
    # experiment should bother watching
    discriminating: list[int] = []
    span = min(len(observed), min((len(s) for s in all_series), default=0))
    for i in range(span):
        seen = {round(s[i], 6) for s in all_series}
        if len(seen) > 1:
            discriminating.append(hours[i])

    resolved: RuleFlags | None = None
    if len(exact) == 1:
        resolved = exact[0].flags
        message = (
            "Exactly one rule combination reproduces the recorded run; the flags are "
            "now pinned.  " + _describe(exact[0].flags)
        )
    elif exact:
        agreed: dict[str, object] = {}
        for name in FACTORY_FLAG_ORDER:
            values = {getattr(c.flags, name) for c in exact}
            if len(values) == 1:
                agreed[name] = values.pop()
        undecided = [f for f in FACTORY_FLAG_ORDER if f not in agreed]
        resolved = RuleFlags(**agreed)
        message = (
            f"{len(exact)} rule combinations reproduce the recorded run exactly. "
            f"Pinned the {len(agreed)} flag(s) they all agree on"
            + (f" ({', '.join(f'{k}={agreed[k]}' for k in agreed)})" if agreed else "")
            + (
                f"; {', '.join(undecided)} stay at their default because this run "
                f"cannot tell them apart."
                if undecided
                else "."
            )
            + (
                f"  Hours {discriminating[:6]} are where the surviving hypotheses "
                f"differ -- watch those next."
                if discriminating
                else ""
            )
        )
    else:
        first_bad = next((d.hour for d in per_hour if not d.ok), None)
        best = candidates[0]
        message = (
            f"No combination of the eight rule flags reproduces this run; the closest "
            f"is off by {best.max_abs_error:g} at worst. The default hypotheses first "
            f"diverge at hour {first_bad}.  Either the board was read wrong "
            f"(a price, a cap, an edge) or the game has a rule we have not modelled. "
            f"Do NOT run the optimizer on this state yet."
        )

    message += partial_note
    keep = exact + [c for c in candidates if not c.exact][:NEAR_MISS_KEPT]
    return CalibrationResult(
        matched=False,
        per_hour=per_hour,
        resolved_flags=resolved,
        candidates=keep,
        discriminating_hours=discriminating,
        message=message,
    )


def _describe(flags: RuleFlags) -> str:
    return ", ".join(f"{name}={getattr(flags, name)}" for name in FACTORY_FLAG_ORDER)


def gate_ok(result: CalibrationResult) -> bool:
    """The optimizer must not run until this is True.

    True means we either reproduce the recorded run exactly, or we have narrowed
    the rules to a set that does.  Anything else means the simulator is telling
    a story the game did not.
    """
    return bool(result.matched or result.resolved_flags is not None)
