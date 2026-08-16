"""Factory optimizer.  Spec section 7.6.

Four layers, in the order they run:

1. BOUND      -- ``bounds.upper_bound``: what is even possible, in milliseconds.
2. SEED       -- back-solve from the seller at the LP throughput and propagate
                 the requirement upstream.  The gap between the bound and this
                 seed is the ramp-up loss, and it is reported.
3. LOCAL      -- simulated annealing over recipe / output / mod moves, evaluated
                 with ``sim.evaluate_fast``.  Anytime: ``on_improve`` fires on
                 every new best and a usable result exists within 200 ms.
4. ENDGAME    -- anything that cannot reach the seller before the horizon is
                 wasted money, so the last ``depth + 2`` hours get their own
                 per-hour output schedule via ``MachineConfig.hourly_overrides``.

THREE PRIORS, computed explicitly rather than assumed:

* Ramp-up dominates at short horizons.  ``one_hour_mod_value`` reports, per
  2-hour machine, the money delta from installing the 1-Hour mod -- measured by
  simulating with and without it, not guessed.
* Priority is exploitable.  Deliberately starving a low-margin branch can raise
  total profit, so "set this machine's output to 0" is an explicit move in the
  neighbourhood and the zeroed machines that survived into the best config are
  reported in ``deliberately_idle``.
* Scoring is a maximum over every tested factory, never the last one.  There is
  no downside to testing aggressively, which is why the archive keeps the top
  20 distinct configs rather than only the winner.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from services.core.rules.dsl import Action, RuleFlags
from services.solvers.factory.bounds import UpperBound, upper_bound
from services.solvers.factory.model import (
    FactoryConfig,
    FactoryState,
    ModKind,
    SimResult,
    WarningKind,
)
from services.solvers.factory.sim import (
    KIND_SELLER,
    MOD_BIT,
    MOD_ORDER,
    CompiledFactory,
    compile_factory,
    config_to_vector,
    evaluate_fast_hourly,
    evaluate_fast_skips,
    simulate_compiled,
    vector_to_config,
)


class ArchiveEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: FactoryConfig
    value: float
    actions: list[Action] = Field(default_factory=list)
    warnings_summary: str = ""


class OptimizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best: FactoryConfig
    best_value: float
    baseline_value: float
    bound: float
    #: top 20 DISTINCT configs, best first
    archive: list[ArchiveEntry] = Field(default_factory=list)
    #: diff from the CURRENT board configuration, ordered, concrete
    actions: list[Action] = Field(default_factory=list)
    top_warning: str | None = None
    elapsed_s: float = 0.0
    iterations: int = 0
    converged: bool = False
    #: per-hour drain schedule (layer 4)
    endgame: list[Action] = Field(default_factory=list)

    # -- explicitly exposed priors and diagnostics -------------------------
    notes: list[str] = Field(default_factory=list)
    #: machine id -> money delta from installing the 1-Hour mod on it
    one_hour_mod_value: dict[str, float] = Field(default_factory=dict)
    #: machines the optimizer deliberately switched off to free up upstream items
    deliberately_idle: list[str] = Field(default_factory=list)
    seed_value: float = 0.0
    ramp_up_loss: float = 0.0
    bound_detail: UpperBound | None = None


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


def validation_errors(state: FactoryState, config: FactoryConfig) -> list[str]:
    """Every way this config would be impossible to dial in on the real board."""
    problems: list[str] = []
    by_id = {m.id: m for m in state.machines}
    for mid, mc in config.machines.items():
        m = by_id.get(mid)
        if m is None:
            problems.append(f"{mid}: no such machine")
            continue
        if mc.machine_id != mid:
            problems.append(f"{mid}: machine_id mismatch ({mc.machine_id})")
        rids = [r.id for r in m.recipes]
        if m.recipes and mc.recipe_id not in rids:
            problems.append(f"{mid}: recipe {mc.recipe_id!r} not in {rids}")
        allowed = set(m.available_mods) | set(m.installed_mods)
        for mod in mc.mods:
            if mod not in allowed:
                problems.append(f"{mid}: mod {mod.value} is not available")
        eff_max = m.output_max * (2 if ModKind.DOUBLE_OUTPUT_MAX in mc.mods else 1)
        if not (0 <= mc.output_setting <= eff_max):
            problems.append(
                f"{mid}: output {mc.output_setting} outside 0..{eff_max}"
            )
        for hour, val in mc.hourly_overrides.items():
            if not (1 <= hour <= state.horizon_hours):
                problems.append(f"{mid}: override for hour {hour} outside the horizon")
            if not (0 <= val <= eff_max):
                problems.append(f"{mid}: override {val} at h{hour} outside 0..{eff_max}")
    for m in state.machines:
        if m.id not in config.machines:
            problems.append(f"{m.id}: missing from the config")
    return problems


def validate_config(state: FactoryState, config: FactoryConfig) -> bool:
    """True when every knob in ``config`` is one the game would actually accept."""
    return not validation_errors(state, config)


# ---------------------------------------------------------------------------
# diffing
# ---------------------------------------------------------------------------


def config_diff(a: FactoryConfig, b: FactoryConfig) -> list[Action]:
    """Concrete UI actions turning config ``a`` into config ``b``."""
    actions: list[Action] = []
    for mid, mb in b.machines.items():
        ma = a.machines.get(mid)
        if ma is None:
            continue
        if ma.recipe_id != mb.recipe_id:
            actions.append(
                Action(
                    target=mid,
                    setting="recipe",
                    value=mb.recipe_id,
                    reason=f"switch {mid} from {ma.recipe_id} to {mb.recipe_id}",
                )
            )
        old_mods = set(ma.mods)
        new_mods = set(mb.mods)
        for mod in MOD_ORDER:
            if mod in new_mods and mod not in old_mods:
                actions.append(
                    Action(
                        target=mid,
                        setting="mod",
                        value=mod.value,
                        reason=f"install {mod.value} on {mid}",
                    )
                )
            elif mod in old_mods and mod not in new_mods:
                actions.append(
                    Action(
                        target=mid,
                        setting="mod",
                        value=f"-{mod.value}",
                        reason=f"remove {mod.value} from {mid}",
                    )
                )
        if ma.output_setting != mb.output_setting:
            why = (
                f"set {mid} output to {mb.output_setting}"
                if mb.output_setting
                else f"switch {mid} off: its branch is worth less than the items it eats"
            )
            actions.append(
                Action(target=mid, setting="output", value=mb.output_setting, reason=why)
            )
    return actions


def _order_actions(cf: CompiledFactory, actions: list[Action]) -> list[Action]:
    """Recipes and mods first, then output dials, furthest-from-the-seller first
    so the chain is already fed by the time the downstream machines spin up."""
    rank = {"recipe": 0, "mod": 1, "output": 2}
    return sorted(
        actions,
        key=lambda a: (
            rank.get(a.setting, 3),
            -cf.hops[cf.index[a.target]] if a.target in cf.index else 0,
            a.target,
        ),
    )


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


def _clamp_vector(cf: CompiledFactory, vec: np.ndarray) -> np.ndarray:
    """Force a vector back inside the game's allowed range."""
    n = cf.n
    for i in range(n):
        nr = len(cf.recipes[i])
        if nr == 0:
            vec[i] = -1
            vec[n + i] = 0
            vec[2 * n + i] = 0
            continue
        vec[i] = int(vec[i]) % nr
        vec[2 * n + i] = int(vec[2 * n + i]) & cf.available_mask[i]
        eff = cf.effective_output_max(i, int(vec[2 * n + i]))
        v = int(vec[n + i])
        vec[n + i] = 0 if v < 0 else (eff if v > eff else v)
    return vec


def _seed_from_bound(cf: CompiledFactory, ub: UpperBound) -> np.ndarray:
    """Back-solve: start at the seller's LP throughput and push the requirement
    upstream, rounding each machine's output setting up to the next whole unit."""
    n = cf.n
    vec = np.zeros(3 * n, dtype=np.int64)
    for i, mid in enumerate(cf.machine_ids):
        rid = ub.recipe_choice.get(mid)
        ids = cf.recipe_ids[i]
        vec[i] = ids.index(rid) if rid in ids else (0 if ids else -1)
        vec[2 * n + i] = cf.installed_mask[i]

    need_out = [0.0] * n  # units per hour this machine must produce
    for i, mid in enumerate(cf.machine_ids):
        if cf.kinds[i] == 2:  # seller: pinned by the LP
            need_out[i] = max(0.0, ub.throughput.get(mid, 0.0))

    for i in sorted(range(n), key=lambda k: cf.hops[k]):
        ri = int(vec[i])
        if ri < 0:
            continue
        r = cf.recipes[i][ri]
        mask = int(vec[2 * n + i])
        hours = cf.effective_hours(i, mask)
        eff_max = cf.effective_output_max(i, mask)
        if cf.kinds[i] == 0:  # supplier: driven entirely by what is asked of it
            want = need_out[i]
        else:
            want = need_out[i]
        if want <= 0:
            setting = 0
        else:
            setting = math.ceil(want * hours / max(1, r.out_qty))
            setting = min(setting, eff_max)
        vec[n + i] = setting
        if setting <= 0:
            continue
        for k in range(len(r.in_items)):
            item = r.in_items[k]
            per_hour = r.in_qtys[k] * setting / hours
            producers = [
                u
                for u in cf.upstream[i]
                if any(rr.out_item == item for rr in cf.recipes[u])
            ]
            if not producers:
                continue
            share = per_hour / len(producers)
            for u in producers:
                need_out[u] += share
    return _clamp_vector(cf, vec)


def _seed_all_max(cf: CompiledFactory, mods: bool) -> np.ndarray:
    n = cf.n
    vec = np.zeros(3 * n, dtype=np.int64)
    for i in range(n):
        vec[i] = 0 if cf.recipes[i] else -1
        vec[2 * n + i] = cf.available_mask[i] if mods else cf.installed_mask[i]
        vec[n + i] = cf.effective_output_max(i, int(vec[2 * n + i]))
    return _clamp_vector(cf, vec)


# ---------------------------------------------------------------------------
# warnings
# ---------------------------------------------------------------------------

_WARN_RANK = {
    WarningKind.NO_MONEY: 0,
    WarningKind.SKIPPED: 1,
    WarningKind.OVERFLOW: 2,
    WarningKind.IDLE_FULL: 3,
    WarningKind.STRANDED: 4,
    WarningKind.UNREACHABLE: 5,
}


def _warning_lines(res: SimResult) -> list[str]:
    """One human sentence per (kind, machine), most costly first."""
    groups: dict[tuple, list] = {}
    for w in res.warnings:
        groups.setdefault((w.kind, w.machine_id), []).append(w)
    lines: list[tuple[tuple, str]] = []
    for (kind, mid), ws in groups.items():
        amount = sum(w.amount for w in ws)
        detail = ws[0].detail
        if kind == WarningKind.SKIPPED:
            text = f"{mid} produced nothing in {len(ws)} hour(s): {detail}"
        elif kind == WarningKind.OVERFLOW:
            text = f"{mid} destroyed {amount:g} item(s) by overflowing its storage"
        elif kind == WarningKind.IDLE_FULL:
            text = f"{mid} idled with a full buffer for {len(ws)} hour(s)"
        elif kind == WarningKind.NO_MONEY:
            text = f"{mid} could not afford to run in {len(ws)} hour(s): {detail}"
        elif kind == WarningKind.STRANDED:
            text = f"{mid} still holds {amount:g} item(s) at the horizon, worth zero"
        else:
            text = f"{mid}: {kind.value} {detail}"
        lines.append(((_WARN_RANK.get(kind, 9), -amount, -len(ws)), text))
    lines.sort(key=lambda p: p[0])
    return [t for _k, t in lines]


def _warning_summary(res: SimResult) -> str:
    counts: dict[str, int] = {}
    for w in res.warnings:
        counts[w.kind.value] = counts.get(w.kind.value, 0) + 1
    return ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "clean"


# ---------------------------------------------------------------------------
# the optimizer
# ---------------------------------------------------------------------------


def optimize(
    state: FactoryState,
    seconds: float = 10.0,
    flags: RuleFlags = RuleFlags(),
    seed_config: FactoryConfig | None = None,
    on_improve: Callable[[OptimizeResult], None] | None = None,
) -> OptimizeResult:
    t_start = time.perf_counter()
    deadline = t_start + max(0.01, seconds)
    cf = compile_factory(state, flags)
    n = cf.n
    rng = random.Random(0xF00D)
    notes: list[str] = []

    # ---- layer 1: bound ---------------------------------------------------
    ub = upper_bound(state, flags)

    # ---- layer 2: seeds ---------------------------------------------------
    current = FactoryConfig.from_state(state)
    base_vec = _clamp_vector(cf, config_to_vector(cf, current))
    baseline_value, baseline_skips = evaluate_fast_skips(cf, base_vec)

    candidates: list[np.ndarray] = [base_vec]
    seed_vec = _seed_from_bound(cf, ub)
    candidates.append(seed_vec)
    candidates.append(_seed_all_max(cf, mods=False))
    candidates.append(_seed_all_max(cf, mods=True))
    if seed_config is not None:
        candidates.append(_clamp_vector(cf, config_to_vector(cf, seed_config)))

    seed_value, _sk = evaluate_fast_skips(cf, seed_vec)

    best_vec = base_vec.copy()
    best_value = baseline_value
    best_skips = baseline_skips
    archive: dict[tuple, tuple[float, np.ndarray]] = {}

    def remember(vec: np.ndarray, value: float) -> None:
        # Knobs on a switched-off machine change nothing, so normalise them back
        # to what the board already shows: the archive is a to-do list for a
        # human and must not ask for pointless clicks.
        canon = vec.copy()
        for i in range(n):
            if int(canon[n + i]) == 0:
                canon[i] = base_vec[i]
                canon[2 * n + i] = base_vec[2 * n + i]
        key = tuple(int(x) for x in canon)
        prev = archive.get(key)
        if prev is None or value > prev[0]:
            archive[key] = (value, canon)
        if len(archive) > 400:  # keep the dict small; trimmed to 20 at the end
            for k in sorted(archive, key=lambda k: archive[k][0])[:200]:
                archive.pop(k, None)

    for vec in candidates:
        val, sk = evaluate_fast_skips(cf, vec)
        remember(vec, val)
        if val > best_value:
            best_vec, best_value, best_skips = vec.copy(), val, sk

    def emit() -> None:
        if on_improve is None:
            return
        try:
            on_improve(
                OptimizeResult(
                    best=vector_to_config(cf, best_vec),
                    best_value=best_value,
                    baseline_value=baseline_value,
                    bound=ub.ceiling,
                    actions=_order_actions(
                        cf, config_diff(current, vector_to_config(cf, best_vec))
                    ),
                    elapsed_s=time.perf_counter() - t_start,
                    iterations=0,
                    converged=False,
                    seed_value=seed_value,
                )
            )
        except Exception:  # pragma: no cover - a callback must never break solving
            pass

    emit()

    # ---- layer 3: iterated local search / annealing ------------------------
    movable = [i for i in range(n) if cf.recipes[i]]
    iterations = 0
    last_improve_iter = 0
    cur_vec = best_vec.copy()
    cur_value = best_value
    cur_skips = best_skips
    scale = max(1.0, abs(best_value) * 0.02, abs(ub.per_hour_profit))
    temp0 = scale
    stall_limit = 400

    while movable and time.perf_counter() < deadline:
        for _ in range(64):
            iterations += 1
            cand = _neighbour(cf, cur_vec, rng, movable)
            value, skips = evaluate_fast_skips(cf, cand)

            # CLIFF GUARD: a move that newly starves a machine is refused unless
            # the money actually went up.  Allocation is all-or-nothing, so a
            # "harmless looking" +1 can zero a whole branch.
            new_starve = skips & ~cur_skips
            if new_starve and value <= cur_value:
                continue

            remember(cand, value)
            delta = value - cur_value
            frac = min(1.0, max(0.0, (time.perf_counter() - t_start) / max(1e-9, seconds)))
            temp = temp0 * (0.001 ** frac)  # geometric cooling
            if delta >= 0 or (temp > 0 and rng.random() < math.exp(delta / temp)):
                cur_vec, cur_value, cur_skips = cand, value, skips
            if value > best_value:
                best_vec, best_value, best_skips = cand.copy(), value, skips
                last_improve_iter = iterations
                emit()
        if iterations - last_improve_iter > stall_limit:
            # restart: kick the incumbent hard, keep the best on the shelf
            cur_vec = best_vec.copy()
            for _ in range(1 + rng.randrange(3)):
                cur_vec = _neighbour(cf, cur_vec, rng, movable)
            cur_value, cur_skips = evaluate_fast_skips(cf, cur_vec)
            last_improve_iter = iterations

    converged = bool(movable) and (iterations - last_improve_iter > stall_limit)

    # ---- prior: what is the 1-Hour mod actually worth? ---------------------
    one_hour_value: dict[str, float] = {}
    for i in range(n):
        if cf.production_hours[i] < 2:
            continue
        if not (cf.available_mask[i] & MOD_BIT[ModKind.ONE_HOUR_PRODUCTION]):
            continue
        with_mod = best_vec.copy()
        with_mod[2 * n + i] |= MOD_BIT[ModKind.ONE_HOUR_PRODUCTION]
        without = best_vec.copy()
        without[2 * n + i] &= ~MOD_BIT[ModKind.ONE_HOUR_PRODUCTION]
        v_with = evaluate_fast_skips(cf, _clamp_vector(cf, with_mod))[0]
        v_without = evaluate_fast_skips(cf, _clamp_vector(cf, without))[0]
        one_hour_value[cf.machine_ids[i]] = round(v_with - v_without, 6)
        if v_with > best_value:
            best_vec, best_value = _clamp_vector(cf, with_mod), v_with
            remember(best_vec, best_value)

    # ---- layer 4: endgame drain schedule -----------------------------------
    best_config = vector_to_config(cf, best_vec)
    endgame_actions, endgame_config, endgame_value = _endgame(cf, best_vec, best_value)
    if endgame_value > best_value + 1e-9:
        best_config = endgame_config
        best_value = endgame_value

    # ---- reporting ---------------------------------------------------------
    best_res = simulate_compiled(cf, best_config, collect_logs=True)
    lines = _warning_lines(best_res)

    ranked = sorted(archive.items(), key=lambda kv: -kv[1][0])[:20]
    entries: list[ArchiveEntry] = []
    for _key, (value, vec) in ranked:
        cfg = vector_to_config(cf, vec)
        res = simulate_compiled(cf, cfg, collect_logs=True)
        entries.append(
            ArchiveEntry(
                config=cfg,
                value=value,
                actions=_order_actions(cf, config_diff(current, cfg)),
                warnings_summary=_warning_summary(res),
            )
        )

    idle = [
        cf.machine_ids[i]
        for i in range(n)
        if cf.recipes[i] and int(best_vec[n + i]) == 0 and cf.kinds[i] != KIND_SELLER
    ]

    notes.extend(ub.notes)
    notes.append(
        f"Ramp-up loss: the steady-state LP says {ub.ceiling:g}; back-solving that "
        f"same throughput into real dials and simulating it reaches {seed_value:g}. "
        f"The {ub.ceiling - seed_value:g} gap is what the LP refuses to see -- the "
        f"hours the chain spends filling itself, plus integer output settings and "
        f"the all-or-nothing allocation rule.  Search closed it to "
        f"{ub.ceiling - best_value:g}."
    )
    if one_hour_value:
        best_mod = max(one_hour_value, key=lambda k: one_hour_value[k])
        notes.append(
            "Prior 1 (ramp-up dominates): the 1-Hour mod is worth "
            + ", ".join(f"{k} {v:+g}" for k, v in sorted(one_hour_value.items()))
            + f" -- {best_mod} is the one to buy first."
        )
    else:
        notes.append(
            "Prior 1 (ramp-up dominates): no 2-hour machine can take the 1-Hour "
            "mod here, so there is no ramp-up shortcut to buy."
        )
    notes.append(
        "Prior 2 (priority is exploitable): 'set output to 0' is an explicit move, "
        + (
            f"and the winning plan deliberately idles {', '.join(idle)}."
            if idle
            else "though on this board no machine was worth starving."
        )
    )
    notes.append(
        "Prior 3 (the score is a maximum over every test): a worse experiment can "
        "never lower the recorded score, so test aggressively -- the archive below "
        "holds 20 distinct configurations worth trying by hand."
    )
    if best_value > ub.ceiling + 1e-6:
        notes.append(
            "WARNING: the simulated result exceeds the LP ceiling.  One of them is "
            "wrong -- do not trust this plan until calibration says otherwise."
        )

    result = OptimizeResult(
        best=best_config,
        best_value=best_value,
        baseline_value=baseline_value,
        bound=ub.ceiling,
        archive=entries,
        actions=_order_actions(cf, config_diff(current, best_config)),
        top_warning=lines[0] if lines else None,
        elapsed_s=time.perf_counter() - t_start,
        iterations=iterations,
        converged=converged,
        endgame=endgame_actions,
        notes=notes,
        one_hour_mod_value=one_hour_value,
        deliberately_idle=idle,
        seed_value=seed_value,
        ramp_up_loss=round(ub.ceiling - seed_value, 6),
        bound_detail=ub,
    )
    return result


def _neighbour(
    cf: CompiledFactory, vec: np.ndarray, rng: random.Random, movable: list[int]
) -> np.ndarray:
    """One local move.  Every move lands inside the game's allowed range."""
    n = cf.n
    out = vec.copy()
    i = rng.choice(movable)
    roll = rng.random()

    if roll < 0.12 and len(cf.recipes[i]) > 1:
        # change a recipe
        out[i] = rng.randrange(len(cf.recipes[i]))
    elif roll < 0.20:
        # set output to zero: deliberately starve a branch (prior 2)
        out[n + i] = 0
    elif roll < 0.28:
        # jump to this machine's ceiling
        out[n + i] = cf.effective_output_max(i, int(out[2 * n + i]))
    elif roll < 0.45 and cf.available_mask[i]:
        # toggle one modification
        bits = [b for b in MOD_ORDER if cf.available_mask[i] & MOD_BIT[b]]
        bit = MOD_BIT[rng.choice(bits)]
        out[2 * n + i] ^= bit
    elif roll < 0.55 and cf.available_mask[i]:
        # swap a modification between two machines
        j = rng.choice(movable)
        bits = [
            b
            for b in MOD_ORDER
            if (cf.available_mask[i] & MOD_BIT[b]) and (cf.available_mask[j] & MOD_BIT[b])
        ]
        if bits and j != i:
            bit = MOD_BIT[rng.choice(bits)]
            if out[2 * n + i] & bit and not out[2 * n + j] & bit:
                out[2 * n + i] &= ~bit
                out[2 * n + j] |= bit
            elif out[2 * n + j] & bit and not out[2 * n + i] & bit:
                out[2 * n + j] &= ~bit
                out[2 * n + i] |= bit
    else:
        # +/- 1 on one output setting (the workhorse move)
        step = 1 if rng.random() < 0.5 else -1
        out[n + i] = int(out[n + i]) + step

    return _clamp_vector(cf, out)


def _endgame(
    cf: CompiledFactory, best_vec: np.ndarray, best_value: float
) -> tuple[list[Action], FactoryConfig, float]:
    """Layer 4.

    A run started at hour ``h`` by machine ``i`` only ever becomes money if
    ``h + lead_time[i] <= horizon``.  Everything after that is items bought and
    paid for that die in a buffer.  We build that drain schedule for the last
    ``depth + 2`` hours, then greedily test each remaining (machine, hour) cell
    -- the schedule is only returned if it actually beats the flat config.
    """
    n = cf.n
    horizon = cf.horizon
    window_start = max(1, horizon - (cf.depth + 2) + 1)
    hourly = np.zeros((n, horizon + 1), dtype=np.int64)
    for i in range(n):
        hourly[i, :] = int(best_vec[n + i])

    changed: list[tuple[int, int]] = []
    for i in range(n):
        if int(best_vec[n + i]) <= 0:
            continue
        lead = cf.lead_time[i]
        if lead >= 10**6:
            continue
        for h in range(window_start, horizon + 1):
            if h + lead > horizon and hourly[i, h] != 0:
                hourly[i, h] = 0
                changed.append((i, h))

    value = evaluate_fast_hourly(cf, best_vec, hourly) if changed else best_value
    if value < best_value:  # the heuristic schedule hurt: back it out
        for i, h in changed:
            hourly[i, h] = int(best_vec[n + i])
        changed = []
        value = best_value

    # greedy pass over the rest of the window
    for i in range(n):
        if int(best_vec[n + i]) <= 0:
            continue
        for h in range(window_start, horizon + 1):
            if hourly[i, h] == 0:
                continue
            keep = int(hourly[i, h])
            hourly[i, h] = 0
            trial = evaluate_fast_hourly(cf, best_vec, hourly)
            if trial > value + 1e-9:
                value = trial
                changed.append((i, h))
            else:
                hourly[i, h] = keep

    config = vector_to_config(cf, best_vec)
    actions: list[Action] = []
    if changed:
        machines = dict(config.machines)
        for i, h in sorted(changed, key=lambda p: (p[1], cf.hops[p[0]])):
            mid = cf.machine_ids[i]
            mc = machines[mid]
            overrides = dict(mc.hourly_overrides)
            overrides[h] = 0
            machines[mid] = mc.model_copy(update={"hourly_overrides": overrides})
            actions.append(
                Action(
                    target=mid,
                    setting="output",
                    value=0,
                    reason=(
                        f"hour {h}: anything {mid} starts now needs "
                        f"{cf.lead_time[i]}h to reach the seller and the run ends at "
                        f"hour {horizon} -- switch it off and keep the money"
                    ),
                )
            )
        config = FactoryConfig(machines=machines)
    return actions, config, value
