"""Factory simulator.  Spec section 7.3.

Pure, deterministic, no I/O.  The public entry point is :func:`simulate`, a thin
wrapper over :func:`compile_factory` + :func:`simulate_compiled`.  The optimizer
uses :func:`evaluate_fast`, which runs the same hour loop with every pydantic
object, dict and string key stripped out.

HOUR LOOP (identical in both paths), for each hour ``h`` in ``1..horizon``:

1. advance in-progress jobs (a job started ``k`` hours ago with
   ``production_hours == k`` matures now);
2. deposit matured output -- makers/suppliers into their OWN storage, sellers
   straight into money at the recipe sale price;
3. order consumers by priority (hops-to-seller, then topmost);
4. allocate: full requirement or nothing, no partial production;
5. charge supplier purchases and mod-inflated production costs;
6. append money.

``money_by_hour[0]`` is the starting money; ``money_by_hour[h]`` is the money at
the end of hour ``h`` (see ``model.SimResult``).

ASSUMPTIONS (semantics the game does not state and no flag covers -- every one
of these is a deliberate choice, listed here because a wrong one silently
changes the answer):

A1. A job started in hour ``h`` deposits at step 2 of hour ``h + production_hours``.
    So even a 1-hour machine has a one-hour deposit lag, and a supplier -> seller
    chain books its first revenue in hour 3.  This is the "ramp-up delay".
A2. ``half_materials`` halves the *total* requirement and rounds up:
    ``ceil(qty_per_unit * output_setting / 2)``.  The alternative reading
    (``ceil(qty_per_unit / 2) * output_setting``) is strictly worse for the
    player and is not used; see ``_requirement``.
A3. (PROMOTED TO A FLAG -- no longer an assumption.)  What a machine does when
    its own output storage is already full is the ``full_storage_behavior``
    rule flag, options ``idle`` (default) and ``produce_and_waste``:

      * ``idle``: the machine does not run.  It consumes nothing, pays nothing,
        and emits ``IDLE_FULL``.  ``OVERFLOW`` can then only ever be a *partial*
        overshoot -- the buffer had room, the deposit was bigger than the room.
      * ``produce_and_waste``: the machine runs anyway.  It consumes its full
        input requirement, pays its per-item production cost, and deposits into
        a buffer with no room, so the entire batch is clipped and logged as
        ``OVERFLOW``.  A machine in this state burns money for nothing, hour
        after hour, which is a very different economic outcome -- which is
        exactly why it is a flag and not a comment.

    NOTE: the flag is read with ``getattr(flags, "full_storage_behavior",
    "idle")`` so this module works whether or not the installed
    ``RuleFlags`` model carries the field yet.  See ``make_flags``.
A4. Mods are maker modifications: they inflate maker production cost only, never
    a supplier's purchase cost.
A5. Affordability is always *decided* at order time (both settings of
    ``supplier_cost_timing``); the flag only moves *when the money leaves the
    account*, which is exactly what a recorded money series can discriminate.
A6. ``insufficient_funds`` governs suppliers.  A maker that cannot afford its
    production cost is skipped under both settings (there is no sensible
    "half a maker run" once inputs are all-or-nothing).
A7. Inputs are drawn from connected upstream storages in edge-declaration order,
    greedily.
A8. Sellers have no storage, never overflow, and sell ``output_setting *
    output_qty`` items per run at ``sale_price``.

PERFORMANCE.  Item quantities live in ``CompiledFactory.scratch_storage``, a
preallocated ``(n_machines, n_items)`` int64 array indexed by integer machine
and item id -- never a dict of strings.  The hour loop works on that array's
flat view (``_run`` lifts it into a flat python list for the duration of the
run and writes it back at the end; CPython list indexing is ~5x cheaper than
numpy scalar indexing at these sizes, and this is a 10^6-evaluations-per-run
hot path).  Everything else -- requirements, upstream flat indices, per-item
costs -- is precomputed once per run into a per-machine "plan" tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Iterable

import numpy as np

from services.core.rules.dsl import RuleFlags
from services.solvers.factory.model import (
    FactoryConfig,
    FactoryState,
    MachineConfig,
    MachineHourLog,
    ModKind,
    SimResult,
    SimWarning,
    WarningKind,
)

KIND_SUPPLIER = 0
KIND_MAKER = 1
KIND_SELLER = 2

_KIND_CODE = {"supplier": KIND_SUPPLIER, "maker": KIND_MAKER, "seller": KIND_SELLER}

#: Canonical mod bit order.  A config's mod set is a 4-bit integer in the vector.
MOD_ORDER: tuple[ModKind, ...] = (
    ModKind.DOUBLE_OUTPUT_MAX,
    ModKind.HALF_MATERIALS,
    ModKind.ONE_HOUR_PRODUCTION,
    ModKind.DOUBLE_STORAGE_MAX,
)
MOD_BIT: dict[ModKind, int] = {m: 1 << i for i, m in enumerate(MOD_ORDER)}

BIT_DOUBLE_OUTPUT = MOD_BIT[ModKind.DOUBLE_OUTPUT_MAX]
BIT_HALF_MATERIALS = MOD_BIT[ModKind.HALF_MATERIALS]
BIT_ONE_HOUR = MOD_BIT[ModKind.ONE_HOUR_PRODUCTION]
BIT_DOUBLE_STORAGE = MOD_BIT[ModKind.DOUBLE_STORAGE_MAX]

_INF_HOPS = 9999
_EPS = 1e-9

#: Defaults for rule flags this simulator honours.  A flag lives in
#: ``services.core.rules.dsl.RuleFlags``; this table only exists so that the
#: simulator keeps working while a newly-promoted flag is still landing in the
#: DSL (the two files are edited by different hands).  Every entry here must
#: also appear in the DSL -- ``flag_of`` is a compatibility shim, not a place to
#: invent rules.
FLAG_DEFAULTS: dict[str, str] = {"full_storage_behavior": "idle"}


def flag_of(flags: RuleFlags, name: str) -> str:
    """Read a rule flag, falling back to its documented default.

    ``RuleFlags`` is frozen and forbids extras, so a flag that has been promoted
    in this package but not yet added to the DSL model would otherwise raise on
    construction.  Reading through here keeps the hour loop honest either way.
    """
    return getattr(flags, name, FLAG_DEFAULTS[name])


def make_flags(**kwargs: str) -> RuleFlags:
    """Build ``RuleFlags``, tolerating flags the installed DSL model lacks.

    Used by ``calibrate`` (which sweeps every flag the simulator honours) and by
    the tests.  Once the DSL carries every flag, this is exactly
    ``RuleFlags(**kwargs)``.
    """
    known = {k: v for k, v in kwargs.items() if k in RuleFlags.model_fields}
    extra = {k: v for k, v in kwargs.items() if k not in RuleFlags.model_fields}
    flags = RuleFlags(**known)
    return flags.model_copy(update=extra) if extra else flags


# ---------------------------------------------------------------------------
# compiled representation
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CompiledRecipe:
    id: str
    in_items: tuple[int, ...]
    in_qtys: tuple[int, ...]
    out_item: int
    out_qty: int
    sale_price: float
    purchase_cost: float
    production_cost: float


@dataclass
class CompiledFactory:
    """Immutable compile of ``(state, flags)`` plus reusable scratch buffers."""

    state: FactoryState
    flags: RuleFlags
    machine_ids: tuple[str, ...]
    index: dict[str, int]
    kinds: tuple[int, ...]
    item_ids: tuple[str, ...]
    item_index: dict[str, int]
    recipes: tuple[tuple[CompiledRecipe, ...], ...]
    recipe_ids: tuple[tuple[str, ...], ...]
    upstream: tuple[tuple[int, ...], ...]
    downstream: tuple[tuple[int, ...], ...]
    storage_max: tuple[int, ...]
    output_max: tuple[int, ...]
    production_hours: tuple[int, ...]
    mod_cost: tuple[tuple[float, ...], ...]
    available_mask: tuple[int, ...]
    installed_mask: tuple[int, ...]
    initial_storage: np.ndarray
    initial_flat: list[int]
    hops: tuple[int, ...]
    pixel_dist: tuple[float, ...]
    ys: tuple[float, ...]
    fixed_order: tuple[int, ...]
    lead_time: tuple[int, ...]
    depth: int
    horizon: int
    starting_money: float
    n: int
    n_items: int
    scratch_storage: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    #: memoized ``(machine, recipe, setting, mods) -> plan`` (see ``_plan``).
    plan_cache: dict = field(repr=False, default_factory=dict)

    # -- small helpers used by bounds/optimize -----------------------------

    def n_recipes(self, i: int) -> int:
        return len(self.recipes[i])

    def effective_output_max(self, i: int, mod_mask: int) -> int:
        return self.output_max[i] * (2 if mod_mask & BIT_DOUBLE_OUTPUT else 1)

    def effective_storage_max(self, i: int, mod_mask: int) -> int:
        return self.storage_max[i] * (2 if mod_mask & BIT_DOUBLE_STORAGE else 1)

    def effective_hours(self, i: int, mod_mask: int) -> int:
        return 1 if mod_mask & BIT_ONE_HOUR else self.production_hours[i]


def _mask_of(mods: Iterable[ModKind]) -> int:
    mask = 0
    for m in mods:
        mask |= MOD_BIT.get(m, 0)
    return mask


def _bfs_hops(n: int, downstream: list[list[int]], sellers: list[int]) -> list[int]:
    """Hops from each machine to the nearest seller (a seller itself is 0)."""
    hops = [_INF_HOPS] * n
    for s in sellers:
        hops[s] = 0
    upstream: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in downstream[i]:
            upstream[j].append(i)
    frontier = list(sellers)
    d = 0
    while frontier:
        d += 1
        nxt: list[int] = []
        for j in frontier:
            for i in upstream[j]:
                if hops[i] > d:
                    hops[i] = d
                    nxt.append(i)
        frontier = nxt
    return hops


def compile_factory(state: FactoryState, flags: RuleFlags = RuleFlags()) -> CompiledFactory:
    machines = list(state.machines)
    n = len(machines)
    machine_ids = tuple(m.id for m in machines)
    index = {mid: i for i, mid in enumerate(machine_ids)}

    item_ids = tuple(state.item_ids())
    item_index = {it: k for k, it in enumerate(item_ids)}
    n_items = max(1, len(item_ids))

    kinds = tuple(_KIND_CODE[m.kind.value] for m in machines)

    recipes: list[tuple[CompiledRecipe, ...]] = []
    recipe_ids: list[tuple[str, ...]] = []
    for m in machines:
        compiled = []
        for r in m.recipes:
            compiled.append(
                CompiledRecipe(
                    id=r.id,
                    in_items=tuple(item_index[k] for k in r.inputs),
                    in_qtys=tuple(int(v) for v in r.inputs.values()),
                    out_item=item_index[r.output_item] if r.output_item else -1,
                    out_qty=int(r.output_qty),
                    sale_price=float(r.sale_price),
                    purchase_cost=float(r.purchase_cost),
                    production_cost=float(r.production_cost),
                )
            )
        recipes.append(tuple(compiled))
        recipe_ids.append(tuple(r.id for r in m.recipes))

    up: list[list[int]] = [[] for _ in range(n)]
    down: list[list[int]] = [[] for _ in range(n)]
    for e in state.edges:
        if e.src in index and e.dst in index:
            up[index[e.dst]].append(index[e.src])
            down[index[e.src]].append(index[e.dst])

    sellers = [i for i in range(n) if kinds[i] == KIND_SELLER]
    hops = _bfs_hops(n, down, sellers)

    pixel_dist: list[float] = []
    for m in machines:
        if not sellers:
            pixel_dist.append(float(_INF_HOPS))
        else:
            pixel_dist.append(
                min(hypot(m.x - machines[s].x, m.y - machines[s].y) for s in sellers)
            )

    initial = np.zeros((n, n_items), dtype=np.int64)
    for mid, per_item in state.initial_storage.items():
        if mid not in index:
            continue
        for item, qty in per_item.items():
            if item in item_index:
                initial[index[mid], item_index[item]] = int(qty)

    prod_hours = tuple(max(1, int(m.production_hours)) for m in machines)

    # lead time: hours from "start a run now" to "money in the bank", along the
    # fastest downstream path.  Drives the endgame drain schedule.
    lead = [10**6] * n
    for s in sellers:
        lead[s] = prod_hours[s]
    for _ in range(n + 1):
        changed = False
        for i in range(n):
            if kinds[i] == KIND_SELLER:
                continue
            best = min((lead[j] for j in down[i]), default=10**6)
            if best < 10**6 and prod_hours[i] + best < lead[i]:
                lead[i] = prod_hours[i] + best
                changed = True
        if not changed:
            break

    reach = [h for h in hops if h < _INF_HOPS]
    depth = (max(reach) + 1) if reach else 1

    cf = CompiledFactory(
        state=state,
        flags=flags,
        machine_ids=machine_ids,
        index=index,
        kinds=kinds,
        item_ids=item_ids,
        item_index=item_index,
        recipes=tuple(recipes),
        recipe_ids=tuple(recipe_ids),
        upstream=tuple(tuple(u) for u in up),
        downstream=tuple(tuple(d) for d in down),
        storage_max=tuple(int(m.storage_max) for m in machines),
        output_max=tuple(int(m.output_max) for m in machines),
        production_hours=prod_hours,
        mod_cost=tuple(
            tuple(float(m.mod_cost.get(mod, 0.0)) for mod in MOD_ORDER) for m in machines
        ),
        available_mask=tuple(
            _mask_of(m.available_mods) | _mask_of(m.installed_mods) for m in machines
        ),
        installed_mask=tuple(_mask_of(m.installed_mods) for m in machines),
        initial_storage=initial,
        initial_flat=initial.reshape(-1).tolist(),
        hops=tuple(hops),
        pixel_dist=tuple(pixel_dist),
        ys=tuple(float(m.y) for m in machines),
        fixed_order=(),
        lead_time=tuple(lead),
        depth=depth,
        horizon=int(state.horizon_hours),
        starting_money=float(state.starting_money),
        n=n,
        n_items=n_items,
        scratch_storage=np.zeros((n, n_items), dtype=np.int64),
    )
    cf.fixed_order = _static_order(cf)
    return cf


def _static_order(cf: CompiledFactory) -> tuple[int, ...]:
    """Consumers closest to the seller first, ties broken topmost (smallest y)."""
    if cf.flags.priority_metric == "pixels":
        return tuple(sorted(range(cf.n), key=lambda i: (cf.pixel_dist[i], cf.ys[i], i)))
    return tuple(sorted(range(cf.n), key=lambda i: (cf.hops[i], cf.ys[i], i)))


def _dynamic_order(cf: CompiledFactory, live: list[bool]) -> tuple[int, ...]:
    """``priority_recompute="per_hour"``.

    Distance is recomputed through *live* machines only: a machine that is busy
    on a 2-hour job or switched off this hour does not currently conduct items
    to the seller, so its upstream neighbours are further from the money than
    the static graph says.  Machines with no live path are served last.
    """
    n = cf.n
    dist = [_INF_HOPS] * n
    for i in range(n):
        if cf.kinds[i] == KIND_SELLER and live[i]:
            dist[i] = 0
    for _ in range(n):
        changed = False
        for i in range(n):
            if cf.kinds[i] == KIND_SELLER:
                continue
            best = _INF_HOPS
            for j in cf.downstream[i]:
                if live[j] and dist[j] < best:
                    best = dist[j]
            if best < _INF_HOPS and best + 1 < dist[i]:
                dist[i] = best + 1
                changed = True
        if not changed:
            break
    if cf.flags.priority_metric == "pixels":
        return tuple(sorted(range(n), key=lambda i: (dist[i], cf.pixel_dist[i], cf.ys[i], i)))
    return tuple(sorted(range(n), key=lambda i: (dist[i], cf.ys[i], i)))


# ---------------------------------------------------------------------------
# config <-> vector
# ---------------------------------------------------------------------------


def config_to_vector(cf: CompiledFactory, config: FactoryConfig) -> np.ndarray:
    """``[recipe_idx | output_setting | mod_mask]``, three contiguous blocks."""
    n = cf.n
    vec = np.zeros(3 * n, dtype=np.int64)
    for i, mid in enumerate(cf.machine_ids):
        mc = config.machines.get(mid)
        if mc is None:
            vec[i] = -1
            continue
        ids = cf.recipe_ids[i]
        vec[i] = ids.index(mc.recipe_id) if mc.recipe_id in ids else (0 if ids else -1)
        vec[n + i] = int(mc.output_setting)
        vec[2 * n + i] = _mask_of(mc.mods)
    return vec


def vector_to_config(cf: CompiledFactory, vector: np.ndarray) -> FactoryConfig:
    n = cf.n
    machines: dict[str, MachineConfig] = {}
    for i, mid in enumerate(cf.machine_ids):
        ri = int(vector[i])
        ids = cf.recipe_ids[i]
        rid = ids[ri] if 0 <= ri < len(ids) else None
        mask = int(vector[2 * n + i])
        machines[mid] = MachineConfig(
            machine_id=mid,
            recipe_id=rid,
            output_setting=int(vector[n + i]),
            mods=[m for m in MOD_ORDER if mask & MOD_BIT[m]],
        )
    return FactoryConfig(machines=machines)


def hourly_matrix(cf: CompiledFactory, config: FactoryConfig) -> np.ndarray | None:
    """``(n, horizon+1)`` per-hour output settings, or ``None`` if constant."""
    if not any(mc.hourly_overrides for mc in config.machines.values()):
        return None
    mat = np.zeros((cf.n, cf.horizon + 1), dtype=np.int64)
    for i, mid in enumerate(cf.machine_ids):
        mc = config.machines.get(mid)
        if mc is None:
            continue
        for h in range(cf.horizon + 1):
            mat[i, h] = mc.output_at(h)
    return mat


# ---------------------------------------------------------------------------
# the hour loop
# ---------------------------------------------------------------------------


def _requirement(qty_per_unit: int, setting: int, half: bool) -> int:
    """Input units needed for one run.

    ASSUMPTION A2: ``half_materials`` halves the *total* and rounds UP --
    ``ceil(qty * setting / 2)``.  The game shows one number ("materials
    needed"), so halving the displayed total is the natural reading; ceil (never
    floor) because no game hands out a free fractional item.
    """
    total = qty_per_unit * setting
    if half:
        return -(-total // 2)
    return total


def _run(
    cf: CompiledFactory,
    recipe_idx,
    out_const,
    mod_mask,
    out_hourly: np.ndarray | None = None,
    collect_logs: bool = False,
):
    """Core loop.

    Returns ``(money_by_hour, warnings, logs, storage, revenue, cost,
    skip_mask)``.  ``warnings``/``logs`` are only built when ``collect_logs``;
    ``skip_mask`` (one int, bit i = machine i starved at least once) is always
    maintained because the optimizer's cliff guard needs it.
    """
    flags = cf.flags
    n = cf.n
    n_items = cf.n_items
    horizon = cf.horizon
    kinds = cf.kinds

    two_hour_late = flags.two_hour_consume_timing == "start_hour_2"
    cost_at_delivery = flags.supplier_cost_timing == "at_delivery"
    clip_before_pull = flags.overflow_timing == "before_pull"
    recompute = flags.priority_recompute == "per_hour"
    # (output_max_meaning and mod_stacking are applied inside ``_plan``, which
    # is where the effective caps and per-item costs are computed and cached.)
    idles_when_full = flag_of(flags, "full_storage_behavior") == "idle"
    partial_funds = flags.insufficient_funds == "partial"

    # ---- per-machine plans -------------------------------------------------
    # A plan freezes one machine's entire hour-4 decision for a given
    # (recipe, output setting, mod mask).  It is memoized on the compiled
    # factory, so a local-search move that touches one machine reuses every
    # other machine's plan for free.
    ris = list(map(int, recipe_idx))
    masks = list(map(int, mod_mask))
    plan = _plan
    if out_hourly is None:
        settings = list(map(int, out_const))
        plans: list[tuple | None] | None = [
            plan(cf, i, ris[i], settings[i], masks[i]) for i in range(n)
        ]
    else:
        plans = None

    # flat working copy of the preallocated storage array, index i*n_items + item
    buf = cf.initial_flat.copy()

    pend_units = [0] * n
    pend_rem = [0] * n
    pend_plan: list[tuple | None] = [None] * n
    pend_cost = [0.0] * n
    awaiting = [False] * n

    money = cf.starting_money
    money_by_hour = [money]
    total_rev = 0.0
    total_cost = 0.0
    skip_mask = 0
    warnings: list[SimWarning] = []
    logs: list[MachineHourLog] = []
    machine_ids = cf.machine_ids
    item_ids = cf.item_ids
    # machines that can never run under this config are dropped from both loops
    if plans is not None:
        live_ids = tuple(i for i in range(n) if plans[i] is not None)
        order = tuple(i for i in cf.fixed_order if plans[i] is not None)
    else:
        live_ids = tuple(range(n))
        order = cf.fixed_order
    live_set = frozenset(live_ids)
    kind_seller = KIND_SELLER
    kind_supplier = KIND_SUPPLIER
    eps = _EPS
    money_append = money_by_hour.append

    for h in range(1, horizon + 1):
        if collect_logs:
            row_produced = [0] * n
            row_dep = [0] * n
            row_waste = [0] * n
            row_rev = [0.0] * n
            row_cost = [0.0] * n
            row_skip = [False] * n
            row_consumed: list[dict[str, int]] = [dict() for _ in range(n)]
        deposited: list[tuple[int, int, int]] = []

        # -- 1. advance in-progress jobs + 2. deposit matured output ----------
        # (one pass; every deposit still lands before any allocation)
        for i in live_ids:
            rem = pend_rem[i]
            if rem > 0:
                rem -= 1
                pend_rem[i] = rem
            if rem != 0 or pend_units[i] <= 0:
                continue
            units = pend_units[i]
            pend_units[i] = 0
            p = pend_plan[i]
            r = p[1]
            if p[9] == kind_seller:
                rev = units * r.sale_price
                money += rev
                total_rev += rev
                if collect_logs:
                    row_rev[i] += rev
                    row_dep[i] += units
            elif p[8] >= 0:
                cap = p[6]
                fi = p[8]
                new = buf[fi] + units
                if clip_before_pull and new > cap:
                    waste = new - cap
                    new = cap
                    if collect_logs:
                        row_waste[i] += waste
                        warnings.append(
                            SimWarning(
                                kind=WarningKind.OVERFLOW,
                                hour=h,
                                machine_id=machine_ids[i],
                                amount=float(waste),
                                detail=f"{waste} x {item_ids[r.out_item]} destroyed "
                                f"(storage cap {cap})",
                            )
                        )
                elif new > cap:
                    # cleared after the pull phase (overflow_timing default)
                    deposited.append((i, fi, cap))
                buf[fi] = new
                if collect_logs:
                    row_dep[i] += units
            if pend_cost[i]:
                c = pend_cost[i]
                pend_cost[i] = 0.0
                money -= c
                total_cost += c
                if collect_logs:
                    row_cost[i] += c

        # -- 3. priority order -------------------------------------------------
        if recompute:
            live = [False] * n
            if plans is not None:
                for i in live_ids:
                    live[i] = pend_rem[i] == 0
            else:
                for i in range(n):
                    live[i] = pend_rem[i] == 0 and int(out_hourly[i, h]) > 0
            order = tuple(i for i in _dynamic_order(cf, live) if i in live_set)

        # -- 4a. deferred consumption for 2-hour machines (start_hour_2) -------
        if two_hour_late:
            for i in order:
                if not awaiting[i]:
                    continue
                awaiting[i] = False
                p = pend_plan[i]
                needs = p[7]
                ok = True
                for item, need, flats in needs:
                    avail = 0
                    for fi in flats:
                        avail += buf[fi]
                    if avail < need:
                        ok = False
                        break
                if not ok:
                    pend_units[i] = 0
                    pend_rem[i] = 0
                    skip_mask |= 1 << i
                    if collect_logs:
                        row_skip[i] = True
                        warnings.append(
                            SimWarning(
                                kind=WarningKind.SKIPPED,
                                hour=h,
                                machine_id=machine_ids[i],
                                detail="2-hour job cancelled: inputs gone by its second hour",
                            )
                        )
                    continue
                for item, need, flats in needs:
                    _draw(buf, flats, need)
                    if collect_logs:
                        nm = item_ids[item]
                        row_consumed[i][nm] = row_consumed[i].get(nm, 0) + need
                c = pend_units[i] * p[5]
                if c:
                    money -= c
                    total_cost += c
                    if collect_logs:
                        row_cost[i] += c

        # -- 4b/5. allocate in priority order, then charge ----------------------
        for i in order:
            if pend_rem[i] > 0 or pend_units[i] > 0:
                continue  # busy on a job
            if plans is not None:
                p = plans[i]
            else:
                p = plan(cf, i, ris[i], int(out_hourly[i, h]), masks[i])
            if p is None:
                continue
            r = p[1]
            units = p[2]
            if units <= 0:
                continue

            # flag full_storage_behavior: a machine whose buffer is already full
            # either stands down (default) or runs anyway and destroys the batch.
            # Under "produce_and_waste" we fall through: it eats its inputs, pays
            # its production cost, and the deposit step clips the lot.
            if idles_when_full and p[9] != kind_seller and p[8] >= 0:
                if buf[p[8]] >= p[6]:
                    if collect_logs:
                        warnings.append(
                            SimWarning(
                                kind=WarningKind.IDLE_FULL,
                                hour=h,
                                machine_id=machine_ids[i],
                                amount=float(p[6]),
                                detail=f"storage full ({p[6]}), machine idle",
                            )
                        )
                    continue

            if p[9] == kind_supplier:
                cost = units * r.purchase_cost
                if cost > money + eps:
                    # ASSUMPTION A5/A6: affordability is decided at order time
                    # under both supplier_cost_timing settings.
                    if partial_funds and r.purchase_cost > 0:
                        afford = int(money // r.purchase_cost)
                        if afford <= 0:
                            skip_mask |= 1 << i
                            if collect_logs:
                                row_skip[i] = True
                                warnings.append(
                                    SimWarning(
                                        kind=WarningKind.NO_MONEY,
                                        hour=h,
                                        machine_id=machine_ids[i],
                                        amount=cost,
                                        detail=f"order costs {cost:g}, have {money:g}",
                                    )
                                )
                            continue
                        units = afford if afford < units else units
                        cost = units * r.purchase_cost
                        if collect_logs:
                            warnings.append(
                                SimWarning(
                                    kind=WarningKind.NO_MONEY,
                                    hour=h,
                                    machine_id=machine_ids[i],
                                    amount=cost,
                                    detail=f"partial order: only {units} affordable",
                                )
                            )
                    else:
                        skip_mask |= 1 << i
                        if collect_logs:
                            row_skip[i] = True
                            warnings.append(
                                SimWarning(
                                    kind=WarningKind.NO_MONEY,
                                    hour=h,
                                    machine_id=machine_ids[i],
                                    amount=cost,
                                    detail=f"order costs {cost:g}, have {money:g}",
                                )
                            )
                        continue
                if cost_at_delivery:
                    pend_cost[i] = cost
                else:
                    money -= cost
                    total_cost += cost
                    if collect_logs:
                        row_cost[i] += cost
                pend_units[i] = units
                pend_plan[i] = p
                pend_rem[i] = p[3]
                if collect_logs:
                    row_produced[i] = units
                continue

            # maker / seller: the full requirement or nothing at all
            simple = p[10]
            ok = True
            short = -1
            short_need = 0
            if simple is not None:
                for item, need, fi in simple:
                    if buf[fi] < need:
                        ok = False
                        short = item
                        short_need = need
                        break
            else:
                for item, need, flats in p[7]:
                    avail = 0
                    for fi in flats:
                        avail += buf[fi]
                    if avail < need:
                        ok = False
                        short = item
                        short_need = need
                        break
            if not ok:
                skip_mask |= 1 << i
                if collect_logs:
                    row_skip[i] = True
                    warnings.append(
                        SimWarning(
                            kind=WarningKind.SKIPPED,
                            hour=h,
                            machine_id=machine_ids[i],
                            amount=float(short_need),
                            detail=f"needs {short_need} x {item_ids[short]}, "
                            f"upstream storage has less -- produced nothing",
                        )
                    )
                continue

            run_cost = units * p[5]
            if run_cost > money + eps:
                skip_mask |= 1 << i
                if collect_logs:
                    row_skip[i] = True
                    warnings.append(
                        SimWarning(
                            kind=WarningKind.NO_MONEY,
                            hour=h,
                            machine_id=machine_ids[i],
                            amount=run_cost,
                            detail=f"production costs {run_cost:g}, have {money:g}",
                        )
                    )
                continue

            if two_hour_late and p[3] == 2:
                # reserve now, consume (and pay) at the start of the second hour
                awaiting[i] = True
                pend_units[i] = units
                pend_plan[i] = p
                pend_rem[i] = 2
                if collect_logs:
                    row_produced[i] = units
                continue

            if simple is not None:
                for item, need, fi in simple:
                    buf[fi] -= need
                    if collect_logs:
                        nm = item_ids[item]
                        row_consumed[i][nm] = row_consumed[i].get(nm, 0) + need
            else:
                for item, need, flats in p[7]:
                    left = need
                    for fi in flats:
                        have = buf[fi]
                        if have <= 0:
                            continue
                        if have >= left:
                            buf[fi] = have - left
                            break
                        buf[fi] = 0
                        left -= have
                    if collect_logs:
                        nm = item_ids[item]
                        row_consumed[i][nm] = row_consumed[i].get(nm, 0) + need
            if run_cost:
                money -= run_cost
                total_cost += run_cost
                if collect_logs:
                    row_cost[i] += run_cost
            pend_units[i] = units
            pend_plan[i] = p
            pend_rem[i] = p[3]
            if collect_logs:
                row_produced[i] = units

        # -- overflow clipped only after production (default) ------------------
        if deposited:
            for i, fi, cap in deposited:
                cur = buf[fi]
                if cur > cap:
                    waste = cur - cap
                    buf[fi] = cap
                    if collect_logs:
                        row_waste[i] += waste
                        warnings.append(
                            SimWarning(
                                kind=WarningKind.OVERFLOW,
                                hour=h,
                                machine_id=machine_ids[i],
                                amount=float(waste),
                                detail=f"{waste} x {item_ids[fi - i * n_items]} destroyed "
                                f"(storage cap {cap})",
                            )
                        )

        # -- 6. append money ----------------------------------------------------
        money_append(money)

        if collect_logs:
            for i in range(n):
                base = i * n_items
                logs.append(
                    MachineHourLog(
                        hour=h,
                        machine_id=machine_ids[i],
                        produced=row_produced[i],
                        consumed=row_consumed[i],
                        deposited=row_dep[i],
                        wasted=row_waste[i],
                        revenue=row_rev[i],
                        cost=row_cost[i],
                        skipped=row_skip[i],
                        storage_after={
                            item_ids[k]: buf[base + k]
                            for k in range(len(item_ids))
                            if buf[base + k]
                        },
                    )
                )

    storage = cf.scratch_storage
    storage.reshape(-1)[:] = buf

    if collect_logs:
        for i in range(n):
            if kinds[i] == KIND_SELLER:
                continue
            base = i * n_items
            left = sum(buf[base : base + len(item_ids)])
            if left > 0:
                detail = ", ".join(
                    f"{buf[base + k]} x {item_ids[k]}"
                    for k in range(len(item_ids))
                    if buf[base + k]
                )
                warnings.append(
                    SimWarning(
                        kind=WarningKind.STRANDED,
                        hour=horizon,
                        machine_id=machine_ids[i],
                        amount=float(left),
                        detail=f"{detail} left in storage at the horizon, worth zero",
                    )
                )

    return money_by_hour, warnings, logs, storage, total_rev, total_cost, skip_mask


_MISS = object()


def _plan(cf: CompiledFactory, i: int, ri: int, setting: int, mask: int) -> tuple | None:
    """Freeze one machine's whole hour-4 decision; memoized on ``cf``.

    ``(i, recipe, units, hours, half, unit_cost, storage_cap, needs, out_flat,
    kind)`` where ``needs`` is ``[(item, required, upstream_flat_indices)]``.
    ``None`` means "this machine cannot run at all under this config".
    """
    key = (i, ri, setting, mask)
    cache = cf.plan_cache
    p = cache.get(key, _MISS)
    if p is not _MISS:
        return p
    if len(cache) > 200_000:  # pathological board; keep the memo bounded
        cache.clear()

    n_items = cf.n_items
    recipes = cf.recipes[i]
    if ri < 0 or ri >= len(recipes) or setting <= 0:
        cache[key] = None
        return None
    r = recipes[ri]

    out_max = cf.output_max[i] * 2 if mask & BIT_DOUBLE_OUTPUT else cf.output_max[i]
    store = cf.storage_max[i] * 2 if mask & BIT_DOUBLE_STORAGE else cf.storage_max[i]
    # output_max_meaning="separate_from_storage": the per-hour output ceiling is
    # tracked apart from the storage bar, so a machine may hold storage_max of
    # stock *plus* one full output batch.
    cap = store + out_max if cf.flags.output_max_meaning == "separate_from_storage" else store
    hours = 1 if mask & BIT_ONE_HOUR else cf.production_hours[i]
    half = bool(mask & BIT_HALF_MATERIALS)

    costs = cf.mod_cost[i]
    add = 0.0
    mult = 1.0
    for b in range(4):
        if mask & (1 << b):
            add += costs[b]
            mult *= 1.0 + costs[b]
    # ASSUMPTION A4 + flag mod_stacking: increments add to the base per-item
    # cost, or multiply it ((1 + increment) each) under "multiplicative".
    unit_cost = (
        r.production_cost * mult
        if cf.flags.mod_stacking == "multiplicative"
        else r.production_cost + add
    )

    s = setting if setting <= out_max else out_max  # output_max is a per-hour ceiling
    if s <= 0:
        cache[key] = None
        return None
    needs = tuple(
        (
            r.in_items[k],
            _requirement(r.in_qtys[k], s, half),
            tuple(u * n_items + r.in_items[k] for u in cf.upstream[i]),
        )
        for k in range(len(r.in_items))
    )
    out_flat = i * n_items + r.out_item if r.out_item >= 0 else -1
    # fast path: every input has exactly one upstream source (the common shape),
    # so check-and-draw is a flat list of (item, need, flat_index).
    simple = (
        tuple((item, need, flats[0]) for item, need, flats in needs)
        if all(len(f) == 1 for _it, _nd, f in needs)
        else None
    )
    p = (i, r, s * r.out_qty, hours, half, unit_cost, cap, needs, out_flat, cf.kinds[i], simple)
    cache[key] = p
    return p


def _draw(buf: list[int], flats: tuple[int, ...], need: int) -> None:
    """Pull ``need`` units out of the upstream storages (ASSUMPTION A7: greedy,
    in edge-declaration order)."""
    left = need
    for fi in flats:
        have = buf[fi]
        if have <= 0:
            continue
        if have >= left:
            buf[fi] = have - left
            return
        buf[fi] = 0
        left -= have


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def simulate_compiled(
    cf: CompiledFactory, config: FactoryConfig, *, collect_logs: bool = True
) -> SimResult:
    vec = config_to_vector(cf, config)
    n = cf.n
    money, warns, logs, storage, rev, cost, _skips = _run(
        cf,
        vec[:n],
        vec[n : 2 * n],
        vec[2 * n : 3 * n],
        out_hourly=hourly_matrix(cf, config),
        collect_logs=collect_logs,
    )
    ending: dict[str, dict[str, int]] = {}
    for i, mid in enumerate(cf.machine_ids):
        row = {
            cf.item_ids[k]: int(storage[i, k])
            for k in range(len(cf.item_ids))
            if storage[i, k]
        }
        if row:
            ending[mid] = row
    return SimResult(
        money_by_hour=[float(m) for m in money],
        final_money=float(money[-1]),
        per_machine_log=logs,
        warnings=warns,
        ending_storage=ending,
        total_revenue=float(rev),
        total_cost=float(cost),
    )


def simulate(
    state: FactoryState, config: FactoryConfig, flags: RuleFlags = RuleFlags()
) -> SimResult:
    """Compile and run.  Pure; no caching, no I/O."""
    return simulate_compiled(compile_factory(state, flags), config, collect_logs=True)


def evaluate_fast(cf: CompiledFactory, vector: np.ndarray) -> float:
    """Hot path: final money only.  No logs, no pydantic, no string keys.

    Per call it allocates only the fixed-size bookkeeping lists (one entry per
    machine) and a copy of the flat storage buffer; every requirement, upstream
    index and per-item cost comes out of the plan memo on ``cf``, so a
    local-search move that touches one machine reuses everything else.
    Measured at ~140 microseconds for a 10-machine, 24-hour board.
    """
    n = cf.n
    money = _run(cf, vector[:n], vector[n : 2 * n], vector[2 * n : 3 * n], collect_logs=False)[0]
    return money[-1]


def evaluate_fast_skips(cf: CompiledFactory, vector: np.ndarray) -> tuple[float, int]:
    """Final money plus a bitmask of machines that starved (the cliff guard)."""
    n = cf.n
    out = _run(cf, vector[:n], vector[n : 2 * n], vector[2 * n : 3 * n], collect_logs=False)
    return out[0][-1], out[6]


def evaluate_fast_hourly(cf: CompiledFactory, vector: np.ndarray, hourly: np.ndarray) -> float:
    """Same as :func:`evaluate_fast` with a ``(n, horizon+1)`` override matrix."""
    n = cf.n
    money = _run(
        cf,
        vector[:n],
        vector[n : 2 * n],
        vector[2 * n : 3 * n],
        out_hourly=hourly,
        collect_logs=False,
    )[0]
    return money[-1]


def money_series(cf: CompiledFactory, config: FactoryConfig) -> list[float]:
    """``money_by_hour`` without building logs -- calibrate.py's 256 runs."""
    vec = config_to_vector(cf, config)
    n = cf.n
    money = _run(
        cf,
        vec[:n],
        vec[n : 2 * n],
        vec[2 * n : 3 * n],
        out_hourly=hourly_matrix(cf, config),
        collect_logs=False,
    )[0]
    return [float(m) for m in money]
