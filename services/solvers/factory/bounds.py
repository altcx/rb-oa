"""Steady-state LP upper bound.  Spec section 7.5.

This is the first tool the agent calls, so it must answer in milliseconds and it
must be honest: the number it returns has to be an upper bound on anything the
simulator can produce, or the optimizer will chase a target it can reach and
report a bogus "converged".

The relaxation, in one paragraph: ignore transients, ignore storage, ignore the
all-or-nothing allocation rule, and let every machine run at a fractional rate.
Each machine gets a per-hour capacity of ``max_output_max / min_production_hours``
-- the *best* it could ever be modded to -- while its per-item cost stays at the
*unmodded* base, and ``half_materials`` is assumed free wherever it is
available.  Every one of those choices is optimistic on purpose: relaxing in
only one direction is what makes the result a bound rather than an estimate.

The only concession to reality is the ramp: nothing can be sold before the
chain has filled, so the horizon is charged ``depth`` dead hours.
"""

from __future__ import annotations

from ortools.linear_solver import pywraplp
from pydantic import BaseModel, ConfigDict, Field

from services.core.rules.dsl import RuleFlags
from services.solvers.factory.model import FactoryState, MachineKind, ModKind
from services.solvers.factory.sim import compile_factory


class UpperBound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: money achievable over the whole horizon, LP relaxation
    ceiling: float
    per_hour_profit: float
    bottleneck_machine: str | None = None
    #: plain language, shown to the user verbatim
    bottleneck_reason: str = ""
    #: constraint name -> shadow price
    duals: dict[str, float] = Field(default_factory=dict)
    #: machine id -> units/hour at the LP optimum
    throughput: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    #: chain depth in machines; the ramp-up allowance charged to the horizon
    depth: int = 1
    #: recipe the LP would run on each machine (a hint for the seed)
    recipe_choice: dict[str, str] = Field(default_factory=dict)
    status: str = "optimal"


def _best_caps(state: FactoryState, m) -> tuple[float, float, bool]:
    """(units/hour ceiling, per-item base cost, half-materials available).

    Optimistic on capacity, pessimistic on cost -- i.e. a relaxation.
    """
    avail = set(m.available_mods) | set(m.installed_mods)
    out_max = m.output_max * (2 if ModKind.DOUBLE_OUTPUT_MAX in avail else 1)
    hours = 1 if ModKind.ONE_HOUR_PRODUCTION in avail else max(1, m.production_hours)
    return out_max / hours, 0.0, ModKind.HALF_MATERIALS in avail


def upper_bound(state: FactoryState, flags: RuleFlags = RuleFlags()) -> UpperBound:
    cf = compile_factory(state, flags)
    notes: list[str] = []
    machines = {m.id: m for m in state.machines}
    sellers = [m for m in state.machines if m.kind == MachineKind.SELLER]

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:  # pragma: no cover - ortools always ships GLOP
        return UpperBound(
            ceiling=state.starting_money,
            per_hour_profit=0.0,
            bottleneck_reason="GLOP unavailable; no bound computed.",
            status="unavailable",
            notes=["OR-Tools GLOP could not be created."],
        )

    inf = solver.infinity()

    # ---- variables: runs per hour, one per (machine, recipe) --------------
    t: dict[tuple[str, str], object] = {}
    for m in state.machines:
        cap, _base, _half = _best_caps(state, m)
        for r in m.recipes:
            t[(m.id, r.id)] = solver.NumVar(0.0, inf if cap > 0 else 0.0, f"t[{m.id},{r.id}]")

    # ---- machine capacity -------------------------------------------------
    cap_cons: dict[str, object] = {}
    for m in state.machines:
        cap, _base, _half = _best_caps(state, m)
        c = solver.Constraint(-inf, float(cap), f"cap:{m.id}")
        for r in m.recipes:
            c.SetCoefficient(t[(m.id, r.id)], 1.0)
        cap_cons[m.id] = c

    # ---- flows on edges ---------------------------------------------------
    produced_items: dict[str, set[str]] = {
        m.id: {r.output_item for r in m.recipes if r.output_item} for m in state.machines
    }
    consumed_items: dict[str, set[str]] = {
        m.id: {k for r in m.recipes for k in r.inputs} for m in state.machines
    }
    flow: dict[tuple[str, str, str], object] = {}
    for e in state.edges:
        if e.src not in machines or e.dst not in machines:
            continue
        for item in produced_items[e.src] & consumed_items[e.dst]:
            flow[(e.src, e.dst, item)] = solver.NumVar(0.0, inf, f"f[{e.src},{e.dst},{item}]")

    # ---- conservation: a consumer's needs must arrive over its edges -------
    for m in state.machines:
        half = ModKind.HALF_MATERIALS in (set(m.available_mods) | set(m.installed_mods))
        for item in consumed_items[m.id]:
            c = solver.Constraint(0.0, 0.0, f"need:{m.id}:{item}")
            for r in m.recipes:
                q = r.inputs.get(item, 0)
                if q:
                    c.SetCoefficient(t[(m.id, r.id)], -(q / 2.0 if half else float(q)))
            any_edge = False
            for src in state.upstream_of(m.id):
                key = (src, m.id, item)
                if key in flow:
                    c.SetCoefficient(flow[key], 1.0)
                    any_edge = True
            if not any_edge:
                # nothing upstream makes this item: the recipes needing it are
                # pinned to zero by the equality with no inflow term.
                notes.append(f"{m.id} has no upstream source of {item!r}.")

    # ---- supply: a producer cannot ship more than it makes -----------------
    for m in state.machines:
        for item in produced_items[m.id]:
            c = solver.Constraint(-inf, 0.0, f"supply:{m.id}:{item}")
            for r in m.recipes:
                if r.output_item == item:
                    c.SetCoefficient(t[(m.id, r.id)], -float(r.output_qty))
            for dst in state.downstream_of(m.id):
                key = (m.id, dst, item)
                if key in flow:
                    c.SetCoefficient(flow[key], 1.0)

    # ---- objective: revenue - purchase cost - production cost --------------
    obj = solver.Objective()
    for m in state.machines:
        for r in m.recipes:
            var = t[(m.id, r.id)]
            if m.kind == MachineKind.SELLER:
                obj.SetCoefficient(var, float(r.output_qty) * float(r.sale_price))
            elif m.kind == MachineKind.SUPPLIER:
                obj.SetCoefficient(var, -float(r.output_qty) * float(r.purchase_cost))
            else:
                obj.SetCoefficient(var, -float(r.output_qty) * float(r.production_cost))
    obj.SetMaximization()

    status = solver.Solve()
    status_name = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
    }.get(status, "unknown")

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return UpperBound(
            ceiling=float(state.starting_money),
            per_hour_profit=0.0,
            bottleneck_reason=(
                "The steady-state LP has no profitable solution: nothing on this "
                "board can turn a supplier's items into a sale."
            ),
            status=status_name,
            depth=cf.depth,
            notes=notes + ["LP status: " + status_name],
        )

    per_hour = float(obj.Value())
    throughput: dict[str, float] = {}
    recipe_choice: dict[str, str] = {}
    for m in state.machines:
        best_r, best_v = None, 0.0
        total = 0.0
        for r in m.recipes:
            v = float(t[(m.id, r.id)].solution_value())
            total += v * float(r.output_qty)
            if v > best_v:
                best_r, best_v = r.id, v
        throughput[m.id] = round(total, 9)
        if best_r:
            recipe_choice[m.id] = best_r

    duals: dict[str, float] = {}
    for c in solver.constraints():
        try:
            d = float(c.dual_value())
        except Exception:  # pragma: no cover - defensive
            continue
        if abs(d) > 1e-9:
            duals[c.name()] = round(d, 9)

    # ---- name the binding constraint in plain language ---------------------
    bottleneck = None
    reason = ""
    cap_duals = {mid: abs(float(c.dual_value())) for mid, c in cap_cons.items()}
    cap_duals = {k: v for k, v in cap_duals.items() if v > 1e-9}
    if cap_duals:
        bottleneck = max(cap_duals, key=lambda k: cap_duals[k])
        m = machines[bottleneck]
        cap, _b, _h = _best_caps(state, m)
        reason = (
            f"{bottleneck} is the bottleneck: it is running flat out at "
            f"{cap:g} units/hour, and every extra unit it could make is worth "
            f"{cap_duals[bottleneck]:g} more money per hour. "
            f"Raising its output max (or halving its production time) is the only "
            f"thing that moves the ceiling."
        )
    elif per_hour <= 0:
        reason = (
            "No machine is a bottleneck because nothing here is profitable at "
            "steady state: the inputs cost at least as much as the sale price."
        )
    else:
        supply_duals = {k: v for k, v in duals.items() if k.startswith("supply:")}
        if supply_duals:
            key = max(supply_duals, key=lambda k: abs(supply_duals[k]))
            _, mid, item = key.split(":", 2)
            bottleneck = mid
            reason = (
                f"{mid} is the bottleneck: the chain is short of {item!r}, and "
                f"every extra unit of it is worth {abs(supply_duals[key]):g} per hour."
            )
        else:
            reason = "The LP optimum is not constrained by any single machine."

    depth = cf.depth
    horizon = int(state.horizon_hours)
    productive_hours = max(0, horizon - depth)

    # Stock already on the board can be sold during the ramp, so it has to be
    # inside the bound too.  An item is worth what a seller pays for it, or --
    # if it is only an ingredient -- what the most valuable thing it can be
    # turned into is worth, ignoring the other ingredients that recipe also
    # needs.  Optimistic on purpose: this has to stay an upper bound.
    best_price: dict[str, float] = {}
    for s in sellers:
        for r in s.recipes:
            for item in r.inputs:
                per_item = float(r.sale_price) * float(r.output_qty) / max(1, r.inputs[item])
                best_price[item] = max(best_price.get(item, 0.0), per_item)
    for _ in range(len(state.item_ids()) + 1):
        changed = False
        for m in state.machines:
            for r in m.recipes:
                if not r.output_item:
                    continue
                out_value = best_price.get(r.output_item, 0.0)
                if out_value <= 0:
                    continue
                for item, qty in r.inputs.items():
                    cand = out_value * float(r.output_qty) / max(1, int(qty))
                    if cand > best_price.get(item, 0.0) + 1e-12:
                        best_price[item] = cand
                        changed = True
        if not changed:
            break
    initial_value = 0.0
    for _mid, per_item_map in state.initial_storage.items():
        for item, qty in per_item_map.items():
            initial_value += best_price.get(item, 0.0) * qty

    ceiling = float(state.starting_money) + max(0.0, per_hour) * productive_hours + initial_value

    notes.append(
        f"Ramp-up allowance: the chain is {depth} machines deep, so the first sale "
        f"cannot land before hour {depth + 1}; the ceiling only counts "
        f"{productive_hours} of the {horizon} hours as productive."
    )
    notes.append(
        "Relaxation: fractional throughput, no storage limits, no all-or-nothing "
        "allocation, every machine already carrying its best mods but paying "
        "un-modded per-item costs.  Real play is strictly below this."
    )
    if initial_value:
        notes.append(
            f"Stock already in storage is valued at {initial_value:g} -- the best "
            f"price it could fetch after any number of conversions -- and added to "
            f"the ceiling, since it can be sold during the ramp."
        )
    if per_hour <= 0:
        notes.append(
            "Steady-state profit is not positive: the best plan may be to run "
            "nothing at all and keep the starting money."
        )

    return UpperBound(
        ceiling=ceiling,
        per_hour_profit=per_hour,
        bottleneck_machine=bottleneck,
        bottleneck_reason=reason,
        duals=duals,
        throughput=throughput,
        notes=notes,
        depth=depth,
        recipe_choice=recipe_choice,
        status=status_name,
    )
