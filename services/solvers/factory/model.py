"""Factory domain model.  Spec section 7.

Pure data.  No I/O, no solving.  The simulator compiles these into preallocated
numpy arrays (``sim.CompiledFactory``); nothing on the hot path touches a dict
of strings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MachineKind(str, Enum):
    SUPPLIER = "supplier"
    MAKER = "maker"
    SELLER = "seller"


class ModKind(str, Enum):
    """Maker modifications (spec 7.1).  Each raises per-item cost."""

    DOUBLE_OUTPUT_MAX = "double_output_max"
    HALF_MATERIALS = "half_materials"
    ONE_HOUR_PRODUCTION = "one_hour_production"
    DOUBLE_STORAGE_MAX = "double_storage_max"


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""


class Recipe(BaseModel):
    """A selectable recipe: input type, input quantity and sale price.

    Quantities are *per unit of output setting*.  A maker at output setting 3
    running a recipe with ``inputs={"ore": 2}`` requires 6 ore that hour and
    deposits ``3 * output_qty`` of ``output_item``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    inputs: dict[str, int] = Field(default_factory=dict)
    output_item: str | None = None
    output_qty: int = 1
    #: Sellers: money received per item sold.
    sale_price: float = 0.0
    #: Suppliers: money paid per item produced/ordered.
    purchase_cost: float = 0.0
    #: Makers: base production cost per item, before mod inflation.
    production_cost: float = 0.0


class Machine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: MachineKind
    name: str = ""
    recipes: list[Recipe] = Field(default_factory=list)
    #: Storage cap.  Sellers have no storage; the field is ignored for them.
    storage_max: int = 0
    #: Per-hour production ceiling (see flag ``output_max_meaning``).
    output_max: int = 0
    #: 1, or 2 for clock-marked machines.
    production_hours: int = 1
    #: Mods installed in the *observed* state.  Configs may differ.
    installed_mods: list[ModKind] = Field(default_factory=list)
    available_mods: list[ModKind] = Field(default_factory=list)
    #: Per-item cost increment charged by each installed mod.
    mod_cost: dict[ModKind, float] = Field(default_factory=dict)
    #: Screen position.  ``y`` breaks priority ties (topmost first);
    #: ``x``/``y`` also feed the ``priority_metric="pixels"`` hypothesis.
    x: float = 0.0
    y: float = 0.0

    def recipe(self, recipe_id: str) -> Recipe:
        for r in self.recipes:
            if r.id == recipe_id:
                return r
        raise KeyError(f"machine {self.id!r} has no recipe {recipe_id!r}")

    def default_recipe_id(self) -> str | None:
        return self.recipes[0].id if self.recipes else None


class Edge(BaseModel):
    """A directed connection: ``src`` feeds ``dst``."""

    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str


class FactoryState(BaseModel):
    """Everything read off the board.  The *world*, not the *configuration*."""

    model_config = ConfigDict(extra="forbid")

    machines: list[Machine] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    starting_money: float = 0.0
    horizon_hours: int = 24
    #: machine id -> item id -> starting quantity in that machine's storage.
    initial_storage: dict[str, dict[str, int]] = Field(default_factory=dict)
    #: Free-form provenance from extraction (capture ids, jury confidence).
    provenance: dict[str, Any] = Field(default_factory=dict)

    def machine(self, machine_id: str) -> Machine:
        for m in self.machines:
            if m.id == machine_id:
                return m
        raise KeyError(f"no machine {machine_id!r}")

    def upstream_of(self, machine_id: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == machine_id]

    def downstream_of(self, machine_id: str) -> list[str]:
        return [e.dst for e in self.edges if e.src == machine_id]

    def item_ids(self) -> list[str]:
        if self.items:
            return [i.id for i in self.items]
        seen: dict[str, None] = {}
        for m in self.machines:
            for r in m.recipes:
                for k in r.inputs:
                    seen.setdefault(k, None)
                if r.output_item:
                    seen.setdefault(r.output_item, None)
        return list(seen)

    def fingerprint(self) -> str:
        """Stable id for *this puzzle instance* (spec 13: never key rules by family)."""
        import hashlib
        import json

        payload = {
            "machines": [
                {
                    "id": m.id,
                    "kind": m.kind.value,
                    "storage_max": m.storage_max,
                    "output_max": m.output_max,
                    "production_hours": m.production_hours,
                    "recipes": [r.model_dump() for r in m.recipes],
                }
                for m in sorted(self.machines, key=lambda m: m.id)
            ],
            "edges": sorted([(e.src, e.dst) for e in self.edges]),
            "horizon": self.horizon_hours,
            "money": self.starting_money,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class MachineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_id: str
    recipe_id: str | None = None
    output_setting: int = 0
    mods: list[ModKind] = Field(default_factory=list)
    #: Endgame overrides (spec 7.6 layer 4): hour index -> output setting.
    hourly_overrides: dict[int, int] = Field(default_factory=dict)

    def output_at(self, hour: int) -> int:
        return self.hourly_overrides.get(hour, self.output_setting)


class FactoryConfig(BaseModel):
    """The knob positions.  This is what the optimizer searches over."""

    model_config = ConfigDict(extra="forbid")

    machines: dict[str, MachineConfig] = Field(default_factory=dict)

    @classmethod
    def from_state(cls, state: FactoryState) -> "FactoryConfig":
        """The configuration currently visible on the board."""
        return cls(
            machines={
                m.id: MachineConfig(
                    machine_id=m.id,
                    recipe_id=m.default_recipe_id(),
                    output_setting=m.output_max,
                    mods=list(m.installed_mods),
                )
                for m in state.machines
            }
        )

    def get(self, machine_id: str) -> MachineConfig:
        return self.machines[machine_id]

    def with_change(self, machine_id: str, **kwargs: Any) -> "FactoryConfig":
        mc = self.machines[machine_id].model_copy(update=kwargs)
        return self.model_copy(update={"machines": {**self.machines, machine_id: mc}})

    def key(self) -> tuple:
        return tuple(
            (
                mid,
                mc.recipe_id,
                mc.output_setting,
                tuple(sorted(m.value for m in mc.mods)),
                tuple(sorted(mc.hourly_overrides.items())),
            )
            for mid, mc in sorted(self.machines.items())
        )


class WarningKind(str, Enum):
    SKIPPED = "skipped"          # requirement not fully met -> produced nothing
    OVERFLOW = "overflow"        # deposit clipped at storage_max, items destroyed
    IDLE_FULL = "idle_full"      # storage full, machine idling
    NO_MONEY = "no_money"        # supplier order unaffordable
    STRANDED = "stranded"        # items in storage at horizon, worth zero
    UNREACHABLE = "unreachable"  # output can't reach a seller before the horizon


class SimWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: WarningKind
    hour: int
    machine_id: str
    detail: str = ""
    amount: float = 0.0

    def as_text(self) -> str:
        return f"h{self.hour} {self.machine_id}: {self.kind.value} {self.detail}".strip()


class MachineHourLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int
    machine_id: str
    produced: int = 0
    consumed: dict[str, int] = Field(default_factory=dict)
    deposited: int = 0
    wasted: int = 0
    revenue: float = 0.0
    cost: float = 0.0
    skipped: bool = False
    storage_after: dict[str, int] = Field(default_factory=dict)


class SimResult(BaseModel):
    """Simulation output.

    ``money_by_hour`` has length ``horizon_hours + 1``: index 0 is the money on
    hand before hour 1 runs (i.e. ``state.starting_money``) and index ``h`` is
    the money at the end of hour ``h``.  A game-reported series of 24 values
    lines up with ``money_by_hour[1:]``; ``calibrate.py`` owns that alignment.
    """

    model_config = ConfigDict(extra="forbid")

    money_by_hour: list[float] = Field(default_factory=list)
    final_money: float = 0.0
    per_machine_log: list[MachineHourLog] = Field(default_factory=list)
    warnings: list[SimWarning] = Field(default_factory=list)
    #: machine id -> item id -> quantity left when the horizon ends.
    ending_storage: dict[str, dict[str, int]] = Field(default_factory=dict)
    total_revenue: float = 0.0
    total_cost: float = 0.0

    def profit(self) -> float:
        return self.final_money - (self.money_by_hour[0] if self.money_by_hour else 0.0)
