"""Extraction wire schemas.  Single source of truth for what the vision models
are asked to return, and for how tiles reassemble into solver state.

Two rules run through this whole file:

1.  Every property carries a ``description``, and every description tells the
    model to emit ``null`` rather than guess (spec 6.2, 13).
2.  Emitted JSON Schema is *strict*: ``additionalProperties: false`` and every
    property listed in ``required`` (OpenAI/OpenRouter strict mode requires
    this; nullability is expressed with a union type, not by omission).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from services.core.rules.dsl import BuilderRules
from services.solvers.builder.model import BuilderPuzzle, Obstacle, Part
from services.solvers.factory.model import (
    Edge,
    FactoryState,
    Item,
    Machine,
    MachineKind,
    ModKind,
    Recipe,
)

NULL_HINT = " Emit null if this is not legible in the image; never guess."


# ---------------------------------------------------------------------------
# Factory extraction
# ---------------------------------------------------------------------------


class RecipeExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, description="Recipe identifier or its exact label." + NULL_HINT)
    name: str | None = Field(default=None, description="Human label shown in the UI." + NULL_HINT)
    inputs: dict[str, int] | None = Field(
        default=None,
        description=(
            "Materials needed per unit of output, as item name -> quantity. "
            "Empty object for a supplier that consumes nothing." + NULL_HINT
        ),
    )
    output_item: str | None = Field(
        default=None, description="Item name this recipe produces." + NULL_HINT
    )
    output_qty: int | None = Field(
        default=None, description="Units produced per production cycle." + NULL_HINT
    )
    sale_price: float | None = Field(
        default=None, description="Money received per item sold (sellers only)." + NULL_HINT
    )
    purchase_cost: float | None = Field(
        default=None, description="Money paid per item ordered (suppliers only)." + NULL_HINT
    )
    production_cost: float | None = Field(
        default=None, description="Base production cost per item (makers only)." + NULL_HINT
    )


class MachinePanelExtraction(BaseModel):
    """One machine tile.  Tiles are extracted concurrently (spec 6.2)."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, description="Machine identifier or on-screen label." + NULL_HINT)
    kind: Literal["supplier", "maker", "seller"] | None = Field(
        default=None, description="Machine kind." + NULL_HINT
    )
    name: str | None = Field(default=None, description="Displayed name." + NULL_HINT)
    recipes: list[RecipeExtraction] | None = Field(
        default=None, description="Every selectable recipe visible in the panel." + NULL_HINT
    )
    selected_recipe_id: str | None = Field(
        default=None, description="Which recipe is currently selected." + NULL_HINT
    )
    output_setting: int | None = Field(
        default=None, description="Current output/production setting." + NULL_HINT
    )
    output_max: int | None = Field(
        default=None, description="Maximum output setting allowed." + NULL_HINT
    )
    storage_max: int | None = Field(
        default=None, description="Storage capacity. Null or 0 for sellers." + NULL_HINT
    )
    production_hours: int | None = Field(
        default=None,
        description=(
            "Hours per production cycle. 2 when the panel shows a clock marker, "
            "otherwise 1." + NULL_HINT
        ),
    )
    installed_mods: list[str] | None = Field(
        default=None,
        description=(
            "Modifications currently installed. Use exactly: double_output_max, "
            "half_materials, one_hour_production, double_storage_max." + NULL_HINT
        ),
    )
    current_storage: dict[str, int] | None = Field(
        default=None, description="Items presently in this machine's storage." + NULL_HINT
    )
    x: float | None = Field(default=None, description="Panel centre x in pixels." + NULL_HINT)
    y: float | None = Field(default=None, description="Panel centre y in pixels." + NULL_HINT)


class TopologyExtraction(BaseModel):
    """One cheap connections-only call over the whole board (spec 6.2)."""

    model_config = ConfigDict(extra="forbid")

    edges: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Directed connections as objects with keys 'src' and 'dst', using the "
            "machine labels shown on screen. Material flows src -> dst." + NULL_HINT
        )
    )


class FactoryHUDExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    money: float | None = Field(default=None, description="Current money on hand." + NULL_HINT)
    horizon_hours: int | None = Field(
        default=None, description="Total hours in the run." + NULL_HINT
    )
    hour_now: int | None = Field(default=None, description="Current hour, if shown." + NULL_HINT)


class FactoryExtraction(BaseModel):
    """The merged result of all factory tiles plus topology."""

    model_config = ConfigDict(extra="forbid")

    hud: FactoryHUDExtraction | None = None
    machines: list[MachinePanelExtraction] = Field(default_factory=list)
    topology: TopologyExtraction | None = None


# ---------------------------------------------------------------------------
# Builder extraction
# ---------------------------------------------------------------------------


class PartExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, description="Part identifier or its exact label." + NULL_HINT)
    name: str | None = Field(default=None, description="Displayed name." + NULL_HINT)
    weight: int | None = Field(default=None, description="Weight of one copy." + NULL_HINT)
    qty_available: int | None = Field(
        default=None, description="How many copies are available. 1 if not shown." + NULL_HINT
    )
    attributes: dict[str, int] | None = Field(
        default=None,
        description=(
            "Every named stat on the part card as name -> integer value. Use the "
            "exact on-screen stat names, lowercased." + NULL_HINT
        ),
    )
    cost: int | None = Field(default=None, description="Purchase cost, if shown." + NULL_HINT)


class ObstacleExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, description="Obstacle identifier or label." + NULL_HINT)
    order: int | None = Field(
        default=None, description="1-based position in the obstacle sequence." + NULL_HINT
    )
    requires: dict[str, int] | None = Field(
        default=None,
        description=(
            "Requirement as stat name -> integer, using the same stat names as the "
            "part cards." + NULL_HINT
        ),
    )
    name: str | None = Field(default=None, description="Displayed name." + NULL_HINT)


class BuilderRulesExtraction(BaseModel):
    """Read off the instructions panel.  Every field may be null; null means the
    instructions did not state it, which is a *result*, not a failure."""

    model_config = ConfigDict(extra="forbid")

    weight_max: int | None = Field(default=None, description="Total weight budget." + NULL_HINT)
    slot_max: int | None = Field(default=None, description="Maximum part count." + NULL_HINT)
    money_max: int | None = Field(default=None, description="Spend budget." + NULL_HINT)
    duplicates_allowed: bool | None = Field(
        default=None,
        description="True only if the text explicitly permits using a part more than once."
        + NULL_HINT,
    )
    attribute_names: list[str] | None = Field(
        default=None, description="Stat names the instructions mention." + NULL_HINT
    )
    obstacle_ordering: Literal["ordered", "unordered"] | None = Field(
        default=None,
        description="'ordered' only if the text says obstacles are faced in sequence." + NULL_HINT,
    )
    obstacle_semantics: Literal["threshold", "consumable", "mixed"] | None = Field(
        default=None,
        description=(
            "'threshold' if an obstacle merely checks a stat, 'consumable' if passing "
            "spends it. Null unless the text is explicit." + NULL_HINT
        ),
    )
    aggregation: Literal["sum", "min", "max"] | None = Field(
        default=None, description="How part stats combine across a build." + NULL_HINT
    )
    failure_mode: Literal["all_must_pass", "count_passed"] | None = Field(
        default=None, description="Whether every obstacle must be cleared." + NULL_HINT
    )
    objective: Literal["count_valid", "enumerate", "max_passed", "min_weight"] | None = Field(
        default=None, description="What the puzzle asks you to produce." + NULL_HINT
    )
    quoted_rule_text: list[str] | None = Field(
        default=None,
        description="Verbatim sentences from the instructions that state a rule." + NULL_HINT,
    )


class BuilderExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[PartExtraction] = Field(default_factory=list)
    obstacles: list[ObstacleExtraction] = Field(default_factory=list)
    rules: BuilderRulesExtraction | None = None


# ---------------------------------------------------------------------------
# Strict JSON Schema emission
# ---------------------------------------------------------------------------


def _strictify(node: Any) -> Any:
    """Make a pydantic-generated schema satisfy OpenAI strict mode."""
    if isinstance(node, list):
        return [_strictify(n) for n in node]
    if not isinstance(node, dict):
        return node
    node = {k: _strictify(v) for k, v in node.items()}
    if node.get("type") == "object" or "properties" in node:
        node.setdefault("additionalProperties", False)
        props = node.get("properties")
        if isinstance(props, dict):
            # strict mode: everything required; optionality is expressed by a
            # null union, which is exactly what we want the model to emit.
            node["required"] = list(props)
    node.pop("default", None)
    return node


def json_schema_for(model: type[BaseModel], name: str | None = None) -> dict[str, Any]:
    """Return an OpenRouter ``response_format`` payload for ``model``."""
    schema = _strictify(model.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or model.__name__,
            "strict": True,
            "schema": schema,
        },
    }


# ---------------------------------------------------------------------------
# Flatten / unflatten (jury voting + "34 fields agreed" UI row)
# ---------------------------------------------------------------------------


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested extraction into dotted field paths.

    Lists of objects key by the object's ``id`` when present so that two models
    listing machines in different orders still vote on the same fields.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = None
            if isinstance(v, dict):
                key = v.get("id") or v.get("name")
            if isinstance(v, BaseModel):
                key = getattr(v, "id", None) or getattr(v, "name", None)
            out.update(flatten(v, f"{prefix}[{key if key else i}]"))
    else:
        out[prefix] = obj
    return out


def unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`flatten` for dict/list structures."""
    import re

    root: dict[str, Any] = {}
    token = re.compile(r"([^.\[\]]+)|\[([^\]]*)\]")
    for path, value in flat.items():
        parts: list[tuple[str, bool]] = []
        for m in token.finditer(path):
            if m.group(1) is not None:
                parts.append((m.group(1), False))
            else:
                parts.append((m.group(2), True))
        cursor: Any = root
        for i, (name, is_index) in enumerate(parts):
            last = i == len(parts) - 1
            if is_index:
                container = cursor.setdefault("__list__", {}) if isinstance(cursor, dict) else cursor
                if last:
                    container[name] = value
                else:
                    cursor = container.setdefault(name, {})
            else:
                if last:
                    cursor[name] = value
                else:
                    cursor = cursor.setdefault(name, {})
    return _collapse_lists(root)


def _collapse_lists(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    if set(node) == {"__list__"}:
        return [_collapse_lists(v) for v in node["__list__"].values()]
    out = {}
    for k, v in node.items():
        if isinstance(v, dict) and set(v) == {"__list__"}:
            out[k] = [_collapse_lists(x) for x in v["__list__"].values()]
        else:
            out[k] = _collapse_lists(v)
    return out


# ---------------------------------------------------------------------------
# Assembly: extraction -> solver state
# ---------------------------------------------------------------------------

_MOD_ALIASES = {
    "2x output max": ModKind.DOUBLE_OUTPUT_MAX,
    "double output max": ModKind.DOUBLE_OUTPUT_MAX,
    "double_output_max": ModKind.DOUBLE_OUTPUT_MAX,
    "half materials needed": ModKind.HALF_MATERIALS,
    "half materials": ModKind.HALF_MATERIALS,
    "half_materials": ModKind.HALF_MATERIALS,
    "1 hour production time": ModKind.ONE_HOUR_PRODUCTION,
    "one hour production": ModKind.ONE_HOUR_PRODUCTION,
    "one_hour_production": ModKind.ONE_HOUR_PRODUCTION,
    "2x storage max": ModKind.DOUBLE_STORAGE_MAX,
    "double storage max": ModKind.DOUBLE_STORAGE_MAX,
    "double_storage_max": ModKind.DOUBLE_STORAGE_MAX,
}


def parse_mod(raw: str) -> ModKind | None:
    return _MOD_ALIASES.get(raw.strip().lower())


def to_factory_state(
    ext: FactoryExtraction,
    *,
    default_horizon: int = 24,
) -> tuple[FactoryState, list[str]]:
    """Assemble a :class:`FactoryState`.  Returns (state, unresolved field paths).

    Fields still ``None`` after the jury are reported, not defaulted silently.
    """
    missing: list[str] = []
    machines: list[Machine] = []
    for mp in ext.machines:
        if not mp.id or not mp.kind:
            missing.append(f"machines[{mp.id or '?'}].id_or_kind")
            continue
        recipes: list[Recipe] = []
        for r in mp.recipes or []:
            if not r.id:
                missing.append(f"machines[{mp.id}].recipes[?].id")
                continue
            recipes.append(
                Recipe(
                    id=r.id,
                    name=r.name or "",
                    inputs=r.inputs or {},
                    output_item=r.output_item,
                    output_qty=r.output_qty if r.output_qty is not None else 1,
                    sale_price=r.sale_price or 0.0,
                    purchase_cost=r.purchase_cost or 0.0,
                    production_cost=r.production_cost or 0.0,
                )
            )
        for field in ("output_max", "storage_max", "production_hours"):
            if getattr(mp, field) is None:
                missing.append(f"machines[{mp.id}].{field}")
        mods = [m for m in (parse_mod(x) for x in (mp.installed_mods or [])) if m]
        machines.append(
            Machine(
                id=mp.id,
                kind=MachineKind(mp.kind),
                name=mp.name or mp.id,
                recipes=recipes,
                storage_max=mp.storage_max or 0,
                output_max=mp.output_max or 0,
                production_hours=mp.production_hours or 1,
                installed_mods=mods,
                available_mods=list(ModKind),
                x=mp.x or 0.0,
                y=mp.y or 0.0,
            )
        )

    edges = [
        Edge(src=e["src"], dst=e["dst"])
        for e in ((ext.topology.edges if ext.topology else None) or [])
        if e.get("src") and e.get("dst")
    ]
    if ext.topology is None or ext.topology.edges is None:
        missing.append("topology.edges")

    item_ids: dict[str, None] = {}
    for m in machines:
        for r in m.recipes:
            for k in r.inputs:
                item_ids.setdefault(k, None)
            if r.output_item:
                item_ids.setdefault(r.output_item, None)

    money = ext.hud.money if ext.hud else None
    if money is None:
        missing.append("hud.money")

    state = FactoryState(
        machines=machines,
        edges=edges,
        items=[Item(id=i, name=i) for i in item_ids],
        starting_money=money or 0.0,
        horizon_hours=(ext.hud.horizon_hours if ext.hud and ext.hud.horizon_hours else default_horizon),
        initial_storage={
            mp.id: (mp.current_storage or {}) for mp in ext.machines if mp.id and mp.current_storage
        },
    )
    return state, missing


def to_builder_puzzle(ext: BuilderExtraction) -> tuple[BuilderPuzzle, list[str]]:
    missing: list[str] = []
    parts: list[Part] = []
    for pe in ext.parts:
        if not pe.id:
            missing.append("parts[?].id")
            continue
        if pe.weight is None:
            missing.append(f"parts[{pe.id}].weight")
        parts.append(
            Part(
                id=pe.id,
                name=pe.name or pe.id,
                weight=pe.weight or 0,
                qty_available=pe.qty_available if pe.qty_available is not None else 1,
                attributes=pe.attributes or {},
                cost=pe.cost or 0,
            )
        )
    obstacles: list[Obstacle] = []
    for i, oe in enumerate(ext.obstacles):
        if not oe.id:
            missing.append("obstacles[?].id")
            continue
        if oe.requires is None:
            missing.append(f"obstacles[{oe.id}].requires")
        obstacles.append(
            Obstacle(
                id=oe.id,
                order=oe.order if oe.order is not None else i + 1,
                requires=oe.requires or {},
                name=oe.name or oe.id,
            )
        )
    re_ = ext.rules
    rules = BuilderRules(
        weight_max=re_.weight_max if re_ else None,
        slot_max=re_.slot_max if re_ else None,
        money_max=re_.money_max if re_ else None,
        duplicates_allowed=re_.duplicates_allowed if re_ else None,
        attribute_names=(re_.attribute_names if re_ and re_.attribute_names else []),
        obstacle_ordering=re_.obstacle_ordering if re_ else None,
        obstacle_semantics=re_.obstacle_semantics if re_ else None,
        aggregation=re_.aggregation if re_ else None,
        failure_mode=re_.failure_mode if re_ else None,
        objective=re_.objective if re_ else None,
    )
    return BuilderPuzzle(parts=parts, obstacles=obstacles, rules=rules), missing
