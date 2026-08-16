"""Builder domain model.  Spec section 8.

Domain-agnostic by construction: nothing in this package says car, ship or
vehicle.  Attributes are opaque named integers; obstacles are opaque named
requirements.  All semantics live in ``BuilderRules`` (spec 5.1).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from services.core.rules.dsl import BuilderRules


class Part(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    weight: int = 0
    qty_available: int = 1
    #: attribute name -> value contributed by one copy of this part.
    attributes: dict[str, int] = Field(default_factory=dict)
    cost: int = 0
    slots: int = 1

    def attr(self, name: str) -> int:
        return self.attributes.get(name, 0)


class Obstacle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    order: int = 0
    #: attribute name -> minimum required (threshold) or amount spent (consumable).
    requires: dict[str, int] = Field(default_factory=dict)
    name: str = ""


class BuilderPuzzle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[Part] = Field(default_factory=list)
    obstacles: list[Obstacle] = Field(default_factory=list)
    rules: BuilderRules = Field(default_factory=BuilderRules)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def attribute_names(self) -> list[str]:
        """Union of every attribute mentioned by rules, parts or obstacles."""
        names: dict[str, None] = {n: None for n in self.rules.attribute_names}
        for o in self.obstacles:
            for k in o.requires:
                names.setdefault(k, None)
        for p in self.parts:
            for k in p.attributes:
                names.setdefault(k, None)
        return list(names)

    def part(self, part_id: str) -> Part:
        for p in self.parts:
            if p.id == part_id:
                return p
        raise KeyError(f"no part {part_id!r}")

    def fingerprint(self) -> str:
        """Stable id for *this puzzle instance* (spec 13)."""
        import hashlib
        import json

        payload = {
            "parts": sorted(
                (p.id, p.weight, p.qty_available, sorted(p.attributes.items()))
                for p in self.parts
            ),
            "obstacles": sorted((o.id, o.order, sorted(o.requires.items())) for o in self.obstacles),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class Build(BaseModel):
    """A concrete build.  ``counts`` maps part id -> how many copies used."""

    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int] = Field(default_factory=dict)
    weight: int = 0
    slots: int = 0
    cost: int = 0
    attributes: dict[str, int] = Field(default_factory=dict)
    obstacles_passed: int = 0

    def part_ids(self) -> list[str]:
        return sorted(self.counts)

    def key(self) -> tuple:
        return tuple(sorted(self.counts.items()))


class MinimalCore(BaseModel):
    """A minimal valid build plus the parts that may be freely added to it.

    Requirements are monotone, so every superset of ``core`` that stays inside
    the weight/slot budget is also valid.  Reporting 7 cores each extendable by
    a named free set beats forty thousand rows (spec 8.4).
    """

    model_config = ConfigDict(extra="forbid")

    core: Build
    freely_addable: list[str] = Field(default_factory=list)
    extensions_count: int = 1


class BuildReport(BaseModel):
    """Return shape of every builder solver (spec 8.4)."""

    model_config = ConfigDict(extra="forbid")

    total_valid: int = 0
    #: Present only when the objective asked for builds, capped by ``max_builds``.
    builds: list[Build] = Field(default_factory=list)
    minimal_builds: list[MinimalCore] = Field(default_factory=list)
    #: Zero-weight parts that violate nothing: count is multiplied by 2**|F|.
    free_parts: list[str] = Field(default_factory=list)
    free_multiplier: int = 1
    #: The obstacle that eliminates the most candidates.  Usually the answer
    #: the user actually wanted.
    binding_obstacle: str | None = None
    binding_obstacle_eliminated: int = 0
    obstacle_elimination: dict[str, int] = Field(default_factory=dict)
    #: Weight budget minus the weight of the lightest valid build.
    weight_slack: int | None = None
    best_build: Build | None = None
    method: str = ""
    elapsed_ms: float = 0.0
    exact: bool = True
    clipped_at: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
