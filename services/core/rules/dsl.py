"""Rule spec DSL.  Spec section 5.1.

Nothing in here names a car, a ship or a vehicle.  Every rule field is nullable
on purpose: ``None`` means *unresolved*, and unresolved is a first-class state
that propagates all the way into solving (see ``rules/resolve.py``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Builder rules (section 5.1)
# --------------------------------------------------------------------------

ObstacleOrdering = Literal["ordered", "unordered"]
ObstacleSemantics = Literal["threshold", "consumable", "mixed"]
Aggregation = Literal["sum", "min", "max"]
FailureMode = Literal["all_must_pass", "count_passed"]
BuilderObjective = Literal["count_valid", "enumerate", "max_passed", "min_weight"]


class BuilderRules(BaseModel):
    """Spans the plausible builder-variant space.  Extraction fills it in."""

    model_config = ConfigDict(extra="forbid")

    weight_max: int | None = None
    slot_max: int | None = None
    money_max: int | None = None
    duplicates_allowed: bool | None = None
    attribute_names: list[str] = Field(default_factory=list)
    obstacle_ordering: ObstacleOrdering | None = None
    obstacle_semantics: ObstacleSemantics | None = None
    aggregation: Aggregation | None = None
    failure_mode: FailureMode | None = None
    objective: BuilderObjective | None = None

    # ---- unresolved-flag machinery -------------------------------------

    #: Option sets for every flag that can be unresolved, ordered so that the
    #: highest-leverage flag comes first (spec 5.1: semantics > aggregation >
    #: duplicates).  ``resolve.py`` walks this in order.
    OPTIONS: dict[str, tuple[Any, ...]] = {}  # populated below (class attr, not a field)

    def unresolved(self) -> list[str]:
        """Names of flags that are still ``None``, highest-leverage first."""
        return [f for f in BUILDER_FLAG_ORDER if getattr(self, f) is None]

    def with_flag(self, flag: str, value: Any) -> "BuilderRules":
        return self.model_copy(update={flag: value})

    def fingerprint_fields(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=False)


#: Highest-leverage first.  ``obstacle_semantics`` changes whether order matters
#: at all; ``aggregation`` changes which parts are substitutes; then duplicates
#: turns subset counting into multiset counting.
BUILDER_FLAG_ORDER: tuple[str, ...] = (
    "obstacle_semantics",
    "aggregation",
    "duplicates_allowed",
    "obstacle_ordering",
    "failure_mode",
    "objective",
)

BUILDER_FLAG_OPTIONS: dict[str, tuple[Any, ...]] = {
    "obstacle_semantics": ("threshold", "consumable", "mixed"),
    "aggregation": ("sum", "min", "max"),
    "duplicates_allowed": (False, True),
    "obstacle_ordering": ("unordered", "ordered"),
    "failure_mode": ("all_must_pass", "count_passed"),
    "objective": ("count_valid", "enumerate", "max_passed", "min_weight"),
}


# --------------------------------------------------------------------------
# Factory rule flags (section 7.2)
# --------------------------------------------------------------------------

TwoHourConsumeTiming = Literal["start_hour_1", "start_hour_2"]
SupplierCostTiming = Literal["at_order", "at_delivery"]
OverflowTiming = Literal["before_pull", "after_production"]
PriorityMetric = Literal["hops", "pixels"]
PriorityRecompute = Literal["fixed", "per_hour"]
OutputMaxMeaning = Literal["per_hour_ceiling", "separate_from_storage"]
ModStacking = Literal["multiplicative", "independent"]
InsufficientFunds = Literal["skipped", "partial"]
FullStorageBehavior = Literal["idle", "produce_and_waste"]


class RuleFlags(BaseModel):
    """The eight unstated factory rules, with the spec's defaults.

    Defaults are *hypotheses*, not facts.  ``calibrate.py`` pins them against a
    real recorded run and persists the result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    two_hour_consume_timing: TwoHourConsumeTiming = "start_hour_1"
    supplier_cost_timing: SupplierCostTiming = "at_order"
    overflow_timing: OverflowTiming = "after_production"
    priority_metric: PriorityMetric = "hops"
    priority_recompute: PriorityRecompute = "fixed"
    output_max_meaning: OutputMaxMeaning = "per_hour_ceiling"
    mod_stacking: ModStacking = "independent"
    insufficient_funds: InsufficientFunds = "skipped"
    #: Not in the spec's table of eight.  The simulator cannot avoid taking a
    #: position on it, and the two positions differ in money (produce_and_waste
    #: burns input and production cost for nothing), so it is a flag rather than
    #: a comment in a header.
    full_storage_behavior: FullStorageBehavior = "idle"


FACTORY_FLAG_ORDER: tuple[str, ...] = (
    "two_hour_consume_timing",
    "supplier_cost_timing",
    "overflow_timing",
    "priority_metric",
    "priority_recompute",
    "output_max_meaning",
    "mod_stacking",
    "insufficient_funds",
    "full_storage_behavior",
)

FACTORY_FLAG_OPTIONS: dict[str, tuple[Any, ...]] = {
    "two_hour_consume_timing": ("start_hour_1", "start_hour_2"),
    "supplier_cost_timing": ("at_order", "at_delivery"),
    "overflow_timing": ("after_production", "before_pull"),
    "priority_metric": ("hops", "pixels"),
    "priority_recompute": ("fixed", "per_hour"),
    "output_max_meaning": ("per_hour_ceiling", "separate_from_storage"),
    "mod_stacking": ("independent", "multiplicative"),
    "insufficient_funds": ("skipped", "partial"),
    "full_storage_behavior": ("idle", "produce_and_waste"),
}


class FactoryRuleState(BaseModel):
    """Which factory flags have been *resolved* vs merely defaulted."""

    model_config = ConfigDict(extra="forbid")

    flags: RuleFlags = Field(default_factory=RuleFlags)
    #: flag name -> how it was pinned ("calibration:run-3", "experiment:e1", ...)
    resolved_by: dict[str, str] = Field(default_factory=dict)

    def unresolved(self) -> list[str]:
        return [f for f in FACTORY_FLAG_ORDER if f not in self.resolved_by]


# --------------------------------------------------------------------------
# Experiments and uncertainty (sections 5.2, 5.3)
# --------------------------------------------------------------------------


class Experiment(BaseModel):
    """The smallest in-game observation that discriminates between options.

    ``instruction`` must be executable by a human in under fifteen seconds.
    """

    model_config = ConfigDict(extra="forbid")

    flag: str
    options: list[Any]
    instruction: str
    observable: str = Field(description="What to look at once the action is done.")
    #: option value -> what you would see if that option is the true rule
    discriminator: dict[str, str] = Field(default_factory=dict)
    estimated_seconds: int = 15


class Unknown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: str
    options: list[Any]
    reason: str = ""
    experiment: Experiment | None = None


class Action(BaseModel):
    """One concrete UI action.  Never a strategy sentence."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(description="Machine id / part id the user must touch.")
    setting: str = Field(description="Which control, e.g. 'output', 'recipe', 'mod'.")
    value: Any = Field(description="The new value to set.")
    reason: str = ""

    def key(self) -> tuple[str, str, str]:
        return (self.target, self.setting, repr(self.value))

    def as_text(self) -> str:
        return f"{self.target}: set {self.setting} -> {self.value}"


class InterpretationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignment: dict[str, Any]
    objective: float | None = None
    actions: list[Action] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class DivergentGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: str = Field(description="The flag that drives this disagreement.")
    by_option: dict[str, list[Action]] = Field(default_factory=dict)


class UncertainResult(BaseModel):
    """Return type of ``solve_all_interpretations`` (spec 5.3)."""

    model_config = ConfigDict(extra="forbid")

    interpretations: list[InterpretationResult] = Field(default_factory=list)
    consensus_actions: list[Action] = Field(default_factory=list)
    divergent_actions: list[DivergentGroup] = Field(default_factory=list)
    highest_leverage_unknown: Unknown | None = None
    value_of_information: float = 0.0
    combinations_tried: int = 0
    truncated: bool = False
    notes: list[str] = Field(default_factory=list)


class PuzzleFamily(str, Enum):
    FACTORY = "factory"
    BUILDER = "builder"
