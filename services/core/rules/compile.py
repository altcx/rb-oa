"""Rule compilation: instruction screenshots -> ``BuilderRules`` + unknowns.

The whole point of this module is that **null is a result**.  A rule the
instructions do not state is not a gap to be filled with a plausible guess; it
is a fact about the puzzle ("the game never says whether obstacles consume the
stat") that must survive all the way into solving, where ``resolve.py`` turns
it into a set of interpretations.

Every unknown ships with an :class:`Experiment`: the smallest in-game action
that tells the two options apart, phrased so the user can run it in under
fifteen seconds.  The library below is hand-written for every flag in
``BUILDER_FLAG_OPTIONS`` and ``FACTORY_FLAG_OPTIONS`` -- an LLM-invented
experiment is exactly the sort of thing that sounds executable and is not.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from services.core.extract.schemas import BuilderRulesExtraction, json_schema_for
from services.core.llm.protocol import (
    EXTRACTION_PROVIDER,
    ChatMessage,
    image_part,
    text_part,
)
from services.core.rules.dsl import (
    BUILDER_FLAG_OPTIONS,
    BUILDER_FLAG_ORDER,
    FACTORY_FLAG_OPTIONS,
    FACTORY_FLAG_ORDER,
    BuilderRules,
    Experiment,
    FactoryRuleState,
    Unknown,
)

#: Latency is irrelevant for the rule compiler (it runs once per puzzle and is
#: cached), so do not sort providers by latency -- just demand schema support.
RULE_COMPILER_PROVIDER: dict[str, Any] = {
    k: v for k, v in EXTRACTION_PROVIDER.items() if k != "sort"
}

SYSTEM_PROMPT = """\
You read a puzzle's instruction panel and turn it into a machine-readable rule
spec. You are not solving the puzzle and you are not guessing at it.

THE ONLY RULE THAT MATTERS: emit null for every field the instructions do not
EXPLICITLY state. Do not infer a rule from the puzzle's genre, from what would
be reasonable, from what similar games do, or from the layout of the board. If
the text does not say it, it is null. A null is a correct, useful answer; an
invented rule silently corrupts every downstream computation.

For every field you DO fill in, add the verbatim sentence from the instructions
that justifies it to `quoted_rule_text`, copied character for character from
the image. If you cannot quote a sentence for a field, that field must be null.

Read every panel supplied, including fine print, tooltips and footnotes.\
"""

USER_PROMPT = """\
Here are the instruction panels for one puzzle. Extract the rule spec.

Reminder: null for anything not explicitly stated, and one verbatim quote in
quoted_rule_text for every non-null field.\
"""


# ---------------------------------------------------------------------------
# Experiment library
# ---------------------------------------------------------------------------


def _exp(
    flag: str,
    options: Sequence[Any],
    instruction: str,
    observable: str,
    discriminator: dict[str, str],
    seconds: int = 15,
) -> Experiment:
    return Experiment(
        flag=flag,
        options=list(options),
        instruction=instruction,
        observable=observable,
        discriminator=dict(discriminator),
        estimated_seconds=seconds,
    )


_BUILDER_EXPERIMENTS: dict[str, Experiment] = {
    "obstacle_semantics": _exp(
        "obstacle_semantics",
        BUILDER_FLAG_OPTIONS["obstacle_semantics"],
        "Make a build whose key stat exactly equals obstacle 1's requirement, "
        "with nothing to spare, and run it against obstacle 1 then obstacle 2.",
        "Whether obstacle 2 still passes after obstacle 1 was cleared.",
        {
            "threshold": "Obstacle 2 still passes: clearing obstacle 1 spent nothing.",
            "consumable": "Obstacle 2 now fails: obstacle 1 consumed the stat.",
            "mixed": "One stat is spent and another is not — the panel shows a "
            "drop for some stats and not others.",
        },
    ),
    "aggregation": _exp(
        "aggregation",
        BUILDER_FLAG_OPTIONS["aggregation"],
        "Add a second part that also carries the key stat to a build that "
        "already has one, and read the build's total for that stat.",
        "The build's displayed total for that stat, before and after.",
        {
            "sum": "Total rises by the second part's value.",
            "min": "Total drops to the smaller of the two parts' values.",
            "max": "Total is unchanged unless the new part is larger, then it "
            "jumps to the new part's value exactly.",
        },
    ),
    "duplicates_allowed": _exp(
        "duplicates_allowed",
        BUILDER_FLAG_OPTIONS["duplicates_allowed"],
        "Pick any part already in the build and try to add a second copy of it.",
        "Whether the part's count goes to 2 or the control refuses.",
        {
            "False": "The part greys out / the add control is disabled after one copy.",
            "True": "The count increments to 2 and the weight total rises accordingly.",
        },
        seconds=8,
    ),
    "obstacle_ordering": _exp(
        "obstacle_ordering",
        BUILDER_FLAG_OPTIONS["obstacle_ordering"],
        "Submit a build that clearly fails obstacle 1 but clearly clears "
        "obstacle 2, and watch the result panel.",
        "Whether obstacle 2 is evaluated at all after obstacle 1 fails.",
        {
            "unordered": "Obstacle 2 is marked passed and obstacle 1 failed — "
            "all obstacles were checked.",
            "ordered": "The run stops at obstacle 1; obstacle 2 shows no result.",
        },
    ),
    "failure_mode": _exp(
        "failure_mode",
        BUILDER_FLAG_OPTIONS["failure_mode"],
        "Submit a build that clears every obstacle except the last one.",
        "The score / validity readout on the result panel.",
        {
            "all_must_pass": "The build reads as invalid, score 0 — one miss kills it.",
            "count_passed": "The build reads as a partial score (n-1 of n cleared).",
        },
    ),
    "objective": _exp(
        "objective",
        BUILDER_FLAG_OPTIONS["objective"],
        "Open the answer/submission box without typing anything and read what "
        "it asks for.",
        "The shape of the input the game accepts.",
        {
            "count_valid": "A single number field: it wants how many builds work.",
            "enumerate": "A list or multi-row entry: it wants the builds themselves.",
            "max_passed": "A score readout tied to obstacles cleared.",
            "min_weight": "A weight field, with weight shown as the thing to minimise.",
        },
        seconds=10,
    ),
}


_FACTORY_EXPERIMENTS: dict[str, Experiment] = {
    "two_hour_consume_timing": _exp(
        "two_hour_consume_timing",
        FACTORY_FLAG_OPTIONS["two_hour_consume_timing"],
        "Give a clock-marked (2-hour) machine exactly one batch of inputs, "
        "start it, and read its input storage at the end of hour 1.",
        "The input count in that machine's storage after one hour.",
        {
            "start_hour_1": "Inputs are already gone at the end of hour 1.",
            "start_hour_2": "Inputs are still sitting there at the end of hour 1 "
            "and disappear at the end of hour 2.",
        },
    ),
    "supplier_cost_timing": _exp(
        "supplier_cost_timing",
        FACTORY_FLAG_OPTIONS["supplier_cost_timing"],
        "Note your money, set one supplier to order, and read money again at "
        "the end of that same hour, before the goods arrive downstream.",
        "When the money leaves your balance.",
        {
            "at_order": "Money drops in the hour the order is placed.",
            "at_delivery": "Money is unchanged that hour and drops when the "
            "items actually land in storage.",
        },
    ),
    "overflow_timing": _exp(
        "overflow_timing",
        FACTORY_FLAG_OPTIONS["overflow_timing"],
        "Fill one maker's storage to its maximum, leave its downstream "
        "consumer pulling the same hour, and run a single hour.",
        "Whether any items are destroyed, and the storage level afterwards.",
        {
            "after_production": "An overflow/waste indicator fires: production "
            "landed in a full store before the downstream pull.",
            "before_pull": "No waste: the downstream machine pulled first and "
            "storage ends below the maximum.",
        },
    ),
    "priority_metric": _exp(
        "priority_metric",
        FACTORY_FLAG_OPTIONS["priority_metric"],
        "Find an hour where two machines compete for the same short input and "
        "the one further down the chain sits higher on the screen. Run one hour "
        "and see which one got fed.",
        "Which competing machine received the limited input.",
        {
            "hops": "The machine fewer hops from the supplier was fed.",
            "pixels": "The machine higher / further left on screen was fed, "
            "even though it is deeper in the chain.",
        },
        seconds=15,
    ),
    "priority_recompute": _exp(
        "priority_recompute",
        FACTORY_FLAG_OPTIONS["priority_recompute"],
        "While a contested allocation is running, change something that would "
        "reorder priority (drag a panel, or add a mod that changes chain "
        "position) and run one more hour.",
        "Whether the allocation order changes on the next hour.",
        {
            "fixed": "The same machine keeps winning: order was frozen at hour 1.",
            "per_hour": "The other machine wins the next hour: order is recomputed.",
        },
        seconds=15,
    ),
    "output_max_meaning": _exp(
        "output_max_meaning",
        FACTORY_FLAG_OPTIONS["output_max_meaning"],
        "Set one maker's output setting to its maximum while its own storage "
        "is nearly full, and run one hour.",
        "How many units it actually produced that hour.",
        {
            "per_hour_ceiling": "It produced the full output setting and the "
            "excess was clipped/wasted at deposit.",
            "separate_from_storage": "It produced only as much as the remaining "
            "storage room allowed; nothing was wasted.",
        },
    ),
    "mod_stacking": _exp(
        "mod_stacking",
        FACTORY_FLAG_OPTIONS["mod_stacking"],
        "Install a second mod on a maker that already has one and read the "
        "per-item production cost before and after.",
        "The per-item cost with one mod versus two.",
        {
            "independent": "The second mod's increment is simply added to the first.",
            "multiplicative": "The cost is larger than the sum of the two "
            "increments — the mods multiply.",
        },
    ),
    "insufficient_funds": _exp(
        "insufficient_funds",
        FACTORY_FLAG_OPTIONS["insufficient_funds"],
        "Let your money fall below the cost of one supplier's full order for a "
        "single hour and read that supplier's output.",
        "How many units the supplier delivered that hour.",
        {
            "skipped": "It delivered zero — the whole order was skipped.",
            "partial": "It delivered as many units as the remaining money covered.",
        },
    ),
    "full_storage_behavior": _exp(
        "full_storage_behavior",
        FACTORY_FLAG_OPTIONS["full_storage_behavior"],
        "Let one maker fill its storage completely while its consumer is switched "
        "off, then watch that maker for one hour with its inputs still available.",
        "Whether the upstream inputs drain and money drops while storage stays at "
        "its cap.",
        {
            "idle": "Inputs are untouched and money is flat — a full machine stops.",
            "produce_and_waste": "Inputs drain and money drops while storage stays "
            "capped — it keeps producing into a full buffer and destroys the output.",
        },
    ),
}

#: Every flag we can write an experiment for.
EXPERIMENTS: dict[str, Experiment] = {**_BUILDER_EXPERIMENTS, **_FACTORY_EXPERIMENTS}

ALL_FLAG_OPTIONS: dict[str, tuple[Any, ...]] = {
    **BUILDER_FLAG_OPTIONS,
    **FACTORY_FLAG_OPTIONS,
}


def experiment_for(flag: str, options: Sequence[Any] | None = None) -> Experiment:
    """The discriminating observation for ``flag``, narrowed to ``options``.

    Always returns an :class:`Experiment` -- an unknown flag with no library
    entry gets a generic-but-still-executable fallback rather than ``None``,
    because the caller's contract is "every unknown carries an experiment".
    """
    opts = list(options) if options is not None else list(ALL_FLAG_OPTIONS.get(flag, ()))
    base = EXPERIMENTS.get(flag)
    if base is None:
        pretty = flag.replace("_", " ")
        return _exp(
            flag,
            opts,
            f"Set up the smallest board state where '{pretty}' changes the "
            f"outcome, run exactly one step, and read the result panel.",
            f"The value the game reports for {pretty}.",
            {str(o): f"The result is consistent with {pretty} = {o}." for o in opts},
            seconds=15,
        )
    if opts and set(map(str, opts)) != set(map(str, base.options)):
        disc = {str(o): base.discriminator.get(str(o), "") for o in opts}
        return base.model_copy(update={"options": opts, "discriminator": disc})
    return base


# ---------------------------------------------------------------------------
# Unknowns
# ---------------------------------------------------------------------------


def unknowns_for(rules: BuilderRules) -> list[Unknown]:
    """Every ``None`` builder flag, highest-leverage first, with experiments."""
    out: list[Unknown] = []
    for flag in BUILDER_FLAG_ORDER:
        if getattr(rules, flag, None) is not None:
            continue
        options = list(BUILDER_FLAG_OPTIONS[flag])
        out.append(
            Unknown(
                flag=flag,
                options=options,
                reason="the instructions do not state this",
                experiment=experiment_for(flag, options),
            )
        )
    return out


def factory_unknowns(flag_state: FactoryRuleState | None = None) -> list[Unknown]:
    """Every factory flag not yet pinned by calibration or an experiment."""
    state = flag_state or FactoryRuleState()
    out: list[Unknown] = []
    for flag in FACTORY_FLAG_ORDER:
        if flag in state.resolved_by:
            continue
        options = list(FACTORY_FLAG_OPTIONS[flag])
        out.append(
            Unknown(
                flag=flag,
                options=options,
                reason="defaulted hypothesis, not confirmed by calibration",
                experiment=experiment_for(flag, options),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Extraction -> rules
# ---------------------------------------------------------------------------


class CompiledRules(BaseModel):
    """Full result; :func:`compile_builder_rules` returns the first two fields."""

    model_config = ConfigDict(extra="forbid")

    rules: BuilderRules = Field(default_factory=BuilderRules)
    unknowns: list[Unknown] = Field(default_factory=list)
    #: verbatim sentences the model quoted to justify the resolved fields
    quotes: list[str] = Field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    error: str | None = None


def rules_from_extraction(ext: BuilderRulesExtraction | dict[str, Any]) -> BuilderRules:
    """Pure: extraction payload -> ``BuilderRules``.  Nulls stay null."""
    if isinstance(ext, dict):
        ext = BuilderRulesExtraction.model_validate(ext)
    return BuilderRules(
        weight_max=ext.weight_max,
        slot_max=ext.slot_max,
        money_max=ext.money_max,
        duplicates_allowed=ext.duplicates_allowed,
        attribute_names=list(ext.attribute_names or []),
        obstacle_ordering=ext.obstacle_ordering,
        obstacle_semantics=ext.obstacle_semantics,
        aggregation=ext.aggregation,
        failure_mode=ext.failure_mode,
        objective=ext.objective,
    )


def build_messages(images: Iterable[str]) -> list[ChatMessage]:
    parts = [text_part(USER_PROMPT)]
    parts.extend(image_part(url) for url in images)
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", parts=parts),
    ]


async def compile_builder_rules_detailed(
    client: Any,
    model: str,
    images: list[str],
    *,
    timeout_s: float | None = 90.0,
    max_tokens: int | None = 1500,
) -> CompiledRules:
    if not images:
        rules = BuilderRules()
        return CompiledRules(
            rules=rules,
            unknowns=unknowns_for(rules),
            error="no instruction images supplied",
        )
    resp = await client.complete(
        model=model,
        messages=build_messages(images),
        response_format=json_schema_for(BuilderRulesExtraction),
        provider=dict(RULE_COMPILER_PROVIDER),
        max_tokens=max_tokens,
        temperature=0.0,
        timeout_s=timeout_s,
    )
    latency = float(getattr(resp, "latency_ms", 0.0) or 0.0)
    error = getattr(resp, "error", None)
    payload = getattr(resp, "parsed", None)
    if error or payload is None:
        rules = BuilderRules()
        return CompiledRules(
            rules=rules,
            unknowns=unknowns_for(rules),
            model=getattr(resp, "model", "") or model,
            latency_ms=latency,
            error=error or "rule compiler returned no parsable JSON",
        )
    try:
        ext = BuilderRulesExtraction.model_validate(payload)
    except Exception as exc:
        rules = BuilderRules()
        return CompiledRules(
            rules=rules,
            unknowns=unknowns_for(rules),
            model=getattr(resp, "model", "") or model,
            latency_ms=latency,
            error=f"schema mismatch: {exc}",
        )
    rules = rules_from_extraction(ext)
    return CompiledRules(
        rules=rules,
        unknowns=unknowns_for(rules),
        quotes=list(ext.quoted_rule_text or []),
        model=getattr(resp, "model", "") or model,
        latency_ms=latency,
    )


async def compile_builder_rules(
    client: Any, model: str, images: list[str]
) -> tuple[BuilderRules, list[Unknown]]:
    """Instruction captures -> (rules, unknowns).  Never raises."""
    out = await compile_builder_rules_detailed(client, model, images)
    return out.rules, out.unknowns
