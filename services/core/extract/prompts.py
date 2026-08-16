"""Extraction prompts (spec 6.2, 13).

Three rules, repeated in every prompt because models forget the one you say
once:

* **Emit null rather than guess.**  A null costs the user one review keystroke;
  a hallucinated number costs the run.  The prompts say exactly that.
* **Read only what is inside the crop.**  Tiles exist so a neighbouring panel's
  numbers cannot bleed into this answer.
* **Short.**  Prompt tokens are prefill latency on every one of the ~20 calls a
  board costs.  The schema already documents the fields; the prompt only adds
  what the schema cannot say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from services.core.extract.schemas import (
    BuilderRulesExtraction,
    FactoryHUDExtraction,
    MachinePanelExtraction,
    ObstacleExtraction,
    PartExtraction,
    TopologyExtraction,
)
from services.core.llm.protocol import ChatMessage, image_part, text_part

# --------------------------------------------------------------------------
# Panel wire models that schemas.py does not need but the wire does.
# --------------------------------------------------------------------------


class PartsPanelExtraction(BaseModel):
    """Several part cards read from one panel crop."""

    model_config = ConfigDict(extra="forbid")

    parts: list[PartExtraction] = Field(
        description="Every part card fully visible in this crop. Emit null for any "
        "field you cannot read; never guess."
    )


class ObstaclesPanelExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obstacles: list[ObstacleExtraction] = Field(
        description="Every obstacle listed in this crop, in the order shown. Emit "
        "null for any field you cannot read; never guess."
    )


# --------------------------------------------------------------------------
# Shared rules
# --------------------------------------------------------------------------

NULL_RULE = (
    "If a value is not clearly legible, return null for it. Never guess and never "
    "infer a plausible number: a null costs the user one review keystroke, a wrong "
    "number costs the whole run."
)

CROP_RULE = (
    "Read ONLY what is inside this crop. Ignore anything cut off at the edges and "
    "anything you assume about panels you cannot see."
)

SYSTEM = (
    "You read game UI panels and return JSON matching the given schema exactly. "
    + NULL_RULE
    + " "
    + CROP_RULE
    + " No prose, no explanation."
)


def _messages(task: str, data_url: str, *, detail: str = "high") -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(
            role="user",
            parts=[image_part(data_url, detail=detail), text_part(task)],
        ),
    ]


def _hint(hint: str | None) -> str:
    return f" This crop is labelled {hint!r} on screen." if hint else ""


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def factory_machine_prompt(data_url: str, *, hint: str | None = None) -> list[ChatMessage]:
    task = (
        "One machine panel."
        + _hint(hint)
        + " Read its id/label, kind (supplier/maker/seller), every selectable recipe "
        "with its inputs and prices, the selected recipe, output setting and max, "
        "storage max, production hours (2 only if a clock marker is shown, else 1), "
        "installed mods, and current storage. "
        + NULL_RULE
    )
    return _messages(task, data_url)


def factory_topology_prompt(
    data_url: str, *, machine_ids: Sequence[str] | None = None, hint: str | None = None
) -> list[ChatMessage]:
    known = (
        f" The machines on this board are: {', '.join(machine_ids)}. Use exactly these ids."
        if machine_ids
        else ""
    )
    task = (
        "Whole board, low resolution: read CONNECTIONS ONLY. List every arrow/pipe as "
        "{'src','dst'} with material flowing src -> dst." + known + " Do not read any "
        "numbers from this image. " + NULL_RULE
    )
    return _messages(task, data_url, detail="low")


def factory_hud_prompt(data_url: str, *, hint: str | None = None) -> list[ChatMessage]:
    task = (
        "The HUD strip." + _hint(hint) + " Read money on hand, the run's total hours, "
        "and the current hour if shown. " + NULL_RULE
    )
    return _messages(task, data_url)


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def builder_part_prompt(data_url: str, *, hint: str | None = None) -> list[ChatMessage]:
    task = (
        "One part card." + _hint(hint) + " Read its id/label, name, weight, quantity "
        "available, cost, and every named stat as name -> integer using the exact "
        "on-screen stat names lowercased. " + NULL_RULE
    )
    return _messages(task, data_url)


def builder_parts_prompt(data_url: str, *, hint: str | None = None) -> list[ChatMessage]:
    task = (
        "A panel of part cards." + _hint(hint) + " For each card fully visible, read "
        "id/label, name, weight, quantity available, cost, and every named stat as "
        "name -> integer using the exact on-screen stat names lowercased. Skip cards "
        "that are cut off. " + NULL_RULE
    )
    return _messages(task, data_url)


def builder_obstacles_prompt(data_url: str, *, hint: str | None = None) -> list[ChatMessage]:
    task = (
        "The obstacle list." + _hint(hint) + " For each obstacle read id/label, its "
        "1-based position in the list, and its requirements as stat name -> integer, "
        "using the same stat names the part cards use. " + NULL_RULE
    )
    return _messages(task, data_url)


def builder_instructions_prompt(data_url: str, *, hint: str | None = None) -> list[ChatMessage]:
    task = (
        "The instructions panel." + _hint(hint) + " Fill a field ONLY if the text "
        "states it outright. Do not infer standard puzzle conventions: a null here "
        "means the instructions were silent, which is a result, not a failure. Quote "
        "verbatim every sentence that states a rule. " + NULL_RULE
    )
    return _messages(task, data_url)


# --------------------------------------------------------------------------
# Delta (spec 6.2: re-read only what changed)
# --------------------------------------------------------------------------

DELTA_SYSTEM = (
    "You compare a previously confirmed state against a fresh screenshot and report "
    "ONLY the fields whose value changed. " + NULL_RULE + " " + CROP_RULE + " "
    "Use the exact dotted field paths from the previous state; never invent a path. "
    "An unchanged field must not appear in your answer."
)


def delta_prompt(
    data_url: str, previous_flat: dict[str, Any], *, hint: str | None = None
) -> list[ChatMessage]:
    import json

    known = json.dumps(previous_flat, sort_keys=True, default=str)
    task = (
        "Previously confirmed state (dotted path -> value):\n"
        + known
        + "\n\nReturn only the paths whose on-screen value differs from the above."
        + _hint(hint)
        + " If nothing changed, return an empty list. "
        + NULL_RULE
    )
    return [
        ChatMessage(role="system", content=DELTA_SYSTEM),
        ChatMessage(role="user", parts=[image_part(data_url), text_part(task)]),
    ]


# --------------------------------------------------------------------------
# Target registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptTarget:
    name: str
    schema: type[BaseModel]
    build: Callable[..., list[ChatMessage]]
    #: where the merged result lands inside the puzzle-level extraction
    slot: str
    many: bool = False
    puzzle: str = "factory"


TARGETS: dict[str, PromptTarget] = {
    "factory_machine": PromptTarget(
        "factory_machine", MachinePanelExtraction, factory_machine_prompt, "machines", True
    ),
    "factory_topology": PromptTarget(
        "factory_topology", TopologyExtraction, factory_topology_prompt, "topology"
    ),
    "factory_hud": PromptTarget(
        "factory_hud", FactoryHUDExtraction, factory_hud_prompt, "hud"
    ),
    "builder_part": PromptTarget(
        "builder_part", PartExtraction, builder_part_prompt, "parts", True, puzzle="builder"
    ),
    "builder_parts": PromptTarget(
        "builder_parts", PartsPanelExtraction, builder_parts_prompt, "parts", True, puzzle="builder"
    ),
    "builder_obstacles": PromptTarget(
        "builder_obstacles",
        ObstaclesPanelExtraction,
        builder_obstacles_prompt,
        "obstacles",
        True,
        puzzle="builder",
    ),
    "builder_instructions": PromptTarget(
        "builder_instructions",
        BuilderRulesExtraction,
        builder_instructions_prompt,
        "rules",
        puzzle="builder",
    ),
}


def target_for(name: str) -> PromptTarget:
    try:
        return TARGETS[name]
    except KeyError:
        raise KeyError(f"unknown extraction target {name!r}; have {sorted(TARGETS)}") from None


def schema_for_target(name: str) -> type[BaseModel]:
    return target_for(name).schema


def build_messages(name: str, data_url: str, **kw: Any) -> list[ChatMessage]:
    return target_for(name).build(data_url, **kw)


__all__ = [
    "NULL_RULE",
    "CROP_RULE",
    "SYSTEM",
    "DELTA_SYSTEM",
    "PartsPanelExtraction",
    "ObstaclesPanelExtraction",
    "PromptTarget",
    "TARGETS",
    "factory_machine_prompt",
    "factory_topology_prompt",
    "factory_hud_prompt",
    "builder_part_prompt",
    "builder_parts_prompt",
    "builder_obstacles_prompt",
    "builder_instructions_prompt",
    "delta_prompt",
    "target_for",
    "schema_for_target",
    "build_messages",
]
