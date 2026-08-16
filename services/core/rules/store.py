"""Persist resolved rules so a resolution is paid for once.

CRITICAL, and the known way this goes wrong: the cache is keyed by **puzzle
instance fingerprint**, never by puzzle family.  Two builder puzzles are both
"builder"; they are not the same puzzle, and a rule resolved by experiment on
one of them says nothing about the other.  Keying by family silently poisons
every later run with a stale answer that *looks* confirmed.

File layout, one file per family at ``data/rules/{family}.json``::

    {
      "family": "builder",
      "instances": {
        "<fingerprint>": {
          "flags": {...},
          "resolved_by": {"obstacle_semantics": "experiment:e1"},
          "unknowns": ["aggregation"],
          "notes": [...],
          "updated_at": 1710000000.0
        }
      }
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from services.core.rules.dsl import BuilderRules, FactoryRuleState, RuleFlags

REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_DIR = REPO_ROOT / "data" / "rules"


class InstanceRules(BaseModel):
    """What we know about one specific puzzle instance."""

    model_config = ConfigDict(extra="allow")

    fingerprint: str = ""
    #: flag name -> resolved value
    flags: dict[str, Any] = Field(default_factory=dict)
    #: flag name -> how it was pinned ("experiment:obstacle_semantics",
    #: "calibration:run-3", "instructions")
    resolved_by: dict[str, str] = Field(default_factory=dict)
    #: flags still unknown for this instance
    unknowns: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    updated_at: float = 0.0

    def merged_with(self, other: "InstanceRules") -> "InstanceRules":
        """``other`` wins on conflict -- it is the newer observation."""
        return InstanceRules(
            fingerprint=other.fingerprint or self.fingerprint,
            flags={**self.flags, **other.flags},
            resolved_by={**self.resolved_by, **other.resolved_by},
            unknowns=[u for u in other.unknowns or self.unknowns],
            notes=list(dict.fromkeys([*self.notes, *other.notes])),
            updated_at=max(self.updated_at, other.updated_at, time.time()),
        )


class RuleFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    family: str = ""
    instances: dict[str, InstanceRules] = Field(default_factory=dict)


def path_for(family: str, root: Path | None = None) -> Path:
    return (Path(root) if root is not None else RULES_DIR) / f"{family}.json"


def load(family: str, root: Path | None = None) -> RuleFile:
    p = path_for(family, root)
    if not p.exists():
        return RuleFile(family=family)
    try:
        return RuleFile.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return RuleFile(family=family)


def save(rule_file: RuleFile, root: Path | None = None) -> Path:
    p = path_for(rule_file.family, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(rule_file.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def merge(
    family: str,
    fingerprint: str,
    entry: InstanceRules | dict[str, Any],
    root: Path | None = None,
) -> InstanceRules:
    """Merge one instance's knowledge into the family file and persist it."""
    if isinstance(entry, dict):
        entry = InstanceRules.model_validate(entry)
    entry = entry.model_copy(update={"fingerprint": fingerprint, "updated_at": time.time()})
    rule_file = load(family, root)
    rule_file.family = family
    existing = rule_file.instances.get(fingerprint)
    merged = existing.merged_with(entry) if existing else entry
    rule_file.instances[fingerprint] = merged
    save(rule_file, root)
    return merged


def resolved_for(family: str, fingerprint: str, root: Path | None = None) -> InstanceRules | None:
    """What we know about *this* instance.  ``None`` when we have never seen it.

    Deliberately no fallback to a sibling instance of the same family.
    """
    return load(family, root).instances.get(fingerprint)


def forget(family: str, fingerprint: str, root: Path | None = None) -> bool:
    rule_file = load(family, root)
    if fingerprint not in rule_file.instances:
        return False
    del rule_file.instances[fingerprint]
    save(rule_file, root)
    return True


def fingerprints(family: str, root: Path | None = None) -> list[str]:
    return sorted(load(family, root).instances)


# ---------------------------------------------------------------------------
# Typed convenience wrappers
# ---------------------------------------------------------------------------


def save_builder_rules(
    puzzle: Any,
    rules: BuilderRules,
    resolved_by: dict[str, str] | None = None,
    unknowns: Iterable[str] = (),
    root: Path | None = None,
) -> InstanceRules:
    return merge(
        "builder",
        puzzle.fingerprint(),
        InstanceRules(
            flags={k: v for k, v in rules.model_dump().items() if v is not None},
            resolved_by=dict(resolved_by or {}),
            unknowns=list(unknowns),
        ),
        root,
    )


def load_builder_rules(puzzle: Any, root: Path | None = None) -> BuilderRules | None:
    entry = resolved_for("builder", puzzle.fingerprint(), root)
    if entry is None:
        return None
    known = {k: v for k, v in entry.flags.items() if k in BuilderRules.model_fields}
    return BuilderRules(**known)


def save_factory_flags(
    state: Any,
    flag_state: FactoryRuleState,
    root: Path | None = None,
) -> InstanceRules:
    from services.core.rules.dsl import FACTORY_FLAG_ORDER

    return merge(
        "factory",
        state.fingerprint(),
        InstanceRules(
            flags=flag_state.flags.model_dump(),
            resolved_by=dict(flag_state.resolved_by),
            unknowns=[f for f in FACTORY_FLAG_ORDER if f not in flag_state.resolved_by],
        ),
        root,
    )


def load_factory_flags(state: Any, root: Path | None = None) -> FactoryRuleState | None:
    entry = resolved_for("factory", state.fingerprint(), root)
    if entry is None:
        return None
    known = {k: v for k, v in entry.flags.items() if k in RuleFlags.model_fields}
    return FactoryRuleState(flags=RuleFlags(**known), resolved_by=dict(entry.resolved_by))
