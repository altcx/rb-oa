"""Model catalog, role assignment and measured latency.

Model slugs churn weekly.  Nothing here hardcodes one: roles are assigned by
*capability*, read live from ``GET /api/v1/models``.

Capability derivation (OpenRouter's schema):

*   vision              -> ``"image" in architecture.input_modalities``
*   structured outputs  -> ``"structured_outputs" in supported_parameters``
*   tool calling        -> ``"tools" in supported_parameters``
*   reasoning           -> ``"reasoning"``/``"include_reasoning"`` in
                           ``supported_parameters`` (OpenRouter's own marker)

The one non-obvious rule is the jury: the three extractor slots must come from
three *different vendor families*, because two models from one family share a
tokenizer, a training corpus and therefore their mistakes.  A jury that agrees
because it is inbred is worse than no jury -- it manufactures confidence.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = REPO_ROOT / "data" / "settings.json"

ROLES: tuple[str, ...] = ("extractor_jury", "rule_compiler", "strategist", "tie_breaker")

#: How many samples we keep per (role, model) before dropping the oldest.
LATENCY_WINDOW = 50


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""
    context_length: int = 0
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)
    supported_parameters: list[str] = Field(default_factory=list)
    prompt_price: float = 0.0
    completion_price: float = 0.0
    image_price: float = 0.0
    created: int = 0

    # -- capabilities ---------------------------------------------------

    @property
    def family(self) -> str:
        """Vendor prefix of the slug -- the correlation unit for the jury."""
        return self.id.split("/", 1)[0] if "/" in self.id else self.id

    @property
    def vision(self) -> bool:
        return "image" in [m.lower() for m in self.input_modalities]

    @property
    def structured_outputs(self) -> bool:
        return "structured_outputs" in self.supported_parameters

    @property
    def tools(self) -> bool:
        return "tools" in self.supported_parameters

    @property
    def reasoning(self) -> bool:
        return any(
            p in self.supported_parameters for p in ("reasoning", "include_reasoning", "thinking")
        )

    @property
    def blended_price(self) -> float:
        """Cost proxy: 3 prompt tokens per completion token is close enough
        for extraction traffic, which is image-heavy and answer-light."""
        return 3.0 * self.prompt_price + self.completion_price

    def strength(self) -> float:
        """Crude 'strong reasoning' proxy that does not name a single model.

        Price and context length are the only vendor-neutral signals in the
        catalog that track capability; the explicit reasoning parameter is the
        strongest of the three.
        """
        score = 0.0
        if self.reasoning:
            score += 3.0
        score += min(self.context_length, 1_000_000) / 200_000.0
        score += min(self.blended_price * 1000.0, 10.0)
        return score


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_model(raw: dict[str, Any]) -> ModelInfo:
    arch = raw.get("architecture") or {}
    pricing = raw.get("pricing") or {}
    modalities = arch.get("input_modalities")
    if not modalities:
        modality = str(arch.get("modality") or "")
        head = modality.split("->")[0]
        modalities = [m for m in head.replace("+", ",").split(",") if m]
    return ModelInfo(
        id=raw.get("id") or "",
        name=raw.get("name") or "",
        context_length=int(raw.get("context_length") or 0),
        input_modalities=list(modalities or []),
        output_modalities=list(arch.get("output_modalities") or []),
        supported_parameters=list(raw.get("supported_parameters") or []),
        prompt_price=_num(pricing.get("prompt")),
        completion_price=_num(pricing.get("completion")),
        image_price=_num(pricing.get("image")),
        created=int(raw.get("created") or 0),
    )


async def fetch_catalog(client: Any, *, timeout_s: float = 20.0) -> list[ModelInfo]:
    """Load the live catalog.

    ``client`` may be an :class:`OpenRouterClient` (uses its ``get_json``) or a
    bare ``httpx.AsyncClient``.
    """
    payload: Any
    if hasattr(client, "get_json"):
        payload = await client.get_json("/models", timeout_s=timeout_s)
    else:  # bare httpx.AsyncClient
        resp = await client.get("https://openrouter.ai/api/v1/models", timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else payload
    out = [parse_model(r) for r in (data or []) if isinstance(r, dict)]
    return [m for m in out if m.id]


# -- capability filters (public: the settings UI uses these directly) -------


def with_vision(catalog: Iterable[ModelInfo]) -> list[ModelInfo]:
    return [m for m in catalog if m.vision]


def with_structured_outputs(catalog: Iterable[ModelInfo]) -> list[ModelInfo]:
    return [m for m in catalog if m.structured_outputs]


def with_tools(catalog: Iterable[ModelInfo]) -> list[ModelInfo]:
    return [m for m in catalog if m.tools]


ROLE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "extractor_jury": {
        "needs": ("vision", "structured_outputs"),
        "prefers": "cheap and fast; three different families",
        "slots": 3,
    },
    "rule_compiler": {
        "needs": ("vision", "structured_outputs"),
        "prefers": "strongest reasoning; runs once per puzzle, cached",
        "slots": 1,
    },
    "strategist": {
        "needs": ("tools",),
        "prefers": "strong reasoning, streaming",
        "slots": 1,
    },
    "tie_breaker": {
        "needs": ("vision",),
        "prefers": "strong; a family the jury does not already contain",
        "slots": 1,
    },
}


def eligible(catalog: Iterable[ModelInfo], role: str) -> list[ModelInfo]:
    needs = ROLE_REQUIREMENTS[role]["needs"]
    return [m for m in catalog if all(getattr(m, n) for n in needs)]


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------


class RoleAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extractor_jury: list[str] = Field(default_factory=list)
    rule_compiler: str | None = None
    strategist: str | None = None
    tie_breaker: str | None = None
    notes: list[str] = Field(default_factory=list)

    def for_role(self, role: str) -> list[str]:
        value = getattr(self, role, None)
        if value is None:
            return []
        return list(value) if isinstance(value, list) else [value]

    def families(self) -> list[str]:
        return [m.split("/", 1)[0] for m in self.extractor_jury]


def pick_jury(catalog: Iterable[ModelInfo], slots: int = 3) -> tuple[list[ModelInfo], list[str]]:
    """Cheapest eligible model from each of ``slots`` distinct families."""
    notes: list[str] = []
    cands = sorted(eligible(catalog, "extractor_jury"), key=lambda m: (m.blended_price, m.id))
    chosen: list[ModelInfo] = []
    seen_families: set[str] = set()
    for m in cands:
        if m.family in seen_families:
            continue
        chosen.append(m)
        seen_families.add(m.family)
        if len(chosen) == slots:
            break
    if len(chosen) < slots:
        notes.append(
            f"only {len(chosen)} distinct vision+structured families available; "
            "jury errors will correlate — verify disputed fields manually"
        )
        for m in cands:
            if len(chosen) >= slots:
                break
            if m not in chosen:
                chosen.append(m)
    return chosen, notes


def auto_assign(catalog: Iterable[ModelInfo]) -> RoleAssignment:
    """Assign every role from capability alone."""
    catalog = list(catalog)
    notes: list[str] = []

    jury, jury_notes = pick_jury(catalog)
    notes.extend(jury_notes)

    compilers = sorted(
        eligible(catalog, "rule_compiler"), key=lambda m: (-m.strength(), m.blended_price, m.id)
    )
    compiler = compilers[0].id if compilers else None
    if compiler is None:
        notes.append("no vision+structured model for rule_compiler")

    strategists = sorted(
        eligible(catalog, "strategist"), key=lambda m: (-m.strength(), m.blended_price, m.id)
    )
    strategist = strategists[0].id if strategists else None
    if strategist is None:
        notes.append("no tool-calling model for strategist")

    jury_families = {m.family for m in jury}
    breakers = sorted(
        eligible(catalog, "tie_breaker"), key=lambda m: (-m.strength(), m.blended_price, m.id)
    )
    outside = [m for m in breakers if m.family not in jury_families]
    tie_breaker = (outside or breakers)[0].id if (outside or breakers) else None
    if breakers and not outside:
        notes.append("tie_breaker shares a family with the jury; its vote is not independent")

    return RoleAssignment(
        extractor_jury=[m.id for m in jury],
        rule_compiler=compiler,
        strategist=strategist,
        tie_breaker=tie_breaker,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Persistence + measured latency
# ---------------------------------------------------------------------------


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    model: str
    count: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    last_ms: float = 0.0


class SettingsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: RoleAssignment = Field(default_factory=RoleAssignment)
    #: role -> model -> recent latency samples in ms
    latency: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    catalog_fetched_at: float = 0.0
    updated_at: float = 0.0


def _path(path: Path | None) -> Path:
    return Path(path) if path is not None else SETTINGS_PATH


def load_settings(path: Path | None = None) -> SettingsFile:
    p = _path(path)
    if not p.exists():
        return SettingsFile()
    try:
        return SettingsFile.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return SettingsFile()


def save_settings(settings: SettingsFile, path: Path | None = None) -> Path:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    settings.updated_at = time.time()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def save_assignment(roles: RoleAssignment, path: Path | None = None) -> Path:
    settings = load_settings(path)
    settings.roles = roles
    settings.catalog_fetched_at = time.time()
    return save_settings(settings, path)


def record_latency(role: str, model: str, ms: float, path: Path | None = None) -> None:
    """Append one measurement so the settings UI can show real per-role speed."""
    settings = load_settings(path)
    bucket = settings.latency.setdefault(role, {}).setdefault(model, [])
    bucket.append(float(ms))
    del bucket[:-LATENCY_WINDOW]
    save_settings(settings, path)


def latency_report(path: Path | None = None) -> dict[str, list[LatencyStats]]:
    """role -> per-model measured latency, fastest first."""
    settings = load_settings(path)
    out: dict[str, list[LatencyStats]] = {}
    for role, models in settings.latency.items():
        rows: list[LatencyStats] = []
        for model, samples in models.items():
            if not samples:
                continue
            ordered = sorted(samples)
            rows.append(
                LatencyStats(
                    role=role,
                    model=model,
                    count=len(samples),
                    p50_ms=statistics.median(ordered),
                    p95_ms=ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
                    mean_ms=statistics.fmean(samples),
                    last_ms=samples[-1],
                )
            )
        out[role] = sorted(rows, key=lambda r: r.p50_ms)
    return out
