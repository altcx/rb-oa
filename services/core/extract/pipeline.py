"""Extraction orchestration (spec 6.2): capture -> reviewable state in < 4 s.

The shape of the win, in order of size:

1.  **Tile fan-out.**  N per-machine crops read concurrently at ~1.5 s each
    beat one whole-board call at 8 s, and a crop containing exactly one panel
    cannot bleed a neighbour's numbers into the answer.  The graph is
    reassembled from the tiles plus ONE cheap topology-only call.
2.  **The jury.**  Every tile goes to three model families at once; ~35 of 40
    fields never reach the user (see ``jury.py``).
3.  **Speculative extraction.**  The work starts the moment a capture lands,
    before the user asks for anything.  If they then ask, ``extract`` awaits the
    in-flight task instead of starting a second copy of it.
4.  **Delta.**  With a confirmed previous state, ask only for what changed.

Everything is structured output (``json_schema_for``) with
``provider.require_parameters = true``, so a provider that cannot honour the
schema is routed around rather than silently returning prose.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from services.core.capture.grab import (
    Capture,
    PreparedImage,
    Rect,
    Tile,
    board_overview,
    prepare,
)
from services.core.extract import prompts
from services.core.extract.delta import DeltaResult, extract_delta_detailed, path_tokens
from services.core.extract.jury import FieldVerdict, JuryResult, run_jury, sort_edges
from services.core.extract.schemas import (
    BuilderExtraction,
    FactoryExtraction,
    TopologyExtraction,
    flatten,
    json_schema_for,
    to_builder_puzzle,
    to_factory_state,
    unflatten,
)
from services.core.llm.protocol import LLMClient
from services.core.rules.dsl import PuzzleFamily
from services.solvers.builder.model import BuilderPuzzle
from services.solvers.factory.model import FactoryState

StageHook = Callable[[str, float], Any] | None

DEFAULT_TIMEOUT_S = 12.0

#: Three families on purpose (spec 6.2): same-family models correlate errors,
#: and correlated errors make agreement meaningless.  Overridden by settings.
DEFAULT_JURY = [
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.5-haiku",
]


class ExtractRoles(BaseModel):
    """Which model plays which part."""

    model_config = ConfigDict(extra="forbid")

    jury: list[str] = Field(default_factory=lambda: list(DEFAULT_JURY))
    #: Only ever called when the jury splits three ways, and only for those fields.
    tie_breaker: str | None = None
    #: Naming a model here downgrades topology to ONE cheap reader, trading the
    #: consensus for two calls' worth of tokens.  Left null -- the default -- the
    #: graph goes through the same three-family jury as every tile.  A flipped
    #: edge does not cost a verification round trip, it invalidates the entire
    #: solve while looking completely plausible, so it is the last read on the
    #: board that should have a single reader.
    topology: str | None = None
    #: Delta re-reads.
    delta: str | None = None

    @classmethod
    def coerce(cls, value: "ExtractRoles | dict | Sequence[str] | None") -> "ExtractRoles":
        if value is None:
            return cls()
        if isinstance(value, ExtractRoles):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        return cls(jury=list(value))

    def topology_model(self) -> str:
        """The single-reader override, or "" when the jury reads the graph."""
        return self.topology or ""

    def delta_model(self) -> str:
        return self.delta or (self.jury[0] if self.jury else "")


class FieldProvenance(BaseModel):
    """Where a field came from and who voted for it (spec 6.2)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    value: Any = None
    status: str = "split"
    confidence: float = 0.0
    votes: dict[str, Any] = Field(default_factory=dict)
    capture_id: str = ""
    tile: str = ""
    crop_id: str | None = None
    box: dict[str, int] = Field(default_factory=dict)
    auto_confirmed: bool = False
    reason: str = ""


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    puzzle_type: str
    #: The merged wire document (FactoryExtraction / BuilderExtraction shaped).
    extraction: dict[str, Any] = Field(default_factory=dict)
    factory_state: FactoryState | None = None
    builder_puzzle: BuilderPuzzle | None = None
    #: Fields the assembler could not fill (nulls that the solver needs).
    unresolved: list[str] = Field(default_factory=list)
    #: Never shown to the user: three families agreed on a non-null value.
    auto_confirmed: list[str] = Field(default_factory=list)
    #: Everything else: majority-flagged, split, or unanimously null.
    disputed: list[str] = Field(default_factory=list)
    provenance: list[FieldProvenance] = Field(default_factory=list)
    jury: dict[str, JuryResult] = Field(default_factory=dict)
    stage_ms: dict[str, float] = Field(default_factory=dict)
    elapsed_ms: float = 0.0
    errors: dict[str, str] = Field(default_factory=dict)
    #: Delta path only.
    changed_paths: list[str] = Field(default_factory=list)
    rejected_paths: list[str] = Field(default_factory=list)
    used_delta: bool = False
    speculated: bool = False
    n_calls: int = 0

    def needs_review(self) -> list[str]:
        return sorted(set(self.disputed) | set(self.unresolved))

    def auto_confirm_rate(self) -> float:
        total = len(self.auto_confirmed) + len(self.disputed)
        return len(self.auto_confirmed) / total if total else 0.0

    def provenance_for(self, path: str) -> FieldProvenance | None:
        for p in self.provenance:
            if p.path == path:
                return p
        return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


async def _emit(on_stage: StageHook, stage: str, ms: float, sink: dict[str, float]) -> None:
    sink[stage] = round(sink.get(stage, 0.0) + ms, 3)
    if on_stage is None:
        return
    out = on_stage(stage, ms)
    if inspect.isawaitable(out):
        await out


def _safe_validate(schema: type[BaseModel], doc: dict[str, Any]) -> BaseModel:
    """Validate ``doc``, filling required-but-missing fields with ``None``.

    A model that dropped a key must not take the whole assembly down; the
    missing value surfaces as an unresolved field instead.
    """
    filled = dict(doc)
    for name, field_info in schema.model_fields.items():
        if name not in filled and field_info.is_required():
            filled[name] = None
    return schema.model_validate(filled)


def default_tiles(capture: Capture, puzzle_type: str) -> list[Tile]:
    """Fallback tiling when the caller has not marked up regions yet."""
    w, h = capture.image.width, capture.image.height
    if puzzle_type == PuzzleFamily.FACTORY.value:
        return [
            Tile("board", Rect(0, 0, w, h), "factory_machine"),
            board_overview((w, h)),
        ]
    return [Tile("panel", Rect(0, 0, w, h), "builder_parts")]


def _path_prefix(target: str, merged: dict[str, Any], tile: Tile) -> str:
    slot = prompts.target_for(target).slot
    if target == "factory_machine":
        return f"machines[{merged.get('id') or tile.name}]."
    if target == "builder_part":
        return f"parts[{merged.get('id') or tile.name}]."
    if target in ("builder_parts", "builder_obstacles"):
        return ""  # the panel schema already nests under parts[...]/obstacles[...]
    return f"{slot}."


def _single_call_result(
    model: str, flat: dict[str, Any], crop_id: str | None, latency_ms: float, error: str | None
) -> JuryResult:
    """Wrap a one-model call in the jury's shape.

    One reader is never a confirmation, so every field is pre-filled and
    flagged: the user presses one key, or fixes it.
    """
    verdicts = [
        FieldVerdict(
            path=path,
            value=value,
            status="majority" if value is not None else "split",
            votes={model: value},
            confidence=0.5 if value is not None else 0.0,
            crop_id=crop_id,
            auto_confirmed=False,
            reason="single cheap read; confirm before solving",
        )
        for path, value in flat.items()
    ]
    return JuryResult(
        merged=unflatten({v.path: v.value for v in verdicts}) if verdicts else {},
        verdicts=verdicts,
        majority_count=sum(1 for v in verdicts if v.status == "majority"),
        split_count=sum(1 for v in verdicts if v.status == "split"),
        disputed=[v.path for v in verdicts],
        per_model_latency_ms={model: round(latency_ms, 2)},
        elapsed_ms=round(latency_ms, 2),
        errors=({model: error} if error else {}),
        voters=[] if error else [model],
        crop_id=crop_id,
    )


async def _topology_call(
    client: LLMClient,
    model: str,
    messages: list,
    crop_id: str | None,
    timeout_s: float,
) -> JuryResult:
    from services.core.extract.jury import _ask  # single-call helper, same wire contract

    t0 = time.perf_counter()
    try:
        flat, ms = await _ask(
            client,
            model,
            messages,
            json_schema_for(TopologyExtraction),
            TopologyExtraction,
            timeout_s,
            None,
        )
        return _single_call_result(model, sort_edges(flat), crop_id, ms, None)
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000.0
        return _single_call_result(model, {}, crop_id, ms, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# the main pass
# --------------------------------------------------------------------------


def _needed_paths(missing: Sequence[str], paths: Iterable[str]) -> set[str]:
    """Which extracted paths the assembler said it could not fill.

    ``to_factory_state`` / ``to_builder_puzzle`` are the authority on what the
    solver reads: a field that never shows up in their ``missing`` list is a
    field nothing downstream touches.  Their entries are patterns rather than
    exact paths -- ``machines[?].id_or_kind`` when the id itself was unreadable,
    ``topology.edges`` standing for the whole edge list -- so match by token with
    ``?`` as a wildcard and a prefix counting as a hit.
    """
    patterns: list[list[tuple[str, bool]]] = []
    for m in missing:
        if m.endswith(".id_or_kind"):
            base = m[: -len("id_or_kind")]
            patterns += [path_tokens(base + "id"), path_tokens(base + "kind")]
        else:
            patterns.append(path_tokens(m))
    out: set[str] = set()
    for path in paths:
        tokens = path_tokens(path)
        for pat in patterns:
            if len(pat) > len(tokens):
                continue
            if all(
                (p_name == t_name or (p_is_idx and p_name == "?" and t_is_idx))
                and p_is_idx == t_is_idx
                for (p_name, p_is_idx), (t_name, t_is_idx) in zip(pat, tokens)
            ):
                out.add(path)
                break
    return out


def _demote_needed_nulls(
    provenance: Sequence[FieldProvenance], missing: Sequence[str]
) -> list[str]:
    """Un-confirm unanimous nulls on fields the solver actually needs.

    Absent is fine.  Silently defaulted is not: if every reader failed on a
    number the simulator will index with, that is the one null a human has to
    resolve.  Returns the demoted paths so they join the unresolved list.
    """
    candidates = [p.path for p in provenance if p.auto_confirmed and p.value is None]
    if not candidates:
        return []
    needed = _needed_paths(missing, candidates)
    # A record the assembler had to drop for an unreadable id takes its whole
    # panel with it: we know nothing about that machine or part, so none of its
    # nulls may be confirmed as "absent".  Otherwise a reviewer who fixes the id
    # inherits a weight of 0 -- a free part -- that nobody was ever asked about.
    dropped = {p.rsplit(".", 1)[0] for p in needed if p.rsplit(".", 1)[-1] in ("id", "kind")}
    needed |= {
        c for c in candidates if any(c.startswith(prefix + ".") for prefix in dropped)
    }
    demoted: list[str] = []
    for p in provenance:
        if p.path in needed and p.auto_confirmed and p.value is None:
            p.auto_confirmed = False
            p.reason = "unreadable, and the solver needs it: fill this in"
            demoted.append(p.path)
    return demoted


async def _run_tile(
    client: LLMClient,
    roles: ExtractRoles,
    tile: Tile,
    prepared: PreparedImage,
    *,
    timeout_s: float,
    on_stage: StageHook,
    stage_ms: dict[str, float],
) -> tuple[Tile, PreparedImage, JuryResult]:
    target = prompts.target_for(tile.target)
    messages = target.build(prepared.data_url, hint=tile.name)
    is_topology = tile.target == "factory_topology"
    t0 = time.perf_counter()
    if is_topology and roles.topology:
        # Explicitly downgraded to one cheap reader.
        result = await _topology_call(
            client, roles.topology, messages, prepared.crop_id, timeout_s
        )
    else:
        result = await run_jury(
            client,
            list(roles.jury),
            messages,
            target.schema,
            timeout_s=timeout_s,
            tie_breaker=roles.tie_breaker,
            crop_id=prepared.crop_id,
            # Edges are read as a set, in a canonical order, so that two models
            # tracing the graph from different corners vote on the same fields.
            normalizer=sort_edges if is_topology else None,
            set_paths=("edges",) if is_topology else (),
        )
    await _emit(on_stage, f"tile:{tile.name}", (time.perf_counter() - t0) * 1000.0, stage_ms)
    return tile, prepared, result


def _clean_edges(merged: dict[str, Any]) -> dict[str, Any]:
    """Drop edges the jury did not settle on.

    A disputed edge merges to ``{"src": null, "dst": null}``, which is not an
    edge and would fail assembly.  Dropping it from the *document* costs
    nothing: the verdict survives in the provenance with its votes, so the user
    is still told that one reader saw a connection the others did not.
    """
    edges = merged.get("edges")
    if not isinstance(edges, list):
        return merged
    kept = [e for e in edges if isinstance(e, dict) and e.get("src") and e.get("dst")]
    return {**merged, "edges": kept}


def _merge_documents(
    puzzle_type: str, pieces: Sequence[tuple[Tile, JuryResult]]
) -> dict[str, Any]:
    """Reassemble the puzzle-level wire document from per-tile answers."""
    doc: dict[str, Any] = (
        {"machines": [], "hud": None, "topology": None}
        if puzzle_type == PuzzleFamily.FACTORY.value
        else {"parts": [], "obstacles": [], "rules": None}
    )
    for tile, result in pieces:
        merged = result.merged or {}
        target = prompts.target_for(tile.target)
        if tile.target == "factory_machine":
            if merged:
                doc["machines"].append(merged)
        elif tile.target == "builder_part":
            if merged:
                doc["parts"].append(merged)
        elif tile.target == "builder_parts":
            doc["parts"].extend(merged.get("parts") or [])
        elif tile.target == "builder_obstacles":
            doc["obstacles"].extend(merged.get("obstacles") or [])
        elif target.slot in ("hud", "topology", "rules"):
            existing = doc.get(target.slot) or {}
            if target.slot == "topology":
                merged = _clean_edges(merged)
            doc[target.slot] = {**existing, **merged}
    return doc


def _assemble(
    puzzle_type: str, doc: dict[str, Any]
) -> tuple[FactoryState | None, BuilderPuzzle | None, list[str], dict[str, str]]:
    errors: dict[str, str] = {}
    if puzzle_type == PuzzleFamily.FACTORY.value:
        try:
            ext = _safe_validate(FactoryExtraction, doc)
            state, missing = to_factory_state(ext)  # type: ignore[arg-type]
            return state, None, missing, errors
        except Exception as exc:
            errors["assemble"] = f"{type(exc).__name__}: {exc}"
            return None, None, ["<assembly failed>"], errors
    try:
        ext = _safe_validate(BuilderExtraction, doc)
        puzzle, missing = to_builder_puzzle(ext)  # type: ignore[arg-type]
        return None, puzzle, missing, errors
    except Exception as exc:
        errors["assemble"] = f"{type(exc).__name__}: {exc}"
        return None, None, ["<assembly failed>"], errors


async def _extract_full(
    client: LLMClient,
    roles: ExtractRoles,
    captures: Sequence[Capture],
    puzzle_type: str,
    *,
    timeout_s: float,
    on_stage: StageHook,
) -> ExtractionResult:
    t_start = time.perf_counter()
    stage_ms: dict[str, float] = {}

    # ---- crop + downscale every tile (CPU, ~ms each) -------------------
    t0 = time.perf_counter()
    jobs: list[tuple[Capture, Tile, PreparedImage]] = []
    errors: dict[str, str] = {}
    for capture in captures:
        tiles = list(capture.tiles) or default_tiles(capture, puzzle_type)
        for tile in tiles:
            try:
                jobs.append(
                    (capture, tile, prepare(capture.image, tile.box, max_long_edge=tile.max_long_edge))
                )
            except Exception as exc:
                errors[f"prepare:{tile.name}"] = f"{type(exc).__name__}: {exc}"
    await _emit(on_stage, "prepare", (time.perf_counter() - t0) * 1000.0, stage_ms)

    # ---- fan out: every tile at once ----------------------------------
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(
            _run_tile(
                client,
                roles,
                tile,
                prep,
                timeout_s=timeout_s,
                on_stage=on_stage,
                stage_ms=stage_ms,
            )
            for cap, tile, prep in jobs
        ),
        return_exceptions=True,
    )
    await _emit(on_stage, "fanout", (time.perf_counter() - t0) * 1000.0, stage_ms)

    pieces: list[tuple[Tile, JuryResult]] = []
    juries: dict[str, JuryResult] = {}
    provenance: list[FieldProvenance] = []
    n_calls = 0
    for (cap, tile, prep), outcome in zip(jobs, results):
        if isinstance(outcome, BaseException):
            errors[f"tile:{tile.name}"] = f"{type(outcome).__name__}: {outcome}"
            continue
        _, prepared, jury_result = outcome
        pieces.append((tile, jury_result))
        juries[tile.name] = jury_result
        n_calls += len(jury_result.per_model_latency_ms)
        for model, msg in jury_result.errors.items():
            errors[f"{tile.name}:{model}"] = msg
        prefix = _path_prefix(tile.target, jury_result.merged or {}, tile)
        for v in jury_result.verdicts:
            path = f"{prefix}{v.path}"
            provenance.append(
                FieldProvenance(
                    path=path,
                    value=v.value,
                    status=v.status,
                    confidence=v.confidence,
                    votes=v.votes,
                    capture_id=cap.capture_id,
                    tile=tile.name,
                    crop_id=prepared.crop_id,
                    box=prepared.source_box.as_dict(),
                    auto_confirmed=v.auto_confirmed,
                    reason=v.reason,
                )
            )

    # ---- reassemble ----------------------------------------------------
    t0 = time.perf_counter()
    doc = _merge_documents(puzzle_type, pieces)
    state, puzzle, missing, asm_errors = _assemble(puzzle_type, doc)
    errors.update(asm_errors)

    # The assembler is the authority on what the solver needs, so the
    # auto-confirm decision for unanimous nulls can only be made here.
    demoted = _demote_needed_nulls(provenance, missing)
    auto = [p.path for p in provenance if p.auto_confirmed]
    disputed = [p.path for p in provenance if not p.auto_confirmed and p.path not in demoted]
    unresolved = sorted(set(missing) | set(demoted))
    await _emit(on_stage, "assemble", (time.perf_counter() - t0) * 1000.0, stage_ms)

    elapsed = (time.perf_counter() - t_start) * 1000.0
    await _emit(on_stage, "total", elapsed, stage_ms)
    stage_ms["total"] = round(elapsed, 3)
    return ExtractionResult(
        puzzle_type=puzzle_type,
        extraction=doc,
        factory_state=state,
        builder_puzzle=puzzle,
        unresolved=unresolved,
        auto_confirmed=sorted(auto),
        disputed=sorted(disputed),
        provenance=provenance,
        jury=juries,
        stage_ms=stage_ms,
        elapsed_ms=round(elapsed, 2),
        errors=errors,
        n_calls=n_calls,
    )


async def _extract_delta_pass(
    client: LLMClient,
    roles: ExtractRoles,
    captures: Sequence[Capture],
    puzzle_type: str,
    previous_state: dict[str, Any],
    *,
    timeout_s: float,
    on_stage: StageHook,
) -> ExtractionResult:
    t_start = time.perf_counter()
    stage_ms: dict[str, float] = {}
    errors: dict[str, str] = {}

    # One crop is enough: the delta call is about what moved, not about detail.
    capture = captures[0]
    tiles = list(capture.tiles) or default_tiles(capture, puzzle_type)
    tile = next((t for t in tiles if t.target == "factory_topology"), tiles[0])
    t0 = time.perf_counter()
    prepared = prepare(capture.image, tile.box, max_long_edge=tile.max_long_edge)
    await _emit(on_stage, "prepare", (time.perf_counter() - t0) * 1000.0, stage_ms)

    schema = FactoryExtraction if puzzle_type == PuzzleFamily.FACTORY.value else BuilderExtraction
    messages = prompts.delta_prompt(prepared.data_url, flatten(previous_state), hint=tile.name)
    t0 = time.perf_counter()
    result: DeltaResult = await extract_delta_detailed(
        client,
        roles.delta_model(),
        previous_state,
        messages,
        schema,
        timeout_s=timeout_s,
    )
    await _emit(on_stage, "delta", (time.perf_counter() - t0) * 1000.0, stage_ms)
    if result.error:
        errors["delta"] = result.error
    if result.rejected:
        errors["delta:rejected_paths"] = (
            "model returned paths that do not exist in the previous state: "
            + ", ".join(sorted(result.rejected))
        )

    t0 = time.perf_counter()
    state, puzzle, missing, asm_errors = _assemble(puzzle_type, result.merged)
    errors.update(asm_errors)
    await _emit(on_stage, "assemble", (time.perf_counter() - t0) * 1000.0, stage_ms)

    provenance = [
        FieldProvenance(
            path=path,
            value=flatten(result.merged).get(path),
            status="majority",
            confidence=0.5,
            votes={roles.delta_model(): flatten(result.merged).get(path)},
            capture_id=capture.capture_id,
            tile=tile.name,
            crop_id=prepared.crop_id,
            box=prepared.source_box.as_dict(),
            auto_confirmed=False,
            reason="changed since the last confirmed state; confirm the new value",
        )
        for path in result.changed
    ]

    elapsed = (time.perf_counter() - t_start) * 1000.0
    await _emit(on_stage, "total", elapsed, stage_ms)
    stage_ms["total"] = round(elapsed, 3)
    unchanged = [p for p in flatten(result.merged) if p not in set(result.changed)]
    return ExtractionResult(
        puzzle_type=puzzle_type,
        extraction=result.merged,
        factory_state=state,
        builder_puzzle=puzzle,
        unresolved=missing,
        auto_confirmed=sorted(unchanged),  # carried over from the confirmed state
        disputed=sorted(result.changed),
        provenance=provenance,
        stage_ms=stage_ms,
        elapsed_ms=round(elapsed, 2),
        errors=errors,
        changed_paths=list(result.changed),
        rejected_paths=list(result.rejected),
        used_delta=True,
        n_calls=1,
    )


# --------------------------------------------------------------------------
# Speculation
# --------------------------------------------------------------------------

_INFLIGHT: dict[str, asyncio.Task] = {}


def speculation_key(
    captures: Sequence[Capture] | Capture,
    puzzle_type: str,
    previous_state: dict[str, Any] | None = None,
) -> str:
    caps = [captures] if isinstance(captures, Capture) else list(captures)
    h = hashlib.sha256()
    h.update(puzzle_type.encode())
    for c in caps:
        h.update(c.capture_id.encode())
        for t in c.tiles:
            h.update(f"{t.name}:{t.box.as_tuple()}:{t.target}".encode())
    h.update(b"delta" if previous_state else b"full")
    return h.hexdigest()[:16]


def speculate(
    client: LLMClient,
    roles: "ExtractRoles | dict | Sequence[str] | None",
    captures: Sequence[Capture] | Capture,
    puzzle_type: str,
    *,
    previous_state: dict[str, Any] | None = None,
    on_stage: StageHook = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> asyncio.Task:
    """Start extracting **now**, before the user asks.

    The capture landing is the trigger; the user's question arrives a second or
    two later and finds the work already done (or already running -- either way
    it is not started twice).
    """
    caps = [captures] if isinstance(captures, Capture) else list(captures)
    key = speculation_key(caps, puzzle_type, previous_state)
    task = _INFLIGHT.get(key)
    if task is not None and not task.cancelled():
        return task
    coro = extract(
        client,
        roles,
        caps,
        puzzle_type,
        previous_state=previous_state,
        on_stage=on_stage,
        timeout_s=timeout_s,
        _speculating=True,
    )
    task = asyncio.ensure_future(coro)
    _INFLIGHT[key] = task

    def _drop(t: asyncio.Task) -> None:
        if t.cancelled() or t.exception() is not None:
            _INFLIGHT.pop(key, None)

    task.add_done_callback(_drop)
    return task


def clear_speculations() -> None:
    for task in list(_INFLIGHT.values()):
        if not task.done():
            task.cancel()
    _INFLIGHT.clear()


def inflight_count() -> int:
    return len(_INFLIGHT)


async def extract(
    client: LLMClient,
    roles: "ExtractRoles | dict | Sequence[str] | None",
    captures: Sequence[Capture] | Capture,
    puzzle_type: str,
    *,
    previous_state: dict[str, Any] | None = None,
    on_stage: StageHook = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    _speculating: bool = False,
) -> ExtractionResult:
    """Capture(s) -> reviewable state.

    Awaits an in-flight speculative pass for the same captures instead of
    duplicating the work.
    """
    caps = [captures] if isinstance(captures, Capture) else list(captures)
    if not caps:
        raise ValueError("extract() needs at least one capture")
    roles_ = ExtractRoles.coerce(roles)

    if not _speculating:
        key = speculation_key(caps, puzzle_type, previous_state)
        task = _INFLIGHT.get(key)
        if task is not None and not task.cancelled():
            result = await asyncio.shield(task)
            return result.model_copy(update={"speculated": True})

    if previous_state:
        return await _extract_delta_pass(
            client,
            roles_,
            caps,
            puzzle_type,
            previous_state,
            timeout_s=timeout_s,
            on_stage=on_stage,
        )
    return await _extract_full(
        client, roles_, caps, puzzle_type, timeout_s=timeout_s, on_stage=on_stage
    )


__all__ = [
    "DEFAULT_JURY",
    "ExtractRoles",
    "ExtractionResult",
    "FieldProvenance",
    "default_tiles",
    "extract",
    "speculate",
    "speculation_key",
    "clear_speculations",
    "inflight_count",
]
