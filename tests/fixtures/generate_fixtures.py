"""Fixture generator + fake vision client (spec 6.3).

The real game UI is not available in this container, so the fixtures are
rendered synthetically with PIL: a panel that *looks* like the game (labelled
fields, numbers, arrows) plus a hand-verifiable ground-truth sidecar in the
exact extraction wire shape.

The directory layout is the same one a real capture uses::

    tests/fixtures/captures/factory/factory_001.png
    tests/fixtures/captures/factory/factory_001.json   <- sidecar

so a genuine screenshot dropped in beside a hand-written sidecar replays
through ``practice.run_practice`` with nothing else changed.

:class:`FakeVisionClient` "reads" a fixture by returning its ground truth for
whichever crop it is shown, with configurable per-model error injection.  That
is what makes the practice tests mean anything: inject a disagreement into one
model and assert the jury *flags* the field instead of auto-confirming it.

Regenerate::

    python -m tests.fixtures.generate_fixtures --out tests/fixtures/captures
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

from PIL import Image, ImageDraw, ImageFont

from services.core.capture.grab import Rect, Tile, board_overview, prepare
from services.core.extract import prompts
from services.core.extract.practice import Fixture, load_fixtures
from services.core.extract.schemas import (
    BuilderRulesExtraction,
    FactoryHUDExtraction,
    MachinePanelExtraction,
    ObstacleExtraction,
    PartExtraction,
    RecipeExtraction,
    TopologyExtraction,
    flatten,
    unflatten,
)
from services.core.llm.protocol import ChatMessage, LLMResponse, StreamEvent, Usage

DEFAULT_OUT = Path("tests/fixtures/captures")
N_PER_PUZZLE = 30

# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

BG = (18, 20, 26)
PANEL = (32, 36, 46)
PANEL_EDGE = (70, 78, 96)
INK = (226, 230, 238)
DIM = (150, 158, 176)
ACCENT = (120, 200, 150)
WARN = (232, 176, 92)

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_FONT_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in _FONT_BOLD if bold else _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _panel(d: ImageDraw.ImageDraw, r: Rect, title: str, sub: str = "", tag: str = "") -> None:
    d.rounded_rectangle(r.as_box(), radius=8, fill=PANEL, outline=PANEL_EDGE, width=2)
    d.rectangle((r.x + 2, r.y + 2, r.x + r.w - 2, r.y + 26), fill=(44, 50, 64))
    d.text((r.x + 10, r.y + 7), title, font=_font(15, bold=True), fill=INK)
    if sub:
        d.text((r.x + r.w - 10, r.y + 8), sub, font=_font(13), fill=ACCENT, anchor="ra")
    if tag:
        # Board serial, drawn in every panel.  It makes each crop unique, which
        # is what lets the fake client match a crop back to its fixture.
        d.text((r.x + r.w - 8, r.y + r.h - 16), tag, font=_font(11), fill=(88, 96, 116), anchor="ra")


def _row(d: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, w: int) -> None:
    d.text((x, y), label, font=_font(13), fill=DIM)
    d.text((x + w, y), value, font=_font(14, bold=True), fill=INK, anchor="ra")


WIRE = (110, 160, 210)


def _arrow_head(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int]) -> None:
    import math

    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for off in (2.6, -2.6):
        d.line(
            [b, (b[0] + 12 * math.cos(ang + off), b[1] + 12 * math.sin(ang + off))],
            fill=WIRE,
            width=3,
        )


def _route(d: ImageDraw.ImageDraw, src: Rect, dst: Rect, gap: int) -> None:
    """Orthogonal wire from ``src`` to ``dst``, routed through the panel gaps."""
    sy = src.y + src.h // 2
    dy = dst.y + dst.h // 2
    if dst.x >= src.x + src.w:  # to the right
        a = (src.x + src.w, sy)
        b = (dst.x, dy)
        mx = (a[0] + b[0]) // 2
    elif dst.x + dst.w <= src.x:  # to the left
        a = (src.x, sy)
        b = (dst.x + dst.w, dy)
        mx = (a[0] + b[0]) // 2
    else:  # same column: go down the side
        a = (src.x + src.w, sy)
        b = (dst.x + dst.w, dy)
        mx = a[0] + gap
    pts = [a, (mx, a[1]), (mx, b[1]), b]
    d.line(pts, fill=WIRE, width=3, joint="curve")
    _arrow_head(d, pts[-2], b)


# --------------------------------------------------------------------------
# Factory fixtures
# --------------------------------------------------------------------------

ITEMS = ["ore", "plate", "gear", "coil", "ingot", "rod", "panel", "chip"]
MODS = ["double_output_max", "half_materials", "one_hour_production", "double_storage_max"]


def _make_factory(idx: int, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    tag = f"#{idx:03d}"
    n_makers = rng.choice([1, 2])
    cols, rows = 3, 2
    pw, ph = 340, 250
    margin_x, margin_y = 40, 110
    gap_x, gap_y = 30, 40
    width = margin_x * 2 + cols * pw + (cols - 1) * gap_x
    height = margin_y + rows * ph + (rows - 1) * gap_y + 40
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    # ---- HUD ---------------------------------------------------------
    money = float(rng.randrange(200, 5000, 25))
    horizon = rng.choice([12, 18, 24, 24, 36])
    hour_now = rng.choice([None, rng.randint(1, horizon)])
    hud_box = Rect(margin_x, 24, width - 2 * margin_x, 60)
    d.rounded_rectangle(hud_box.as_box(), radius=8, fill=(26, 30, 40), outline=PANEL_EDGE)
    d.text((hud_box.x + 16, hud_box.y + 8), f"FACTORY  {tag}", font=_font(16, bold=True), fill=INK)
    d.text((hud_box.x + 16, hud_box.y + 32), f"MONEY  ${money:,.0f}", font=_font(15), fill=ACCENT)
    d.text(
        (hud_box.x + 260, hud_box.y + 32), f"HOURS  {horizon}", font=_font(15), fill=INK
    )
    if hour_now is not None:
        d.text(
            (hud_box.x + 420, hud_box.y + 32), f"NOW  h{hour_now}", font=_font(15), fill=WARN
        )

    # ---- machines ----------------------------------------------------
    kinds = ["supplier"] + ["maker"] * n_makers + ["seller"]
    while len(kinds) < rng.choice([4, 5, 6]):
        kinds.insert(-1, "maker")
    kinds = kinds[:6]
    machines: list[dict[str, Any]] = []
    regions: list[Tile] = []
    region_truth: dict[str, Any] = {}
    centers: dict[str, tuple[int, int]] = {}
    boxes: dict[str, Rect] = {}
    chain = rng.sample(ITEMS, k=min(len(kinds) + 1, len(ITEMS)))

    for i, kind in enumerate(kinds):
        c, r = i % cols, i // cols
        box = Rect(margin_x + c * (pw + gap_x), margin_y + r * (ph + gap_y), pw, ph)
        mid = f"M{i + 1}"
        centers[mid] = (box.x + box.w // 2, box.y + box.h // 2)
        in_item = chain[max(0, i - 1)]

        recipes: list[RecipeExtraction] = []
        n_recipes = rng.choice([1, 1, 2])
        for k in range(n_recipes):
            rid = f"{mid}-r{k + 1}"
            if kind == "supplier":
                recipes.append(
                    RecipeExtraction(
                        id=rid,
                        name=f"buy {chain[i]}",
                        inputs={},
                        output_item=chain[i],
                        output_qty=rng.choice([1, 1, 2]),
                        purchase_cost=float(rng.randrange(2, 20)),
                    )
                )
            elif kind == "maker":
                recipes.append(
                    RecipeExtraction(
                        id=rid,
                        name=f"{in_item}->{chain[i]}",
                        inputs={in_item: rng.choice([1, 2, 3])},
                        output_item=chain[i],
                        output_qty=rng.choice([1, 1, 2]),
                        production_cost=float(rng.randrange(1, 12)),
                    )
                )
            else:
                recipes.append(
                    RecipeExtraction(
                        id=rid,
                        name=f"sell {in_item}",
                        inputs={in_item: 1},
                        output_item=None,
                        output_qty=1,
                        sale_price=float(rng.randrange(10, 90)),
                    )
                )
        output_max = rng.choice([2, 3, 4, 6, 8])
        mp = MachinePanelExtraction(
            id=mid,
            kind=kind,  # type: ignore[arg-type]
            name=f"{kind.title()} {i + 1}",
            recipes=recipes,
            selected_recipe_id=recipes[0].id,
            output_setting=rng.randint(0, output_max),
            output_max=output_max,
            storage_max=0 if kind == "seller" else rng.choice([10, 20, 40, 60]),
            production_hours=rng.choice([1, 1, 1, 2]),
            installed_mods=rng.sample(MODS, k=rng.choice([0, 0, 1])),
            current_storage=None,
            x=float(centers[mid][0]),
            y=float(centers[mid][1]),
        )

        machines.append(mp.model_dump())
        region_truth[mid] = mp.model_dump()
        regions.append(Tile(name=mid, box=box, target="factory_machine"))
        boxes[mid] = box

    # ---- topology, drawn first so the wiring runs *behind* the panels --
    ids = [m["id"] for m in machines]
    edges = [{"src": a, "dst": b} for a, b in zip(ids, ids[1:])]
    if len(ids) > 3 and rng.random() < 0.5:
        edges.append({"src": ids[0], "dst": ids[2]})
    for e in edges:
        _route(d, boxes[e["src"]], boxes[e["dst"]], gap_x // 2)

    # ---- panels --------------------------------------------------------
    for mp_dump in machines:
        mp = MachinePanelExtraction.model_validate(mp_dump)
        mid = str(mp.id)
        box = boxes[mid]
        kind = str(mp.kind)
        recipes = mp.recipes or []
        _panel(d, box, f"{mid}  {mp.name}", kind.upper(), tag=tag)
        y = box.y + 36
        _row(d, box.x + 12, y, "KIND", str(kind), box.w - 24)
        y += 22
        for rec in recipes:
            ins = ", ".join(f"{v}x {k}" for k, v in (rec.inputs or {}).items()) or "-"
            mark = ">" if rec.id == mp.selected_recipe_id else " "
            d.text(
                (box.x + 12, y),
                f"{mark} {rec.id}: {ins} -> {rec.output_item or 'money'} x{rec.output_qty}",
                font=_font(12),
                fill=INK if mark == ">" else DIM,
            )
            y += 18
            price = (
                f"buy ${rec.purchase_cost:.0f}"
                if kind == "supplier"
                else (
                    f"sell ${rec.sale_price:.0f}"
                    if kind == "seller"
                    else f"cost ${rec.production_cost:.0f}"
                )
            )
            d.text((box.x + 26, y), price, font=_font(12), fill=DIM)
            y += 20
        y += 4
        _row(d, box.x + 12, y, "OUTPUT", f"{mp.output_setting} / {mp.output_max}", box.w - 24)
        y += 20
        _row(d, box.x + 12, y, "STORAGE MAX", str(mp.storage_max), box.w - 24)
        y += 20
        _row(
            d,
            box.x + 12,
            y,
            "PRODUCTION",
            f"{mp.production_hours} h" + ("  (clock)" if mp.production_hours == 2 else ""),
            box.w - 24,
        )
        y += 20
        _row(d, box.x + 12, y, "MODS", ", ".join(mp.installed_mods or []) or "none", box.w - 24)

    topo = TopologyExtraction(edges=edges)
    hud = FactoryHUDExtraction(money=money, horizon_hours=horizon, hour_now=hour_now)

    regions.append(Tile(name="hud", box=hud_box, target="factory_hud"))
    regions.append(board_overview((width, height)))
    region_truth["hud"] = hud.model_dump()
    region_truth["topology"] = topo.model_dump()

    ground_truth = {
        "hud": hud.model_dump(),
        "machines": machines,
        "topology": topo.model_dump(),
    }
    sidecar = {
        "regions": regions,
        "region_truth": region_truth,
        "ground_truth": ground_truth,
    }
    return img, sidecar


# --------------------------------------------------------------------------
# Builder fixtures
# --------------------------------------------------------------------------

STATS = ["speed", "armor", "grip", "power", "lift", "range"]
PART_WORDS = ["frame", "wheel", "engine", "plate", "boom", "hook", "tank", "rotor", "brace"]


def _make_builder(idx: int, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    tag = f"#{idx:03d}"
    stats = rng.sample(STATS, k=rng.choice([2, 3]))
    n_parts = rng.choice([5, 6])
    width, height = 1120, 820
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 24), f"BUILDER  {tag}", font=_font(18, bold=True), fill=INK)

    regions: list[Tile] = []
    region_truth: dict[str, Any] = {}
    parts: list[dict[str, Any]] = []

    card_w, card_h = 320, 112 + 18 * len(stats)
    for i in range(n_parts):
        c, r = i % 2, i // 2
        box = Rect(40 + c * (card_w + 24), 70 + r * (card_h + 20), card_w, card_h)
        pid = f"P{i + 1}"
        pe = PartExtraction(
            id=pid,
            name=f"{rng.choice(PART_WORDS)}-{rng.randint(10, 99)}",
            weight=rng.randint(1, 12),
            qty_available=rng.choice([1, 1, 1, 2, 3]),
            attributes={s: rng.randint(0, 9) for s in stats},
            cost=rng.choice([0, 5, 10, 15, 25]),
        )
        _panel(d, box, f"{pid}  {pe.name}", f"WT {pe.weight}", tag=tag)
        y = box.y + 34
        _row(d, box.x + 12, y, "WEIGHT", str(pe.weight), box.w - 24)
        y += 20
        _row(d, box.x + 12, y, "QTY", str(pe.qty_available), box.w - 24)
        y += 20
        _row(d, box.x + 12, y, "COST", str(pe.cost), box.w - 24)
        y += 20
        for s in stats:
            _row(d, box.x + 12, y, s.upper(), str(pe.attributes[s]), box.w - 24)
            y += 18
        parts.append(pe.model_dump())
        region_truth[pid] = pe.model_dump()
        regions.append(Tile(name=pid, box=box, target="builder_part"))

    # ---- obstacles ----------------------------------------------------
    n_obs = rng.choice([3, 4])
    obs_box = Rect(740, 70, 340, 60 + n_obs * 56)
    _panel(d, obs_box, "OBSTACLES", f"{n_obs}", tag=tag)
    obstacles: list[dict[str, Any]] = []
    y = obs_box.y + 36
    for i in range(n_obs):
        oe = ObstacleExtraction(
            id=f"O{i + 1}",
            order=i + 1,
            name=f"stage {i + 1}",
            requires={s: rng.randint(1, 14) for s in rng.sample(stats, k=min(2, len(stats)))},
        )
        req = "  ".join(f"{k} {v}" for k, v in (oe.requires or {}).items())
        d.text((obs_box.x + 12, y), f"{oe.id}. {oe.name}", font=_font(13, bold=True), fill=INK)
        d.text((obs_box.x + 12, y + 18), req, font=_font(13), fill=DIM)
        y += 56
        obstacles.append(oe.model_dump())
    regions.append(Tile(name="obstacles", box=obs_box, target="builder_obstacles"))
    region_truth["obstacles"] = {"obstacles": obstacles}

    # ---- instructions -------------------------------------------------
    weight_max = rng.randint(12, 40)
    slot_max = rng.choice([None, 3, 4])
    duplicates = rng.choice([None, True, False])
    ordering = rng.choice([None, "ordered", "unordered"])
    quoted = [f"Total weight may not exceed {weight_max}."]
    if slot_max:
        quoted.append(f"Use at most {slot_max} parts.")
    if duplicates is True:
        quoted.append("A part may be used more than once.")
    elif duplicates is False:
        quoted.append("Each part may be used at most once.")
    if ordering == "ordered":
        quoted.append("Obstacles are faced in the order listed.")
    elif ordering == "unordered":
        quoted.append("Obstacles may be faced in any order.")
    rules = BuilderRulesExtraction(
        weight_max=weight_max,
        slot_max=slot_max,
        money_max=None,
        duplicates_allowed=duplicates,
        attribute_names=stats,
        obstacle_ordering=ordering,  # type: ignore[arg-type]
        obstacle_semantics=None,
        aggregation=None,
        failure_mode=None,
        objective="count_valid",
        quoted_rule_text=quoted,
    )
    ins_box = Rect(740, obs_box.y + obs_box.h + 24, 340, 260)
    _panel(d, ins_box, "INSTRUCTIONS", tag=tag)
    y = ins_box.y + 36
    for line in quoted + ["How many valid builds are possible?"]:
        for chunk in _wrap(line, 44):
            d.text((ins_box.x + 12, y), chunk, font=_font(13), fill=INK)
            y += 18
        y += 4
    regions.append(Tile(name="rules", box=ins_box, target="builder_instructions"))
    region_truth["rules"] = rules.model_dump()

    ground_truth = {"parts": parts, "obstacles": obstacles, "rules": rules.model_dump()}
    return img, {
        "regions": regions,
        "region_truth": region_truth,
        "ground_truth": ground_truth,
    }


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def generate(out_dir: Path | str = DEFAULT_OUT, n: int = N_PER_PUZZLE, seed: int = 20240816) -> list[Path]:
    """Render ``n`` fixtures per puzzle and write PNG + sidecar for each."""
    out = Path(out_dir)
    written: list[Path] = []
    for puzzle, maker in (("factory", _make_factory), ("builder", _make_builder)):
        pdir = out / puzzle
        pdir.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            rng = random.Random(f"{seed}:{puzzle}:{i}")
            img, payload = maker(i, rng)
            fid = f"{puzzle}_{i:03d}"
            img_path = pdir / f"{fid}.png"
            img.save(img_path, format="PNG", optimize=True)
            fixture = Fixture(
                fixture_id=fid,
                puzzle_type=puzzle,
                image_path=img_path,
                regions=payload["regions"],
                ground_truth=payload["ground_truth"],
                region_truth=payload["region_truth"],
                source="synthetic",
                notes={
                    "generator": "tests/fixtures/generate_fixtures.py",
                    "verified": "ground truth is the render input, so it is exact by construction",
                },
            )
            sidecar = pdir / f"{fid}.json"
            sidecar.write_text(json.dumps(fixture.sidecar(), indent=1, default=str))
            written.append(sidecar)
    return written


# --------------------------------------------------------------------------
# Fake vision client
# --------------------------------------------------------------------------


@dataclass
class ErrorSpec:
    """One injected reading error for one model.

    ``paths`` are dotted paths *within the region document* (``output_max``,
    ``recipes[M1-r1].sale_price``, ``*``) and accept fnmatch wildcards.
    """

    paths: tuple[str, ...] = ("*",)
    mode: str = "corrupt"  # corrupt | null | set | drop
    value: Any = None
    regions: tuple[str, ...] = ()
    fixtures: tuple[str, ...] = ()

    def matches(self, fixture_id: str, region: str) -> bool:
        if self.regions and not any(fnmatch.fnmatch(region, r) for r in self.regions):
            return False
        if self.fixtures and not any(fnmatch.fnmatch(fixture_id, f) for f in self.fixtures):
            return False
        return True

    def hits(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, p) for p in self.paths)


def _corrupt(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 7
    if isinstance(value, float):
        return value + 7.0
    if isinstance(value, str):
        return value + "X"
    return 0


class FakeVisionClient:
    """An :class:`LLMClient` that reads fixtures instead of pixels.

    It indexes every region crop exactly as the pipeline prepares it (same
    box, same downscale, same PNG), so a call is matched to a region by the
    bytes the pipeline actually sent -- not by a side channel.  A crop it does
    not recognise comes back all-null, which is the honest answer for an
    illegible image and is what the "nulls, not guesses" test asserts.
    """

    def __init__(
        self,
        fixtures: Sequence[Fixture] = (),
        *,
        errors: dict[str, Sequence[ErrorSpec]] | None = None,
        latency_ms: float = 0.0,
        per_model_latency_ms: dict[str, float] | None = None,
        hang_models: Sequence[str] = (),
        illegible: Sequence[str] = (),
        delta_changes: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.errors = {m: list(specs) for m, specs in (errors or {}).items()}
        self.latency_ms = latency_ms
        self.per_model_latency_ms = dict(per_model_latency_ms or {})
        self.hang_models = set(hang_models)
        self.illegible = set(illegible)
        self.delta_changes = list(delta_changes or [])
        self.calls: list[dict[str, Any]] = []
        self._index: dict[str, tuple[str, str]] = {}
        self._truth: dict[str, dict[str, Any]] = {}
        for fx in fixtures:
            self.add_fixture(fx)

    # -- indexing -------------------------------------------------------
    @classmethod
    def from_dir(
        cls,
        fixture_dir: Path | str,
        puzzle_type: str | None = None,
        *,
        limit: int | None = None,
        **kw: Any,
    ) -> "FakeVisionClient":
        return cls(load_fixtures(fixture_dir, puzzle_type, limit=limit), **kw)

    def add_fixture(self, fixture: Fixture) -> None:
        self._truth[fixture.fixture_id] = fixture.region_truth
        image = fixture.load_image()
        for tile in fixture.regions:
            prepared = prepare(image, tile.box, max_long_edge=tile.max_long_edge)
            key = _digest(prepared.data_url)
            clash = self._index.get(key)
            if clash is not None and clash != (fixture.fixture_id, tile.name):
                # Two fixtures that render byte-identical crops would silently
                # answer for each other and the accuracy numbers would be
                # fiction.  Fail loudly instead.
                raise ValueError(
                    f"fixture crop collision: {fixture.fixture_id}:{tile.name} renders "
                    f"identically to {clash[0]}:{clash[1]}; make the fixtures distinct"
                )
            self._index[key] = (fixture.fixture_id, tile.name)

    # -- LLMClient ------------------------------------------------------
    async def complete(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        response_format: dict[str, Any] | None = None,
        **kw: Any,
    ) -> LLMResponse:
        schema_name = (response_format or {}).get("json_schema", {}).get("name", "")
        data_url = _first_image(messages)
        key = _digest(data_url) if data_url else ""
        located = self._index.get(key)
        self.calls.append({"model": model, "schema": schema_name, "region": located})

        if model in self.hang_models:
            await asyncio.sleep(3600)

        delay = self.per_model_latency_ms.get(model, self.latency_ms)
        if delay:
            await asyncio.sleep(delay / 1000.0)

        if schema_name == "DeltaChanges":
            doc: dict[str, Any] = {"changes": [dict(c) for c in self.delta_changes]}
        else:
            doc = self._document(model, schema_name, located)

        return LLMResponse(
            content=json.dumps(doc, default=str),
            parsed=doc,
            model=model,
            provider="fake",
            usage=Usage(prompt_tokens=800, completion_tokens=200),
            finish_reason="stop",
            latency_ms=delay,
        )

    def stream(self, **kw: Any) -> AsyncIterator[StreamEvent]:  # pragma: no cover
        raise NotImplementedError("FakeVisionClient does not stream")

    # -- answer construction --------------------------------------------
    def _document(
        self, model: str, schema_name: str, located: tuple[str, str] | None
    ) -> dict[str, Any]:
        schema = _schema_by_name(schema_name)
        if located is None:
            return _null_document(schema)
        fixture_id, region = located
        if region in self.illegible or f"{fixture_id}:{region}" in self.illegible:
            return _null_document(schema)
        truth = (self._truth.get(fixture_id) or {}).get(region)
        if truth is None:
            return _null_document(schema)
        doc = json.loads(json.dumps(truth, default=str))
        specs = [s for s in self.errors.get(model, ()) if s.matches(fixture_id, region)]
        if not specs:
            return doc
        flat = flatten(doc)
        for spec in specs:
            for path in list(flat):
                if not spec.hits(path):
                    continue
                if spec.mode == "null":
                    flat[path] = None
                elif spec.mode == "set":
                    flat[path] = spec.value
                elif spec.mode == "drop":
                    flat.pop(path)
                else:
                    flat[path] = _corrupt(flat[path])
        return unflatten(flat)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _first_image(messages: Sequence[ChatMessage]) -> str | None:
    for msg in messages:
        for part in msg.parts or ():
            if part.get("type") == "image_url":
                return part.get("image_url", {}).get("url")
    return None


_SCHEMAS: dict[str, Any] = {}


def _schema_by_name(name: str) -> Any:
    if not _SCHEMAS:
        for target in prompts.TARGETS.values():
            _SCHEMAS[target.schema.__name__] = target.schema
        from services.core.extract.delta import DeltaChanges

        _SCHEMAS["DeltaChanges"] = DeltaChanges
    return _SCHEMAS.get(name)


def _null_document(schema: Any) -> dict[str, Any]:
    """"I cannot read this" in the shape the schema demands."""
    if schema is None:
        return {}
    out: dict[str, Any] = {}
    for fname, finfo in schema.model_fields.items():
        ann = str(finfo.annotation)
        out[fname] = [] if ann.startswith("list[") else None
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="generate_fixtures")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=N_PER_PUZZLE)
    ap.add_argument("--seed", type=int, default=20240816)
    args = ap.parse_args(argv)
    written = generate(args.out, args.n, args.seed)
    print(f"wrote {len(written)} fixtures under {args.out}")
    return 0


__all__ = [
    "ErrorSpec",
    "FakeVisionClient",
    "generate",
    "DEFAULT_OUT",
    "N_PER_PUZZLE",
]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
