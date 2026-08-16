"""Practice mode (spec 6.3): the latency work that happens before the clock starts.

Replays recorded captures through the *whole* pipeline -- same tiles, same
jury, same assembly -- and reports two things:

* **per-stage latency** (p50/p90), so you know which stage to attack next; and
* **extraction accuracy** against hand-verified ground truth.

The metric that matters is neither of those, though.  It is
**wrong auto-confirms**: a field the jury agreed on unanimously whose value is
not the truth.  A low agreement rate costs keystrokes; a wrong auto-confirm
costs the run, silently, because by construction the user never sees it.  Any
non-zero count is a hard failure and names the offending fixture.

CLI::

    python -m services.core.extract.practice --fixtures tests/fixtures/captures --puzzle factory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from services.core.capture.grab import Capture, Rect, Tile, capture_from_image
from services.core.extract.pipeline import ExtractionResult, ExtractRoles, extract
from services.core.extract.schemas import flatten
from services.core.llm.protocol import LLMClient

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@dataclass
class Fixture:
    """One recorded board: an image, its regions, and verified ground truth.

    Synthetic fixtures and real captures use the same sidecar shape, so a real
    PNG dropped next to a real JSON replays through this harness unchanged.
    """

    fixture_id: str
    puzzle_type: str
    image_path: Path
    regions: list[Tile]
    ground_truth: dict[str, Any]
    #: region name -> the part of the ground truth that region shows
    region_truth: dict[str, Any] = field(default_factory=dict)
    source: str = "synthetic"
    scale: float = 1.0
    notes: dict[str, Any] = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        with Image.open(self.image_path) as im:
            return im.convert("RGB")

    def as_capture(self) -> Capture:
        image = self.load_image()
        return capture_from_image(
            image,
            region=Rect(0, 0, image.width, image.height),
            scale=self.scale,
            puzzle_type=self.puzzle_type,
            capture_id=f"fixture_{self.fixture_id}",
            tiles=self.regions,
            source="fixture",
        )

    def sidecar(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "puzzle_type": self.puzzle_type,
            "image": self.image_path.name,
            "scale": self.scale,
            "source": self.source,
            "verified": True,
            "regions": [t.as_dict() for t in self.regions],
            "region_truth": self.region_truth,
            "ground_truth": self.ground_truth,
            "notes": self.notes,
        }


def load_fixture(path: Path) -> Fixture:
    data = json.loads(Path(path).read_text())
    base = Path(path).parent
    return Fixture(
        fixture_id=data["fixture_id"],
        puzzle_type=data["puzzle_type"],
        image_path=base / data["image"],
        regions=[Tile.from_dict(r) for r in data.get("regions", [])],
        ground_truth=data.get("ground_truth", {}),
        region_truth=data.get("region_truth", {}),
        source=data.get("source", "synthetic"),
        scale=float(data.get("scale", 1.0)),
        notes=data.get("notes", {}),
    )


def load_fixtures(
    fixture_dir: Path | str, puzzle_type: str | None = None, *, limit: int | None = None
) -> list[Fixture]:
    """Load every sidecar under ``fixture_dir`` (recursively)."""
    root = Path(fixture_dir)
    if not root.exists():
        raise FileNotFoundError(f"no fixture directory at {root}")
    out: list[Fixture] = []
    for p in sorted(root.rglob("*.json")):
        try:
            fx = load_fixture(p)
        except (KeyError, json.JSONDecodeError):
            continue
        if puzzle_type and fx.puzzle_type != puzzle_type:
            continue
        out.append(fx)
    out.sort(key=lambda f: f.fixture_id)
    return out[:limit] if limit else out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


class StageStat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    n: int
    p50_ms: float
    p90_ms: float
    max_ms: float


class WrongAutoConfirm(BaseModel):
    """The failure that matters: agreed, confident, and wrong."""

    model_config = ConfigDict(extra="forbid")

    fixture: str
    path: str
    confirmed: Any = None
    expected: Any = None
    votes: dict[str, Any] = Field(default_factory=dict)
    crop_id: str | None = None
    tile: str = ""

    def as_text(self) -> str:
        return (
            f"{self.fixture}: {self.path} auto-confirmed as {self.confirmed!r} "
            f"but truth is {self.expected!r} (votes: {self.votes})"
        )


class FixtureOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    fields_total: int = 0
    fields_correct: int = 0
    fields_missing: int = 0
    auto_confirmed: int = 0
    disputed: int = 0
    wrong_auto_confirms: int = 0
    elapsed_ms: float = 0.0
    n_calls: int = 0
    errors: dict[str, str] = Field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.fields_correct / self.fields_total if self.fields_total else 0.0


class PracticeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    puzzle_type: str
    fixtures: int = 0
    stages: list[StageStat] = Field(default_factory=list)
    p50_total_ms: float = 0.0
    p90_total_ms: float = 0.0
    field_accuracy: float = 0.0
    auto_confirm_rate: float = 0.0
    disputed_count: int = 0
    auto_confirmed_count: int = 0
    fields_total: int = 0
    fields_correct: int = 0
    wrong_auto_confirm_count: int = 0
    wrong_auto_confirms: list[WrongAutoConfirm] = Field(default_factory=list)
    per_fixture: list[FixtureOutcome] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """A run passes only with zero wrong auto-confirms."""
        return self.wrong_auto_confirm_count == 0

    def summary(self) -> str:
        lines = [
            f"practice: {self.puzzle_type}  fixtures={self.fixtures}",
            f"  total latency   p50 {self.p50_total_ms:8.1f} ms   p90 {self.p90_total_ms:8.1f} ms",
        ]
        for s in self.stages:
            lines.append(
                f"  {s.stage:<22} p50 {s.p50_ms:8.1f} ms   p90 {s.p90_ms:8.1f} ms  (n={s.n})"
            )
        lines += [
            f"  field accuracy      {self.field_accuracy:6.1%}"
            f"  ({self.fields_correct}/{self.fields_total})",
            f"  auto-confirm rate   {self.auto_confirm_rate:6.1%}"
            f"  ({self.auto_confirmed_count} auto / {self.disputed_count} reviewed)",
        ]
        if self.wrong_auto_confirm_count:
            lines.append(f"  WRONG AUTO-CONFIRMS: {self.wrong_auto_confirm_count}  <-- HARD FAILURE")
            for w in self.wrong_auto_confirms[:20]:
                lines.append(f"    ! {w.as_text()}")
        else:
            lines.append("  wrong auto-confirms 0  (ok)")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


_EDGE_RE = re.compile(r"^(?P<pre>.*edges)\[(?P<key>[^\]]+)\]\.(?P<leaf>src|dst)$")


def canonical_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Re-key edge lists by ``src->dst`` so position never counts as a difference.

    Two readings of the same graph that start from different corners are the
    same graph; scoring them as different would punish the pipeline for being
    order-independent, which is exactly what we asked it to be.
    """
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    out: dict[str, Any] = {}
    for path, value in flat.items():
        m = _EDGE_RE.match(path)
        if m:
            edges.setdefault((m["pre"], m["key"]), {})[m["leaf"]] = value
        else:
            out[path] = value
    for (pre, _), edge in edges.items():
        key = f"{edge.get('src')}->{edge.get('dst')}"
        out[f"{pre}[{key}].src"] = edge.get("src")
        out[f"{pre}[{key}].dst"] = edge.get("dst")
    return out


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return str(a) == str(b)


def score_fixture(
    fixture: Fixture, result: ExtractionResult
) -> tuple[FixtureOutcome, list[WrongAutoConfirm]]:
    truth = canonical_flat(flatten(fixture.ground_truth))
    got = canonical_flat(flatten(result.extraction))
    correct = 0
    missing = 0
    for path, expected in truth.items():
        if path not in got:
            missing += 1
            continue
        if _same(got[path], expected):
            correct += 1

    wrongs: list[WrongAutoConfirm] = []
    for path in result.auto_confirmed:
        prov = result.provenance_for(path)
        value = got.get(path, prov.value if prov else None)
        if path in truth and _same(value, truth[path]):
            continue
        wrongs.append(
            WrongAutoConfirm(
                fixture=fixture.fixture_id,
                path=path,
                confirmed=value,
                expected=truth.get(path),
                votes=(prov.votes if prov else {}),
                crop_id=(prov.crop_id if prov else None),
                tile=(prov.tile if prov else ""),
            )
        )

    outcome = FixtureOutcome(
        fixture_id=fixture.fixture_id,
        fields_total=len(truth),
        fields_correct=correct,
        fields_missing=missing,
        auto_confirmed=len(result.auto_confirmed),
        disputed=len(result.disputed),
        wrong_auto_confirms=len(wrongs),
        elapsed_ms=result.elapsed_ms,
        n_calls=result.n_calls,
        errors=dict(result.errors),
    )
    return outcome, wrongs


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


async def run_practice(
    client: LLMClient,
    roles: "ExtractRoles | dict | Sequence[str] | None",
    fixture_dir: Path | str,
    puzzle_type: str,
    *,
    limit: int | None = None,
    concurrency: int = 1,
    timeout_s: float = 12.0,
) -> PracticeReport:
    """Replay ``fixture_dir`` through the full pipeline and grade the result."""
    fixtures = load_fixtures(fixture_dir, puzzle_type, limit=limit)
    roles_ = ExtractRoles.coerce(roles)
    report = PracticeReport(puzzle_type=puzzle_type, fixtures=len(fixtures))
    if not fixtures:
        report.errors["fixtures"] = f"no {puzzle_type} fixtures under {fixture_dir}"
        return report

    stage_samples: dict[str, list[float]] = {}
    totals: list[float] = []
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(fx: Fixture) -> tuple[Fixture, ExtractionResult | BaseException]:
        async with sem:
            try:
                # Fresh capture per fixture; no speculation cache, so the timings
                # measure the real pass rather than a cache hit.
                return fx, await extract(
                    client, roles_, [fx.as_capture()], fx.puzzle_type, timeout_s=timeout_s
                )
            except Exception as exc:  # a broken fixture must not stop the run
                return fx, exc

    for fx, outcome in await asyncio.gather(*(one(f) for f in fixtures)):
        if isinstance(outcome, BaseException):
            report.errors[fx.fixture_id] = f"{type(outcome).__name__}: {outcome}"
            continue
        for stage, ms in outcome.stage_ms.items():
            key = "tile:*" if stage.startswith("tile:") else stage
            stage_samples.setdefault(key, []).append(ms)
        totals.append(outcome.elapsed_ms)
        fo, wrongs = score_fixture(fx, outcome)
        report.per_fixture.append(fo)
        report.wrong_auto_confirms.extend(wrongs)
        report.fields_total += fo.fields_total
        report.fields_correct += fo.fields_correct
        report.auto_confirmed_count += fo.auto_confirmed
        report.disputed_count += fo.disputed

    report.per_fixture.sort(key=lambda f: f.fixture_id)
    report.stages = [
        StageStat(
            stage=stage,
            n=len(vals),
            p50_ms=_pct(vals, 0.5),
            p90_ms=_pct(vals, 0.9),
            max_ms=round(max(vals), 2),
        )
        for stage, vals in sorted(stage_samples.items())
    ]
    report.p50_total_ms = _pct(totals, 0.5)
    report.p90_total_ms = _pct(totals, 0.9)
    report.field_accuracy = (
        report.fields_correct / report.fields_total if report.fields_total else 0.0
    )
    reviewed = report.auto_confirmed_count + report.disputed_count
    report.auto_confirm_rate = report.auto_confirmed_count / reviewed if reviewed else 0.0
    report.wrong_auto_confirm_count = len(report.wrong_auto_confirms)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_client(args: argparse.Namespace, fixture_dir: Path, puzzle: str) -> LLMClient:
    if args.fake:
        try:
            from tests.fixtures.generate_fixtures import FakeVisionClient
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"--fake needs the test fixtures package: {exc}") from None
        return FakeVisionClient.from_dir(fixture_dir, puzzle_type=puzzle)
    try:
        from services.core.llm.client import OpenRouterClient  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "No LLM client available (services/core/llm/client.py). "
            f"Run with --fake to practise against the fixture ground truth. ({exc})"
        ) from None
    return OpenRouterClient()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="practice", description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", required=True, help="directory of fixture sidecars")
    ap.add_argument("--puzzle", default="factory", choices=["factory", "builder"])
    ap.add_argument("--models", default="", help="comma-separated jury models")
    ap.add_argument("--tie-breaker", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--fake", action="store_true", help="replay with the fixture fake client")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    fixture_dir = Path(args.fixtures)
    roles = ExtractRoles()
    if args.models:
        roles = ExtractRoles(jury=[m.strip() for m in args.models.split(",") if m.strip()])
    if args.tie_breaker:
        roles.tie_breaker = args.tie_breaker

    client = _build_client(args, fixture_dir, args.puzzle)
    report = asyncio.run(
        run_practice(
            client,
            roles,
            fixture_dir,
            args.puzzle,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    )
    if args.json:
        print(json.dumps(report.model_dump(), indent=2, default=str))
    else:
        print(report.summary())
    return 0 if report.passed else 1


__all__ = [
    "Fixture",
    "FixtureOutcome",
    "PracticeReport",
    "StageStat",
    "WrongAutoConfirm",
    "load_fixture",
    "load_fixtures",
    "run_practice",
    "score_fixture",
    "main",
]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
