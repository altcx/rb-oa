"""Latency budgets.  Spec section 6: every number there is a budget with a test.

These are deliberately generous relative to the spec's targets — a shared CI box
is not the user's desktop — but they fail loudly on a regression of the *shape*
of the pipeline (a serialized fan-out, a solver with no anytime path, a hot loop
that allocates).  Each test PRINTS its measurement so `-s` gives you the real
numbers rather than just a pass.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytestmark = pytest.mark.latency


def _factory_fixture():
    """A ten-machine chain, the size the spec quotes its simulator budget for."""
    from services.solvers.factory.model import (
        Edge,
        FactoryState,
        Item,
        Machine,
        MachineKind,
        Recipe,
    )

    machines = [
        Machine(
            id="s0",
            kind=MachineKind.SUPPLIER,
            recipes=[Recipe(id="r0", output_item="i0", output_qty=4, purchase_cost=1.0)],
            storage_max=40,
            output_max=4,
            y=0,
        )
    ]
    edges: list[Edge] = []
    for k in range(1, 9):
        machines.append(
            Machine(
                id=f"m{k}",
                kind=MachineKind.MAKER,
                recipes=[
                    Recipe(
                        id=f"r{k}",
                        inputs={f"i{k - 1}": 1},
                        output_item=f"i{k}",
                        output_qty=1,
                        production_cost=0.5,
                    )
                ],
                storage_max=40,
                output_max=4,
                y=k,
            )
        )
        edges.append(Edge(src=machines[-2].id, dst=machines[-1].id))
    machines.append(
        Machine(
            id="sell",
            kind=MachineKind.SELLER,
            recipes=[Recipe(id="rs", inputs={"i8": 1}, sale_price=12.0)],
            storage_max=0,
            output_max=4,
            y=9,
        )
    )
    edges.append(Edge(src="m8", dst="sell"))
    return FactoryState(
        machines=machines,
        edges=edges,
        items=[Item(id=f"i{i}") for i in range(9)],
        starting_money=500.0,
        horizon_hours=24,
    )


def test_simulator_hot_path_is_microseconds():
    """The optimizer needs 1e5..1e6 evaluations inside a 30 s budget, so a single
    ten-machine 24-hour simulation has to cost microseconds, not milliseconds."""
    sim = pytest.importorskip("services.solvers.factory.sim")
    state = _factory_fixture()
    from services.solvers.factory.model import FactoryConfig

    config = FactoryConfig.from_state(state)
    cf = sim.compile_factory(state)
    vec = sim.config_to_vector(cf, config)

    sim.evaluate_fast(cf, vec)  # warm
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        sim.evaluate_fast(cf, vec)
    per_call_us = (time.perf_counter() - t0) / n * 1e6
    print(f"\nevaluate_fast: {per_call_us:.1f} us/call ({1e6 / per_call_us:,.0f} evals/s)")
    assert per_call_us < 2000, f"{per_call_us:.0f} us/call is too slow for local search"


def test_upper_bound_returns_in_milliseconds():
    """The agent calls the bound first precisely because it is cheap."""
    bounds = pytest.importorskip("services.solvers.factory.bounds")
    state = _factory_fixture()
    bounds.upper_bound(state)  # warm the LP solver import
    t0 = time.perf_counter()
    result = bounds.upper_bound(state)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\nupper_bound: {ms:.1f} ms, ceiling={result.ceiling:.2f}")
    assert ms < 500
    assert result.bottleneck_machine is not None or result.notes


def test_optimizer_emits_a_usable_result_at_200ms():
    """Anytime everywhere: the user never sees a spinner (spec 6.3)."""
    optimize_mod = pytest.importorskip("services.solvers.factory.optimize")
    state = _factory_fixture()
    t0 = time.perf_counter()
    result = optimize_mod.optimize(state, seconds=0.2)
    elapsed = time.perf_counter() - t0
    print(f"\noptimize(0.2s): returned in {elapsed:.2f}s, best={result.best_value:.2f}")
    assert elapsed < 3.0, "a 200 ms budget must not overrun by an order of magnitude"
    assert result.best is not None and result.best_value is not None


def test_optimizer_streams_improvements_before_it_finishes():
    optimize_mod = pytest.importorskip("services.solvers.factory.optimize")
    state = _factory_fixture()
    stamps: list[float] = []
    t0 = time.perf_counter()
    optimize_mod.optimize(state, seconds=1.5, on_improve=lambda r: stamps.append(time.perf_counter() - t0))
    print(f"\non_improve fired {len(stamps)} times, first at {stamps[0] if stamps else float('nan'):.3f}s")
    assert stamps, "the optimizer never reported a best-so-far"
    assert stamps[0] < 1.0, "first improvement arrived too late to be an anytime result"


def test_builder_realistic_instance_under_ten_seconds():
    solve_mod = pytest.importorskip("services.solvers.builder.solve")
    from services.core.rules.dsl import BuilderRules
    from services.solvers.builder.model import BuilderPuzzle, Obstacle, Part

    rng_parts = [
        Part(
            id=f"p{i}",
            weight=1 + (i * 7) % 9,
            qty_available=1,
            attributes={"a": (i * 3) % 5, "b": (i * 5) % 4, "c": (i * 11) % 3},
        )
        for i in range(30)
    ]
    obstacles = [
        Obstacle(id=f"o{j}", order=j + 1, requires={"a": 6 + j, "b": 4 + j, "c": 2})
        for j in range(5)
    ]
    puzzle = BuilderPuzzle(
        parts=rng_parts,
        obstacles=obstacles,
        rules=BuilderRules(
            weight_max=30,
            duplicates_allowed=False,
            obstacle_semantics="threshold",
            aggregation="sum",
            failure_mode="all_must_pass",
            objective="count_valid",
        ),
    )
    t0 = time.perf_counter()
    report = solve_mod.solve_builder(puzzle, budget_s=10.0)
    elapsed = time.perf_counter() - t0
    print(f"\nsolve_builder(30 parts, 5 obstacles): {elapsed:.2f}s via {report.method}, "
          f"{report.total_valid} valid")
    assert elapsed < 10.0


def test_crop_and_downscale_budget():
    """Never send a full screen.  Crop + LANCZOS downscale to a 1568px long edge."""
    grab = pytest.importorskip("services.core.capture.grab")
    from PIL import Image

    img = Image.new("RGB", (3840, 2160), (30, 30, 30))
    grab.prepare(img, (100, 100, 1000, 700))  # warm
    t0 = time.perf_counter()
    prepared = grab.prepare(img, (100, 100, 1000, 700))
    ms = (time.perf_counter() - t0) * 1000
    print(f"\ncrop+downscale of 4K: {ms:.1f} ms")
    assert ms < 400
    assert prepared.data_url.startswith("data:image/")


async def test_capture_to_reviewable_state_under_four_seconds():
    """The M6 budget: capture to reviewable state under 4 s at p90.

    Driven with a fake client pinned at 800 ms per vision call — slower than the
    1.5 s target divided across a jury, and deliberately slow enough that a
    serialized fan-out (six tiles x three jurors x 800 ms = 14 s) blows the
    budget outright.  Passing therefore proves the *shape* of the pipeline, not
    the speed of this machine.
    """
    pipeline = pytest.importorskip("services.core.extract.pipeline")
    grab = pytest.importorskip("services.core.capture.grab")
    from PIL import Image

    from services.core.llm.protocol import LLMResponse

    concurrent = 0
    peak = 0

    class SleepyClient:
        async def complete(self, **kw):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            try:
                await asyncio.sleep(0.8)
            finally:
                concurrent -= 1
            return LLMResponse(content="{}", parsed={}, latency_ms=800.0)

        def stream(self, **kw):  # pragma: no cover - narration is tested elsewhere
            raise NotImplementedError

    capture = grab.Capture(
        image=Image.new("RGB", (1920, 1080), (20, 20, 20)),
        region=grab.Rect(0, 0, 1920, 1080),
        puzzle_type="factory",
        tiles=[grab.Tile(name=f"m{i}", box=grab.Rect(i * 300, 0, 300, 400)) for i in range(6)],
    )
    pipeline.clear_speculations()
    t0 = time.perf_counter()
    await pipeline.extract(SleepyClient(), None, capture, "factory")
    elapsed = time.perf_counter() - t0
    print(f"\ncapture -> reviewable state: {elapsed:.2f}s at 800 ms/call, peak concurrency {peak}")
    assert elapsed < 4.0, "extraction is serialized somewhere"
    assert peak > 3, "tiles and jurors are not running concurrently"
