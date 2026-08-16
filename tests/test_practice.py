"""Practice mode: replay fixtures, measure, and above all catch wrong auto-confirms."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.core.extract.pipeline import ExtractRoles, clear_speculations
from services.core.extract.practice import (
    load_fixtures,
    run_practice,
    score_fixture,
)
from tests.fixtures.generate_fixtures import DEFAULT_OUT, ErrorSpec, FakeVisionClient, generate

MODELS = ["famA/one", "famB/two", "famC/three"]
ROLES = ExtractRoles(jury=MODELS, topology="famA/one", delta="famA/one")


@pytest.fixture(autouse=True)
def _clean():
    clear_speculations()
    yield
    clear_speculations()


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("captures")
    generate(out, n=3)
    return out


# --------------------------------------------------------------------------
# the committed fixture set
# --------------------------------------------------------------------------


def test_committed_fixture_set_has_at_least_thirty_per_puzzle():
    for puzzle in ("factory", "builder"):
        fixtures = load_fixtures(DEFAULT_OUT, puzzle)
        assert len(fixtures) >= 30, f"only {len(fixtures)} {puzzle} fixtures"
        for fx in fixtures:
            assert fx.image_path.exists()
            assert fx.regions and fx.ground_truth
            # every region carries hand-verifiable truth of its own
            assert set(fx.region_truth) >= {t.name for t in fx.regions if t.name != "topology"}


@pytest.mark.slow
async def test_the_whole_committed_fixture_set_replays_cleanly(capsys):
    """End to end over all 60 boards: a perfect reader must score perfectly.

    Building the client also asserts that no two fixture crops are
    byte-identical, which would let one fixture answer for another and make
    every accuracy number here fiction.
    """
    for puzzle in ("factory", "builder"):
        client = FakeVisionClient.from_dir(DEFAULT_OUT, puzzle)
        report = await run_practice(client, ROLES, DEFAULT_OUT, puzzle)
        with capsys.disabled():
            print("\n" + report.summary())
        assert report.fixtures >= 30
        assert report.field_accuracy == 1.0
        assert report.wrong_auto_confirm_count == 0
        assert report.passed


def test_fixture_sidecars_are_the_same_shape_a_real_capture_would_use():
    fx = load_fixtures(DEFAULT_OUT, "factory", limit=1)[0]
    side = fx.sidecar()
    assert set(side) >= {
        "fixture_id",
        "puzzle_type",
        "image",
        "regions",
        "region_truth",
        "ground_truth",
        "source",
    }
    assert side["regions"][0]["box"].keys() == {"x", "y", "w", "h"}


# --------------------------------------------------------------------------
# a clean replay
# --------------------------------------------------------------------------


async def test_clean_replay_is_accurate_fast_and_confirms_most_fields(fixtures_dir, capsys):
    client = FakeVisionClient.from_dir(fixtures_dir, "factory")
    report = await run_practice(client, ROLES, fixtures_dir, "factory")

    with capsys.disabled():
        print("\n" + report.summary())

    assert report.fixtures == 3
    assert report.field_accuracy == 1.0
    assert report.wrong_auto_confirm_count == 0
    assert report.passed is True
    assert report.auto_confirm_rate > 0.6
    stages = {s.stage for s in report.stages}
    assert {"prepare", "fanout", "assemble", "tile:*", "total"} <= stages
    assert all(s.p50_ms <= s.p90_ms <= s.max_ms for s in report.stages)
    assert report.p90_total_ms >= report.p50_total_ms > 0


async def test_builder_fixtures_replay_too(fixtures_dir):
    client = FakeVisionClient.from_dir(fixtures_dir, "builder")
    report = await run_practice(client, ROLES, fixtures_dir, "builder")
    assert report.fixtures == 3
    assert report.field_accuracy == 1.0
    assert report.passed and report.wrong_auto_confirm_count == 0


# --------------------------------------------------------------------------
# injected disagreement
# --------------------------------------------------------------------------


async def test_one_model_misreading_a_field_is_flagged_not_auto_confirmed(fixtures_dir):
    """The point of the jury: a lone bad reading costs a keystroke, not the run."""
    client = FakeVisionClient.from_dir(
        fixtures_dir,
        "factory",
        errors={MODELS[2]: [ErrorSpec(paths=("output_max", "storage_max"), mode="corrupt")]},
    )
    report = await run_practice(client, ROLES, fixtures_dir, "factory")

    assert report.disputed_count > 0
    assert report.wrong_auto_confirm_count == 0
    assert report.passed is True
    # the majority still produced the right value, so accuracy is untouched
    assert report.field_accuracy == 1.0

    fixtures = load_fixtures(fixtures_dir, "factory")
    from services.core.extract.pipeline import extract

    result = await extract(client, ROLES, [fixtures[0].as_capture()], "factory")
    flagged = [p for p in result.disputed if p.endswith(".output_max")]
    assert flagged, "the misread field must reach the user"
    prov = result.provenance_for(flagged[0])
    assert prov.status == "majority"
    assert prov.votes[MODELS[0]] == prov.value != prov.votes[MODELS[2]]
    assert prov.auto_confirmed is False


async def test_two_models_disagreeing_three_ways_leaves_the_field_empty(fixtures_dir):
    client = FakeVisionClient.from_dir(
        fixtures_dir,
        "factory",
        errors={
            MODELS[1]: [ErrorSpec(paths=("output_max",), mode="set", value=111)],
            MODELS[2]: [ErrorSpec(paths=("output_max",), mode="set", value=222)],
        },
    )
    fixtures = load_fixtures(fixtures_dir, "factory", limit=1)
    from services.core.extract.pipeline import extract

    result = await extract(client, ROLES, [fixtures[0].as_capture()], "factory")
    split = [p for p in result.disputed if p.endswith(".output_max")]
    assert split
    prov = result.provenance_for(split[0])
    assert prov.status == "split" and prov.value is None  # shown empty, with the crop
    assert prov.crop_id and prov.box["w"] > 0
    assert set(prov.votes.values()) == {prov.votes[MODELS[0]], 111, 222}

    outcome, wrongs = score_fixture(fixtures[0], result)
    assert wrongs == []  # a split is honest, not wrong
    assert outcome.fields_correct < outcome.fields_total


# --------------------------------------------------------------------------
# nulls, not guesses
# --------------------------------------------------------------------------


async def test_an_illegible_crop_produces_nulls_rather_than_guesses(fixtures_dir):
    client = FakeVisionClient.from_dir(fixtures_dir, "factory", illegible=["M1"])
    fixtures = load_fixtures(fixtures_dir, "factory", limit=1)
    from services.core.extract.pipeline import extract

    result = await extract(client, ROLES, [fixtures[0].as_capture()], "factory")

    m1 = [p for p in result.provenance if p.tile == "M1"]
    assert m1, "the M1 tile must still appear"
    assert all(p.value is None for p in m1), "an illegible crop must not be guessed at"
    assert all(p.status == "unanimous" for p in m1)
    # the fields the solver needs are held back for a human...
    needed = [p for p in m1 if p.path.endswith((".id", ".kind"))]
    assert needed and all(not p.auto_confirmed for p in needed)
    assert all(p.path in result.unresolved for p in needed)
    # ...and the rest are simply absent, costing nobody a keystroke
    assert any(p.auto_confirmed for p in m1)
    # the assembler reports the hole rather than inventing a machine
    assert result.factory_state is not None
    assert "M1" not in [m.id for m in result.factory_state.machines]
    assert any("id_or_kind" in u for u in result.unresolved)

    # Practice mode still calls this out, and should: a field that WAS on the
    # screen and that nobody could read is a real alarm before the clock starts.
    # What matters is that the alarm is about nulls, never about invented values.
    _, wrongs = score_fixture(fixtures[0], result)
    assert wrongs, "an unreadable crop must not pass practice silently"
    assert all(w.confirmed is None for w in wrongs), "practice saw a fabricated value"
    assert all(w.tile == "M1" for w in wrongs)  # scoped to the illegible crop


# --------------------------------------------------------------------------
# the detector itself
# --------------------------------------------------------------------------


async def test_the_two_safety_properties_hold_under_every_failure_mode(fixtures_dir):
    """No auto-confirmed field is ever fabricated, and no field the solver needs
    is ever silently defaulted -- clean, with a misreading juror, and with a
    crop nobody can read."""
    from services.core.extract.pipeline import extract

    scenarios: dict[str, dict] = {
        "clean": {},
        "one juror misreads": {
            "errors": {MODELS[2]: [ErrorSpec(paths=("*weight", "*output_max"), mode="corrupt")]}
        },
        "an illegible panel": {"illegible": ["M1", "P1"]},
    }
    for puzzle in ("factory", "builder"):
        for label, kw in scenarios.items():
            client = FakeVisionClient.from_dir(fixtures_dir, puzzle, **kw)
            for fx in load_fixtures(fixtures_dir, puzzle):
                result = await extract(client, ROLES, [fx.as_capture()], puzzle)
                needed = set(result.unresolved)
                silent = [p.path for p in result.provenance if p.auto_confirmed and p.path in needed]
                assert not silent, f"{puzzle}/{label}: solver-critical fields defaulted: {silent}"
                _, wrongs = score_fixture(fx, result)
                fabricated = [w for w in wrongs if w.confirmed is not None]
                assert not fabricated, f"{puzzle}/{label}: {fabricated[0].as_text()}"


async def test_a_dropped_record_takes_its_whole_panel_into_review(fixtures_dir):
    """An unreadable id drops the part; its other nulls must not read as 'absent'.

    Otherwise a reviewer who fixes the id inherits weight 0 -- a free part the
    solver will happily use -- without ever being asked.
    """
    from services.core.extract.pipeline import extract

    client = FakeVisionClient.from_dir(fixtures_dir, "builder", illegible=["P1"])
    fx = load_fixtures(fixtures_dir, "builder", limit=1)[0]
    result = await extract(client, ROLES, [fx.as_capture()], "builder")

    p1 = [p for p in result.provenance if p.tile == "P1"]
    assert p1 and not any(p.auto_confirmed for p in p1)
    assert {"parts[P1].id", "parts[P1].weight"} <= set(result.unresolved)
    # a legible neighbour is untouched: the blast radius is one panel
    assert sum(1 for p in result.provenance if p.tile == "P2" and p.auto_confirmed) >= 5


async def test_a_correlated_error_across_all_three_models_is_reported_as_a_hard_failure(
    fixtures_dir,
):
    """If all three families make the same mistake the jury cannot save you --
    so practice mode has to be able to see it."""
    spec = ErrorSpec(paths=("output_max",), mode="set", value=999)
    client = FakeVisionClient.from_dir(
        fixtures_dir, "factory", errors={m: [spec] for m in MODELS}
    )
    report = await run_practice(client, ROLES, fixtures_dir, "factory")

    assert report.wrong_auto_confirm_count > 0
    assert report.passed is False
    worst = report.wrong_auto_confirms[0]
    assert worst.fixture.startswith("factory_")  # names the offending fixture
    assert worst.path.endswith(".output_max")
    assert worst.confirmed == 999 and worst.expected != 999
    assert set(worst.votes) == set(MODELS)
    assert worst.crop_id and worst.tile
    assert "HARD FAILURE" in report.summary()
    assert worst.as_text() in report.summary()


async def test_missing_fixtures_are_reported_not_crashed(tmp_path):
    report = await run_practice(FakeVisionClient(), ROLES, tmp_path, "factory")
    assert report.fixtures == 0 and "fixtures" in report.errors
    assert report.passed is True  # nothing ran, nothing was wrongly confirmed


def test_cli_entry_point_exists():
    from services.core.extract import practice

    assert callable(practice.main)
    with pytest.raises(SystemExit):
        practice.main(["--help"])
