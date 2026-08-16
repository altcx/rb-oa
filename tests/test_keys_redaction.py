"""Key validation, storage backend reporting, and — the one that matters —
that a key can never reach a log sink, including through a traceback.
"""

from __future__ import annotations

import io
import logging

import httpx
import pytest
import respx

from services.core.settings import keys

KEY = "sk-or-v1-" + "a1b2c3d4e5f6a7b8" * 4
GENERIC_KEY = "sk-proj-" + "Z" * 40
KEY_URL = "https://openrouter.ai/api/v1/key"


@pytest.fixture
def sinks(caplog):
    """A StringIO handler on the root logger, plus caplog, both filtered."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)
    keys.install_redaction_filter()
    try:
        yield stream, caplog
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redact_covers_both_patterns():
    assert KEY not in keys.redact(f"using {KEY} now")
    assert GENERIC_KEY not in keys.redact(f"header: Bearer {GENERIC_KEY}")
    assert keys.REDACTED in keys.redact(KEY)
    assert keys.redact("no secrets here 12345") == "no secrets here 12345"


def test_key_in_message_is_redacted(sinks):
    stream, caplog = sinks
    logging.getLogger("test.direct").warning("authenticating with %s", KEY)
    assert KEY not in stream.getvalue()
    assert KEY not in caplog.text
    assert keys.REDACTED in stream.getvalue()


def test_key_in_exception_traceback_never_reaches_any_sink(sinks):
    """The real leak path: the key is an argument to the call that raised, so
    it shows up in the traceback rather than in the log message."""
    stream, caplog = sinks
    log = logging.getLogger("test.errorpath")
    try:
        raise RuntimeError(f"OpenRouter rejected credential {KEY}")
    except RuntimeError:
        log.exception("key validation failed")

    captured = stream.getvalue()
    assert KEY not in captured
    assert KEY not in caplog.text
    assert "Traceback" in captured
    assert keys.REDACTED in captured
    assert keys.REDACTED in caplog.text
    # and the record itself is clean, whatever a later handler does with it
    record = caplog.records[-1]
    assert KEY not in (record.exc_text or "")


def test_key_inside_structured_args_is_redacted(sinks):
    stream, caplog = sinks
    logging.getLogger("test.args").error("request %s", {"headers": {"Authorization": KEY}})
    assert KEY not in stream.getvalue()
    assert KEY not in caplog.text


def test_install_is_idempotent():
    before = len(logging.getLogger().filters)
    keys.install_redaction_filter()
    keys.install_redaction_filter()
    after = len(logging.getLogger().filters)
    assert after - before <= 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@respx.mock
async def test_validate_key_returns_label_usage_and_remaining():
    respx.get(KEY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "label": "copilot-laptop",
                    "usage": 3.25,
                    "limit": 20.0,
                    "limit_remaining": 16.75,
                    "is_free_tier": False,
                    "rate_limit": {"requests": 200, "interval": "10s"},
                }
            },
        )
    )
    info = await keys.validate_key(KEY)
    assert info.valid
    assert info.label == "copilot-laptop"
    assert info.usage == 3.25
    assert info.limit == 20.0
    assert info.remaining == 16.75


@respx.mock
async def test_remaining_is_derived_when_only_limit_is_reported():
    respx.get(KEY_URL).mock(
        return_value=httpx.Response(200, json={"data": {"label": "x", "usage": 2.0, "limit": 5.0}})
    )
    info = await keys.validate_key(KEY)
    assert info.remaining == pytest.approx(3.0)


@respx.mock
async def test_rejected_key_is_not_valid_and_never_raises():
    respx.get(KEY_URL).mock(return_value=httpx.Response(401, json={"error": "nope"}))
    info = await keys.validate_key(KEY)
    assert not info.valid
    assert "401" in (info.error or "")


@respx.mock
async def test_network_failure_is_reported_without_the_key():
    respx.get(KEY_URL).mock(side_effect=httpx.ConnectError(f"failed for {KEY}"))
    info = await keys.validate_key(KEY)
    assert not info.valid
    assert KEY not in (info.error or "")


def test_unvalidated_keys_are_refused():
    with pytest.raises(keys.KeyValidationError):
        keys.store_key(KEY, keys.KeyInfo(valid=False, error="never checked"))


def test_storage_backend_is_reportable():
    backend = keys.storage_backend()
    assert isinstance(backend, str)
    assert backend.startswith("keyring:") or backend.startswith("file:")


def test_fallback_file_roundtrip_is_0600(tmp_path, monkeypatch):
    monkeypatch.setattr(keys, "SECRETS_DIR", tmp_path / ".secrets")
    monkeypatch.setattr(keys, "FALLBACK_KEY_PATH", tmp_path / ".secrets" / "openrouter.key")
    monkeypatch.setattr(keys, "_keyring", lambda: None)
    backend = keys.store_key(KEY, keys.KeyInfo(valid=True, label="ok"))
    assert backend.startswith("file:")
    assert keys.load_key() == KEY
    mode = keys.FALLBACK_KEY_PATH.stat().st_mode & 0o777
    assert mode == 0o600
    keys.delete_key()
    assert not keys.FALLBACK_KEY_PATH.exists()


# ---------------------------------------------------------------------------
# settings/models.py — capability-based role assignment (no hardcoded slugs)
# ---------------------------------------------------------------------------

CATALOG_PAYLOAD = {
    "data": [
        {
            "id": "alpha/nano",
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
            "supported_parameters": ["structured_outputs", "tools"],
        },
        {
            # same family as alpha/nano and even cheaper on completion: must NOT
            # take a second jury slot, because its errors correlate
            "id": "alpha/nano-lite",
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0.00000011", "completion": "0.0000002"},
            "supported_parameters": ["structured_outputs", "tools"],
        },
        {
            "id": "beta/small",
            "context_length": 200000,
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
            "supported_parameters": ["structured_outputs", "tools"],
        },
        {
            "id": "gamma/mini",
            "context_length": 64000,
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0.0000003", "completion": "0.0000009"},
            "supported_parameters": ["structured_outputs"],
        },
        {
            "id": "delta/big",
            "context_length": 1000000,
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0.00001", "completion": "0.00003"},
            "supported_parameters": ["structured_outputs", "tools", "reasoning"],
        },
        {
            "id": "eps/text-only",
            "context_length": 32000,
            "architecture": {"input_modalities": ["text"]},
            "pricing": {"prompt": "0.000005", "completion": "0.00001"},
            "supported_parameters": ["tools", "reasoning"],
        },
    ]
}


@respx.mock
async def test_fetch_catalog_reads_capabilities_off_the_live_list():
    from services.core.llm.client import OpenRouterClient
    from services.core.settings import models as M

    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json=CATALOG_PAYLOAD)
    )
    client = OpenRouterClient(KEY)
    catalog = await M.fetch_catalog(client)
    await client.aclose()

    by_id = {m.id: m for m in catalog}
    assert by_id["alpha/nano"].vision
    assert by_id["alpha/nano"].structured_outputs
    assert by_id["alpha/nano"].tools
    assert not by_id["eps/text-only"].vision
    assert not by_id["gamma/mini"].tools
    assert by_id["delta/big"].reasoning
    assert by_id["beta/small"].family == "beta"


def test_auto_assign_picks_three_distinct_families_for_the_jury():
    from services.core.settings import models as M

    catalog = [M.parse_model(r) for r in CATALOG_PAYLOAD["data"]]
    roles = M.auto_assign(catalog)

    assert len(roles.extractor_jury) == 3
    assert len(set(roles.families())) == 3, "same-family jurors correlate their errors"
    # exactly one alpha model, and it is the cheaper of the two
    assert [m for m in roles.extractor_jury if m.startswith("alpha/")] == ["alpha/nano-lite"]
    assert set(roles.extractor_jury) == {"alpha/nano-lite", "beta/small", "gamma/mini"}
    # strongest reasoning for the roles that need it
    assert roles.rule_compiler == "delta/big"
    assert roles.strategist == "delta/big"
    # tie breaker must be outside the jury's families
    assert roles.tie_breaker.split("/")[0] not in set(roles.families())


def test_auto_assign_says_so_when_families_run_out():
    from services.core.settings import models as M

    catalog = [
        M.parse_model(r)
        for r in CATALOG_PAYLOAD["data"]
        if r["id"].startswith(("alpha/", "beta/"))
    ]
    roles = M.auto_assign(catalog)
    assert len(roles.extractor_jury) == 3
    assert any("correlate" in n for n in roles.notes)


def test_settings_persistence_and_latency_report(tmp_path):
    from services.core.settings import models as M

    path = tmp_path / "settings.json"
    catalog = [M.parse_model(r) for r in CATALOG_PAYLOAD["data"]]
    roles = M.auto_assign(catalog)
    M.save_assignment(roles, path)
    assert M.load_settings(path).roles.extractor_jury == roles.extractor_jury

    for ms in (900.0, 1100.0, 1000.0):
        M.record_latency("extractor_jury", "alpha/nano", ms, path)
    M.record_latency("extractor_jury", "beta/small", 2500.0, path)
    M.record_latency("rule_compiler", "delta/big", 8000.0, path)

    report = M.latency_report(path)
    jury = report["extractor_jury"]
    assert [r.model for r in jury] == ["alpha/nano", "beta/small"]  # fastest first
    assert jury[0].count == 3
    assert jury[0].p50_ms == 1000.0
    assert jury[0].last_ms == 1000.0
    assert report["rule_compiler"][0].mean_ms == 8000.0
