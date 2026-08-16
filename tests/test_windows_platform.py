"""Windows behaviour, exercised from Linux.

The tool ships on Windows; it was written and tested on headless Linux.  Every
Windows-only branch is therefore reachable by monkeypatching ``os.name`` rather
than by calling into the platform, so these tests run in CI on any OS and still
fail when a Windows path regresses.

What is *not* covered here, and cannot be: whether ``icacls`` actually denies a
second local account, whether ``SetConsoleMode`` really flips a legacy conhost,
and whether ``mss``/``pynput`` behave on a real multi-monitor desktop.  Those
need a Windows box.  Everything below tests the code's decisions, which is the
part that can be wrong in a way a Windows box would not obviously reveal.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path

import pytest

from services.core.settings import keys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_acceptance():
    """Import ``scripts/acceptance.py`` as a module (it is a script, not a package)."""
    path = REPO_ROOT / "scripts" / "acceptance.py"
    name = "_acceptance_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations via sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


acceptance = _load_acceptance()


# ---------------------------------------------------------------------------
# 1. Interpreter discovery: Scripts on Windows, bin on POSIX
# ---------------------------------------------------------------------------


@pytest.fixture
def both_layouts(tmp_path):
    """A fake project root carrying *both* venv layouts, so the choice is the test."""
    scripts = tmp_path / ".venv" / "Scripts"
    binv = tmp_path / ".venv" / "bin"
    scripts.mkdir(parents=True)
    binv.mkdir(parents=True)
    (scripts / "python.exe").write_text("", encoding="utf-8")
    (binv / "python").write_text("", encoding="utf-8")
    return tmp_path


def test_finds_scripts_layout_when_windows(both_layouts, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    found = acceptance.find_python(both_layouts)
    assert found == both_layouts / ".venv" / "Scripts" / "python.exe"


def test_finds_bin_layout_when_posix(both_layouts, monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    found = acceptance.find_python(both_layouts)
    assert found == both_layouts / ".venv" / "bin" / "python"


def test_windows_still_finds_a_bin_layout_venv(tmp_path, monkeypatch):
    """A venv copied from WSL, or a MSYS Python, has the other layout."""
    binv = tmp_path / ".venv" / "bin"
    binv.mkdir(parents=True)
    (binv / "python.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(os, "name", "nt")
    assert acceptance.find_python(tmp_path) == binv / "python.exe"


@pytest.mark.parametrize("osname", ["nt", "posix"])
def test_falls_back_to_the_running_interpreter(tmp_path, monkeypatch, osname):
    """No .venv at all: run the tests with whatever is running us."""
    monkeypatch.setattr(os, "name", osname)
    # str(), not Path(): os.name is patched, so Path() would build a WindowsPath.
    assert str(acceptance.find_python(tmp_path)) == sys.executable


def test_the_real_repo_resolves_to_something_executable():
    found = acceptance.find_python()
    assert found.exists(), f"{found} does not exist"


# ---------------------------------------------------------------------------
# 2. The fallback key file must never be written unprotected on Windows
# ---------------------------------------------------------------------------


@pytest.fixture
def windows_fallback(tmp_path, monkeypatch):
    """Fallback storage pointed at a temp dir, mocked as Windows, no keyring."""
    monkeypatch.setattr(keys, "SECRETS_DIR", tmp_path / ".secrets")
    monkeypatch.setattr(keys, "FALLBACK_KEY_PATH", tmp_path / ".secrets" / "openrouter.key")
    monkeypatch.setattr(keys, "_keyring", lambda: None)
    monkeypatch.setattr(keys.os, "name", "nt")
    monkeypatch.setenv("USERNAME", "puzzler")
    monkeypatch.setenv("USERDOMAIN", "DESKTOP-1")
    return tmp_path


VALID = keys.KeyInfo(valid=True, label="ok")
SECRET = "sk-or-v1-" + "0123456789abcdef" * 4


def test_refuses_to_store_when_icacls_fails(windows_fallback, monkeypatch):
    """icacls itself failed.  Storing anyway would be a lie in the settings UI."""
    monkeypatch.setattr(keys, "_windows_restrict_acl", lambda p, container=False: False)
    monkeypatch.setattr(keys, "_windows_acl_is_restricted", lambda p: False)

    with pytest.raises(keys.KeyStorageError):
        keys.store_key(SECRET, VALID)

    assert not keys.FALLBACK_KEY_PATH.exists(), "no key file may survive a failed lockdown"


def test_refuses_to_store_when_the_acl_readback_disagrees(windows_fallback, monkeypatch):
    """icacls returned 0 but the ACL is still open -- verification is the point."""
    monkeypatch.setattr(keys, "_windows_restrict_acl", lambda p, container=False: True)
    monkeypatch.setattr(keys, "_windows_acl_is_restricted", lambda p: False)

    with pytest.raises(keys.KeyStorageError):
        keys.store_key(SECRET, VALID)

    assert not keys.FALLBACK_KEY_PATH.exists()


def test_the_refusal_names_the_fix(windows_fallback, monkeypatch):
    monkeypatch.setattr(keys, "_windows_restrict_acl", lambda p, container=False: False)
    with pytest.raises(keys.KeyStorageError) as excinfo:
        keys.store_key(SECRET, VALID)
    message = str(excinfo.value).lower()
    assert "keyring" in message, "the user needs to be told what to actually fix"


def test_a_failed_lockdown_never_leaks_the_key_to_disk(windows_fallback, monkeypatch):
    """The empty-then-restrict-then-write order is what makes this true."""
    monkeypatch.setattr(keys, "_windows_restrict_acl", lambda p, container=False: True)
    monkeypatch.setattr(keys, "_windows_acl_is_restricted", lambda p: False)

    with pytest.raises(keys.KeyStorageError):
        keys.store_key(SECRET, VALID)

    leaked = [
        p
        for p in (windows_fallback / ".secrets").rglob("*")
        if p.is_file() and SECRET in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert leaked == []


def test_successful_acl_reports_the_restricted_backend(windows_fallback, monkeypatch):
    monkeypatch.setattr(keys, "_windows_restrict_acl", lambda p, container=False: True)
    monkeypatch.setattr(keys, "_windows_acl_is_restricted", lambda p: True)

    backend = keys.store_key(SECRET, VALID)

    assert backend.startswith("file:")
    assert "ACL-restricted" in backend
    assert keys.storage_backend() == backend
    assert keys.load_key() == SECRET


def test_backend_string_says_unprotected_if_the_acl_ever_lapses(windows_fallback, monkeypatch):
    """A file that exists but is not restricted must report that, not hide it."""
    monkeypatch.setattr(keys, "_windows_restrict_acl", lambda p, container=False: True)
    monkeypatch.setattr(keys, "_windows_acl_is_restricted", lambda p: True)
    keys.store_key(SECRET, VALID)

    monkeypatch.setattr(keys, "_windows_acl_is_restricted", lambda p: False)
    assert "UNPROTECTED" in keys.storage_backend()


# -- the icacls readback parser ---------------------------------------------
#
# Verified against the documented `icacls <file>` output shape.  Not verified
# on a real Windows host; see the module docstring.

LOCKED_DOWN = "{path} DESKTOP-1\\puzzler:(F)\n\nSuccessfully processed 1 files; Failed processing 0 files\n"
STILL_INHERITED = (
    "{path} BUILTIN\\Users:(I)(RX)\n"
    "        DESKTOP-1\\puzzler:(F)\n"
    "\nSuccessfully processed 1 files; Failed processing 0 files\n"
)
EVERYONE = (
    "{path} Everyone:(F)\n"
    "        DESKTOP-1\\puzzler:(F)\n"
    "\nSuccessfully processed 1 files; Failed processing 0 files\n"
)
SOMEONE_ELSE = "{path} DESKTOP-1\\otheruser:(F)\n\nSuccessfully processed 1 files\n"


def _icacls_returning(text: str, path: Path, returncode: int = 0):
    class Proc:
        pass

    def run(args):
        proc = Proc()
        proc.returncode = returncode
        proc.stdout = text.format(path=path)
        proc.stderr = ""
        return proc

    return run


@pytest.mark.parametrize(
    "output,expected",
    [
        (LOCKED_DOWN, True),
        (STILL_INHERITED, False),
        (EVERYONE, False),
        (SOMEONE_ELSE, False),
    ],
    ids=["locked-down", "inheritance-not-removed", "everyone-can-read", "not-our-account"],
)
def test_acl_readback_only_accepts_a_lone_owner_ace(tmp_path, monkeypatch, output, expected):
    monkeypatch.setenv("USERNAME", "puzzler")
    monkeypatch.setenv("USERDOMAIN", "DESKTOP-1")
    target = tmp_path / "openrouter.key"
    monkeypatch.setattr(keys, "_run_icacls", _icacls_returning(output, target))
    assert keys._windows_acl_is_restricted(target) is expected


def test_acl_readback_is_false_when_icacls_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("USERNAME", "puzzler")

    def boom(args):
        raise FileNotFoundError("icacls")

    monkeypatch.setattr(keys, "_run_icacls", boom)
    assert keys._windows_acl_is_restricted(tmp_path / "k") is False


def test_icacls_is_never_invoked_through_a_shell(tmp_path, monkeypatch):
    """A path with a space must not be re-split by a shell."""
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return Proc()

    monkeypatch.setattr(keys.subprocess, "run", fake_run)
    keys._run_icacls([str(tmp_path / "My Data" / "openrouter.key")])

    assert isinstance(seen["cmd"], list), "argv form, never a joined string"
    assert seen["cmd"][0] == "icacls"
    assert seen["kwargs"].get("shell") is False


def test_posix_still_uses_mode_bits(tmp_path, monkeypatch):
    """The Windows work must not have changed what POSIX does."""
    monkeypatch.setattr(keys, "SECRETS_DIR", tmp_path / ".secrets")
    monkeypatch.setattr(keys, "FALLBACK_KEY_PATH", tmp_path / ".secrets" / "openrouter.key")
    monkeypatch.setattr(keys, "_keyring", lambda: None)
    monkeypatch.setattr(keys.os, "name", "posix")

    def explode(*a, **k):  # pragma: no cover - fails the test if reached
        raise AssertionError("POSIX must never shell out to icacls")

    monkeypatch.setattr(keys, "_run_icacls", explode)

    backend = keys.store_key(SECRET, VALID)
    assert "mode 0600" in backend
    assert keys.FALLBACK_KEY_PATH.stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# 3. Console encoding
# ---------------------------------------------------------------------------

NON_ASCII = "Puzzle Copilot — café → █ 日本語"


def _cp1252_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


def test_a_cp1252_stream_cannot_take_the_report_as_it_stands():
    """Establishes the bug being fixed: without the reconfigure this raises."""
    stream = _cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        stream.write(NON_ASCII)
        stream.flush()


def test_reconfigure_makes_a_cp1252_stream_accept_non_ascii():
    stream = _cp1252_stream()
    assert acceptance.enable_utf8_console(stream) is True
    assert stream.encoding.lower().replace("-", "") == "utf8"

    stream.write(NON_ASCII)  # must not raise
    stream.flush()
    assert "café" in stream.buffer.getvalue().decode("utf-8")


def test_errors_replace_means_a_stubborn_stream_still_does_not_raise():
    """Belt and braces: even if the encoding stays cp1252, replacement holds."""
    stream = _cp1252_stream()
    stream.reconfigure(errors="replace")
    stream.write(NON_ASCII)  # would raise under errors="strict"
    stream.flush()


def test_reconfigure_reports_failure_instead_of_raising():
    """A stream that refuses (detached, wrapped by pytest, a plain file) is not fatal."""

    class Stubborn:
        def reconfigure(self, **kwargs):
            raise io.UnsupportedOperation("not reconfigurable")

    assert acceptance.enable_utf8_console(Stubborn()) is False


def test_a_stream_without_reconfigure_is_handled():
    assert acceptance.enable_utf8_console(io.StringIO()) is False


def test_practice_console_guard_matches():
    """``practice.py`` cannot import from ``scripts/``, so it carries its own."""
    from services.core.extract import practice

    real_out, real_err = sys.stdout, sys.stderr
    out, err = _cp1252_stream(), _cp1252_stream()
    sys.stdout, sys.stderr = out, err
    try:
        assert practice._enable_utf8_console() is True
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    out.write(NON_ASCII)  # must not raise
    out.flush()


def test_the_acceptance_report_body_is_pure_ascii():
    """Reconfiguration is the belt; ASCII output is the braces."""
    source = (REPO_ROOT / "scripts" / "acceptance.py").read_text(encoding="utf-8")
    offenders = [
        (n, line)
        for n, line in enumerate(source.splitlines(), 1)
        if any(ord(ch) > 127 for ch in line)
    ]
    assert offenders == [], f"non-ASCII in the acceptance report: {offenders}"


# ---------------------------------------------------------------------------
# 4. ANSI colour
# ---------------------------------------------------------------------------


def test_colour_is_off_when_not_a_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert acceptance.use_color(io.StringIO()) is False


def test_colour_respects_no_color(monkeypatch):
    class Tty:
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    assert acceptance.use_color(Tty()) is False


def test_colour_on_a_posix_tty(monkeypatch):
    class Tty:
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(os, "name", "posix")
    assert acceptance.use_color(Tty()) is True


def test_windows_tty_colour_depends_on_vt_processing(monkeypatch):
    """On Windows, colour is only claimed once VT processing is actually on."""

    class Tty:
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(os, "name", "nt")

    monkeypatch.setattr(acceptance, "enable_vt_mode", lambda: False)
    assert acceptance.use_color(Tty()) is False, "raw ESC[32m is worse than no colour"

    monkeypatch.setattr(acceptance, "enable_vt_mode", lambda: True)
    assert acceptance.use_color(Tty()) is True


def test_enable_vt_mode_is_a_noop_off_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert acceptance.enable_vt_mode() is True


def test_enable_vt_mode_does_not_raise_when_ctypes_has_no_windll(monkeypatch):
    """Running the Windows branch on Linux: must degrade, not explode."""
    monkeypatch.setattr(os, "name", "nt")
    assert acceptance.enable_vt_mode() is False  # no ctypes.windll here


def test_set_color_blanks_every_code():
    acceptance.set_color(False)
    try:
        assert acceptance.GREEN == acceptance.RED == acceptance.OFF == ""
        assert acceptance.BOLD == acceptance.DIM == ""
    finally:
        acceptance.set_color(False)


def test_set_color_restores_every_code():
    try:
        acceptance.set_color(True)
        assert acceptance.GREEN.startswith("\033[")
        assert acceptance.OFF == "\033[0m"
    finally:
        acceptance.set_color(False)


# ---------------------------------------------------------------------------
# 5. Non-ASCII round trip through the persistence helpers in this file's scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "sk-or-v1-café—naïve",  # accented + em dash
        "sk-or-v1-日本語テスト",  # CJK, outside cp1252 entirely
        "sk-or-v1-€£¥",  # currency symbols that differ across code pages
    ],
    ids=["latin-1-plus-dash", "cjk", "currency"],
)
def test_non_ascii_survives_the_fallback_file_round_trip(tmp_path, monkeypatch, value):
    """Explicit UTF-8 on both ends, so the ANSI code page never gets a vote.

    Without ``encoding="utf-8"`` this is the exact bug: on Windows the write
    goes out in cp1252, the CJK case raises, and the currency case comes back
    as different characters.
    """
    monkeypatch.setattr(keys, "SECRETS_DIR", tmp_path / ".secrets")
    monkeypatch.setattr(keys, "FALLBACK_KEY_PATH", tmp_path / ".secrets" / "openrouter.key")
    monkeypatch.setattr(keys, "_keyring", lambda: None)
    monkeypatch.setattr(keys.os, "name", "posix")

    keys.store_key(value, VALID)

    assert keys.load_key() == value
    assert keys.FALLBACK_KEY_PATH.read_bytes().decode("utf-8") == value


def test_non_ascii_round_trip_on_the_windows_write_path(windows_fallback, monkeypatch):
    """The Windows branch is a separate write; it must be UTF-8 too."""
    monkeypatch.setattr(keys, "_windows_restrict_acl", lambda p, container=False: True)
    monkeypatch.setattr(keys, "_windows_acl_is_restricted", lambda p: True)

    value = "sk-or-v1-café-日本語"
    keys.store_key(value, VALID)

    assert keys.load_key() == value
    assert keys.FALLBACK_KEY_PATH.read_bytes().decode("utf-8") == value


def test_a_non_ascii_key_is_still_redacted_from_logs():
    """Redaction is byte-shape based; a unicode tail must not slip past it."""
    weird = "sk-or-v1-" + "a1b2c3d4e5f6a7b8" * 2 + "—café"
    scrubbed = keys.redact(f"authorization: Bearer {weird}")
    assert "a1b2c3d4e5f6a7b8" not in scrubbed
    assert keys.REDACTED in scrubbed
