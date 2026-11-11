"""Every spelling of this machine's host name is replaced in a diagnostics bundle.

The bundle scrub (`diagnostics._scrub_bundle`) strips hidden characters, redacts
credentials, collapses home prefixes and then replaces the host name, so these
tests drive that whole pipeline on one member's text.
"""

from __future__ import annotations

import socket
import zipfile

import pytest
from test_diagnostics import _isolate

from kiro_crew import diagnostics
from kiro_crew.dashboard import crash_dump_store


def _host(monkeypatch, name: str) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: name)
    monkeypatch.setattr(crash_dump_store, "_pid_domain", lambda: f"{name}/pid:[1]")


def _scrubbed(text: str) -> str:
    return diagnostics._scrub_bundle(text)[0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("boot\fbox ok", "boot<host> ok"),
        ("boot\x0bbox ok", "boot<host> ok"),
        ("boot\u200bbox ok", "boot<host> ok"),
        ("GET /a%2Fbox HTTP", "GET /a%2F<host> HTTP"),
        ("%62ox", "<host>"),
        ("C:\\Users\\box\\logs", "C:\\Users\\<host>\\logs"),
        ("\\\\box\\share", "\\\\<host>\\share"),
        ('{"p":"C:\\\\Users\\\\BOX"}', '{"p":"C:\\\\Users\\\\<host>"}'),
    ],
    ids=[
        "form-feed",
        "vertical-tab",
        "zwsp",
        "percent-slash",
        "percent-letter",
        "windows-path",
        "unc-path",
        "json-escaped-path",
    ],
)
def test_a_short_host_name_is_replaced_in_every_spelling(monkeypatch, raw, expected):
    _host(monkeypatch, "box")
    assert _scrubbed(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "jane\\u002dlaptop",
        "jane\\u002Dlaptop",
        "jane\\x2dlaptop",
        "jane%2Dlaptop",
        "JANE%2dLAPTOP",
        "%6Aane-laptop",
    ],
)
def test_a_long_host_name_is_replaced_with_an_encoded_character(monkeypatch, raw):
    _host(monkeypatch, "jane-laptop")
    assert _scrubbed(f"on {raw} now") == "on <host> now"


@pytest.mark.parametrize(
    ("host", "raw"),
    [
        ("bücher-pc", "xn--bcher-pc-65a"),
        ("xn--bcher-pc-65a", "BÜCHER-PC"),
        ("XN--BCHER-PC-65A", "bücher-pc"),
        ("bücher-pc", "b%C3%BCcher-pc"),
        ("bücher-pc", "b%c3%bccher%2Dpc"),
        ("bücher-pc", "b\\xc3\\xbccher-pc"),
    ],
    ids=[
        "unicode-name-punycode-text",
        "punycode-name-unicode-text",
        "upper-case-punycode-name",
        "utf8-percent-encoded",
        "utf8-percent-lower-case-hex",
        "utf8-byte-escapes",
    ],
)
def test_the_other_idna_spelling_is_replaced(monkeypatch, host, raw):
    _host(monkeypatch, host)
    assert _scrubbed(f"host={raw};") == "host=<host>;"


def test_a_byte_that_starts_no_utf8_sequence_still_decodes(monkeypatch):
    """A stray or truncated byte reads as one character, and the run after it is
    still decoded."""
    _host(monkeypatch, "jane-laptop")
    assert _scrubbed("%FF%C3jane%2Dlaptop \\xe2\\x82") == "%FF%C3<host> \\xe2\\x82"


def test_ordinary_words_and_other_machines_survive(monkeypatch):
    _host(monkeypatch, "box")
    text = "boxes inbox my-box vm_box 1box toolbox\n"
    assert _scrubbed(text) == text


def test_a_one_letter_host_name_leaves_escapes_and_paths_intact(monkeypatch):
    _host(monkeypatch, "t")
    text = '{"stderr":"failure\\t","path":"C:\\\\tmp\\\\t"}'
    assert _scrubbed(text) == '{"stderr":"failure\\t","path":"C:\\\\tmp\\\\<host>"}'


def test_a_joined_host_name_keeps_a_credential_redacted(monkeypatch):
    """The join is read before the scrub, the replacement made after it."""
    _host(monkeypatch, "token")
    raw = "boot\ftoken\nAuthorization: Token abc-123-def-456-ghijklmno\n"
    clean = _scrubbed(raw)
    assert clean.startswith("boot<host>\n"), clean
    assert "abc-123-def-456-ghijklmno" not in clean, clean


def test_the_bundle_replaces_a_short_name_a_form_feed_joined_on(tmp_path, monkeypatch):
    """The issue's reproduction: `boot<FF>box` in gateway.log on a machine named box."""
    home = tmp_path / "home"
    _isolate(monkeypatch, home)
    _host(monkeypatch, "box")
    (home / "gateway.log").write_text("boot\fbox\n")

    r = diagnostics.collect_bundle(output_dir=tmp_path / "out")

    with zipfile.ZipFile(r.zip_path) as z:
        assert z.read("gateway.log").decode() == "boot<host>\n"
