"""Replace this machine's host names in a diagnostics bundle member with ``<host>``.

A crash-dump header and the ``last-healthy-boot`` marker record a PID domain that
starts with the host name, and a work machine's name often carries its user's name
or an asset tag, so every text member of the bundle is passed through
:func:`replace_host_names` before it is written.

The matcher reads two copies of the text:

- the text as written, and
- a decoded view: ``\\uXXXX``, ``\\xXX`` and ``%XX`` each read as the one
  character they encode, a serialized escape such as ``\\n`` reads as one
  non-name character, and ``\\\\`` reads as one backslash. Every view character
  keeps the span of the original characters it came from, so a match in the view
  replaces exactly those characters.

A host name is a run of name characters (ASCII letters and digits, ``-``, ``_``).
A match ends where the next character is not one, so ``jane-laptop2`` is another
machine while ``jane-laptop.lan`` still matches. A name of
:data:`HOST_ANYWHERE_MIN` characters or more is replaced wherever it starts; a
shorter one only where it starts a segment, so a generic short name does not
rewrite ordinary words.

A short name starts a segment after a non-name character, after an ANSI control
sequence (``[32m``), after a literal ``\\`` when the name has two characters or
more (a Windows path separator; a one-letter name would take the letter of a
serialized ``\\t``), and where a hidden character the credential scrub removes
joined it onto the word in front (:func:`find_joined_host_words`).
"""

from __future__ import annotations

import bisect
import re
import socket
import unicodedata
from dataclasses import dataclass

from kiro_crew.dashboard import crash_dump_store

#: A host name this long is replaced anywhere it appears; a shorter one only where
#: it starts a segment.
HOST_ANYWHERE_MIN = 6

_NAME_CHARS = "A-Za-z0-9_-"
_HOST_END = rf"(?![{_NAME_CHARS}])"
_NAME_CHAR = re.compile(rf"[{_NAME_CHARS}]")
#: An encoded run of the decoded view. ``\\\\`` comes first so the ``u`` of an
#: escaped backslash followed by ``u002d`` is read as a letter. A run of ``%XX`` or
#: ``\\xXX`` bytes is read as UTF-8 (:func:`_byte_run`).
_ENCODED = re.compile(
    r"\\\\|\\u[0-9A-Fa-f]{4}|(?:\\x[0-9A-Fa-f]{2})+|(?:%[0-9A-Fa-f]{2})+|\\[A-Za-z]"
)
#: An ANSI control sequence ending right before a position, with its ESC stripped
#: or serialized: the name after it starts a segment.
_CSI_TAIL = re.compile(r"\[[0-9;?]*[A-Za-z]\Z")
#: How far back the CSI test looks; a longer parameter run is not a color code.
_CSI_LOOKBACK = 24
#: Stands for a serialized escape in the view: a non-name character.
_ESCAPE_STAND_IN = "\x00"


def host_names() -> list[str]:
    """This machine's host names, longest first.

    The live name and the one the crash-dump writer cached at boot, each in full and
    in its short form before the first ``.``, plus the other IDNA spelling of each
    (``xn--`` for a Unicode name, Unicode for an ``xn--`` one).
    """
    own = crash_dump_store.current_process_identity()[0].partition("/")[0]
    names: set[str] = set()
    for full in (socket.gethostname(), own):
        for name in (full, full.partition(".")[0]):
            if name:
                names.add(name)
                names.update(_idna_spellings(name))
    return sorted(names, key=lambda n: (-len(n), n))


def _idna_spellings(name: str) -> set[str]:
    """The other IDNA spelling of *name*, when it has one that the codec accepts."""
    try:
        if not name.isascii():
            return {name.encode("idna").decode("ascii")}
        if "xn--" in name.lower():
            # The codec decodes only a lower-case ``xn--`` label.
            return {name.lower().encode("ascii").decode("idna")}
    except UnicodeError:
        pass
    return set()


@dataclass(frozen=True)
class JoinedHostWord:
    """A short host name a stripped hidden character joined onto *prefix*."""

    prefix: str
    name: str


def find_joined_host_words(raw: str, names: list[str]) -> set[JoinedHostWord]:
    """Short names in *raw* that the hidden-character strip would join onto a word.

    The credential scrub removes hidden characters before it redacts, so a short
    name behind a form feed (``boot<FF>box``) reaches the host pass as ``bootbox``
    with no boundary left. This reads *raw* before that strip and records each
    joined word, so :func:`replace_host_names` can replace the name inside it.
    """
    short = [n for n in names if len(n) < HOST_ANYWHERE_MIN]
    if not short:
        return set()
    pattern = re.compile(
        rf"(?<![{_NAME_CHARS}])([{_NAME_CHARS}]+)({_MAYBE_HIDDEN}+)"
        rf"(?=((?:{_alternation(short)}){_HOST_END}))",
        re.IGNORECASE,
    )
    found: set[JoinedHostWord] = set()
    for m in pattern.finditer(raw):
        if all(_is_stripped(ch) for ch in m.group(2)):
            found.add(JoinedHostWord(m.group(1).lower(), m.group(3).lower()))
    return found


#: A character the hidden-character strip may remove: an ASCII control other than
#: ``\\n``, ``\\r`` and ``\\t``, or any non-ASCII character (:func:`_is_stripped`
#: decides).
_MAYBE_HIDDEN = r"(?:[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|[^\x00-\x7f])"


def _is_stripped(ch: str) -> bool:
    """Whether the hidden-character strip removes *ch* (control, format, surrogate)."""
    return ch not in "\n\r\t" and unicodedata.category(ch) in ("Cc", "Cf", "Cs")


class _View:
    """The decoded view of a text, with each view character's original span."""

    def __init__(self, text: str) -> None:
        pieces: list[str] = []
        self._token_view: list[int] = []
        self._token_orig: list[int] = []
        self._token_len: list[int] = []
        last = 0
        view_len = 0
        for m in _ENCODED.finditer(text):
            pieces.append(text[last : m.start()])
            view_len += m.start() - last
            token = m.group()
            is_bytes = token[0] == "%" or (token[1] == "x" and len(token) > 2)
            decoded = _byte_run(token) if is_bytes else [_decode(token)]
            offset = m.start()
            for ch, length in decoded:
                self._token_view.append(view_len)
                self._token_orig.append(offset)
                self._token_len.append(length)
                pieces.append(ch)
                view_len += 1
                offset += length
            last = m.end()
        pieces.append(text[last:])
        self.text = "".join(pieces)

    def start(self, v: int) -> int:
        """Original offset of view character *v*."""
        i = bisect.bisect_right(self._token_view, v) - 1
        if i < 0:
            return v
        if self._token_view[i] == v:
            return self._token_orig[i]
        return self._token_orig[i] + self._token_len[i] + (v - self._token_view[i] - 1)

    def end(self, v: int) -> int:
        """Original offset just past view character *v*."""
        i = bisect.bisect_right(self._token_view, v) - 1
        if i >= 0 and self._token_view[i] == v:
            return self._token_orig[i] + self._token_len[i]
        return self.start(v) + 1


def _decode(token: str) -> tuple[str, int]:
    """The view character a ``\\\\``, ``\\uXXXX`` or ``\\<letter>`` token stands
    for, with the token's length."""
    if token == "\\\\":
        return "\\", 2
    if len(token) > 2:
        ch = chr(int(token[2:], 16))
        return (_ESCAPE_STAND_IN if "\ud800" <= ch <= "\udfff" else ch), len(token)
    return _ESCAPE_STAND_IN, len(token)


def _byte_run(token: str) -> list[tuple[str, int]]:
    """The view characters of a ``%XX`` or ``\\xXX`` run, each with its length.

    The bytes are read as UTF-8, so ``%C3%BC`` is one ``ü`` spanning six original
    characters. A byte that starts no valid sequence reads as its Latin-1 character.
    """
    unit = 3 if token[0] == "%" else 4
    data = bytes(int(token[i + unit - 2 : i + unit], 16) for i in range(0, len(token), unit))
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(data):
        lead = data[i]
        size = (
            2
            if 0xC2 <= lead <= 0xDF
            else 3 if 0xE0 <= lead <= 0xEF else 4 if 0xF0 <= lead <= 0xF4 else 1
        )
        try:
            ch = data[i : i + size].decode("utf-8") if size > 1 else ""
        except UnicodeDecodeError:
            ch = ""
        if len(ch) != 1:
            ch, size = chr(lead), 1
        out.append((ch, size * unit))
        i += size
    return out


def _starts_segment(text: str, at: int) -> bool:
    """Whether a short name starting at *at* in *text* starts a segment."""
    if at == 0 or not _NAME_CHAR.match(text[at - 1]):
        return True
    return bool(_CSI_TAIL.search(text, max(0, at - _CSI_LOOKBACK), at))


def replace_host_names(
    text: str,
    names: list[str],
    joined: set[JoinedHostWord] | frozenset[JoinedHostWord] = frozenset(),
) -> tuple[str, int]:
    """Replace every spelling of *names* in *text* with ``<host>``; return (text, count).

    *names* comes from :func:`host_names`, *joined* from :func:`find_joined_host_words`
    on the same member before it was scrubbed.
    """
    if not names or not text:
        return text, 0
    long_names = [n for n in names if len(n) >= HOST_ANYWHERE_MIN]
    short_names = [n for n in names if len(n) < HOST_ANYWHERE_MIN]
    view = _View(text)
    spans: list[tuple[int, int]] = []

    def in_view(m_start: int, m_end: int) -> None:
        spans.append((view.start(m_start), view.end(m_end - 1)))

    if long_names:
        anywhere = re.compile(rf"(?:{_alternation(long_names)}){_HOST_END}", re.IGNORECASE)
        spans.extend(m.span() for m in anywhere.finditer(text))
        for m in anywhere.finditer(view.text):
            in_view(*m.span())
    if short_names:
        each = re.compile(rf"(?=((?:{_alternation(short_names)}){_HOST_END}))", re.IGNORECASE)
        for m in each.finditer(view.text):
            if _starts_segment(view.text, m.start(1)):
                in_view(*m.span(1))
        after_backslash = [n for n in short_names if len(n) >= 2]
        if after_backslash:
            windows = re.compile(
                rf"(?<=\\)(?:{_alternation(after_backslash)}){_HOST_END}", re.IGNORECASE
            )
            spans.extend(m.span() for m in windows.finditer(text))
    for word in joined:
        glued = re.compile(
            rf"(?={re.escape(word.prefix)}({re.escape(word.name)}){_HOST_END})", re.IGNORECASE
        )
        for m in glued.finditer(view.text):
            in_view(*m.span(1))
    return _apply(text, spans)


def _alternation(names: list[str]) -> str:
    """A regex alternation of *names*, longest first so a full name wins."""
    return "|".join(map(re.escape, sorted(names, key=len, reverse=True)))


def _apply(text: str, spans: list[tuple[int, int]]) -> tuple[str, int]:
    """Replace the union of *spans* in *text* with ``<host>``, one per merged span."""
    if not spans:
        return text, 0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[str] = []
    last = 0
    for start, end in merged:
        out.append(text[last:start])
        out.append("<host>")
        last = end
    out.append(text[last:])
    return "".join(out), len(merged)
