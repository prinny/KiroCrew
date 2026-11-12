"""Golden tests for the app-server → AcpEvent translator.

The fixtures under ``test/fixtures/meta_api/`` are Layer-1 recordings of
the REAL pinned Codex app-server (0.167.2) talking to the Layer-2 stub
Meta API — inbound frames only, paths and thread ids scrubbed. Replaying
them through TurnTranslator must reproduce the event sequences the live
probe observed, including the two declared limits the recordings proved:
a refusal content part completes EMPTY (no refusal event is fabricated),
and a 401 surfaces only as the terminal error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiro_crew.acp.types import EVENT_COMPLETE, EVENT_PERMISSION_REQUEST
from kiro_crew.providers.meta_api_events import TurnTranslator

FIXTURES = Path(__file__).parent / "fixtures" / "meta_api"


def _replay(name: str, model_id: str = "muse-spark-1.3") -> tuple[TurnTranslator, list]:
    translator = TurnTranslator(model_id=model_id)
    events = []
    for line in (FIXTURES / f"{name}.ndjson").read_text().splitlines():
        events.extend(translator.feed(json.loads(line)))
    return translator, events


def _kinds(events) -> list[str]:
    return [e.kind for e in events]


def _completes(events) -> list:
    return [e for e in events if e.kind == EVENT_COMPLETE]


def test_plain_turn_streams_text_and_completes_once():
    translator, events = _replay("plain")
    assert _kinds(events) == ["text_chunk", "complete"]
    (complete,) = _completes(events)
    assert complete.stop_reason == "end_turn"
    # One API call: 1000 in (200 cached) / 100 out at Spark Standard rates.
    assert translator.usage.input_tokens == 1000
    assert translator.usage.output_tokens == 100
    assert translator.usage.cache_read_tokens == 200
    assert translator.usage.cost_usd == pytest.approx(0.001455)


def test_approval_request_becomes_a_permission_request():
    translator, events = _replay("approve")
    assert _kinds(events) == [
        "tool_call",
        "permission_request",
        "tool_result",
        "text_chunk",
        "complete",
    ]
    (request,) = [e for e in events if e.kind == EVENT_PERMISSION_REQUEST]
    assert [o["optionId"] for o in request.options] == [
        "accept",
        "acceptForSession",
        "decline",
        "cancel",
    ]
    # The pending entry waits for the provider to answer it.
    assert list(translator.pending) == ["0"]
    # Two API calls (tool call, then the answer) — usage accumulates.
    assert translator.usage.input_tokens == 2000
    assert translator.usage.cost_usd == pytest.approx(2 * 0.001455)


def test_declined_tool_still_completes_the_turn():
    _translator, events = _replay("deny")
    assert _kinds(events)[-1] == "complete"
    assert _completes(events)[0].stop_reason == "end_turn"


def test_compaction_items_become_compaction_status():
    _translator, events = _replay("compact")
    kinds = _kinds(events)
    assert kinds.count("compaction_status") == 2
    # The fixture holds two turns (the priming turn + the compaction turn);
    # each terminates exactly once.
    assert len(_completes(events)) == 2
    assert all(e.stop_reason == "end_turn" for e in _completes(events))


def test_interrupted_turn_completes_as_cancelled():
    _translator, events = _replay("interrupt")
    (complete,) = _completes(events)
    assert complete.stop_reason == "cancelled"


def test_refusal_part_completes_empty_no_fabricated_refusal():
    """Declared limit, pinned: on the pinned binary a refusal content part
    reaches the client as an EMPTY completed message. The translator must
    not invent a RefusalInfo the loop never sent."""
    _translator, events = _replay("refusal")
    (complete,) = _completes(events)
    assert complete.stop_reason == "end_turn"
    assert all(e.refusal is None for e in events)
    assert "text_chunk" not in _kinds(events)


def test_401_surfaces_only_as_the_terminal_error():
    translator, events = _replay("fail401")
    (complete,) = _completes(events)
    assert complete.stop_reason == "error"
    assert "401" in complete.text or "401" in translator.last_error
    assert translator.usage.input_tokens == 0


def test_process_death_mid_turn_synthesises_exactly_one_complete():
    translator = TurnTranslator(model_id="muse-spark-1.3")
    translator.turn_open = True
    events = translator.process_died()
    assert [e.kind for e in events] == ["complete"]
    assert events[0].synthetic_completion is True
    # A second death notice must not produce a second terminal event.
    assert translator.process_died() == []


def test_process_death_with_no_open_turn_is_silent():
    translator = TurnTranslator(model_id="muse-spark-1.3")
    assert translator.process_died() == []
