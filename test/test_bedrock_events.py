"""Translator tests for the native Bedrock provider.

Pins the Converse -> AcpEvent mapping: text/thinking deltas, toolUse
input fragments concatenated per content block and parsed only at
block stop, reasoning signatures accumulated into the journal blocks
for unchanged resubmission, the typed refusal signal
(``content_filtered`` / ``guardrail_intervened`` -> refusal +
RefusalInfo), ``max_tokens`` keeping its spelling with the truncated
flag (and classifying as succeeded), usage accumulation into the
turn's shared billing object, and the non-streaming feeder producing
the same outcome as the streamed one.
"""

from __future__ import annotations

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    EVENT_THINKING_CHUNK,
    EVENT_TOOL_CALL,
    EVENT_TOOL_CALL_UPDATE,
    STOP_CLASS_SUCCEEDED,
    TurnUsage,
    classify_stop_reason,
)
from kiro_crew.providers.bedrock_events import ConverseStreamTranslator

TEXT_STREAM = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello "}}},
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "world"}}},
    {"contentBlockStop": {"contentBlockIndex": 0}},
    {"messageStop": {"stopReason": "end_turn"}},
    {
        "metadata": {
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "metrics": {"latencyMs": 42},
        }
    },
]

TOOL_STREAM = [
    {"messageStart": {"role": "assistant"}},
    {
        "contentBlockStart": {
            "contentBlockIndex": 1,
            "start": {"toolUse": {"toolUseId": "tu-1", "name": "shell"}},
        }
    },
    {
        "contentBlockDelta": {
            "contentBlockIndex": 1,
            "delta": {"toolUse": {"input": '{"comm'}},
        }
    },
    {
        "contentBlockDelta": {
            "contentBlockIndex": 1,
            "delta": {"toolUse": {"input": 'and": "echo hi"}'}},
        }
    },
    {"contentBlockStop": {"contentBlockIndex": 1}},
    {"messageStop": {"stopReason": "tool_use"}},
    {"metadata": {"usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28}}},
]

REASONING_STREAM = [
    {"messageStart": {"role": "assistant"}},
    {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"text": "thinking..."}},
        }
    },
    {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"signature": "sig-abc"}},
        }
    },
    {"contentBlockStop": {"contentBlockIndex": 0}},
    {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "answer"}}},
    {"contentBlockStop": {"contentBlockIndex": 1}},
    {"messageStop": {"stopReason": "end_turn"}},
    {"metadata": {"usage": {"inputTokens": 30, "outputTokens": 12, "totalTokens": 42}}},
]


def _run(events, model_id="us.amazon.nova-pro-v1:0"):
    usage = TurnUsage()
    translator = ConverseStreamTranslator(model_id, usage)
    out = []
    for event in events:
        out += translator.feed(event)
    return translator, out, usage


def test_text_stream_events_and_outcome():
    translator, events, usage = _run(TEXT_STREAM)
    texts = [e.text for e in events if e.kind == EVENT_TEXT_CHUNK]
    assert texts == ["Hello ", "world"]
    outcome = translator.finish_call()
    assert outcome.stop_reason == "end_turn"
    assert outcome.crew_stop_reason == "end_turn"
    assert outcome.assistant_blocks == [{"text": "Hello world"}]
    assert usage.input_tokens == 10 and usage.output_tokens == 5
    assert usage.num_turns == 1
    assert translator.last_call_input_tokens == 10
    assert outcome.latency_ms == 42
    assert outcome.rated is True and usage.cost_usd > 0


def test_tool_use_fragments_assembled_and_parsed_at_stop():
    translator, events, _usage = _run(TOOL_STREAM)
    kinds = [e.kind for e in events]
    assert EVENT_TOOL_CALL in kinds
    update = next(e for e in events if e.kind == EVENT_TOOL_CALL_UPDATE)
    assert '"command": "echo hi"' in update.tool_input
    outcome = translator.finish_call()
    assert outcome.stop_reason == "tool_use"
    assert len(outcome.tool_uses) == 1
    assert outcome.tool_uses[0].name == "shell"
    assert outcome.tool_uses[0].input == {"command": "echo hi"}
    assert outcome.assistant_blocks == [
        {"toolUse": {"toolUseId": "tu-1", "name": "shell", "input": {"command": "echo hi"}}}
    ]


def test_partial_json_never_parsed_early():
    usage = TurnUsage()
    translator = ConverseStreamTranslator("us.amazon.nova-pro-v1:0", usage)
    for event in TOOL_STREAM[:3]:  # block start + first fragment only
        translator.feed(event)
    outcome = translator.finish_call()
    # The fragment so far ('{"comm') is not valid JSON; assembly yields
    # {} rather than raising or guessing at the rest.
    assert outcome.tool_uses[0].input == {}


def test_reasoning_signature_accumulated_for_resubmission():
    translator, events, _usage = _run(REASONING_STREAM)
    thinking = [e for e in events if e.kind == EVENT_THINKING_CHUNK]
    assert thinking and thinking[0].text == "thinking..."
    outcome = translator.finish_call()
    assert outcome.assistant_blocks[0] == {
        "reasoningContent": {"reasoningText": {"text": "thinking...", "signature": "sig-abc"}}
    }
    assert outcome.assistant_blocks[1] == {"text": "answer"}


def test_content_filtered_is_a_typed_refusal():
    stream = [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": "content_filtered"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 0}}},
    ]
    translator, _events, _usage = _run(stream)
    outcome = translator.finish_call()
    assert outcome.crew_stop_reason == "refusal"
    assert outcome.refusal is not None
    assert outcome.refusal.category == "content_filtered"


def test_guardrail_intervened_refusal_carries_trace():
    stream = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "partial"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "guardrail_intervened"}},
        {
            "metadata": {
                "usage": {"inputTokens": 5, "outputTokens": 1},
                "trace": {"guardrail": {"inputAssessment": {"g1": {}}}},
            }
        },
    ]
    translator, _events, _usage = _run(stream)
    outcome = translator.finish_call()
    assert outcome.crew_stop_reason == "refusal"
    assert outcome.refusal is not None
    assert outcome.refusal.category == "guardrail_intervened"
    assert outcome.guardrail_trace == {"inputAssessment": {"g1": {}}}


def test_max_tokens_spelling_truncated_and_succeeded_class():
    stream = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "cut off"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "max_tokens"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 4096}}},
    ]
    translator, _events, _usage = _run(stream)
    outcome = translator.finish_call()
    assert outcome.crew_stop_reason == "max_tokens"
    assert outcome.truncated is True
    assert classify_stop_reason("max_tokens").name == STOP_CLASS_SUCCEEDED


def test_context_window_exceeded_is_error_family():
    stream = [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": "model_context_window_exceeded"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 0}}},
    ]
    translator, _events, _usage = _run(stream)
    outcome = translator.finish_call()
    assert outcome.crew_stop_reason.startswith("error:")


def test_stream_exception_is_classified_not_raised():
    translator, _events, _usage = _run(
        [
            {"messageStart": {"role": "assistant"}},
            {"ThrottlingException": {"message": "too fast"}},
        ]
    )
    outcome = translator.finish_call()
    assert outcome.error_class == "ThrottlingException"
    assert outcome.error_message == "too fast"


def test_cache_tokens_accumulate_into_usage():
    stream = [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 100,
                    "cacheReadInputTokens": 900,
                    "cacheWriteInputTokens": 50,
                    "outputTokens": 10,
                }
            }
        },
    ]
    translator, _events, usage = _run(stream, model_id="us.anthropic.claude-sonnet-4-5")
    assert usage.input_tokens == 100
    assert usage.cache_read_tokens == 900
    assert usage.cache_creation_tokens == 50
    assert translator.last_call_input_tokens == 1050


def test_non_streaming_response_matches_streamed_outcome():
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello world"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "metrics": {"latencyMs": 42},
    }
    usage = TurnUsage()
    translator = ConverseStreamTranslator("us.amazon.nova-pro-v1:0", usage)
    events = translator.feed_response(response)
    texts = [e.text for e in events if e.kind == EVENT_TEXT_CHUNK]
    assert texts == ["Hello world"]
    outcome = translator.finish_call()
    assert outcome.assistant_blocks == [{"text": "Hello world"}]
    assert outcome.crew_stop_reason == "end_turn"
    assert usage.input_tokens == 10


def test_stream_died_synthesizes_one_complete():
    usage = TurnUsage()
    translator = ConverseStreamTranslator("us.amazon.nova-pro-v1:0", usage)
    translator.feed({"messageStart": {"role": "assistant"}})
    died = translator.stream_died()
    assert len(died) == 1 and died[0].kind == EVENT_COMPLETE
    assert died[0].synthetic_completion is True
    assert translator.stream_died() == []
