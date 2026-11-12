"""Translate Bedrock Converse event streams into ``AcpEvent``s.

Pure translation, no I/O: the translator consumes the event dicts a
``converse_stream`` response yields (or a synthesized equivalent built
from a non-streaming ``converse`` response) and produces the session
layer's events plus, at the end of each call, a :class:`CallOutcome`
carrying everything the provider's tool loop needs that events do not:
the assembled assistant message (text, signed reasoning and toolUse
blocks, in wire order, ready to journal and resubmit), the raw stop
reason, the call's usage counts, and the guardrail trace when one rode
the stream.

Bedrock's refusal signal is typed, and this module keeps it that way:
``content_filtered`` and ``guardrail_intervened`` map onto the shared
``refusal`` stop reason carrying a :class:`RefusalInfo`, strictly more
than a streamed-empty-message provider can say. ``max_tokens`` keeps
its own spelling and flags the terminal event ``truncated``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    EVENT_THINKING_CHUNK,
    EVENT_TOOL_CALL,
    EVENT_TOOL_CALL_UPDATE,
    STOP_REASON_END_TURN,
    STOP_REASON_REFUSAL,
    AcpEvent,
    RefusalInfo,
    TurnUsage,
)
from kiro_crew.providers.bedrock_rates import call_cost_usd, is_rated

#: Converse stop reasons that end a turn (``tool_use`` does not: the
#: provider loop answers the calls and continues).
TERMINAL_STOP_REASONS = frozenset(
    {
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "guardrail_intervened",
        "content_filtered",
        "model_context_window_exceeded",
    }
)

#: Stream-carried exception keys the translator classifies instead of
#: choking on: a throttled or refused call surfaces as one of these
#: dicts inside the event stream rather than as a raised error.
STREAM_EXCEPTION_KEYS = (
    "ThrottlingException",
    "AccessDeniedException",
    "ValidationException",
    "ModelNotFoundException",
    "ModelStreamErrorException",
    "InternalServerException",
    "ServiceUnavailableException",
)

#: Crew stop-reason spelling for a context-window refusal: the
#: error-family prefix classes it failed/retryable, and the text names
#: the cause so a caller can compact and retry with open eyes.
STOP_REASON_CONTEXT_EXCEEDED = "error: model context window exceeded"


@dataclass
class ToolUseRequest:
    """One assembled toolUse block: fragments joined, input parsed."""

    tool_use_id: str
    name: str
    input: dict[str, Any]


@dataclass
class CallOutcome:
    """Everything one Converse call produced, for the provider loop."""

    stop_reason: str = ""
    crew_stop_reason: str = ""
    usage: TurnUsage = field(default_factory=TurnUsage)
    tool_uses: list[ToolUseRequest] = field(default_factory=list)
    assistant_blocks: list[dict[str, Any]] = field(default_factory=list)
    guardrail_trace: dict[str, Any] | None = None
    refusal: RefusalInfo | None = None
    truncated: bool = False
    error_class: str = ""
    error_message: str = ""
    latency_ms: int = 0
    rated: bool = True


@dataclass
class _BlockState:
    index: int
    kind: str = ""  # "text" | "tool" | "reasoning"
    text_parts: list[str] = field(default_factory=list)
    signature: str = ""
    tool_use_id: str = ""
    tool_name: str = ""
    input_fragments: list[str] = field(default_factory=list)


def crew_stop_reason_for(raw: str, refusal: RefusalInfo | None) -> tuple[str, bool]:
    """Map a raw Converse stop reason onto the Crew vocabulary.

    Returns (crew spelling, truncated). ``tool_use`` never reaches
    here: the provider loop treats it as a continuation, not a
    terminal. Unknown raw spellings pass through unchanged so the
    classifier reports them as unknown rather than success.
    """
    if raw in ("end_turn", "stop_sequence"):
        return STOP_REASON_END_TURN, False
    if raw == "max_tokens":
        return "max_tokens", True
    if raw in ("content_filtered", "guardrail_intervened"):
        return STOP_REASON_REFUSAL, False
    if raw == "model_context_window_exceeded":
        return STOP_REASON_CONTEXT_EXCEEDED, False
    return raw, False


class ConverseStreamTranslator:
    """Stateful per-call translator from Converse stream events."""

    def __init__(self, model_id: str, usage: TurnUsage) -> None:
        self.model_id = model_id
        #: The turn's shared billing object: counts accumulate across
        #: every Converse call the turn makes (the ABC identity rule —
        #: the object the first event carries is the object read back).
        self.usage = usage
        self.turn_open = False
        self._blocks: dict[int, _BlockState] = {}
        self._stop_reason = ""
        self._refusal: RefusalInfo | None = None
        self._guardrail_trace: dict[str, Any] | None = None
        self._error_class = ""
        self._error_message = ""
        self._latency_ms = 0
        self._call_input_tokens = 0

    # -- entry points -----------------------------------------------------

    def feed(self, event: dict) -> list[AcpEvent]:
        """Translate one stream event dict."""
        if not isinstance(event, dict):
            return []
        for key in STREAM_EXCEPTION_KEYS:
            if key in event:
                self._error_class = key
                detail = event.get(key) or {}
                self._error_message = str(detail.get("message") or detail.get("Message") or key)
                self.turn_open = False
                return []
        if "messageStart" in event:
            self.turn_open = True
            return []
        if "contentBlockStart" in event:
            return self._on_block_start(event["contentBlockStart"] or {})
        if "contentBlockDelta" in event:
            return self._on_block_delta(event["contentBlockDelta"] or {})
        if "contentBlockStop" in event:
            return self._on_block_stop(event["contentBlockStop"] or {})
        if "messageStop" in event:
            self._stop_reason = str((event["messageStop"] or {}).get("stopReason") or "")
            self.turn_open = False
            return []
        if "metadata" in event:
            self._on_metadata(event["metadata"] or {})
            return []
        return []

    def feed_response(self, response: dict) -> list[AcpEvent]:
        """Translate a non-streaming ``converse`` response.

        The response is re-expressed as the equivalent stream events so
        both feeders share one assembly path and one set of semantics.
        """
        events: list[AcpEvent] = []
        output = response.get("output") or {}
        message = output.get("message") or {}
        events += self.feed({"messageStart": {"role": message.get("role", "assistant")}})
        for index, block in enumerate(message.get("content") or []):
            events += self._feed_whole_block(index, block)
        events += self.feed({"messageStop": {"stopReason": response.get("stopReason") or ""}})
        usage = response.get("usage") or {}
        metadata: dict[str, Any] = {"usage": usage}
        if response.get("metrics"):
            metadata["metrics"] = response["metrics"]
        if response.get("trace"):
            metadata["trace"] = response["trace"]
        events += self.feed({"metadata": metadata})
        return events

    def note_error(self, code: str, message: str) -> None:
        """Record a classified call failure on the outcome channel.

        Used when the failure arrives as a raised, classified error
        rather than as a stream event dict (the initial call, or an
        exception raised while iterating the stream).
        """
        self._error_class = code
        self._error_message = message
        self.turn_open = False

    def stream_died(self) -> list[AcpEvent]:
        """The stream ended mid-call with no terminal event: exactly
        one synthetic ``complete``, never a hang."""
        if not self.turn_open:
            return []
        self.turn_open = False
        return [
            AcpEvent(
                kind=EVENT_COMPLETE,
                stop_reason="error: bedrock stream ended mid-call",
                synthetic_completion=True,
                usage=self.usage,
            )
        ]

    # -- block assembly -----------------------------------------------------

    def _feed_whole_block(self, index: int, block: dict) -> list[AcpEvent]:
        events: list[AcpEvent] = []
        if "toolUse" in block:
            tool = block["toolUse"] or {}
            events += self.feed(
                {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {
                            "toolUse": {
                                "toolUseId": tool.get("toolUseId") or "",
                                "name": tool.get("name") or "",
                            }
                        },
                    }
                }
            )
            events += self.feed(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {
                            "toolUse": {"input": json.dumps(tool.get("input") or {}, default=str)}
                        },
                    }
                }
            )
            events += self.feed({"contentBlockStop": {"contentBlockIndex": index}})
            return events
        if "reasoningContent" in block:
            reasoning = (block["reasoningContent"] or {}).get("reasoningText") or {}
            events += self.feed(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {
                            "reasoningContent": {
                                "text": reasoning.get("text") or "",
                                "signature": reasoning.get("signature") or "",
                            }
                        },
                    }
                }
            )
            events += self.feed({"contentBlockStop": {"contentBlockIndex": index}})
            return events
        if "text" in block:
            events += self.feed(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {"text": block.get("text") or ""},
                    }
                }
            )
            events += self.feed({"contentBlockStop": {"contentBlockIndex": index}})
            return events
        return events

    def _block(self, index: int) -> _BlockState:
        state = self._blocks.get(index)
        if state is None:
            state = _BlockState(index=index)
            self._blocks[index] = state
        return state

    def _on_block_start(self, start: dict) -> list[AcpEvent]:
        index = int(start.get("contentBlockIndex") or 0)
        state = self._block(index)
        tool = (start.get("start") or {}).get("toolUse")
        if tool is not None:
            state.kind = "tool"
            state.tool_use_id = str(tool.get("toolUseId") or "")
            state.tool_name = str(tool.get("name") or "")
            return [
                AcpEvent(
                    kind=EVENT_TOOL_CALL,
                    tool_call_id=state.tool_use_id,
                    title=state.tool_name or "Tool call",
                    tool_kind="other",
                )
            ]
        return []

    def _on_block_delta(self, delta_event: dict) -> list[AcpEvent]:
        index = int(delta_event.get("contentBlockIndex") or 0)
        delta = delta_event.get("delta") or {}
        state = self._block(index)
        if "text" in delta:
            state.kind = state.kind or "text"
            text = str(delta.get("text") or "")
            state.text_parts.append(text)
            return [AcpEvent(kind=EVENT_TEXT_CHUNK, text=text)] if text else []
        if "reasoningContent" in delta:
            state.kind = "reasoning"
            reasoning = delta.get("reasoningContent") or {}
            text = str(reasoning.get("text") or "")
            if reasoning.get("signature"):
                state.signature += str(reasoning["signature"])
            if text:
                state.text_parts.append(text)
                return [AcpEvent(kind=EVENT_THINKING_CHUNK, text=text)]
            return []
        if "toolUse" in delta:
            state.kind = "tool"
            fragment = (delta.get("toolUse") or {}).get("input")
            if fragment:
                state.input_fragments.append(str(fragment))
            return []
        return []

    def _on_block_stop(self, stop: dict) -> list[AcpEvent]:
        index = int(stop.get("contentBlockIndex") or 0)
        state = self._blocks.get(index)
        if state is None or state.kind != "tool":
            return []
        raw_input = "".join(state.input_fragments)
        try:
            parsed = json.loads(raw_input) if raw_input.strip() else {}
        except ValueError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        return [
            AcpEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                tool_call_id=state.tool_use_id,
                title=state.tool_name or "Tool call",
                tool_kind="other",
                tool_input=json.dumps(parsed, default=str),
            )
        ]

    # -- metadata / outcome ---------------------------------------------------

    def _on_metadata(self, metadata: dict) -> None:
        usage = metadata.get("usage") or {}
        input_tokens = int(usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("outputTokens") or 0)
        cache_read = int(usage.get("cacheReadInputTokens") or 0)
        cache_write = int(usage.get("cacheWriteInputTokens") or 0)
        self._call_input_tokens = input_tokens + cache_read + cache_write
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.cache_read_tokens += cache_read
        self.usage.cache_creation_tokens += cache_write
        self.usage.num_turns += 1
        self.usage.context_sum_tokens += self._call_input_tokens
        self.usage.cost_usd += call_cost_usd(
            self.model_id,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens=output_tokens,
        )
        metrics = metadata.get("metrics") or {}
        if metrics.get("latencyMs"):
            self._latency_ms = int(metrics["latencyMs"])
        trace = metadata.get("trace") or {}
        if trace.get("guardrail"):
            self._guardrail_trace = trace["guardrail"]

    def finish_call(self) -> CallOutcome:
        """Assemble the call outcome: journal-ready blocks, tool uses,
        the mapped stop reason and any typed refusal."""
        blocks: list[dict[str, Any]] = []
        tool_uses: list[ToolUseRequest] = []
        for index in sorted(self._blocks):
            state = self._blocks[index]
            if state.kind == "tool":
                raw_input = "".join(state.input_fragments)
                try:
                    parsed = json.loads(raw_input) if raw_input.strip() else {}
                except ValueError:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
                tool_uses.append(
                    ToolUseRequest(
                        tool_use_id=state.tool_use_id,
                        name=state.tool_name,
                        input=parsed,
                    )
                )
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": state.tool_use_id,
                            "name": state.tool_name,
                            "input": parsed,
                        }
                    }
                )
            elif state.kind == "reasoning":
                reasoning_text: dict[str, Any] = {"text": "".join(state.text_parts)}
                if state.signature:
                    reasoning_text["signature"] = state.signature
                blocks.append({"reasoningContent": {"reasoningText": reasoning_text}})
            elif state.kind == "text":
                blocks.append({"text": "".join(state.text_parts)})
        refusal = self._refusal
        if self._stop_reason == "content_filtered" and refusal is None:
            refusal = RefusalInfo(
                category="content_filtered",
                explanation="Bedrock reported the response as content filtered.",
            )
        if self._stop_reason == "guardrail_intervened" and refusal is None:
            detail = ""
            if self._guardrail_trace:
                detail = json.dumps(self._guardrail_trace, default=str)[:500]
            refusal = RefusalInfo(
                category="guardrail_intervened",
                explanation=(
                    "A Bedrock guardrail intervened on this call."
                    + (f" Trace: {detail}" if detail else "")
                ),
            )
        crew_reason, truncated = crew_stop_reason_for(self._stop_reason, refusal)
        return CallOutcome(
            stop_reason=self._stop_reason,
            crew_stop_reason=crew_reason,
            usage=self.usage,
            tool_uses=tool_uses,
            assistant_blocks=blocks,
            guardrail_trace=self._guardrail_trace,
            refusal=refusal if crew_reason == STOP_REASON_REFUSAL else None,
            truncated=truncated,
            error_class=self._error_class,
            error_message=self._error_message,
            latency_ms=self._latency_ms,
            rated=is_rated(self.model_id),
        )

    @property
    def last_call_input_tokens(self) -> int:
        """Input tokens (incl. cache) of the most recent call: the
        session's current context occupancy from the service's own
        accounting."""
        return self._call_input_tokens
