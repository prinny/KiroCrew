"""Pure translation: Codex app-server frames -> ``AcpEvent``s.

One :class:`TurnTranslator` serves one turn. It is a pure function over
recorded payloads — no process, no I/O — so the golden tests replay the
recorded Layer-1 fixture streams through it and compare the event
sequence. Method names and payload shapes are the pinned app-server
protocol's (see ``meta_api_client.PINNED_CODEX_VERSION``); every shape
here was verified against recorded frames from that binary.

What the pinned protocol does NOT carry, this module does not invent:

- A Responses ``refusal`` content part produces an empty agent message
  from the pinned app-server — no refusal signal reaches the client — so
  no ``RefusalInfo`` is fabricated here. A refusal the API returns as an
  error fails the turn, and the failure path below is where it surfaces.
- Retryable stream errors (``error`` notifications with ``willRetry``)
  are the loop recovering, not events; only the turn's terminal carries
  the failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kiro_crew.acp.types import (
    EVENT_COMPACTION_STATUS,
    EVENT_COMPLETE,
    EVENT_PERMISSION_REQUEST,
    EVENT_TEXT_CHUNK,
    EVENT_THINKING_CHUNK,
    EVENT_TODO_UPDATE,
    EVENT_TOOL_CALL,
    EVENT_TOOL_CALL_UPDATE,
    EVENT_TOOL_RESULT,
    STOP_REASON_CANCELLED,
    STOP_REASON_END_TURN,
    AcpEvent,
    TurnUsage,
)
from kiro_crew.providers.meta_api_rates import turn_cost_usd

#: Display bound for a tool result carried on an event, matching the
#: spirit of the ACP path's raw-output bound: the full output stays with
#: the loop; the event carries enough to render.
TOOL_OUTPUT_EVENT_LIMIT = 16_000

#: App-server server-request methods that ask the user to approve work.
APPROVAL_METHODS = frozenset(
    {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "item/permissions/requestApproval",
    }
)

#: The decision vocabulary the pinned protocol accepts for an approval.
DECISION_ACCEPT = "accept"
DECISION_ACCEPT_FOR_SESSION = "acceptForSession"
DECISION_DECLINE = "decline"
DECISION_CANCEL = "cancel"


@dataclass
class PendingApproval:
    """One unanswered app-server approval request."""

    request_id: str | int
    method: str
    item_id: str = ""


@dataclass
class TurnTranslator:
    """Stateful per-turn translator from app-server frames to events."""

    model_id: str
    pending: dict[str, PendingApproval] = field(default_factory=dict)
    usage: TurnUsage = field(default_factory=TurnUsage)
    last_error: str = ""
    turn_open: bool = False
    _calls: int = 0

    # -- frame entry points -------------------------------------------------

    def feed(self, frame: dict) -> list[AcpEvent]:
        """Translate one inbound frame (notification or server request)."""
        method = frame.get("method")
        if not isinstance(method, str):
            return []
        params = frame.get("params") or {}
        if "id" in frame and method in APPROVAL_METHODS:
            return [self._approval_event(frame["id"], method, params)]
        handler = self._NOTIFICATIONS.get(method)
        if handler is None:
            return []
        return handler(self, params)

    def process_died(self) -> list[AcpEvent]:
        """The loop process ended. An open turn must still terminate:
        exactly one synthetic ``complete``, never a hang."""
        if not self.turn_open:
            return []
        self.turn_open = False
        return [
            AcpEvent(
                kind=EVENT_COMPLETE,
                stop_reason="error: app-server process exited",
                synthetic_completion=True,
                usage=self.usage,
            )
        ]

    # -- approvals ------------------------------------------------------------

    def _approval_event(self, request_id: str | int, method: str, params: dict) -> AcpEvent:
        key = str(request_id)
        item_id = str(params.get("itemId") or "")
        self.pending[key] = PendingApproval(request_id=request_id, method=method, item_id=item_id)
        if method == "item/fileChange/requestApproval":
            title = "File change"
            tool_kind = "edit"
        elif method == "item/permissions/requestApproval":
            title = "Permission request"
            tool_kind = "other"
        else:
            title = str(params.get("command") or "Command")
            tool_kind = "execute"
        return AcpEvent(
            kind=EVENT_PERMISSION_REQUEST,
            request_id=request_id,
            tool_call_id=item_id,
            title=title,
            tool_kind=tool_kind,
            tool_input=json.dumps(params, default=str),
            options=[
                {"optionId": DECISION_ACCEPT, "name": "Accept", "kind": "allow_once"},
                {
                    "optionId": DECISION_ACCEPT_FOR_SESSION,
                    "name": "Accept for session",
                    "kind": "allow_always",
                },
                {"optionId": DECISION_DECLINE, "name": "Decline", "kind": "reject_once"},
                {"optionId": DECISION_CANCEL, "name": "Cancel", "kind": "reject_once"},
            ],
        )

    def take_pending(self, request_id: str | int) -> PendingApproval | None:
        return self.pending.pop(str(request_id), None)

    # -- notifications ----------------------------------------------------------

    def _on_agent_delta(self, params: dict) -> list[AcpEvent]:
        return [AcpEvent(kind=EVENT_TEXT_CHUNK, text=str(params.get("delta") or ""))]

    def _on_reasoning_delta(self, params: dict) -> list[AcpEvent]:
        return [AcpEvent(kind=EVENT_THINKING_CHUNK, text=str(params.get("delta") or ""))]

    def _on_plan_delta(self, params: dict) -> list[AcpEvent]:
        return [AcpEvent(kind=EVENT_TODO_UPDATE, text=str(params.get("delta") or ""))]

    def _on_item_started(self, params: dict) -> list[AcpEvent]:
        item = params.get("item") or {}
        item_type = item.get("type")
        if item_type == "commandExecution":
            return [
                AcpEvent(
                    kind=EVENT_TOOL_CALL,
                    tool_call_id=str(item.get("id") or ""),
                    title=str(item.get("command") or "Command"),
                    tool_kind="execute",
                    tool_input=json.dumps(
                        {"command": item.get("command"), "cwd": item.get("cwd")},
                        default=str,
                    ),
                )
            ]
        if item_type == "fileChange":
            return [
                AcpEvent(
                    kind=EVENT_TOOL_CALL,
                    tool_call_id=str(item.get("id") or ""),
                    title="File change",
                    tool_kind="edit",
                    tool_input=json.dumps(item.get("changes") or [], default=str),
                )
            ]
        if item_type == "mcpToolCall":
            title = f"{item.get('server') or ''}/{item.get('tool') or ''}".strip("/")
            return [
                AcpEvent(
                    kind=EVENT_TOOL_CALL,
                    tool_call_id=str(item.get("id") or ""),
                    title=title or "MCP tool",
                    tool_kind="other",
                    tool_input=json.dumps(item.get("arguments") or {}, default=str),
                )
            ]
        if item_type == "contextCompaction":
            return [AcpEvent(kind=EVENT_COMPACTION_STATUS, text="started")]
        return []

    def _on_item_completed(self, params: dict) -> list[AcpEvent]:
        item = params.get("item") or {}
        item_type = item.get("type")
        if item_type in ("commandExecution", "fileChange", "mcpToolCall"):
            output = item.get("aggregatedOutput")
            if output is None:
                output = item.get("output") or ""
            return [
                AcpEvent(
                    kind=EVENT_TOOL_RESULT,
                    tool_call_id=str(item.get("id") or ""),
                    tool_output=str(output)[:TOOL_OUTPUT_EVENT_LIMIT],
                    tool_final=True,
                    tool_status=str(item.get("status") or ""),
                )
            ]
        if item_type == "contextCompaction":
            return [AcpEvent(kind=EVENT_COMPACTION_STATUS, text="completed")]
        return []

    def _on_output_delta(self, params: dict) -> list[AcpEvent]:
        return [
            AcpEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                tool_call_id=str(params.get("itemId") or ""),
                tool_output=str(params.get("delta") or ""),
            )
        ]

    def _on_token_usage(self, params: dict) -> list[AcpEvent]:
        usage = (params.get("tokenUsage") or {}).get("last") or {}
        input_tokens = int(usage.get("inputTokens") or 0)
        cached = int(usage.get("cachedInputTokens") or 0)
        output = int(usage.get("outputTokens") or 0)
        self._calls += 1
        self.usage.input_tokens += input_tokens
        self.usage.cache_read_tokens += cached
        self.usage.output_tokens += output
        self.usage.num_turns = self._calls
        self.usage.cost_usd += turn_cost_usd(
            self.model_id,
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output,
        )
        return []

    def _on_error(self, params: dict) -> list[AcpEvent]:
        if not params.get("willRetry"):
            error = params.get("error") or {}
            self.last_error = str(error.get("message") or "")
        return []

    def _on_turn_started(self, params: dict) -> list[AcpEvent]:
        self.turn_open = True
        return []

    def _on_turn_completed(self, params: dict) -> list[AcpEvent]:
        self.turn_open = False
        turn = params.get("turn") or {}
        status = str(turn.get("status") or "")
        if status == "completed":
            stop_reason = STOP_REASON_END_TURN
            text = ""
        elif status == "interrupted":
            stop_reason = STOP_REASON_CANCELLED
            text = ""
        else:
            stop_reason = "error"
            error = turn.get("error") or {}
            text = str(error.get("message") or self.last_error or "the turn failed")
        duration = turn.get("durationMs")
        if isinstance(duration, (int, float)):
            self.usage.duration_ms = int(duration)
        return [
            AcpEvent(
                kind=EVENT_COMPLETE,
                stop_reason=stop_reason,
                text=text,
                usage=self.usage,
            )
        ]

    def _on_compacted(self, params: dict) -> list[AcpEvent]:
        return [AcpEvent(kind=EVENT_COMPACTION_STATUS, text="completed")]

    _NOTIFICATIONS = {
        "item/agentMessage/delta": _on_agent_delta,
        "item/reasoning/textDelta": _on_reasoning_delta,
        "item/reasoning/summaryTextDelta": _on_reasoning_delta,
        "item/plan/delta": _on_plan_delta,
        "item/started": _on_item_started,
        "item/completed": _on_item_completed,
        "item/commandExecution/outputDelta": _on_output_delta,
        "item/fileChange/outputDelta": _on_output_delta,
        "thread/tokenUsage/updated": _on_token_usage,
        "error": _on_error,
        "turn/started": _on_turn_started,
        "turn/completed": _on_turn_completed,
        "thread/compacted": _on_compacted,
    }
