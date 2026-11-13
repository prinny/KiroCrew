"""kirocrew-bedrock-acp: an ACP agent whose model is Amazon Bedrock.

Speaks ACP v1 (newline-delimited JSON-RPC over stdio) to Kiro Crew on one
side; on the other, drives Bedrock's Converse API through boto3's default
credential chain, and connects the session's MCP servers itself to offer
their tools to the model. It is a standalone adapter in the muse-acp shape:
Crew spawns it per session because an operator registered it under
``agent.custom_backends`` -- the fork carries no Bedrock code for it.

Permission posture: the model may only run an MCP tool after the client
answers ``session/request_permission``, unless the session's approval mode
says otherwise. The mode is a session config option the client sets at
session start, so Crew's own approval flow is what asks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import uuid
from typing import Any, Optional

from kirocrew_bedrock_acp import bedrock, mcp_client

logger = logging.getLogger("kirocrew_bedrock_acp")

AGENT_NAME = "kirocrew-bedrock-acp"
AGENT_VERSION = "0.1.0"

APPROVAL_PROMPT = "promptUnmatched"
APPROVAL_DENY = "denyUnmatched"
_MAX_TOOL_STEPS = 25
_PERMISSION_TIMEOUT_S = 600.0

_STOP_REASONS = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "content_filtered": "refusal",
    "guardrail_intervened": "refusal",
}


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Session:
    def __init__(self, session_id: str, model: str, models: list) -> None:
        self.id = session_id
        self.model = model
        self.models = models  # [(value, name)]
        self.approval_mode = APPROVAL_PROMPT
        self.messages: list = []
        self.servers: dict = {}
        self.tool_specs: list = []
        self.tool_map: dict = {}
        self.lock = threading.Lock()
        self.cancel = threading.Event()


class Adapter:
    """The ACP server: one process, any number of sessions."""

    def __init__(
        self,
        driver: bedrock.BedrockDriver,
        args: argparse.Namespace,
        out: Any = None,
    ) -> None:
        self._driver = driver
        self._args = args
        # The frame stream: stdio in production, a collector in tests. Held
        # as an object rather than looked up per send, so a host that swaps
        # sys.stdout (a test runner, a daemon redirect) cannot reroute frames
        # mid-session.
        self._out = out if out is not None else sys.stdout
        self._sessions: dict = {}
        self._catalogue: Optional[list] = None
        self._catalogue_error: str = ""
        self._write_lock = threading.Lock()
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        self._next_agent_id = 0

    # ── Wire ──

    def _send(self, payload: dict) -> None:
        line = json.dumps(payload)
        with self._write_lock:
            self._out.write(line + "\n")
            self._out.flush()

    def _notify_update(self, session_id: str, update: dict) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": session_id, "update": update},
            }
        )

    def _request_permission(self, session: Session, tool_call: dict) -> bool:
        """Ask the client about one tool call; True only on an explicit allow."""
        with self._pending_lock:
            self._next_agent_id += 1
            request_id = f"perm-{self._next_agent_id}"
        event = threading.Event()
        box: list = []
        self._pending[request_id] = (event, box)
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/request_permission",
                "params": {
                    "sessionId": session.id,
                    "toolCall": tool_call,
                    "options": [
                        {"optionId": "allow_once", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
                    ],
                },
            }
        )
        if not event.wait(_PERMISSION_TIMEOUT_S):
            self._pending.pop(request_id, None)
            return False
        if not box:
            return False
        outcome = (box[0].get("result") or {}).get("outcome") or {}
        return outcome.get("outcome") == "selected" and outcome.get("optionId") == "allow_once"

    # ── Catalogue / config options ──

    def _models(self) -> list:
        """The model catalogue, read once; explicit ``--models`` wins and repairs."""
        if self._args.models:
            return [(m, m) for m in self._args.models]
        if self._catalogue is None:
            try:
                self._catalogue = self._driver.catalogue()
            except Exception as exc:  # noqa: BLE001 - reported to the client verbatim
                self._catalogue = []
                self._catalogue_error = f"{type(exc).__name__}: {exc}"[:400]
        if not self._catalogue:
            raise RpcError(
                -32603,
                "could not list Bedrock models "
                f"({self._catalogue_error or 'the catalogue was empty'}); "
                "fix the AWS credential chain for this process, or start the "
                "adapter with --models naming the model ids to offer",
            )
        return self._catalogue

    def _default_model(self, models: list) -> str:
        if self._args.model:
            return self._args.model
        values = [value for value, _name in models]
        for preferred in ("amazon.nova-pro-v1:0",):
            if preferred in values:
                return preferred
        return values[0]

    def _config_options(self, session: Session) -> list:
        return [
            {
                "id": "model",
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": session.model,
                "options": [{"value": value, "name": name} for value, name in session.models],
            },
            {
                "id": "approval_mode",
                "name": "Approval mode",
                "category": "mode",
                "type": "select",
                "currentValue": session.approval_mode,
                "options": [
                    {"value": APPROVAL_PROMPT, "name": "Ask before tool calls"},
                    {"value": APPROVAL_DENY, "name": "Deny tool calls"},
                ],
            },
        ]

    # ── Method handlers ──

    def _initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": 1,
            "agentInfo": {"name": AGENT_NAME, "version": AGENT_VERSION},
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": False,
                },
                "mcpCapabilities": {"http": False, "sse": False},
            },
            "authMethods": [],
        }

    def _session_new(self, params: dict) -> dict:
        models = self._models()
        session = Session(uuid.uuid4().hex, self._default_model(models), models)
        elements = params.get("mcpServers") or []
        session.servers = mcp_client.connect_servers(elements)
        for server_name, server in session.servers.items():
            for tool in server.tools:
                tool_name = str(tool.get("name", ""))
                if not tool_name:
                    continue
                wire_name = bedrock.converse_tool_name(server_name, tool_name)
                session.tool_map[wire_name] = (server, tool_name)
                session.tool_specs.append(
                    {
                        "name": wire_name,
                        "description": str(tool.get("description", ""))[:1024],
                        # Converse takes the schema as a tagged union: the
                        # document rides under its "json" key, never bare.
                        "inputSchema": {
                            "json": tool.get("inputSchema") or {"type": "object", "properties": {}}
                        },
                    }
                )
        self._sessions[session.id] = session
        return {
            "sessionId": session.id,
            "configOptions": self._config_options(session),
        }

    def _session_set_config_option(self, params: dict) -> dict:
        session = self._session(params)
        config_id = params.get("configId")
        value = params.get("value")
        if config_id == "model":
            if value not in {v for v, _n in session.models}:
                raise RpcError(-32602, f"Invalid value for config option model: {value}")
            session.model = str(value)
        elif config_id == "approval_mode":
            if value not in (APPROVAL_PROMPT, APPROVAL_DENY):
                raise RpcError(-32602, f"Invalid value for config option approval_mode: {value}")
            session.approval_mode = str(value)
        else:
            raise RpcError(-32602, f"unknown config option: {config_id}")
        return {"configOptions": self._config_options(session)}

    def _session(self, params: dict) -> Session:
        session = self._sessions.get(str(params.get("sessionId", "")))
        if session is None:
            raise RpcError(-32602, "unknown session")
        return session

    # ── Prompt ──

    def _prompt_async(self, request_id: Any, params: dict) -> None:
        try:
            result = self._prompt(params)
            self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except RpcError as exc:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": exc.code, "message": exc.message},
                }
            )
        except Exception as exc:  # noqa: BLE001 - one prompt must not kill the agent
            logger.exception("prompt failed")
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"[:400]},
                }
            )

    def _prompt(self, params: dict) -> dict:
        session = self._session(params)
        with session.lock:
            return self._prompt_locked(session, params)

    def _prompt_locked(self, session: Session, params: dict) -> dict:
        session.cancel.clear()
        text = "\n".join(
            str(block.get("text", ""))
            for block in params.get("prompt", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        session.messages.append({"role": "user", "content": [{"text": text}]})
        usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        stop_reason = "end_turn"
        for _step in range(_MAX_TOOL_STEPS):
            if session.cancel.is_set():
                stop_reason = "cancelled"
                break
            try:
                stop, text_out, tool_uses, turn_usage = bedrock.run_converse_turn(
                    self._driver,
                    model_id=session.model,
                    messages=session.messages,
                    tools=session.tool_specs,
                    emit_text=lambda chunk: self._notify_update(
                        session.id,
                        {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": chunk},
                        },
                    ),
                    cancelled=session.cancel.is_set,
                    max_tokens=self._args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - the Converse error is the answer
                raise RpcError(-32603, f"Bedrock Converse failed: {exc}"[:400]) from exc
            for key in usage:
                usage[key] += int(turn_usage.get(key, 0))
            session.messages.append(bedrock.assistant_message(text_out, tool_uses))
            if stop != "tool_use" or not tool_uses:
                stop_reason = _STOP_REASONS.get(stop, "end_turn")
                break
            results = [self._run_tool(session, tool_use) for tool_use in tool_uses]
            session.messages.append(bedrock.tool_results_message(results))
        else:
            stop_reason = "end_turn"
        return {"stopReason": stop_reason, "usage": usage}

    def _run_tool(self, session: Session, tool_use: dict) -> tuple:
        tool_call_id = str(tool_use.get("toolUseId", ""))
        wire_name = str(tool_use.get("name", ""))
        arguments = tool_use.get("input", {})
        self._notify_update(
            session.id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "title": wire_name,
                "kind": "other",
                "status": "pending",
                "rawInput": arguments,
            },
        )
        denied: Optional[str] = None
        if session.approval_mode == APPROVAL_DENY:
            denied = "Tool use denied by the session's approval mode (denyUnmatched)."
        elif not self._request_permission(
            session,
            {
                "toolCallId": tool_call_id,
                "title": wire_name,
                "kind": "other",
                "rawInput": arguments,
            },
        ):
            denied = "Tool use was not approved."
        if denied is not None:
            text, is_error = denied, True
        else:
            target = session.tool_map.get(wire_name)
            if target is None:
                text, is_error = f"unknown tool: {wire_name}", True
            else:
                server, tool_name = target
                try:
                    text, is_error = server.call_tool(tool_name, arguments)
                except Exception as exc:  # noqa: BLE001 - the tool's failure is a result
                    text, is_error = f"tool call failed: {exc}"[:400], True
        self._notify_update(
            session.id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": "failed" if is_error else "completed",
                "content": [{"type": "content", "content": {"type": "text", "text": text}}],
            },
        )
        return (tool_call_id, text, is_error)

    # ── Dispatch ──

    def handle(self, message: dict) -> None:
        """Route one inbound frame. Responses are never answered."""
        method = message.get("method")
        request_id = message.get("id")
        if method is None:
            # A response to one of this agent's own requests.
            with self._pending_lock:
                pending = self._pending.pop(request_id, None)
            if pending is not None:
                event, box = pending
                box.append(message)
                event.set()
            return
        params = message.get("params") or {}
        if request_id is None:
            if method == "session/cancel":
                session = self._sessions.get(str(params.get("sessionId", "")))
                if session is not None:
                    session.cancel.set()
            return
        if method == "session/prompt":
            # A prompt runs long and asks questions mid-flight; it gets its
            # own thread so the read loop stays free for the answers.
            threading.Thread(
                target=self._prompt_async, args=(request_id, params), daemon=True
            ).start()
            return
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "session/new":
                result = self._session_new(params)
            elif method == "session/set_config_option":
                result = self._session_set_config_option(params)
            elif method == "session/load":
                raise RpcError(-32601, "session/load is not supported: loadSession is false")
            else:
                raise RpcError(-32601, f"method not found: {method}")
        except RpcError as exc:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": exc.code, "message": exc.message},
                }
            )
            return
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def serve(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("dropping a non-JSON line on stdin")
                continue
            if isinstance(message, dict):
                self.handle(message)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=AGENT_NAME,
        description="An ACP agent that drives Amazon Bedrock's Converse API.",
    )
    parser.add_argument(
        "--profile", default=None, help="AWS profile name (default chain when omitted)"
    )
    parser.add_argument(
        "--region", default=None, help="AWS region (the SDK chain's region when omitted)"
    )
    parser.add_argument(
        "--model", default=None, help="model id or inference profile id to start on"
    )
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model ids to offer instead of reading the catalogue",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    args.models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    driver = bedrock.BedrockDriver(profile=args.profile, region=args.region)
    Adapter(driver, args).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
