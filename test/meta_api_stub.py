"""Layer-2 stub Meta Model API for the native provider's tests.

Implements POST /v1/responses (SSE) and GET /v1/models with scripted,
controllable behaviour. State lives in a JSON file the driver rewrites
between requests: {"mode": "plain"|"tool"|"refusal"|"fail401"|"fail429"|
"truncate", "usage": {...}, "tool_command": "..."}.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mb/stub-state.json"
LOG_PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mb/stub-requests.log"

MODELS = {
    "object": "list",
    "data": [
        {"id": "muse-spark-1.3", "object": "model", "created": 1762000000, "owned_by": "meta"},
        {
            "id": "muse-spark-1.3-contributor",
            "object": "model",
            "created": 1762000000,
            "owned_by": "meta",
        },
        {"id": "muse-wonder-1.0", "object": "model", "created": 1762000000, "owned_by": "meta"},
        {"id": "muse-ember-1.0", "object": "model", "created": 1762000000, "owned_by": "meta"},
    ],
}

DEFAULT_USAGE = {
    "input_tokens": 1000,
    "input_tokens_details": {"cached_tokens": 200},
    "output_tokens": 100,
    "output_tokens_details": {"reasoning_tokens": 40},
    "total_tokens": 1100,
}


def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def sse(events: list[dict]) -> bytes:
    out = []
    for ev in events:
        out.append(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n")
    return "".join(out).encode()


def message_response(rid: str, text: str, usage: dict, model: str) -> list[dict]:
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Thinking briefly."}],
    }
    return [
        {
            "type": "response.created",
            "response": {
                "id": rid,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "summary": []},
        },
        {
            "type": "response.reasoning_summary_part.added",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "Thinking briefly.",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "text": "Thinking briefly.",
        },
        {
            "type": "response.reasoning_summary_part.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "Thinking briefly."},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []},
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": text,
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {
            "type": "response.completed",
            "response": {
                "id": rid,
                "object": "response",
                "status": "completed",
                "model": model,
                "output": [item],
                "usage": usage,
            },
        },
    ]


def function_call_response(rid: str, call: dict, usage: dict, model: str) -> list[dict]:
    args = json.dumps(call["arguments"])
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": call["call_id"],
        "name": call["name"],
        "arguments": args,
        "status": "completed",
    }
    return [
        {
            "type": "response.created",
            "response": {
                "id": rid,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": args,
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "arguments": args,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {
            "type": "response.completed",
            "response": {
                "id": rid,
                "object": "response",
                "status": "completed",
                "model": model,
                "output": [item],
                "usage": usage,
            },
        },
    ]


def refusal_response(rid: str, usage: dict, model: str) -> list[dict]:
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "I cannot help with that request."}],
    }
    return [
        {
            "type": "response.created",
            "response": {
                "id": rid,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []},
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "refusal", "refusal": ""},
        },
        {
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "I cannot help with that request.",
        },
        {
            "type": "response.refusal.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "refusal": "I cannot help with that request.",
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "refusal", "refusal": "I cannot help with that request."},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {
            "type": "response.completed",
            "response": {
                "id": rid,
                "object": "response",
                "status": "completed",
                "model": model,
                "output": [item],
                "usage": usage,
            },
        },
    ]


def pick_tool(tools: list[dict], command: str) -> dict | None:
    """Choose the shell-like tool from the request's own tool list and build
    arguments from its declared schema."""
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        name = tool.get("name", "")
        if "ping" in name:
            # An MCP tool surfaced by the loop (P-4): call it bare.
            props = (tool.get("parameters", {}) or {}).get("properties", {}) or {}
            args = {"text": "ping"} if "text" in props else {}
            return {"call_id": "call_stub_mcp", "name": name, "arguments": args}
        params = tool.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        if "cmd" in props or "command" in props:
            args: dict = {}
            if "cmd" in props:
                args["cmd"] = command
            else:
                args["command"] = command
            # Escalation flag the tool offers, so the approval path is exercised.
            for flag in ("sandbox_permissions",):
                if flag in props:
                    args[flag] = "require_escalated"
                    args["justification"] = "The stub scenario requires approval to proceed."
            return {"call_id": "call_stub_1", "name": name, "arguments": args}
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, json.dumps(MODELS).encode(), "application/json")
        else:
            self._send(404, b"{}", "application/json")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({"path": self.path, "request": req}) + "\n")
        if not self.path.startswith("/v1/responses"):
            self._send(404, b"{}", "application/json")
            return
        state = load_state()
        mode = state.get("mode", "plain")
        usage = state.get("usage", DEFAULT_USAGE)
        model = req.get("model", "muse-spark-1.3")
        if mode == "fail401":
            self._send(
                401,
                json.dumps(
                    {
                        "error": {
                            "message": "invalid api key",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                        }
                    }
                ).encode(),
                "application/json",
            )
            return
        if mode == "fail429":
            self._send(
                429,
                json.dumps(
                    {
                        "error": {
                            "message": "rate limited",
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                        }
                    }
                ).encode(),
                "application/json",
            )
            return
        inputs = req.get("input") or []
        has_tool_output = any(
            isinstance(i, dict) and i.get("type") == "function_call_output" for i in inputs
        )
        has_tool_call = any(
            isinstance(i, dict) and i.get("type") == "function_call" for i in inputs
        )
        rid = "resp_stub_1"
        if mode == "refusal" and not has_tool_output:
            events = refusal_response(rid, usage, model)
        elif mode == "tool" and not has_tool_call and not has_tool_output:
            call = pick_tool(
                req.get("tools") or [],
                state.get("tool_command", "echo stub-tool-ran > stub_tool.txt"),
            )
            if call is None:
                events = message_response(rid, "no shell tool offered", usage, model)
            else:
                events = function_call_response(rid, call, usage, model)
        elif has_tool_output:
            out_text = ""
            for i in inputs:
                if isinstance(i, dict) and i.get("type") == "function_call_output":
                    out_text = str(i.get("output", ""))[:200]
            events = message_response(rid, f"Tool output received: {out_text}", usage, model)
        else:
            events = message_response(
                rid, state.get("text", "Hello from the stub Meta API."), usage, model
            )
        body = sse(events)
        if mode == "truncate":
            body = body[: len(body) // 2]
        self._send(200, body, "text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("STUB_PORT", "0"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"STUB_PORT={server.server_address[1]}", flush=True)
    server.serve_forever()
