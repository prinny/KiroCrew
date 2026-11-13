"""A minimal stdio MCP client: just enough of the protocol for tool use.

The adapter receives MCP servers the ACP way -- a ``mcpServers`` array on
``session/new`` -- and the models it drives can only use those servers as
Converse tools if the adapter connects to them itself. This client speaks
newline-delimited JSON-RPC to a stdio server subprocess: ``initialize``,
``tools/list``, ``tools/call``. Nothing else of MCP is implemented, because
nothing else is needed to put a server's tools in front of a model.

HTTP/SSE elements are not connected (the adapter advertises no HTTP MCP
capability); they are skipped with a line on stderr, never silently.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from typing import Any, Optional

logger = logging.getLogger("kirocrew_bedrock_acp.mcp")

_PROTOCOL_VERSION = "2024-11-05"
_CALL_TIMEOUT_S = 120.0


class McpError(RuntimeError):
    """An MCP server failed a call, or could not be started."""


class McpServer:
    """One connected stdio MCP server."""

    def __init__(self, name: str, command: str, args: list, env: dict) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._pending: dict = {}
        self._next_id = 0
        self._dead: Optional[str] = None
        merged_env = dict(os.environ)
        merged_env.update(env)
        try:
            self._proc = subprocess.Popen(
                [command, *args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=merged_env,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise McpError(f"could not start MCP server {name!r}: {exc}") from exc
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.tools: list = self._initialize()

    # ── Wire ──

    def _read_loop(self) -> None:
        stdout = self._proc.stdout
        assert stdout is not None
        for line in stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            request_id = message.get("id")
            if request_id is None or request_id not in self._pending:
                continue  # a notification, or a response already timed out
            event, box = self._pending.pop(request_id)
            box.append(message)
            event.set()
        self._dead = "the server closed its output"
        for event, _box in list(self._pending.values()):
            event.set()

    def _send(self, payload: dict) -> None:
        stdin = self._proc.stdin
        assert stdin is not None
        with self._lock:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()

    def _call(self, method: str, params: Optional[dict] = None) -> Any:
        if self._dead:
            raise McpError(f"MCP server {self.name!r} is gone: {self._dead}")
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
        event = threading.Event()
        box: list = []
        self._pending[request_id] = (event, box)
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        if not event.wait(_CALL_TIMEOUT_S):
            self._pending.pop(request_id, None)
            raise McpError(f"MCP server {self.name!r} did not answer {method} in time")
        if not box:
            raise McpError(f"MCP server {self.name!r} died during {method}")
        message = box[0]
        if "error" in message:
            error = message["error"] or {}
            raise McpError(
                f"MCP server {self.name!r} refused {method}: {error.get('message', error)}"
            )
        return message.get("result")

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    # ── Protocol ──

    def _initialize(self) -> list:
        self._call(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "kirocrew-bedrock-acp", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized")
        result = self._call("tools/list") or {}
        return list(result.get("tools", []))

    def call_tool(self, tool: str, arguments: dict) -> tuple:
        """Run one tool; returns ``(text, is_error)``."""
        result = self._call("tools/call", {"name": tool, "arguments": arguments}) or {}
        parts = []
        for block in result.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts), bool(result.get("isError"))

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
        except OSError:
            pass


def connect_servers(elements: list) -> dict:
    """Connect every stdio element of an ACP ``mcpServers`` array.

    Returns ``{server_name: McpServer}``. An element with a ``url`` (HTTP or
    SSE transport) is skipped and logged: this adapter advertises no HTTP
    MCP capability, and a server skipped silently would read as a server
    with no tools. A stdio element that fails to start is likewise skipped
    and logged -- one dead server must not cost the session the others.
    """
    servers: dict = {}
    for element in elements:
        if not isinstance(element, dict):
            continue
        name = str(element.get("name", ""))
        command = element.get("command")
        if not name:
            continue
        if not command:
            logger.warning("MCP server %r has no stdio command; skipping it", name)
            continue
        args = [str(a) for a in element.get("args", [])]
        env = {}
        for pair in element.get("env", []):
            if isinstance(pair, dict) and "name" in pair:
                env[str(pair["name"])] = str(pair.get("value", ""))
        try:
            servers[name] = McpServer(name, str(command), args, env)
        except McpError as exc:
            logger.warning("%s", exc)
    return servers
