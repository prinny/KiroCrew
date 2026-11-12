"""The gateway-side tool executor for the native Bedrock provider.

A native provider has no harness to execute tools for it, and the tree
has no gateway-callable local executor: ``agent_sdk/tool_gate.py`` is a
DECISION gate for harness-executed tools, not an executor. This module
is the executor the Bedrock design composes for that gap, from the
tree's existing confinement machinery at a new call site:

* subprocess tools run through ``sandbox_spawn.wrap_argv_async`` (the
  session's sandbox tier) with an environment produced by
  ``sandbox_spawn.scrub_agent_subprocess_env`` — the CHILD posture of
  design §5.2 rule 3: a tool subprocess observes no ``AWS_SECRET*`` /
  ``AWS_SESSION*`` names even while the provider's own signing works
  gateway-side;
* file tools are confined to the session work dir by path resolution
  (an escaping path is refused, never followed);
* MCP tools route to operator-configured MCP servers through a minimal
  in-provider client, so the session's MCP set is the model's tool set
  (design §5.2 rule 2) instead of a declared gap.

Execution is reachable ONLY through the provider's approval registry
(design §5.2 rule 1): nothing in this module is wired to a model-facing
surface, and the provider calls :meth:`ToolRouter.execute` from exactly
one gated call site.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Output bounds: a tool result rides back into the model's context, so
#: it is capped like every other provider's tool output.
MAX_TOOL_OUTPUT_CHARS = 32_000
MAX_FILE_READ_BYTES = 256_000
SHELL_TIMEOUT_SECONDS = 120.0

_TOOL_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def converse_tool_name(raw: str) -> str:
    """A Converse-legal tool name ([a-zA-Z0-9_-], <= 64) for *raw*."""
    return _TOOL_NAME_SAFE.sub("_", raw)[:64]


# -- local tool set -----------------------------------------------------------

LOCAL_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "toolSpec": {
            "name": "shell",
            "description": (
                "Run a shell command inside the session sandbox, in the "
                "session work directory. Returns combined stdout/stderr."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the session work directory.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "write_file",
            "description": (
                "Write a UTF-8 text file inside the session work directory, "
                "creating parent directories."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "list_dir",
            "description": "List a directory inside the session work directory.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                }
            },
        }
    },
]

LOCAL_TOOL_NAMES = frozenset(
    spec["toolSpec"]["name"] for spec in LOCAL_TOOL_SPECS
)

_SANDBOX_WRAP_MODES = frozenset({"auto", "standard", "cc", "strict"})


class LocalToolExecutor:
    """The session's local file/shell tool set, executed in-posture."""

    def __init__(self, work_dir: Path, sandbox_mode: str = "") -> None:
        self._work_dir = work_dir
        self._sandbox_mode = sandbox_mode or "auto"

    def _confined_path(self, raw: str) -> Path:
        candidate = (self._work_dir / raw).resolve()
        root = self._work_dir.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"path escapes the session work directory: {raw}")
        return candidate

    async def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Run one local tool; returns (result_text, is_error)."""
        try:
            if name == "shell":
                return await self._shell(str(tool_input.get("command") or ""))
            if name == "read_file":
                return await asyncio.to_thread(
                    self._read_file, str(tool_input.get("path") or "")
                )
            if name == "write_file":
                return await asyncio.to_thread(
                    self._write_file,
                    str(tool_input.get("path") or ""),
                    str(tool_input.get("content") or ""),
                )
            if name == "list_dir":
                return await asyncio.to_thread(
                    self._list_dir, str(tool_input.get("path") or ".")
                )
        except ValueError as exc:
            return str(exc), True
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}", True
        return f"unknown local tool: {name}", True

    def _read_file(self, raw_path: str) -> tuple[str, bool]:
        path = self._confined_path(raw_path)
        data = path.read_bytes()[:MAX_FILE_READ_BYTES]
        return data.decode("utf-8", errors="replace"), False

    def _write_file(self, raw_path: str, content: str) -> tuple[str, bool]:
        path = self._confined_path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {raw_path}", False

    def _list_dir(self, raw_path: str) -> tuple[str, bool]:
        path = self._confined_path(raw_path)
        rows = []
        for entry in sorted(path.iterdir(), key=lambda p: p.name):
            kind = "dir" if entry.is_dir() else "file"
            rows.append(f"{kind}\t{entry.name}")
        return "\n".join(rows) or "(empty directory)", False

    async def _shell(self, command: str) -> tuple[str, bool]:
        if not command.strip():
            return "empty command", True
        from kiro_crew.sandbox_spawn import scrub_agent_subprocess_env

        env = scrub_agent_subprocess_env(dict(os.environ))
        argv = ["/bin/bash", "-c", command]
        if self._sandbox_mode != "off":
            from kiro_crew.sandbox_spawn import wrap_argv_async

            mode = self._sandbox_mode if self._sandbox_mode in _SANDBOX_WRAP_MODES else "auto"
            argv, _launcher = await wrap_argv_async(
                argv, mode, exec_cwd=str(self._work_dir)
            )
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self._work_dir),
            env=env,
        )
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(), timeout=SHELL_TIMEOUT_SECONDS
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return f"command timed out after {SHELL_TIMEOUT_SECONDS:.0f}s", True
        text = (output or b"").decode("utf-8", errors="replace")
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n[output truncated]"
        return text, process.returncode not in (0, None)


# -- MCP routing ----------------------------------------------------------------


class _McpServerConnection:
    """One MCP server, spoken to over stdio JSON-RPC or HTTP POST."""

    def __init__(self, name: str, spec: dict[str, Any]) -> None:
        self.name = name
        self.spec = spec
        self.tools: list[dict[str, Any]] = []
        self.error = ""
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        try:
            await self._request("initialize", {"protocolVersion": "2024-11-05"})
            result = await self._request("tools/list", {})
            self.tools = [
                tool for tool in (result or {}).get("tools", []) if isinstance(tool, dict)
            ]
        except Exception as exc:  # noqa: BLE001 - named in the session report
            self.error = f"{type(exc).__name__}: {exc}"[:300]

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        try:
            result = await self._request(
                "tools/call", {"name": tool_name, "arguments": arguments}
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a tool error
            return f"{type(exc).__name__}: {exc}"[:MAX_TOOL_OUTPUT_CHARS], True
        parts: list[str] = []
        for block in (result or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        text = "\n".join(parts) or json.dumps(result, default=str)
        return text[:MAX_TOOL_OUTPUT_CHARS], bool((result or {}).get("isError"))

    async def close(self) -> None:
        if self._process is not None:
            with _suppress():
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            self._process = None

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self._next_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": method,
                "params": params,
            }
            if self.spec.get("url"):
                return await asyncio.to_thread(self._http_request, payload)
            return await self._stdio_request(payload)

    def _http_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            str(self.spec["url"]),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30.0) as response:
            body = response.read().decode("utf-8", errors="replace")
        if body.lstrip().startswith("event:") or "\ndata:" in body:
            for line in body.splitlines():
                if line.startswith("data:"):
                    body = line[5:].strip()
                    break
        parsed = json.loads(body)
        if parsed.get("error"):
            raise RuntimeError(str(parsed["error"])[:300])
        return parsed.get("result") or {}

    async def _stdio_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process is None or self._process.returncode is not None:
            from kiro_crew.sandbox_spawn import scrub_agent_subprocess_env

            env = scrub_agent_subprocess_env(dict(os.environ))
            for key, value in (self.spec.get("env") or {}).items():
                env[str(key)] = str(value)
            argv = [str(self.spec["command"])] + [
                str(arg) for arg in self.spec.get("args") or []
            ]
            self._process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self._process.stdin.drain()
        while True:
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=60.0)
            if not line:
                raise RuntimeError("mcp server closed its stdio stream")
            try:
                parsed = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if parsed.get("id") == payload["id"]:
                if parsed.get("error"):
                    raise RuntimeError(str(parsed["error"])[:300])
                return parsed.get("result") or {}


class _suppress:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


class ToolRouter:
    """The session's full tool set: local tools plus MCP server tools.

    Names are Converse-legal and unique: local tools keep their bare
    names; MCP tools are ``<server>_<tool>`` sanitized, with the routing
    map kept here. A server that fails to start is NAMED (in
    :meth:`failed_servers` and the provider's session report), never
    silently absent — design §5.2 rule 2.
    """

    def __init__(
        self,
        work_dir: Path,
        sandbox_mode: str = "",
        mcp_servers: dict[str, dict] | None = None,
    ) -> None:
        self._local = LocalToolExecutor(work_dir, sandbox_mode)
        self._servers: dict[str, _McpServerConnection] = {}
        self._routes: dict[str, tuple[_McpServerConnection, str]] = {}
        for name, spec in (mcp_servers or {}).items():
            self._servers[name] = _McpServerConnection(name, spec)

    async def start(self) -> None:
        for connection in self._servers.values():
            await connection.start()
            for tool in connection.tools:
                tool_name = str(tool.get("name") or "")
                if not tool_name:
                    continue
                offered = converse_tool_name(f"{connection.name}_{tool_name}")
                if offered in LOCAL_TOOL_NAMES or offered in self._routes:
                    offered = converse_tool_name(f"{offered}_{connection.name}")[:64]
                self._routes[offered] = (connection, tool_name)

    async def close(self) -> None:
        for connection in self._servers.values():
            await connection.close()

    def tool_specs(self) -> list[dict[str, Any]]:
        specs = list(LOCAL_TOOL_SPECS)
        for offered, (connection, tool_name) in sorted(self._routes.items()):
            tool = next(
                (t for t in connection.tools if t.get("name") == tool_name), {}
            )
            specs.append(
                {
                    "toolSpec": {
                        "name": offered,
                        "description": str(tool.get("description") or offered),
                        "inputSchema": {
                            "json": tool.get("inputSchema")
                            or {"type": "object", "properties": {}}
                        },
                    }
                }
            )
        return specs

    def declared_servers(self) -> tuple[str, ...]:
        return tuple(sorted(self._servers))

    def failed_servers(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, conn in self._servers.items() if conn.error)
        )

    def ready_servers(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, conn in self._servers.items() if not conn.error)
        )

    async def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Run one APPROVED tool call. The single execution entry point:
        the provider's gated call site is its only caller."""
        if name in LOCAL_TOOL_NAMES:
            return await self._local.execute(name, tool_input)
        route = self._routes.get(name)
        if route is None:
            return f"unknown tool: {name}", True
        connection, tool_name = route
        return await connection.call(tool_name, tool_input)
