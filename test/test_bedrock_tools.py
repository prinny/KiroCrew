"""Executor tests for the native Bedrock provider (design §5.2).

Pins the executor's posture: file tools confined to the session work
dir, shell tools running with the CHILD environment (the scrub removes
AWS_SECRET*/AWS_SESSION* while the provider signs gateway-side), the
tool set offered being exactly local + MCP tools, and an MCP server
that fails to start being NAMED in the report rather than silently
absent (rule 2).
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

from kiro_crew.providers.bedrock_tools import ToolRouter

FAKE_MCP_SERVER = textwrap.dedent(
    """
    import json, sys

    for line in sys.stdin:
        request = json.loads(line)
        method = request.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {}}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo the input text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                ]
            }
        elif method == "tools/call":
            text = (request["params"].get("arguments") or {}).get("text", "")
            result = {"content": [{"type": "text", "text": "echo:" + text}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result}), flush=True)
    """
)


def _router(tmp_path: Path, **kwargs) -> ToolRouter:
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    return ToolRouter(tmp_path / "work", "off", kwargs.get("mcp_servers"))


def test_local_tool_set_is_the_spec(tmp_path):
    router = _router(tmp_path)
    names = [s["toolSpec"]["name"] for s in router.tool_specs()]
    assert names == ["shell", "read_file", "write_file", "list_dir"]


def test_file_tools_round_trip_and_confine(tmp_path):
    async def main():
        router = _router(tmp_path)
        text, is_error = await router.execute(
            "write_file", {"path": "a/b.txt", "content": "hello"}
        )
        assert not is_error
        text, is_error = await router.execute("read_file", {"path": "a/b.txt"})
        assert text == "hello" and not is_error
        text, is_error = await router.execute("list_dir", {"path": "a"})
        assert "b.txt" in text and not is_error
        # Confinement: an escaping path is refused, never followed.
        text, is_error = await router.execute(
            "read_file", {"path": "../outside.txt"}
        )
        assert is_error and "escapes" in text
        text, is_error = await router.execute(
            "write_file", {"path": "/etc/bedrock-test-nope", "content": "x"}
        )
        assert is_error

    asyncio.run(main())


def test_shell_tool_runs_with_scrubbed_child_env(tmp_path, monkeypatch):
    async def main():
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-the-tool")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "must-not-reach-the-tool")
        router = _router(tmp_path)
        text, is_error = await router.execute(
            "shell",
            {"command": "echo \"secret=[$AWS_SECRET_ACCESS_KEY] token=[$AWS_SESSION_TOKEN]\""},
        )
        assert not is_error
        assert "secret=[]" in text and "token=[]" in text
        text, is_error = await router.execute("shell", {"command": "exit 3"})
        assert is_error

    asyncio.run(main())


def test_unknown_tool_is_an_error_not_a_crash(tmp_path):
    async def main():
        router = _router(tmp_path)
        text, is_error = await router.execute("nope", {})
        assert is_error and "unknown tool" in text

    asyncio.run(main())


def test_mcp_server_tools_are_offered_and_callable(tmp_path):
    async def main():
        server = tmp_path / "fake_mcp.py"
        server.write_text(FAKE_MCP_SERVER, encoding="utf-8")
        router = _router(
            tmp_path,
            mcp_servers={"fake": {"command": sys.executable, "args": [str(server)]}},
        )
        await router.start()
        try:
            assert router.ready_servers() == ("fake",)
            names = [s["toolSpec"]["name"] for s in router.tool_specs()]
            assert "fake_echo" in names
            text, is_error = await router.execute("fake_echo", {"text": "hi"})
            assert text == "echo:hi" and not is_error
        finally:
            await router.close()

    asyncio.run(main())


def test_unstartable_mcp_server_is_named_not_silent(tmp_path):
    async def main():
        router = _router(
            tmp_path,
            mcp_servers={"broken": {"command": "/nonexistent/mcp-binary-xyz"}},
        )
        await router.start()
        try:
            assert router.failed_servers() == ("broken",)
            assert router.declared_servers() == ("broken",)
            # Its tools are absent; local tools remain.
            names = [s["toolSpec"]["name"] for s in router.tool_specs()]
            assert names == ["shell", "read_file", "write_file", "list_dir"]
        finally:
            await router.close()

    asyncio.run(main())
