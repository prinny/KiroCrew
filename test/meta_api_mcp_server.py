"""A harmless stdio MCP server for the native provider's P-4 test.

Speaks just enough MCP (newline-delimited JSON-RPC) to be registered by
the app-server and to answer one read-only tool, ``ping``. Every tools/call
is appended to the log file named by argv[1] so the test can prove the
call travelled through the loop.
"""

from __future__ import annotations

import json
import sys

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/meta-api-mcp-calls.log"


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method = msg.get("method")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {
                        "protocolVersion": (msg.get("params") or {}).get(
                            "protocolVersion", "2025-03-26"
                        ),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "meta-api-test-mcp", "version": "0.0.1"},
                    },
                }
            )
        elif method == "tools/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "ping",
                                "description": "Harmless read-only ping.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                },
                            }
                        ]
                    },
                }
            )
        elif method == "tools/call":
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(msg.get("params") or {}) + "\n")
            text = ((msg.get("params") or {}).get("arguments") or {}).get("text", "")
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {
                        "content": [{"type": "text", "text": f"pong: {text}"}],
                        "isError": False,
                    },
                }
            )
        elif "id" in msg:  # anything else with an id gets an empty result
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})


if __name__ == "__main__":
    main()
