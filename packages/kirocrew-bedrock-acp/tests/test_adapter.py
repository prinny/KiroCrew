"""The adapter's protocol and Bedrock translation, off fake AWS and MCP peers.

No network, no credentials: the Bedrock driver is a stub returning canned
Converse event streams, and the MCP server is a small Python script speaking
stdio JSON-RPC, so what is pinned is the adapter's own behaviour -- the ACP
frames it emits, the permission round trip, and the catalogue join.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import pytest
from kirocrew_bedrock_acp import bedrock
from kirocrew_bedrock_acp import main as adapter_main

FAKE_MCP_SERVER = r"""
import json, sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    if "id" not in msg:
        continue
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {},
                  "serverInfo": {"name": "fake", "version": "0"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "echo text back",
                             "inputSchema": {"type": "object",
                                             "properties": {"text": {"type": "string"}}}}]}
    elif method == "tools/call":
        text = (msg.get("params") or {}).get("arguments", {}).get("text", "")
        result = {"content": [{"type": "text", "text": "echo:" + text}], "isError": False}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


def _tool_turn():
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "fake__echo"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"text": "hi"}'}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8}}},
    ]


def _text_turn():
    return [
        {"contentBlockDelta": {"delta": {"text": "done"}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 9, "outputTokens": 2, "totalTokens": 11}}},
    ]


class FakeDriver:
    """The BedrockDriver shape: a catalogue and canned Converse streams."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []

    def catalogue(self):
        return [("amazon.nova-pro-v1:0", "Nova Pro"), ("deepseek.v3.2", "DeepSeek v3.2")]

    def converse_stream(self, **kwargs):
        import copy

        # A snapshot: the session's message list keeps growing after the
        # call, and the assertions read what the call was actually given.
        self.calls.append(copy.deepcopy(kwargs))
        return iter(self._turns.pop(0))


class Collector:
    """sys.stdout stand-in: collects complete frames under a lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self.frames = []

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                with self._lock:
                    self.frames.append(json.loads(line))

    def flush(self):
        pass

    def find(self, predicate, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for frame in self.frames:
                    if predicate(frame):
                        return frame
            time.sleep(0.02)
        raise AssertionError(f"no matching frame in {self.frames!r}")


@pytest.fixture()
def adapter():
    collector = Collector()
    driver = FakeDriver([_tool_turn(), _text_turn()])
    args = argparse.Namespace(model=None, models=[], max_tokens=1024)
    return adapter_main.Adapter(driver, args, out=collector), collector, driver


def _request(adapter, request_id, method, params):
    adapter.handle({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def _new_session(adapter, collector, tmp_path=None, mcp_servers=None):
    _request(adapter, "new-1", "session/new", {"cwd": "/tmp", "mcpServers": mcp_servers or []})
    frame = collector.find(lambda f: f.get("id") == "new-1")
    return frame["result"]["sessionId"], frame["result"]["configOptions"]


def test_initialize_advertises_no_session_load(adapter):
    _a, collector, _d = adapter
    _request(_a, "i1", "initialize", {"protocolVersion": 1})
    frame = collector.find(lambda f: f.get("id") == "i1")
    result = frame["result"]
    assert result["protocolVersion"] == 1
    assert result["agentCapabilities"]["loadSession"] is False


def test_session_new_offers_the_catalogue_as_a_model_option(adapter):
    _a, collector, _d = adapter
    _sid, options = _new_session(_a, collector)
    model = next(o for o in options if o["id"] == "model")
    assert model["currentValue"] == "amazon.nova-pro-v1:0"
    assert {o["value"] for o in model["options"]} == {
        "amazon.nova-pro-v1:0",
        "deepseek.v3.2",
    }
    approval = next(o for o in options if o["id"] == "approval_mode")
    assert approval["currentValue"] == "promptUnmatched"


def test_set_config_option_switches_model_and_refuses_unknown_values(adapter):
    _a, collector, _d = adapter
    sid, _options = _new_session(_a, collector)
    _request(
        _a,
        "set-1",
        "session/set_config_option",
        {"sessionId": sid, "configId": "model", "value": "deepseek.v3.2"},
    )
    frame = collector.find(lambda f: f.get("id") == "set-1")
    model = next(o for o in frame["result"]["configOptions"] if o["id"] == "model")
    assert model["currentValue"] == "deepseek.v3.2"

    _request(
        _a,
        "set-2",
        "session/set_config_option",
        {"sessionId": sid, "configId": "model", "value": "not-a-model"},
    )
    frame = collector.find(lambda f: f.get("id") == "set-2")
    assert frame["error"]["code"] == -32602
    assert "Invalid value for config option model" in frame["error"]["message"]


def test_prompt_runs_a_tool_only_after_permission(adapter, tmp_path):
    _a, collector, driver = adapter
    script = tmp_path / "fake_mcp.py"
    script.write_text(FAKE_MCP_SERVER)
    sid, _options = _new_session(
        _a,
        collector,
        mcp_servers=[{"name": "fake", "command": sys.executable, "args": [str(script)], "env": []}],
    )
    _request(
        _a, "p1", "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]}
    )
    # The adapter asks before the tool runs; answer allow.
    request = collector.find(lambda f: f.get("method") == "session/request_permission")
    _a.handle(
        {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}},
        }
    )
    frame = collector.find(lambda f: f.get("id") == "p1")
    assert frame["result"]["stopReason"] == "end_turn"
    assert frame["result"]["usage"]["totalTokens"] == 19
    # The tool reached Converse in its wire shape: the schema under "json".
    spec = driver.calls[0]["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "fake__echo"
    assert set(spec["inputSchema"]) == {"json"}
    assert spec["inputSchema"]["json"]["properties"] == {"text": {"type": "string"}}
    # The tool result rode back into the second Converse call.
    second = driver.calls[1]
    tool_result = second["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["content"][0]["text"] == "echo:hi"
    # And a tool_call update was emitted for the call.
    updates = [f for f in collector.frames if f.get("method") == "session/update"]
    assert any(f["params"]["update"].get("sessionUpdate") == "tool_call" for f in updates)


def test_prompt_in_deny_mode_never_asks_and_denies(adapter, tmp_path):
    _a, collector, driver = adapter
    script = tmp_path / "fake_mcp.py"
    script.write_text(FAKE_MCP_SERVER)
    sid, _options = _new_session(
        _a,
        collector,
        mcp_servers=[{"name": "fake", "command": sys.executable, "args": [str(script)], "env": []}],
    )
    _request(
        _a,
        "set-deny",
        "session/set_config_option",
        {"sessionId": sid, "configId": "approval_mode", "value": "denyUnmatched"},
    )
    collector.find(lambda f: f.get("id") == "set-deny")
    _request(
        _a, "p2", "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]}
    )
    frame = collector.find(lambda f: f.get("id") == "p2")
    assert frame["result"]["stopReason"] == "end_turn"
    assert not any(f.get("method") == "session/request_permission" for f in collector.frames)
    tool_result = driver.calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["status"] == "error"
    assert "denyUnmatched" in tool_result["content"][0]["text"]


def test_session_load_is_method_not_found(adapter):
    _a, collector, _d = adapter
    _request(_a, "l1", "session/load", {"sessionId": "x", "cwd": "/tmp", "mcpServers": []})
    frame = collector.find(lambda f: f.get("id") == "l1")
    assert frame["error"]["code"] == -32601


# ── Catalogue join ──


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **_kwargs):
        return iter(self._pages)


class _FakeControl:
    def get_paginator(self, name):
        if name == "list_foundation_models":
            return _FakePaginator(
                [
                    {
                        "modelSummaries": [
                            {
                                "modelId": "amazon.nova-pro-v1:0",
                                "modelName": "Nova Pro",
                                "modelArn": "arn:model:nova",
                                "inferenceTypesSupported": ["ON_DEMAND"],
                            },
                            {
                                "modelId": "example.provisioned-only-v1:0",
                                "modelName": "Provisioned Only",
                                "modelArn": "arn:model:provisioned",
                                "inferenceTypesSupported": ["PROVISIONED"],
                            },
                        ]
                    }
                ]
            )
        return _FakePaginator(
            [
                {
                    "inferenceProfileSummaries": [
                        {
                            "inferenceProfileId": "us.amazon.nova-pro-v1:0",
                            "inferenceProfileName": "US Nova Pro",
                            "status": "ACTIVE",
                            "models": [{"modelArn": "arn:model:nova"}],
                        },
                        {
                            "inferenceProfileId": "us.retired",
                            "inferenceProfileName": "Retired",
                            "status": "INACTIVE",
                            "models": [{"modelArn": "arn:model:nova"}],
                        },
                    ]
                }
            ]
        )


class _FakeSession:
    def client(self, name):
        assert name == "bedrock"
        return _FakeControl()


def test_catalogue_joins_profiles_on_model_arn():
    driver = bedrock.BedrockDriver(session_factory=lambda **_kw: _FakeSession())
    models = driver.catalogue()
    assert ("amazon.nova-pro-v1:0", "Nova Pro") in models
    assert ("us.amazon.nova-pro-v1:0", "US Nova Pro (Nova Pro)") in models
    assert all(value != "us.retired" for value, _name in models)
    # A provisioned-throughput-only foundation model is not offered bare:
    # it cannot be invoked on demand, so it is not a selectable id.
    assert all(value != "example.provisioned-only-v1:0" for value, _name in models)


def test_converse_tool_name_is_sanitised():
    assert bedrock.converse_tool_name("my server", "do.thing") == "my_server__do_thing"


# ── Credential chain (hermetic: fixture config/credentials files only) ──
#
# The adapter resolves the boto3 default chain inside its own process.
# These tests point the SDK at fixture files in a temp directory, disable
# instance-metadata lookup, and assert which source each profile resolves
# from -- no network, no real credentials.


def _aws_fixtures(tmp_path, monkeypatch):
    proc_script = tmp_path / "proc_creds.py"
    proc_script.write_text(
        "import json, sys\n"
        "sys.stdout.write(json.dumps({"
        "'Version': 1, "
        "'AccessKeyId': 'AKIAPROCFIXTURE01', "
        "'SecretAccessKey': 'proc-fixture-secret', "
        "'Expiration': '2999-01-01T00:00:00Z'}))\n"
    )
    config = tmp_path / "config"
    config.write_text(
        "[profile staticprof]\nregion = us-west-2\n"
        "[profile procprof]\nregion = us-west-2\n"
        f"credential_process = {sys.executable} {proc_script}\n"
    )
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[staticprof]\n"
        "aws_access_key_id = AKIASTATICFIXTURE1\n"
        "aws_secret_access_key = static-fixture-secret\n"
        "aws_session_token = static-fixture-token\n"
    )
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def test_chain_resolves_a_static_profile_with_its_session_token(tmp_path, monkeypatch):
    _aws_fixtures(tmp_path, monkeypatch)
    driver = bedrock.BedrockDriver(profile="staticprof", region="us-west-2")
    creds = driver._ensure_session().get_credentials()
    assert creds.access_key == "AKIASTATICFIXTURE1"
    assert creds.secret_key == "static-fixture-secret"
    assert creds.token == "static-fixture-token"
    assert creds.method == "shared-credentials-file"


def test_chain_resolves_a_credential_process_profile(tmp_path, monkeypatch):
    _aws_fixtures(tmp_path, monkeypatch)
    driver = bedrock.BedrockDriver(profile="procprof", region="us-west-2")
    creds = driver._ensure_session().get_credentials()
    assert creds.access_key == "AKIAPROCFIXTURE01"
    assert creds.method == "custom-process"


def test_chain_selects_the_profile_from_the_sdk_environment(tmp_path, monkeypatch):
    _aws_fixtures(tmp_path, monkeypatch)
    monkeypatch.setenv("AWS_PROFILE", "procprof")
    driver = bedrock.BedrockDriver(region="us-west-2")
    creds = driver._ensure_session().get_credentials()
    assert creds.access_key == "AKIAPROCFIXTURE01"


def test_chain_resolves_environment_static_keys(tmp_path, monkeypatch):
    _aws_fixtures(tmp_path, monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENVFIXTURE0001")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-fixture-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env-fixture-token")
    driver = bedrock.BedrockDriver(region="us-west-2")
    creds = driver._ensure_session().get_credentials()
    assert creds.access_key == "AKIAENVFIXTURE0001"
    assert creds.token == "env-fixture-token"
    assert creds.method == "env"
