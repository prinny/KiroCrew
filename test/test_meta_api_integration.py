"""Integration tests: MetaApiProvider + the REAL pinned app-server + stub API.

Layer-2 strategy (design section 5.2): the provider drives the actual
Codex app-server binary (pinned 0.167.2) whose forced custom provider
points at ``test/meta_api_stub.py`` — a scripted local Responses
endpoint. CI spends nothing: no network, no real key.

Skipped unless a ``codex`` binary is on PATH. The scenarios are the
design's pod probes: P-1 (approval gate), P-2 (end-to-end cost), P-3
(cancel/resume), P-4 (MCP through the loop), P-6 (catalogue
independence), plus the explicit compact path the recordings validated.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("codex") is None, reason="no codex binary on PATH")

STUB = Path(__file__).parent / "meta_api_stub.py"
MCP_SERVER = Path(__file__).parent / "meta_api_mcp_server.py"


class StubServer:
    def __init__(self, tmp_path: Path):
        self.state_path = tmp_path / "stub-state.json"
        self.log_path = tmp_path / "stub-requests.log"
        self.set_state(mode="plain")
        self.proc = subprocess.Popen(
            [sys.executable, str(STUB), str(self.state_path), str(self.log_path)],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "STUB_PORT": "0"},
        )
        line = self.proc.stdout.readline()
        self.base_url = f"http://127.0.0.1:{int(line.strip().split('=')[1])}/v1"

    def set_state(self, **kw):
        self.state_path.write_text(json.dumps(kw))

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)


@pytest.fixture()
def stub(tmp_path):
    server = StubServer(tmp_path)
    yield server
    server.stop()


def _provider(tmp_path: Path, base_url: str, **kw) -> object:
    from kiro_crew.providers.meta_api import MetaApiProvider

    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    args = {
        "work_dir": work_dir,
        "model": "muse-spark-1.3",
        "credential_ref": "",
        "api_base_url": base_url,
        "data_home": tmp_path / "home",
        "mcp_servers": {},
    }
    args.update(kw)
    return MetaApiProvider(**args)


@pytest.fixture(autouse=True)
def _env_key(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "integration-stub-key")


async def _run_turn(provider, message="hello", on_event=None):
    events = []
    async for event in provider.stream(message):
        events.append(event)
        if on_event is not None:
            await on_event(event)
    return events


def _completes(events):
    from kiro_crew.acp.types import EVENT_COMPLETE

    return [e for e in events if e.kind == EVENT_COMPLETE]


def test_p1_approval_gate_and_p2_cost_end_to_end(stub, tmp_path):
    """P-1: an unanswered approval executes NOTHING; on accept the command
    runs. P-2: the turn's cost equals the stub's injected usage priced by
    the rate card, exactly (2 calls x 0.001455)."""
    from kiro_crew.acp.types import EVENT_PERMISSION_REQUEST

    stub.set_state(mode="tool", tool_command="echo stub-tool-ran > stub_tool.txt")
    provider = _provider(tmp_path, stub.base_url)
    target = tmp_path / "work" / "stub_tool.txt"

    async def main():
        await provider.start()
        gate_observed = []

        async def on_event(event):
            if event.kind == EVENT_PERMISSION_REQUEST:
                gate_observed.append(target.exists())
                await provider.approve_tool(event.request_id)

        events = await _run_turn(provider, "use the shell tool", on_event)
        assert gate_observed == [False]  # nothing ran before the answer
        assert target.exists()  # and it ran after acceptance
        (complete,) = _completes(events)
        assert complete.stop_reason == "end_turn"
        assert provider.billing_stats().usage.cost_usd == pytest.approx(0.00291)
        await provider.shutdown()

    asyncio.run(main())


def test_p1_decline_executes_nothing(stub, tmp_path):
    from kiro_crew.acp.types import EVENT_PERMISSION_REQUEST

    stub.set_state(mode="tool", tool_command="echo stub-tool-ran > stub_tool.txt")
    provider = _provider(tmp_path, stub.base_url)
    target = tmp_path / "work" / "stub_tool.txt"

    async def main():
        await provider.start()

        async def on_event(event):
            if event.kind == EVENT_PERMISSION_REQUEST:
                await provider.reject_tool(event.request_id)

        events = await _run_turn(provider, "use the shell tool", on_event)
        assert not target.exists()
        (complete,) = _completes(events)
        assert complete.stop_reason == "end_turn"
        await provider.shutdown()

    asyncio.run(main())


def test_p3_cancel_during_approval_then_resume(stub, tmp_path):
    from kiro_crew.acp.types import EVENT_PERMISSION_REQUEST

    stub.set_state(mode="tool", tool_command="echo stub-tool-ran > stub_tool.txt")
    provider = _provider(tmp_path, stub.base_url)
    target = tmp_path / "work" / "stub_tool.txt"

    async def main():
        await provider.start()
        outcome = []

        async def on_event(event):
            if event.kind == EVENT_PERMISSION_REQUEST:
                outcome.append(await provider.cancel())

        events = await _run_turn(provider, "use the shell tool", on_event)
        assert outcome == ["acked"]
        (complete,) = _completes(events)
        assert complete.stop_reason == "cancelled"
        assert not target.exists()
        # Resume: a fresh prompt on the same provider completes normally.
        stub.set_state(mode="plain")
        events = await _run_turn(provider, "hello again")
        (complete,) = _completes(events)
        assert complete.stop_reason == "end_turn"
        await provider.shutdown()

    asyncio.run(main())


def test_p6_catalogue_comes_from_crew_not_the_loop(stub, tmp_path):
    provider = _provider(tmp_path, stub.base_url)

    async def main():
        await provider.start()
        ids = [row["modelId"] for row in provider.available_models()]
        assert ids[:4] == [
            "muse-spark-1.3",
            "muse-spark-1.3-contributor",
            "muse-wonder-1.0",
            "muse-ember-1.0",
        ]
        assert provider.served_model == "muse-spark-1.3"
        await provider.shutdown()

    asyncio.run(main())


async def _wait_for_compaction(provider, timeout):
    return await provider.wait_for_compaction(timeout=timeout)


def test_explicit_compact_path(stub, tmp_path):
    provider = _provider(tmp_path, stub.base_url)

    async def main():
        await provider.start()
        await _run_turn(provider, "hello")
        await provider.compact()
        result = await _wait_for_compaction(provider, 30.0)
        assert result.get("type") != "timeout"
        await provider.shutdown()

    asyncio.run(main())


def test_p4_mcp_server_registers_and_reports(stub, tmp_path):
    """P-4, the part the stub evidence supports: an MCP server configured
    for the session is passed to the loop at spawn, the loop launches it,
    and its startup state reaches the provider's report as ready.

    The remaining half of P-4 — the served model actually CALLING the
    tool — is a declared gap in this build (providers.md): with the
    loop's fallback metadata for an unknown model id, the app-server
    offered the model no function tools at all in the recorded probe, so
    a model->MCP call cannot be evidenced against the stub. The earlier
    full-call attempt is superseded by this test.
    """
    calls_log = tmp_path / "mcp-calls.log"
    provider = _provider(
        tmp_path,
        stub.base_url,
        mcp_servers={
            "meta-api-test": {
                "command": sys.executable,
                "args": [str(MCP_SERVER), str(calls_log)],
            }
        },
    )

    async def main():
        await provider.start()
        events = await _run_turn(provider, "hello")
        (complete,) = _completes(events)
        assert complete.stop_reason == "end_turn"
        report = provider.mcp_session_report()
        assert report is not None
        payload = report.payload()
        assert payload["servers"]["meta-api-test"]["status"] == "ready"
        assert report.problem_summary() == ""
        await provider.shutdown()

    asyncio.run(main())
