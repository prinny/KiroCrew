"""Contract tests for MetaApiProvider over a scripted fake app-server client.

Mirrors the ACP provider contract by name: exactly one terminal complete
per turn, approve-before-pending is a no-op False, cancel with no turn is
"no_turn", billing_stats() returns a DIFFERENT object per turn,
cleanup_session succeeds with the child already dead, and the key value
never reaches the transcript journal.
"""

from __future__ import annotations

import asyncio

import pytest

import kiro_crew.providers.meta_api as meta_api_mod
from kiro_crew.acp.types import EVENT_COMPLETE, EVENT_PERMISSION_REQUEST
from kiro_crew.providers.meta_api import MetaApiProvider

USAGE_FRAME = {
    "method": "thread/tokenUsage/updated",
    "params": {
        "threadId": "thread-1",
        "turnId": "turn-1",
        "tokenUsage": {
            "last": {
                "totalTokens": 1100,
                "inputTokens": 1000,
                "cachedInputTokens": 200,
                "outputTokens": 100,
                "reasoningOutputTokens": 40,
            },
            "total": {},
            "modelContextWindow": 258400,
        },
    },
}

PLAIN_TURN_FRAMES = [
    {"method": "turn/started", "params": {"threadId": "thread-1", "turn": {"id": "turn-1"}}},
    {
        "method": "item/agentMessage/delta",
        "params": {"itemId": "msg-1", "delta": "hello"},
    },
    USAGE_FRAME,
    {
        "method": "turn/completed",
        "params": {
            "threadId": "thread-1",
            "turn": {"id": "turn-1", "status": "completed", "items": []},
        },
    },
]

APPROVAL_FRAME = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "item/commandExecution/requestApproval",
    "params": {
        "threadId": "thread-1",
        "turnId": "turn-1",
        "itemId": "call-1",
        "command": "touch stub_tool.txt",
        "cwd": "/work",
    },
}


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.frame_handler = None
        self.exit_handler = None
        self.requests: list[tuple] = []
        self.responses: list[tuple] = []
        self.scripts: dict[str, list[dict]] = {}
        self.alive = True
        self.pid = 4242
        self.process_instance = "fake-instance"
        self.exit_code = None
        FakeClient.instances.append(self)

    def set_frame_handler(self, handler):
        self.frame_handler = handler

    def set_exit_handler(self, handler):
        self.exit_handler = handler

    async def start(self):
        return None

    async def close(self):
        self.alive = False

    def is_alive(self):
        return self.alive

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "thread/start":
            result = {"thread": {"id": "thread-1"}}
        elif method == "turn/start":
            result = {"turn": {"id": "turn-1"}}
        else:
            result = {}
        for frame in self.scripts.get(method, []):
            asyncio.get_running_loop().create_task(self._feed(frame))
        return result

    async def _feed(self, frame):
        await self.frame_handler(frame)

    async def respond(self, request_id, result):
        self.responses.append((request_id, result))


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    FakeClient.instances.clear()
    monkeypatch.setattr(meta_api_mod, "AppServerClient", FakeClient)
    monkeypatch.setattr(meta_api_mod, "codex_binary", lambda: "/fake/codex")
    monkeypatch.setenv("MODEL_API_KEY", "contract-test-key")
    work_dir = tmp_path / "work"
    work_dir.mkdir()  # the factory's session work dir exists before start
    return MetaApiProvider(
        work_dir=work_dir,
        model="muse-spark-1.3",
        credential_ref="",
        api_base_url="http://127.0.0.1:1/v1",  # catalogue refresh fails fast
        data_home=tmp_path / "home",
        mcp_servers={},
    )


async def _collect(provider, message="hi"):
    return [event async for event in provider.stream(message)]


def test_plain_turn_exactly_one_terminal_complete(provider):
    async def main():
        await provider.start()
        client = FakeClient.instances[-1]
        client.scripts["turn/start"] = list(PLAIN_TURN_FRAMES)
        events = await _collect(provider)
        completes = [e for e in events if e.kind == EVENT_COMPLETE]
        assert len(completes) == 1
        assert completes[0].stop_reason == "end_turn"
        assert provider.billing_stats().usage.cost_usd == pytest.approx(0.001455)

    asyncio.run(main())


def test_billing_stats_object_differs_per_turn(provider):
    async def main():
        await provider.start()
        client = FakeClient.instances[-1]
        client.scripts["turn/start"] = list(PLAIN_TURN_FRAMES)
        await _collect(provider, "one")
        first = provider.billing_stats()
        await _collect(provider, "two")
        second = provider.billing_stats()
        assert first is not second
        assert second.usage.input_tokens == 1000

    asyncio.run(main())


def test_approve_before_pending_is_noop_false(provider):
    async def main():
        await provider.start()
        assert await provider.approve_tool("nope") is False
        assert await provider.cancel() == "no_turn"

    asyncio.run(main())


def test_approval_round_trip_through_the_provider(provider):
    async def main():
        await provider.start()
        client = FakeClient.instances[-1]
        frames = [
            PLAIN_TURN_FRAMES[0],
            {
                "method": "item/started",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "id": "call-1",
                        "command": "touch stub_tool.txt",
                    }
                },
            },
            APPROVAL_FRAME,
        ]
        client.scripts["turn/start"] = frames
        seen = []
        async for event in provider.stream("run it"):
            seen.append(event)
            if event.kind == EVENT_PERMISSION_REQUEST:
                assert await provider.approve_tool(event.request_id) is True
                # Answer the rest of the turn so the stream can terminate.
                for frame in [
                    USAGE_FRAME,
                    PLAIN_TURN_FRAMES[-1],
                ]:
                    await client.frame_handler(frame)
        assert client.responses == [(0, {"decision": "accept"})]
        assert [e.kind for e in seen if e.kind == EVENT_COMPLETE]

    asyncio.run(main())


async def _approve_with_deadline(provider, deadline):
    return await provider.approve_tool("0", deadline=deadline)


def test_expired_deadline_raises_grant_ended(provider):
    from kiro_crew.trust_patterns import GrantDeadline, GrantEnded

    async def main():
        await provider.start()
        expired = GrantDeadline(wall=0.0, mono=0.0)
        with pytest.raises(GrantEnded):
            await _approve_with_deadline(provider, expired)

    asyncio.run(main())


def test_cleanup_session_with_dead_child(provider, tmp_path):
    async def main():
        await provider.start()
        client = FakeClient.instances[-1]
        client.scripts["turn/start"] = list(PLAIN_TURN_FRAMES)
        await _collect(provider, "leave a journal behind")
        journal = tmp_path / "work" / ".meta-api-journal.jsonl"
        assert journal.exists()
        client.alive = False  # the child is already gone
        await provider.cleanup_session("session-1")  # must not raise
        assert not journal.exists()

    asyncio.run(main())


def test_context_usage_unknown_until_first_usage_frame(provider):
    assert provider.context_usage_unknown() is True
    assert provider.context_usage_pct() == 0.0


def test_journal_never_contains_the_key(provider, tmp_path):
    async def main():
        await provider.start()
        client = FakeClient.instances[-1]
        client.scripts["turn/start"] = list(PLAIN_TURN_FRAMES)
        await _collect(provider, "a user message")
        journal = (tmp_path / "work" / ".meta-api-journal.jsonl").read_text()
        assert "a user message" in journal
        assert "contract-test-key" not in journal

    asyncio.run(main())


def test_capability_declarations_are_deliberate(provider):
    """Parity checklist row 5: every H14 capability property on the native
    provider returns a DELIBERATE value. A future ABC addition that this
    provider should answer differently fails here rather than silently
    inheriting."""
    assert provider.is_meta_api_backend is True
    assert provider.is_claude_backend is False
    assert provider.context_provider_type == "meta_api"
    assert provider.compaction_self_managed is True
    assert provider.harness_managed_compaction_backend == "meta_api"
    # The explicit-compact path is parity-tested on the pinned binary.
    assert provider.manual_compact_unsupported_backend is None
    assert provider.supports_steer is False
    assert provider.supports_refusal_steer is False
    assert provider.last_steer_monotonic == 0.0
    assert provider.background_launch() is None
    assert provider.reports_turn_usage is True
    assert provider.approval_withdrawable is False
    assert provider.turn_survives_disconnect is False
    assert provider.mcp_config_hot_reload is False
    assert provider.mcp_config_wire_reinject is False
    assert provider.mcp_config_relaunch_reload is False


def test_process_death_mid_turn_yields_one_synthetic_complete(provider):
    async def main():
        await provider.start()
        client = FakeClient.instances[-1]
        # The turn opens and streams one delta, then never completes.
        client.scripts["turn/start"] = [PLAIN_TURN_FRAMES[0], PLAIN_TURN_FRAMES[1]]
        events = []
        async for event in provider.stream("hi"):
            events.append(event)
            client.exit_handler()  # the process dies mid-turn
        completes = [e for e in events if e.kind == EVENT_COMPLETE]
        assert len(completes) == 1
        assert completes[0].synthetic_completion is True

    asyncio.run(main())
