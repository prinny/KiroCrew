"""Contract + loop tests for BedrockProvider over a scripted fake boto3
session (bedrock_stub).

Mirrors the ACP/Meta provider contracts: exactly one terminal complete
per turn, billing_stats() returns a DIFFERENT object per turn while the
usage object a turn's events carry is the one read back, approve of an
unknown id is False, cancel with no turn is "no_turn", cleanup_session
succeeds with nothing running. The loop tests pin design §5.2: a tool
call executes ONLY after approval (rule 1), a denial lands in the next
call's messages as an error toolResult the model can see, the tool set
offered is the session's tool set, and the step cap ends the turn.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from bedrock_stub import FakeRuntime, make_provider, text_stream, tool_stream

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_PERMISSION_REQUEST,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_RESULT,
)


async def _collect(provider, message="hello", *, approve=True):
    events = []

    async def drive():
        async for event in provider.stream(message):
            events.append(event)
            if event.kind == EVENT_PERMISSION_REQUEST:
                if approve:
                    await provider.approve_tool(event.request_id)
                else:
                    await provider.reject_tool(event.request_id)

    await asyncio.wait_for(drive(), timeout=15)
    return events


def test_plain_turn_single_complete_and_text(tmp_path):
    async def main():
        runtime = FakeRuntime([text_stream("Hello world")])
        provider, _ = make_provider(tmp_path, runtime)
        events = await _collect(provider)
        completes = [e for e in events if e.kind == EVENT_COMPLETE]
        assert len(completes) == 1
        assert completes[0].stop_reason == "end_turn"
        texts = "".join(e.text for e in events if e.kind == EVENT_TEXT_CHUNK)
        assert texts == "Hello world"
        assert provider.served_model == "us.amazon.nova-pro-v1:0"
        assert provider.context_usage_unknown() is False
        assert provider.billing_stats().usage.input_tokens == 10

    asyncio.run(main())


def test_billing_object_is_fresh_per_turn_but_stable_within_one(tmp_path):
    async def main():
        runtime = FakeRuntime([text_stream("one"), text_stream("two")])
        provider, _ = make_provider(tmp_path, runtime)
        events1 = await _collect(provider, "first")
        stats1 = provider.billing_stats()
        complete1 = next(e for e in events1 if e.kind == EVENT_COMPLETE)
        assert complete1.usage is stats1.usage
        await _collect(provider, "second")
        stats2 = provider.billing_stats()
        assert stats2 is not stats1
        assert stats2.usage.input_tokens == 10  # not accumulated across turns

    asyncio.run(main())


def test_declarations_are_bedrocks_own(tmp_path):
    provider, _ = make_provider(tmp_path)
    assert provider.is_bedrock_backend is True
    assert provider.is_meta_api_backend is False
    assert provider.context_provider_type == "bedrock"
    assert provider.manual_compact_unsupported_backend == "bedrock"
    assert provider.supports_steer is False
    assert provider.supports_refusal_steer is False
    assert provider.approval_withdrawable is False
    assert provider.background_launch() is None
    assert provider.reports_turn_usage is True
    assert provider.mcp_config_hot_reload is False
    assert provider.get_valid_effort_levels() == []
    assert provider.supports_effort() is False
    assert provider.process_instance == ""
    assert provider.exit_code is None


def test_approve_unknown_id_is_false_and_cancel_without_turn(tmp_path):
    async def main():
        provider, _ = make_provider(tmp_path)
        assert await provider.approve_tool("nope") is False
        assert await provider.cancel() == "no_turn"

    asyncio.run(main())


def test_session_label_is_bedrock_not_acp(tmp_path):
    # providers/acp.py provider_label answers native providers by TYPE:
    # without the branch a Bedrock session persists under the ACP label.
    from kiro_crew.providers.acp import provider_label

    provider, _ = make_provider(tmp_path)
    assert provider_label(provider) == "bedrock"


def test_turn_lifecycle_reports_an_open_turn(tmp_path):
    async def main():
        provider, _ = make_provider(tmp_path)
        assert provider.has_active_turn() is False
        assert await provider.wait_turn_done(0.01) == ""
        runtime = FakeRuntime([tool_stream(), text_stream("done")])
        provider2, _ = make_provider(tmp_path, runtime)

        async def drive():
            async for event in provider2.stream("run it"):
                if event.kind == EVENT_PERMISSION_REQUEST:
                    assert provider2.has_active_turn() is True
                    assert provider2.has_unfinished_turn() is True
                    await provider2.approve_tool(event.request_id)

        await drive()
        assert provider2.has_active_turn() is False
        assert await provider2.wait_turn_done(0.5) == "end_turn"

    asyncio.run(main())


def test_tool_turn_executes_only_after_approval(tmp_path):
    async def main():
        runtime = FakeRuntime([tool_stream(), text_stream("done")])
        provider, _ = make_provider(tmp_path, runtime)
        events = await _collect(provider, "run it")
        kinds = [e.kind for e in events]
        assert EVENT_PERMISSION_REQUEST in kinds
        result_event = next(e for e in events if e.kind == EVENT_TOOL_RESULT)
        assert "bedrock-tool-ok" in result_event.tool_output
        # The second Converse call carried the tool result back.
        second_call_messages = runtime.calls[1]["messages"]
        last = second_call_messages[-1]
        assert last["role"] == "user"
        tool_result = last["content"][0]["toolResult"]
        assert tool_result["status"] == "success"
        assert "bedrock-tool-ok" in tool_result["content"][0]["text"]
        completes = [e for e in events if e.kind == EVENT_COMPLETE]
        assert len(completes) == 1 and completes[0].stop_reason == "end_turn"

    asyncio.run(main())


def test_denied_tool_executes_nothing_and_model_sees_the_denial(tmp_path):
    async def main():
        marker = tmp_path / "work" / "denied-marker.txt"
        script = tool_stream(
            tool_input_json=json.dumps(
                {"command": f"touch {marker}"}
            )
        )
        runtime = FakeRuntime([script, text_stream("understood")])
        provider, _ = make_provider(tmp_path, runtime)
        events = await _collect(provider, "run it", approve=False)
        assert not marker.exists()  # rule 1: no approval, no execution
        assert not any(e.kind == EVENT_TOOL_RESULT for e in events)
        second_call_messages = runtime.calls[1]["messages"]
        tool_result = second_call_messages[-1]["content"][0]["toolResult"]
        assert tool_result["status"] == "error"
        assert "declined" in tool_result["content"][0]["text"]

    asyncio.run(main())


def test_tool_step_cap_ends_the_turn(tmp_path):
    async def main():
        runtime = FakeRuntime([tool_stream(), tool_stream(), tool_stream()])
        provider, _ = make_provider(tmp_path, runtime, max_tool_steps=2)
        events = await _collect(provider, "loop forever")
        complete = next(e for e in events if e.kind == EVENT_COMPLETE)
        assert complete.stop_reason == "error: tool step limit reached"
        assert len(runtime.calls) == 3  # initial + 2 steps; the cap stops the 4th

    asyncio.run(main())


def test_journal_records_full_transcript_and_identity_header(tmp_path):
    async def main():
        runtime = FakeRuntime([tool_stream(), text_stream("done")])
        provider, _ = make_provider(tmp_path, runtime)
        await _collect(provider, "run it")
        journal = tmp_path / "work" / ".bedrock-journal.jsonl"
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        header = records[0]
        assert header["type"] == "header"
        assert header["identity_account"] == "123456789012"
        assert header["identity_source_class"]
        bodies = [r["message"] for r in records if r.get("type") == "message"]
        roles = [m["role"] for m in bodies]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert "toolUse" in bodies[1]["content"][0]
        assert "toolResult" in bodies[2]["content"][0]
        # No credential-shaped material anywhere in the journal.
        assert "AWS_SECRET" not in journal.read_text()

    asyncio.run(main())


def test_cleanup_session_removes_journal(tmp_path):
    async def main():
        runtime = FakeRuntime([text_stream()])
        provider, _ = make_provider(tmp_path, runtime)
        await _collect(provider)
        journal = tmp_path / "work" / ".bedrock-journal.jsonl"
        assert journal.exists()
        await provider.cleanup_session(provider.session_id)
        assert not journal.exists()

    asyncio.run(main())


def test_refusal_turn_surfaces_typed_refusal(tmp_path):
    async def main():
        script = [
            {"messageStart": {"role": "assistant"}},
            {"messageStop": {"stopReason": "content_filtered"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 0}}},
        ]
        runtime = FakeRuntime([script])
        provider, _ = make_provider(tmp_path, runtime)
        events = await _collect(provider, "something refused")
        complete = next(e for e in events if e.kind == EVENT_COMPLETE)
        assert complete.stop_reason == "refusal"
        assert complete.refusal is not None
        assert complete.refusal.category == "content_filtered"

    asyncio.run(main())


def test_throttled_call_retries_then_succeeds(tmp_path):
    async def main():
        import botocore.exceptions

        class ThrottleOnceRuntime(FakeRuntime):
            def __init__(self):
                super().__init__([text_stream("after retry")])
                self.attempts = 0

            def converse_stream(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise botocore.exceptions.ClientError(
                        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
                        "ConverseStream",
                    )
                return super().converse_stream(**kwargs)

        runtime = ThrottleOnceRuntime()
        provider, _ = make_provider(tmp_path, runtime)
        events = await _collect(provider, "hi")
        complete = next(e for e in events if e.kind == EVENT_COMPLETE)
        assert complete.stop_reason == "end_turn"
        assert runtime.attempts == 2

    asyncio.run(main())


def test_model_pin_to_invoke_only_model_refused(tmp_path):
    async def main():
        from bedrock_stub import FakeControl, FakeSession

        class ControlWithJ2(FakeControl):
            def list_foundation_models(self, **kwargs):
                page = super().list_foundation_models(**kwargs)
                page["modelSummaries"].append(
                    {
                        "modelId": "ai21.j2-ultra-v1",
                        "modelName": "Jurassic-2 Ultra",
                        "inferenceTypesSupported": ["ON_DEMAND"],
                        "responseStreamingSupported": False,
                    }
                )
                return page

        session = FakeSession()
        session._clients["bedrock"] = ControlWithJ2()
        provider, _ = make_provider(
            tmp_path,
            session_factory=lambda profile: session,
            model="ai21.j2-ultra-v1",
        )
        from kiro_crew.providers.meta_api_client import AppServerError

        with pytest.raises(AppServerError, match="InvokeModel-only"):
            await provider.start()
        assert provider.model_pin_refused.startswith("Model ai21.j2-ultra-v1")

    asyncio.run(main())
