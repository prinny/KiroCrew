"""Scripted fake boto3 session for the native Bedrock provider tests.

One fake stands in for the whole boto3 surface the provider touches:
STS (identity), the bedrock control client (discovery) and the
bedrock-runtime client (Converse calls, scripted per test). No network,
no credentials: the fake's identity is a fixed test account.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

TEST_ACCOUNT = "123456789012"
TEST_ARN = "arn:aws:iam::123456789012:user/kirocrew-test"

FOUNDATION = [
    {
        "modelId": "amazon.nova-pro-v1:0",
        "modelName": "Nova Pro",
        "inferenceTypesSupported": ["INFERENCE_PROFILE"],
        "responseStreamingSupported": True,
    },
    {
        "modelId": "deepseek.v3.2",
        "modelName": "DeepSeek V3.2",
        "inferenceTypesSupported": ["ON_DEMAND"],
        "responseStreamingSupported": True,
    },
    {
        "modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "modelName": "Claude Sonnet 4.5",
        "inferenceTypesSupported": ["INFERENCE_PROFILE", "ON_DEMAND"],
        "responseStreamingSupported": True,
    },
]

PROFILES = [
    {
        "inferenceProfileId": "us.amazon.nova-pro-v1:0",
        "inferenceProfileName": "Nova Pro (US)",
        "status": "ACTIVE",
        "type": "SYSTEM_DEFINED",
        "models": [
            {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-pro-v1:0"}
        ],
    },
    {
        "inferenceProfileId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "inferenceProfileName": "Claude Sonnet 4.5 (US)",
        "status": "ACTIVE",
        "type": "SYSTEM_DEFINED",
        "models": [
            {
                "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"
            }
        ],
    },
]


def text_stream(text: str = "Hello world", *, input_tokens: int = 10, output_tokens: int = 5):
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                },
                "metrics": {"latencyMs": 42},
            }
        },
    ]


def tool_stream(
    tool_name: str = "shell", tool_input_json: str = '{"command": "echo bedrock-tool-ok"}'
):
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tu-1", "name": tool_name}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": tool_input_json}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28}}},
    ]


class FakeSts:
    def __init__(self, account: str = TEST_ACCOUNT) -> None:
        self._account = account

    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": self._account, "Arn": TEST_ARN, "UserId": "AIDAEXAMPLE"}


class FakeControl:
    def list_foundation_models(self, **_kwargs: Any) -> dict:
        return {"modelSummaries": list(FOUNDATION)}

    def list_inference_profiles(self, **_kwargs: Any) -> dict:
        return {"inferenceProfileSummaries": list(PROFILES)}


class FakeRuntime:
    """Scripts are consumed one per Converse call, in order."""

    def __init__(self, scripts: list | None = None) -> None:
        self.scripts = list(scripts or [])
        self.calls: list[dict] = []

    def converse_stream(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        script = self.scripts.pop(0) if self.scripts else text_stream()
        return {"stream": iter(script)}

    def converse(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        script = self.scripts.pop(0) if self.scripts else {}
        return script


class FakeSession:
    def __init__(
        self,
        runtime: FakeRuntime | None = None,
        *,
        account: str = TEST_ACCOUNT,
        credential_method: str = "shared-credentials-file",
    ) -> None:
        self.profile_name: str | None = None
        self.region_name: str | None = "us-west-2"
        self._runtime = runtime or FakeRuntime()
        self._clients = {
            "sts": FakeSts(account),
            "bedrock": FakeControl(),
            "bedrock-runtime": self._runtime,
        }
        self._credential_method = credential_method

    def client(self, name: str, region_name: str | None = None) -> Any:
        return self._clients[name]

    def get_credentials(self) -> Any:
        return SimpleNamespace(method=self._credential_method)


def make_provider(tmp_path, runtime: FakeRuntime | None = None, **overrides):
    """A BedrockProvider wired to a FakeSession, rooted in tmp dirs."""
    from kiro_crew.providers.bedrock import BedrockProvider

    session = FakeSession(runtime)
    kwargs: dict[str, Any] = {
        "work_dir": tmp_path / "work",
        "model": "auto",
        "sandbox_mode": "off",
        "session_factory": lambda profile: session,
        "data_home": tmp_path / "data",
        # Hermetic: never resolve the host's real mcp.json in tests.
        "mcp_servers": {},
    }
    kwargs.update(overrides)
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    provider = BedrockProvider(**kwargs)
    return provider, session
