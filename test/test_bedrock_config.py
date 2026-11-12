"""Config-surface tests for the native Bedrock provider.

Covers the pure-config surfaces: the widened ``agent.provider`` enum
(round-trip), the bedrock_* field defaults, the plaintext-refusal
floor for the three credential-reference fields at write time, the
factory dispatch (bedrock builds the native provider; member/worker
sessions are refused; unknown values fail closed), and the
``to_provider_id`` determination — Bedrock ids and application-profile
ARNs pass through unchanged, so no registry change is needed.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from config_loader_helpers import _load_from_dict

from kiro_crew.config.loader import AgentConfig
from kiro_crew.config.loader_document import write_config_atomically
from kiro_crew.model_registry import to_provider_id
from kiro_crew.providers.bedrock import BedrockProvider

FAKE_MATERIAL = "AKIAEXAMPLE00000000-not-a-real-key"


def test_provider_enum_is_three_values_acp_first():
    provider = next(f for f in fields(AgentConfig) if f.name == "provider")
    assert provider.default == "acp"
    assert provider.metadata["enum"] == ["acp", "meta_api", "bedrock"]


def test_bedrock_fields_have_safe_defaults():
    by_name = {f.name: f for f in fields(AgentConfig)}
    assert by_name["bedrock_region"].default == ""
    assert by_name["bedrock_profile"].default == ""
    assert by_name["bedrock_access_key_id_ref"].default == ""
    assert by_name["bedrock_secret_access_key_ref"].default == ""
    assert by_name["bedrock_session_token_ref"].default == ""
    assert by_name["bedrock_guardrail_id"].default == ""
    assert by_name["bedrock_guardrail_version"].default == ""
    assert by_name["bedrock_max_tool_steps"].default == 50
    assert by_name["bedrock_expected_account_id"].default == ""
    assert by_name["bedrock_monthly_budget_usd"].default == 0.0


def test_bedrock_config_round_trips():
    cfg = _load_from_dict(
        {
            "agent": {
                "provider": "bedrock",
                "bedrock_region": "us-west-2",
                "bedrock_profile": "crew-bedrock",
                "bedrock_access_key_id_ref": "secret://AWS_ACCESS_KEY_ID",
                "bedrock_max_tool_steps": 25,
                "bedrock_monthly_budget_usd": 10.0,
            }
        }
    )
    assert cfg.agent.provider == "bedrock"
    assert cfg.agent.bedrock_region == "us-west-2"
    assert cfg.agent.bedrock_profile == "crew-bedrock"
    assert cfg.agent.bedrock_access_key_id_ref == "secret://AWS_ACCESS_KEY_ID"
    assert cfg.agent.bedrock_max_tool_steps == 25
    assert cfg.agent.bedrock_monthly_budget_usd == 10.0


@pytest.mark.parametrize(
    "field_name",
    [
        "bedrock_access_key_id_ref",
        "bedrock_secret_access_key_ref",
        "bedrock_session_token_ref",
    ],
)
def test_plaintext_bedrock_ref_refused_at_write(tmp_path, field_name):
    path = tmp_path / "config.json"
    with pytest.raises(ValueError, match=field_name):
        write_config_atomically(path, {"agent": {field_name: FAKE_MATERIAL}})
    assert not path.exists()


def test_secret_reference_bedrock_refs_write_fine(tmp_path):
    path = tmp_path / "config.json"
    write_config_atomically(
        path,
        {
            "agent": {
                "bedrock_access_key_id_ref": "secret://AWS_ACCESS_KEY_ID",
                "bedrock_secret_access_key_ref": "secret://AWS_SECRET_ACCESS_KEY",
                "bedrock_session_token_ref": "secret://AWS_SESSION_TOKEN",
            }
        },
    )
    assert path.exists()


def test_factory_builds_bedrock_provider(tmp_path):
    cfg = _load_from_dict({"agent": {"provider": "bedrock", "bedrock_region": "us-west-2"}})
    factory = cfg.create_provider_factory()
    provider = factory(session_key="some-session", cwd=str(tmp_path))
    assert isinstance(provider, BedrockProvider)


def test_factory_refuses_member_sessions():
    cfg = _load_from_dict({"agent": {"provider": "bedrock"}})
    factory = cfg.create_provider_factory()
    with pytest.raises(ValueError, match="member or worker"):
        factory(session_key="some-session", member_context=True)


def test_factory_unknown_provider_fails_closed():
    cfg = _load_from_dict({"agent": {"provider": "acp"}})
    cfg.agent.provider = "carrier-pigeon"
    with pytest.raises(ValueError, match="Unknown agent.provider"):
        cfg.create_provider_factory()


def test_bedrock_ids_pass_through_to_provider_id():
    # The determination behind the (non-)change to sections_agent_fields:
    # unrecognized values — a regional inference-profile id, an
    # application-profile ARN — pass through unchanged, so an operator's
    # explicit Bedrock id is never rewritten.
    assert (
        to_provider_id("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock")
        == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123"
    assert to_provider_id(arn, "bedrock") == arn
