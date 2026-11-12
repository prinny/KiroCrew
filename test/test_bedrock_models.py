"""Catalogue-join tests for the native Bedrock provider.

Discovery exposes no capability fields, so the join's honesty rules are
the contract: profile-carried models are offered through their ACTIVE
profiles, on-demand models directly, provisioned-only models never,
known InvokeModel-only models annotated and unserved, and a model the
capability overlay does not know carries tool_use=False / window 0 —
the contract's "unknown", never a guess.
"""

from __future__ import annotations

from pathlib import Path

from kiro_crew.providers.bedrock_models import (
    VIA_APPLICATION_PROFILE,
    VIA_INFERENCE_PROFILE,
    VIA_ON_DEMAND,
    BedrockModel,
    catalogue_model,
    join_catalogue,
    load_snapshot,
    resolve_model_id,
    save_snapshot,
    snapshot_is_fresh,
)

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
    {
        "modelId": "ai21.j2-ultra-v1",
        "modelName": "Jurassic-2 Ultra",
        "inferenceTypesSupported": ["ON_DEMAND"],
        "responseStreamingSupported": False,
    },
    {
        "modelId": "amazon.titan-text-premier-v1:0",
        "modelName": "Titan Text Premier",
        "inferenceTypesSupported": ["PROVISIONED"],
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
    {
        "inferenceProfileId": "us.meta.llama3-3-70b-instruct-v1:0",
        "inferenceProfileName": "Llama 3.3 (inactive)",
        "status": "INACTIVE",
        "type": "SYSTEM_DEFINED",
        "models": [
            {
                "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-3-70b-instruct-v1:0"
            }
        ],
    },
    {
        "inferenceProfileArn": "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123",
        "inferenceProfileName": "My app profile",
        "status": "ACTIVE",
        "type": "APPLICATION",
        "models": [
            {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-lite-v1:0"}
        ],
    },
]


def _catalogue() -> list[BedrockModel]:
    return join_catalogue(FOUNDATION, PROFILES, region="us-west-2")


def test_profiles_come_first_in_discovery_order():
    catalogue = _catalogue()
    assert catalogue[0].model_id == "us.amazon.nova-pro-v1:0"
    assert catalogue[0].via == VIA_INFERENCE_PROFILE
    assert catalogue[0].base_model_id == "amazon.nova-pro-v1:0"


def test_inactive_profile_is_not_offered():
    ids = [m.model_id for m in _catalogue()]
    assert "us.meta.llama3-3-70b-instruct-v1:0" not in ids


def test_model_with_profile_and_on_demand_offered_both_ways():
    catalogue = _catalogue()
    profile_row = catalogue_model(catalogue, "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    direct_row = catalogue_model(catalogue, "anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert profile_row is not None and profile_row.via == VIA_INFERENCE_PROFILE
    assert direct_row is not None and direct_row.via == VIA_ON_DEMAND


def test_on_demand_only_model_offered_directly():
    row = catalogue_model(_catalogue(), "deepseek.v3.2")
    assert row is not None and row.via == VIA_ON_DEMAND


def test_provisioned_only_model_never_offered():
    assert catalogue_model(_catalogue(), "amazon.titan-text-premier-v1:0") is None


def test_invoke_model_only_row_annotated_and_unserved():
    row = catalogue_model(_catalogue(), "ai21.j2-ultra-v1")
    assert row is not None
    assert row.served is False
    assert "InvokeModel only" in row.picker_row()["description"]


def test_application_profile_offered_by_arn():
    catalogue = _catalogue()
    row = next(m for m in catalogue if m.via == VIA_APPLICATION_PROFILE)
    assert row.model_id.endswith("application-inference-profile/abc123")
    assert row.base_model_id == "amazon.nova-lite-v1:0"


def test_overlay_capability_and_unknown_defaults():
    catalogue = _catalogue()
    nova = catalogue_model(catalogue, "us.amazon.nova-pro-v1:0")
    assert nova is not None and nova.tool_use is True and nova.context_window == 300_000
    deepseek = catalogue_model(catalogue, "deepseek.v3.2")
    assert deepseek is not None and deepseek.tool_use is True
    unknown = BedrockModel(model_id="acme.future-v9", name="Future")
    assert unknown.tool_use is False and unknown.context_window == 0


def test_every_discovered_row_annotated_listed():
    assert all(m.availability == "listed for this account" for m in _catalogue())


def test_resolve_model_id_rules():
    catalogue = _catalogue()
    assert resolve_model_id("auto", catalogue) == "us.amazon.nova-pro-v1:0"
    assert resolve_model_id("", catalogue) == "us.amazon.nova-pro-v1:0"
    assert resolve_model_id(None, catalogue) == "us.amazon.nova-pro-v1:0"
    assert resolve_model_id("deepseek.v3.2", catalogue) == "deepseek.v3.2"
    assert resolve_model_id("auto", []) is None


def test_snapshot_round_trip_and_freshness(tmp_path: Path):
    path = tmp_path / "catalogue.json"
    catalogue = _catalogue()
    save_snapshot(path, region="us-west-2", identity_account="123456789012", models=catalogue)
    loaded = load_snapshot(path)
    assert loaded is not None
    meta, models = loaded
    assert [m.model_id for m in models] == [m.model_id for m in catalogue]
    assert snapshot_is_fresh(meta, region="us-west-2", identity_account="123456789012")
    assert not snapshot_is_fresh(meta, region="eu-west-1", identity_account="123456789012")
    assert not snapshot_is_fresh(meta, region="us-west-2", identity_account="999")
