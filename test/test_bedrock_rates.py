"""Rate-table tests for the native Bedrock provider.

The table is dated data keyed by family x Region class; these tests pin
the resolution rules (Region prefix stripping, longest family prefix,
US inheritance) and the Bedrock usage accounting (inputTokens EXCLUDE
cache tokens, so reads and writes bill additively at their own rates).
An unrated family costs exactly 0.0 — never an invented price.
"""

from __future__ import annotations

import pytest

from kiro_crew.providers.bedrock_rates import (
    base_model_id,
    call_cost_usd,
    family_for,
    is_rated,
    region_class,
    tier_for,
)


def test_base_model_id_strips_region_prefix():
    assert base_model_id("us.anthropic.claude-sonnet-4-5") == "anthropic.claude-sonnet-4-5"
    assert base_model_id("eu.amazon.nova-pro-v1:0") == "amazon.nova-pro-v1:0"
    assert base_model_id("anthropic.claude-opus-4-5") == "anthropic.claude-opus-4-5"
    assert (
        base_model_id("arn:aws:bedrock:us-west-2:123:inference-profile/us.deepseek.v3.2")
        == "deepseek.v3.2"
    )


def test_region_class():
    assert region_class("us.amazon.nova-pro-v1:0") == "us"
    assert region_class("global.anthropic.claude-sonnet-4-5") == "global"
    assert region_class("deepseek.v3.2") == "us"


def test_family_longest_prefix_wins():
    assert family_for("us.anthropic.claude-opus-4-5-20251101-v1:0") == "anthropic.claude-opus"
    assert family_for("anthropic.claude-sonnet-4-5") == "anthropic.claude-sonnet"
    assert family_for("anthropic.claude-v2:1") == "anthropic.claude"
    assert family_for("amazon.nova-pro-v1:0") == "amazon.nova-pro"
    assert family_for("cohere.command-r-plus-v1:0") == "cohere.command-r-plus"
    assert family_for("writer.palmyra-x5") is None


def test_unrated_family_costs_zero():
    assert not is_rated("writer.palmyra-x5")
    assert tier_for("writer.palmyra-x5") is None
    assert (
        call_cost_usd(
            "writer.palmyra-x5",
            input_tokens=1_000_000,
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=1_000_000,
        )
        == 0.0
    )


def test_cost_math_splits_cache_additively():
    # Nova Pro: 0.80 in / 0.20 cache read / 0.80 cache write / 3.20 out.
    cost = call_cost_usd(
        "us.amazon.nova-pro-v1:0",
        input_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(0.80 + 0.20 + 0.80 + 3.20)


def test_claude_cache_premium_and_discount():
    # Sonnet: 3.00 in / 0.30 read (0.1x) / 3.75 write (1.25x) / 15.00 out.
    tier = tier_for("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert tier is not None
    assert tier.cache_read_per_m == pytest.approx(0.30)
    assert tier.cache_write_per_m == pytest.approx(3.75)


def test_non_us_region_class_inherits_us_card():
    assert tier_for("eu.anthropic.claude-sonnet-4-5") == tier_for("us.anthropic.claude-sonnet-4-5")
    assert tier_for("global.amazon.nova-pro-v1:0") is not None
