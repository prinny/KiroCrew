"""Unit goldens for the Meta Model API rate card (meta_api_rates).

Every expected value is hand-computed from the published card the module
carries (RATES_AS_OF 2026-11-12): Spark Standard $1.25 in / $0.15 cached /
$4.25 out per MTok; Contributor $0.10 / $0.002 / $0.20; Wonder $4.00 /
$0.20 / $20.00; Ember $0.10 / $0.004 / $0.50 at or below 200k input tokens
and $0.30 / $0.012 / $1.50 above — the WHOLE request steps, not just the
tokens past the threshold.
"""

from __future__ import annotations

import pytest

from kiro_crew.providers.meta_api_rates import turn_cost_usd


def test_spark_standard_mixed_call():
    # 800 uncached in (0.001) + 200 cached (0.00003) + 100 out (0.000425)
    assert turn_cost_usd(
        "muse-spark-1.3", input_tokens=1000, output_tokens=100, cached_input_tokens=200
    ) == pytest.approx(0.001455)


def test_spark_standard_no_cache():
    assert turn_cost_usd(
        "muse-spark-1.3", input_tokens=1_000_000, output_tokens=1_000_000, cached_input_tokens=0
    ) == pytest.approx(1.25 + 4.25)


def test_contributor_rates():
    assert turn_cost_usd(
        "muse-spark-1.3-contributor",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cached_input_tokens=500_000,
    ) == pytest.approx(0.5 * 0.10 + 0.5 * 0.002 + 0.20)


def test_wonder_rates():
    assert turn_cost_usd(
        "muse-wonder-1.0", input_tokens=100_000, output_tokens=10_000, cached_input_tokens=0
    ) == pytest.approx(0.4 + 0.2)


def test_ember_at_threshold_uses_low_tier():
    # Exactly 200k input stays on the low tier.
    assert turn_cost_usd(
        "muse-ember-1.0", input_tokens=200_000, output_tokens=0, cached_input_tokens=0
    ) == pytest.approx(0.02)


def test_ember_above_threshold_steps_the_whole_request():
    # 200,001 input: EVERY token prices at the high tier, output included.
    cost = turn_cost_usd(
        "muse-ember-1.0", input_tokens=200_001, output_tokens=100_000, cached_input_tokens=0
    )
    assert cost == pytest.approx(200_001 * 0.30 / 1e6 + 100_000 * 1.50 / 1e6)


def test_ember_cached_split_both_tiers():
    low = turn_cost_usd(
        "muse-ember-1.0", input_tokens=100_000, output_tokens=0, cached_input_tokens=50_000
    )
    assert low == pytest.approx(50_000 * 0.10 / 1e6 + 50_000 * 0.004 / 1e6)
    high = turn_cost_usd(
        "muse-ember-1.0", input_tokens=300_000, output_tokens=0, cached_input_tokens=50_000
    )
    assert high == pytest.approx(250_000 * 0.30 / 1e6 + 50_000 * 0.012 / 1e6)


def test_cached_tokens_clamped_to_input():
    # A malformed frame claiming more cached than input must not bill
    # negative uncached tokens.
    cost = turn_cost_usd(
        "muse-spark-1.3", input_tokens=100, output_tokens=0, cached_input_tokens=500
    )
    assert cost == pytest.approx(100 * 0.15 / 1e6)


def test_unrated_model_costs_zero_never_an_invented_price():
    assert (
        turn_cost_usd(
            "some-future-model", input_tokens=10**6, cached_input_tokens=0, output_tokens=10**6
        )
        == 0.0
    )
    assert turn_cost_usd("", input_tokens=10**6, cached_input_tokens=0, output_tokens=10**6) == 0.0
