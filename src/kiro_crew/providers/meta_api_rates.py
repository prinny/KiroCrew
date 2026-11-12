"""The Meta Model API rate card, as dated data, and the cost math for it.

This module is the ONLY place per-token prices for the native Meta provider
live (design: ``docs/system-specs/modules/providers.md``, MetaApiProvider).
Rates are Meta's published prices, verified 2026-11-12; when Meta moves a
price, this table is updated by a change that says so, and ``RATES_AS_OF``
moves with it. Cost for a turn is computed from the API's own usage counts
(input, cached-input and output tokens), never estimated from text length.

Ember's long-context step is whole-request: when a request's input token
count passes the step threshold, the ENTIRE request bills at the higher
input and output rates, so the step is evaluated on the turn's total input
tokens, not per call.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The date the prices below were verified against Meta's published card.
RATES_AS_OF = "2026-11-12"

#: Input-token count above which Ember's long-context rates apply to the
#: whole request.
EMBER_LONG_CONTEXT_THRESHOLD = 200_000


@dataclass(frozen=True)
class RateTier:
    """One model's prices, in USD per million tokens."""

    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


@dataclass(frozen=True)
class ModelRates:
    """A model's standard tier plus, for Ember, its long-context tier."""

    standard: RateTier
    long_context: RateTier | None = None


#: Published rate card (USD per million tokens), verified 2026-11-12.
#: The Contributor tier is priced here because turns on it must still be
#: costed honestly; it is never a default, a fallback, or an "auto" pick
#: (see meta_api_models), because it grants the API operator training
#: rights over the traffic in exchange for the lower price.
RATE_CARD: dict[str, ModelRates] = {
    "muse-spark-1.3": ModelRates(standard=RateTier(1.25, 0.15, 4.25)),
    "muse-spark-1.3-contributor": ModelRates(standard=RateTier(0.10, 0.002, 0.20)),
    "muse-wonder-1.0": ModelRates(standard=RateTier(4.00, 0.20, 20.00)),
    "muse-ember-1.0": ModelRates(
        standard=RateTier(0.10, 0.004, 0.50),
        long_context=RateTier(0.30, 0.012, 1.50),
    ),
}


def tier_for(model_id: str, total_input_tokens: int) -> RateTier | None:
    """The rate tier a request on *model_id* bills at, or None if unrated.

    None means the model is not on the card: the caller records the turn's
    tokens with a zero cost rather than inventing a price.
    """
    rates = RATE_CARD.get(model_id)
    if rates is None:
        return None
    if rates.long_context is not None and total_input_tokens > EMBER_LONG_CONTEXT_THRESHOLD:
        return rates.long_context
    return rates.standard


def turn_cost_usd(
    model_id: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float:
    """USD cost of one turn (or one model call) from its usage counts.

    ``input_tokens`` is the full input count INCLUDING the cached part (the
    Responses API reports cached tokens as a subset of input tokens), so
    the uncached remainder bills at the input rate and the cached part at
    the cached rate. A cached count larger than the input count is clamped
    rather than producing a negative uncached remainder.
    """
    tier = tier_for(model_id, input_tokens)
    if tier is None:
        return 0.0
    cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
    uncached = max(input_tokens, 0) - cached
    return (
        uncached * tier.input_per_m
        + cached * tier.cached_input_per_m
        + max(output_tokens, 0) * tier.output_per_m
    ) / 1_000_000.0
