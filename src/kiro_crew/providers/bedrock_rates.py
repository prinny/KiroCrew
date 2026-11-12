"""The Bedrock rate table, as dated data, and the cost math for it.

This module is the ONLY place per-token prices for the native Bedrock
provider live (design: ``docs/system-specs/modules/providers.md``,
BedrockProvider). Rates are ESTIMATES keyed by model family and Region
class, taken from AWS's published on-demand list prices; AWS billing is
the ground truth for what a turn actually cost, and a turn on an unrated
family records its tokens with a zero cost and is labelled partial by the
caller rather than priced from an invented number. When AWS moves a
price, this table is updated by a change that says so, and
``RATES_AS_OF`` moves with it.

Bedrock's usage accounting differs from the Responses API's: the
``inputTokens`` a Converse call reports EXCLUDE cache tokens, which are
reported separately as ``cacheReadInputTokens`` and
``cacheWriteInputTokens``. The cost math below follows that accounting:
uncached input, cache reads and cache writes each bill at their own
rate, with no subset subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The date the prices below were verified against AWS's published card.
RATES_AS_OF = "2026-11-12"

#: Region prefixes that can lead a Bedrock model or profile id. The part
#: after the prefix is the base model id the family table keys on.
REGION_PREFIXES = ("us", "eu", "apac", "jp", "au", "global", "gov")


@dataclass(frozen=True)
class RateTier:
    """One family's prices, in USD per million tokens."""

    input_per_m: float
    cache_read_per_m: float
    cache_write_per_m: float
    output_per_m: float


#: Published on-demand list prices (USD per million tokens), verified
#: 2026-11-12 for the US Region class. Claude cache pricing follows the
#: published Bedrock prompt-caching card (reads at 0.1x input, writes at
#: 1.25x input); Nova cache reads are priced at 0.25x input with writes
#: at the input rate; families with no published cache discount carry
#: cache at the input rate rather than an invented discount. The EU,
#: APAC and global Region classes inherit the US card in this version —
#: the class dimension exists so a divergent regional price lands as
#: data, not as a code change.
RATE_CARD: dict[tuple[str, str], RateTier] = {
    ("anthropic.claude-opus", "us"): RateTier(5.00, 0.50, 6.25, 25.00),
    ("anthropic.claude-sonnet", "us"): RateTier(3.00, 0.30, 3.75, 15.00),
    ("anthropic.claude-haiku", "us"): RateTier(1.00, 0.10, 1.25, 5.00),
    ("anthropic.claude", "us"): RateTier(3.00, 0.30, 3.75, 15.00),
    ("amazon.nova-premier", "us"): RateTier(2.50, 0.625, 2.50, 12.50),
    ("amazon.nova-pro", "us"): RateTier(0.80, 0.20, 0.80, 3.20),
    ("amazon.nova-lite", "us"): RateTier(0.06, 0.015, 0.06, 0.24),
    ("amazon.nova-micro", "us"): RateTier(0.035, 0.009, 0.035, 0.14),
    ("amazon.titan", "us"): RateTier(0.80, 0.80, 0.80, 1.60),
    ("meta.llama", "us"): RateTier(0.72, 0.72, 0.72, 0.72),
    ("mistral.mistral-large", "us"): RateTier(2.00, 2.00, 2.00, 6.00),
    ("mistral.mistral-small", "us"): RateTier(0.10, 0.10, 0.10, 0.30),
    ("cohere.command-r-plus", "us"): RateTier(0.30, 0.30, 0.30, 1.25),
    ("cohere.command-r", "us"): RateTier(0.15, 0.15, 0.15, 0.60),
    ("deepseek.r1", "us"): RateTier(1.35, 1.35, 1.35, 5.40),
    ("deepseek", "us"): RateTier(0.30, 0.30, 0.30, 0.45),
    ("ai21.jamba-large", "us"): RateTier(2.00, 2.00, 2.00, 8.00),
    ("ai21.jamba", "us"): RateTier(0.20, 0.20, 0.20, 0.40),
}


def base_model_id(model_id: str) -> str:
    """The id with any leading Region-class prefix stripped.

    ``us.anthropic.claude-sonnet-4-5`` -> ``anthropic.claude-sonnet-4-5``;
    an id with no known prefix is returned unchanged. Application
    inference profile ARNs resolve to their final path segment first.
    """
    candidate = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
    head, dot, rest = candidate.partition(".")
    if dot and head in REGION_PREFIXES:
        return rest
    return candidate


def region_class(model_id: str) -> str:
    """The Region class an id's prefix names (``us`` when unprefixed)."""
    candidate = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
    head, dot, _ = candidate.partition(".")
    if dot and head in REGION_PREFIXES:
        return head
    return "us"


def family_for(model_id: str) -> str | None:
    """The rate-card family a model id belongs to, or None if unrated.

    Longest matching family prefix wins, so ``anthropic.claude-opus``
    outranks the ``anthropic.claude`` fallback row.
    """
    base = base_model_id(model_id)
    best: str | None = None
    for family, _region in RATE_CARD:
        if base == family or base.startswith(family + "-") or base.startswith(family + "."):
            if best is None or len(family) > len(best):
                best = family
    return best


def tier_for(model_id: str) -> RateTier | None:
    """The rate tier a call on *model_id* bills at, or None if unrated.

    None means the family is not on the card: the caller records the
    call's tokens with a zero cost rather than inventing a price. A
    Region class with no card of its own inherits the US card.
    """
    family = family_for(model_id)
    if family is None:
        return None
    tier = RATE_CARD.get((family, region_class(model_id)))
    if tier is None:
        tier = RATE_CARD.get((family, "us"))
    return tier


def is_rated(model_id: str) -> bool:
    return tier_for(model_id) is not None


def call_cost_usd(
    model_id: str,
    *,
    input_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
) -> float:
    """USD cost of one Converse call from its reported usage counts.

    ``input_tokens`` is the uncached input count (Bedrock reports cache
    tokens separately, never as a subset). An unrated family costs 0.0;
    the caller labels the turn partial in that case.
    """
    tier = tier_for(model_id)
    if tier is None:
        return 0.0
    return (
        max(input_tokens, 0) * tier.input_per_m
        + max(cache_read_tokens, 0) * tier.cache_read_per_m
        + max(cache_write_tokens, 0) * tier.cache_write_per_m
        + max(output_tokens, 0) * tier.output_per_m
    ) / 1_000_000.0
