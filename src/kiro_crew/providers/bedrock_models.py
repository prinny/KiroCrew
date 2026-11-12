"""The Bedrock model catalogue for the native provider: live discovery.

The picker reads THIS catalogue and nothing else. Bedrock discovery
exposes no capability fields — nothing in ``ListFoundationModels`` or
``ListInferenceProfiles`` says whether a model takes Converse tool use
or how large its context window is — so capability comes from a small
maintained overlay keyed by base model id, and a model the overlay does
not know is offered with ``tool_use=False`` and context window 0, which
the provider contract already reads as "unknown": never a guessed
window, never a silent zero-tool session passed off as capable.

The catalogue is a JOIN, not a list: foundation models that carry an
ACTIVE inference profile are offered through their profiles (the
profile id is the callable id); models that support on-demand
invocation are offered directly; provisioned-throughput-only models are
discovered by the join's inputs but never offered, because there is no
callable id for them without a provisioned ARN an operator supplies.
Known InvokeModel-only text models stay in the catalogue annotated as
such — visible, honest, and never served by this provider in v1.

Discovery is account-scoped: every row it returns is a row the
resolving account can see, so each discovered row carries the
``listed for this account`` availability annotation. Annotation is
never a filter.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Availability annotation every discovered row carries: discovery is
#: account-scoped, so a listed row is a row this account can see. It is
#: an annotation, never a filter (design §3.2).
AVAILABILITY_LISTED = "listed for this account"

#: How a catalogue row is reached when a turn runs.
VIA_INFERENCE_PROFILE = "inference_profile"
VIA_ON_DEMAND = "on_demand"
VIA_APPLICATION_PROFILE = "application_profile"


@dataclass(frozen=True)
class BedrockModel:
    """One catalogue entry, in the shape the picker consumes."""

    model_id: str
    name: str
    base_model_id: str = ""
    via: str = VIA_ON_DEMAND
    tool_use: bool = False
    context_window: int = 0
    max_output_tokens: int = 0
    streaming: bool = True
    served: bool = True
    availability: str = ""
    region: str = ""

    def picker_row(self) -> dict[str, str]:
        row = {"modelId": self.model_id, "name": self.name}
        notes: list[str] = []
        if not self.served:
            notes.append("InvokeModel only — not served by the native provider")
        elif self.via == VIA_INFERENCE_PROFILE:
            notes.append("inference profile")
        elif self.via == VIA_APPLICATION_PROFILE:
            notes.append("application inference profile")
        if self.availability:
            notes.append(self.availability)
        if notes:
            row["description"] = "; ".join(notes)
        return row


@dataclass(frozen=True)
class OverlayEntry:
    """Maintained capability facts for one base-model family prefix."""

    tool_use: bool
    context_window: int
    converse: bool = True
    #: The family's output-token ceiling — a separate, smaller number
    #: than the context window, and the value inferenceConfig.maxTokens
    #: is capped by (Converse refuses a request above it). 0 = unknown.
    max_output_tokens: int = 0


#: The capability overlay, keyed by base-model-id prefix, maintained
#: from vendor documentation (design §3.4). Longest matching prefix
#: wins. A base id with no entry is UNKNOWN capability: tool_use False,
#: context window 0 — the contract's "unknown" values, never a guess.
CAPABILITY_OVERLAY: dict[str, OverlayEntry] = {
    "anthropic.claude": OverlayEntry(True, 200_000, max_output_tokens=8192),
    "amazon.nova-premier": OverlayEntry(True, 1_000_000, max_output_tokens=8192),
    "amazon.nova": OverlayEntry(True, 300_000, max_output_tokens=5000),
    "meta.llama4": OverlayEntry(True, 1_000_000, max_output_tokens=4096),
    "meta.llama": OverlayEntry(True, 128_000, max_output_tokens=2048),
    "mistral.mistral-large": OverlayEntry(True, 128_000, max_output_tokens=8192),
    "mistral.mistral-small": OverlayEntry(True, 128_000, max_output_tokens=8192),
    "cohere.command-r": OverlayEntry(True, 128_000, max_output_tokens=4096),
    "ai21.jamba": OverlayEntry(True, 256_000, max_output_tokens=4096),
    "deepseek.v3": OverlayEntry(True, 128_000, max_output_tokens=8192),
    "deepseek": OverlayEntry(False, 128_000, max_output_tokens=8192),
    # Known InvokeModel-only text models: listed, annotated, unserved.
    "ai21.j2": OverlayEntry(False, 0, converse=False),
    "anthropic.claude-v2": OverlayEntry(False, 0, converse=False),
    "anthropic.claude-instant": OverlayEntry(False, 0, converse=False),
}


def overlay_for(base_model_id: str) -> OverlayEntry | None:
    """The overlay entry for a base model id (longest prefix wins)."""
    best: OverlayEntry | None = None
    best_len = -1
    for prefix, entry in CAPABILITY_OVERLAY.items():
        if (
            base_model_id == prefix
            or base_model_id.startswith(prefix + "-")
            or (base_model_id.startswith(prefix + "."))
        ):
            if len(prefix) > best_len:
                best = entry
                best_len = len(prefix)
    return best


def _model_entry(
    *,
    model_id: str,
    name: str,
    base_model_id: str,
    via: str,
    region: str,
    streaming: bool = True,
) -> BedrockModel:
    overlay = overlay_for(base_model_id)
    return BedrockModel(
        model_id=model_id,
        name=name or model_id,
        base_model_id=base_model_id,
        via=via,
        tool_use=overlay.tool_use if overlay else False,
        context_window=overlay.context_window if overlay else 0,
        max_output_tokens=overlay.max_output_tokens if overlay else 0,
        streaming=streaming,
        served=overlay.converse if overlay else True,
        availability=AVAILABILITY_LISTED,
        region=region,
    )


def join_catalogue(
    foundation_models: list[dict],
    inference_profiles: list[dict],
    *,
    region: str,
) -> list[BedrockModel]:
    """Join discovery output into the offered catalogue.

    Inputs are the raw summary dicts of ``ListFoundationModels`` (already
    filtered to text-output models by the caller) and
    ``ListInferenceProfiles``. Order is deliberate and is the picker's
    order: inference profiles first (in discovery order), then on-demand
    models (in discovery order), then InvokeModel-only rows, annotated
    and unserved. A model with an ACTIVE profile is offered through the
    profile AND directly when it also supports on-demand invocation —
    both ids are genuinely callable, and the picker labels which is
    which. Provisioned-throughput-only models produce no row.
    """
    offered: list[BedrockModel] = []
    profiled_bases: set[str] = set()
    for profile in inference_profiles:
        if str(profile.get("status") or "ACTIVE") != "ACTIVE":
            continue
        models = profile.get("models") or []
        base = ""
        if models and isinstance(models[0], dict):
            base = str(models[0].get("modelArn") or "").rsplit("/", 1)[-1]
        if str(profile.get("type") or "SYSTEM_DEFINED") == "APPLICATION":
            offered.append(
                _model_entry(
                    model_id=str(profile.get("inferenceProfileArn") or ""),
                    name=str(profile.get("inferenceProfileName") or ""),
                    base_model_id=base,
                    via=VIA_APPLICATION_PROFILE,
                    region=region,
                )
            )
            continue
        profile_id = str(profile.get("inferenceProfileId") or "")
        if not profile_id:
            continue
        profiled_bases.add(base)
        offered.append(
            _model_entry(
                model_id=profile_id,
                name=str(profile.get("inferenceProfileName") or profile_id),
                base_model_id=base,
                via=VIA_INFERENCE_PROFILE,
                region=region,
            )
        )
    on_demand: list[BedrockModel] = []
    invoke_only: list[BedrockModel] = []
    for summary in foundation_models:
        model_id = str(summary.get("modelId") or "")
        if not model_id:
            continue
        inference_types = summary.get("inferenceTypesSupported") or []
        entry = _model_entry(
            model_id=model_id,
            name=str(summary.get("modelName") or model_id),
            base_model_id=model_id,
            via=VIA_ON_DEMAND,
            region=region,
            streaming=bool(summary.get("responseStreamingSupported", True)),
        )
        if not entry.served:
            invoke_only.append(entry)
        elif "ON_DEMAND" in inference_types:
            on_demand.append(entry)
        elif not inference_types and model_id not in profiled_bases:
            # A summary with no declared inference types and no profile
            # cannot be called; it is discovered but never offered.
            continue
    return offered + on_demand + invoke_only


def resolve_model_id(requested: str | None, catalogue: list[BedrockModel]) -> str | None:
    """Resolve a configured model value to the catalogue id a turn runs.

    ``""`` / ``"auto"`` / None resolve to the first SERVED row in
    discovery order; an empty or all-unserved catalogue resolves to
    None and the provider fails closed. A concrete id passes through
    unchanged (AWS, not the catalogue, is the authority on whether it
    can be served; a refusal is reported, never silently re-pointed).
    """
    if requested and requested != "auto":
        return requested
    for model in catalogue:
        if model.served:
            return model.model_id
    return None


def catalogue_model(catalogue: list[BedrockModel], model_id: str) -> BedrockModel | None:
    for model in catalogue:
        if model.model_id == model_id:
            return model
    return None


@dataclass
class BedrockConnectionReport:
    """The outcome of a read-only Bedrock connection check.

    Carries no credential material: success/failure, the error CLASS on
    failure, the resolved identity (account id and principal ARN — the
    STS announcement, not a secret), the Region, the discovered model
    count, and latency.
    """

    ok: bool
    model_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    error_class: str = ""
    identity_account: str = ""
    identity_arn: str = ""
    identity_source: str = ""
    region: str = ""

    @property
    def model_count(self) -> int:
        return len(self.model_ids)


# -- catalogue snapshot -----------------------------------------------------

#: Snapshot freshness window: a snapshot inside it is reused at start();
#: past it, start() re-discovers (design §3.5).
SNAPSHOT_TTL_SECONDS = 24 * 3600


def save_snapshot(
    path: Path,
    *,
    region: str,
    identity_account: str,
    models: list[BedrockModel],
) -> None:
    payload = {
        "region": region,
        "identity_account": identity_account,
        "fetched_at": time.time(),
        "models": [asdict(model) for model in models],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def load_snapshot(path: Path) -> tuple[dict, list[BedrockModel]] | None:
    """(metadata, models) from a snapshot file, or None if unusable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = [BedrockModel(**row) for row in payload.get("models", [])]
    except (OSError, ValueError, TypeError):
        return None
    meta = {
        "region": str(payload.get("region") or ""),
        "identity_account": str(payload.get("identity_account") or ""),
        "fetched_at": float(payload.get("fetched_at") or 0.0),
    }
    return meta, models


def snapshot_is_fresh(meta: dict, *, region: str, identity_account: str) -> bool:
    if meta.get("region") != region or meta.get("identity_account") != identity_account:
        return False
    return (time.time() - float(meta.get("fetched_at") or 0.0)) < SNAPSHOT_TTL_SECONDS
