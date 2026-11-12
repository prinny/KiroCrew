"""The Meta Model API model catalogue for the native provider.

The picker reads THIS catalogue and nothing else. The agent loop's own
model list is the loop vendor's built-in catalogue (verified against the
pinned app-server: ``model/list`` answers its built-in ids and never the
provider's), so it must never narrow or replace what is offered here.

The catalogue is a static registry (windows, effort levels, display names)
merged over a live ``GET /v1/models`` snapshot when one can be fetched: a
model the API lists but the registry does not know is offered with the
default window; a registry model the API does not list stays offered, so
the picker keeps working when the API is unreachable or a snapshot is
stale. The Contributor model is listed (an operator may choose it
deliberately) but is never the default and is annotated wherever the
catalogue is rendered: its price buys the API operator training rights
over the traffic.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

#: Every Muse model on the published card shares this context window.
MUSE_CONTEXT_WINDOW = 1_048_576

#: Window assumed for an API-listed model the static registry does not know.
DEFAULT_CONTEXT_WINDOW = MUSE_CONTEXT_WINDOW

#: The model "auto" resolves to on the native provider: the flagship
#: Standard tier. Never the Contributor id.
DEFAULT_MODEL_ID = "muse-spark-1.3"

CONTRIBUTOR_MODEL_ID = "muse-spark-1.3-contributor"

_EFFORT_STANDARD = ("low", "medium", "high", "max")
_EFFORT_CONTRIBUTOR = ("low", "medium", "high")


@dataclass(frozen=True)
class MetaModel:
    """One catalogue entry, in the shape the picker consumes."""

    model_id: str
    name: str
    context_window: int = MUSE_CONTEXT_WINDOW
    effort_levels: tuple[str, ...] = _EFFORT_STANDARD
    contributor: bool = False
    description: str = ""

    def picker_row(self) -> dict[str, str]:
        row = {"modelId": self.model_id, "name": self.name}
        if self.description:
            row["description"] = self.description
        return row


#: The static registry. Windows verified against Meta's published model
#: catalogue, 2026-11-12.
STATIC_MODELS: tuple[MetaModel, ...] = (
    MetaModel(
        model_id="muse-spark-1.3",
        name="Muse Spark 1.3",
        description="Meta's flagship agentic model (Standard tier).",
    ),
    MetaModel(
        model_id="muse-spark-1.3-contributor",
        name="Muse Spark 1.3 Contributor",
        effort_levels=_EFFORT_CONTRIBUTOR,
        contributor=True,
        description=(
            "Contributor tier: far cheaper because it grants the API "
            "operator training rights over the traffic. Never a default."
        ),
    ),
    MetaModel(
        model_id="muse-wonder-1.0",
        name="Muse Wonder 1.0",
        description="Meta's highest-capability model.",
    ),
    MetaModel(
        model_id="muse-ember-1.0",
        name="Muse Ember 1.0",
        description="Meta's fast, low-cost model; long-context pricing steps up past 200k input tokens.",
    ),
)

_STATIC_BY_ID = {m.model_id: m for m in STATIC_MODELS}


def static_model(model_id: str) -> MetaModel | None:
    return _STATIC_BY_ID.get(model_id)


def resolve_model_id(requested: str | None) -> str:
    """Resolve a configured model value to the catalogue id a turn runs.

    ``""`` / ``"auto"`` / None resolve to the Standard default — never to
    the Contributor id. A concrete id passes through unchanged (the API,
    not the catalogue, is the authority on whether it can be served; a
    refusal is reported, never silently re-pointed).
    """
    if not requested or requested == "auto":
        return DEFAULT_MODEL_ID
    return requested


def merge_catalogue(api_model_ids: list[str] | None) -> list[MetaModel]:
    """The offered catalogue: static entries, plus API-only ids appended.

    Order is deliberate: the static registry order is the picker's order,
    with the default first. An unreachable API (``None``) yields exactly
    the static registry.
    """
    offered = list(STATIC_MODELS)
    if api_model_ids:
        known = set(_STATIC_BY_ID)
        for model_id in api_model_ids:
            if model_id and model_id not in known:
                offered.append(MetaModel(model_id=model_id, name=model_id))
                known.add(model_id)
    return offered


@dataclass
class ConnectionReport:
    """The outcome of a read-only ``GET /v1/models`` connection check.

    Carries no response body and no credential: success/failure, the model
    count and ids, latency, and on failure the error CLASS only.
    """

    ok: bool
    model_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    error_class: str = ""

    @property
    def model_count(self) -> int:
        return len(self.model_ids)


def fetch_models(base_url: str, api_key: str, *, timeout: float = 10.0) -> ConnectionReport:
    """GET ``<base_url>/models`` and classify the outcome.

    The key travels only in the Authorization header of this one request;
    it is never logged, and no error path here carries the response body
    (an echo of request headers in an error body must not be able to leak
    it). Failure classes: ``unauthorised`` (401/403), ``rate_limited``
    (429), ``unreachable`` (network/timeout), ``bad_response`` (anything
    else, including an unparseable body).
    """
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        latency = int((time.monotonic() - started) * 1000)
        if exc.code in (401, 403):
            return ConnectionReport(False, latency_ms=latency, error_class="unauthorised")
        if exc.code == 429:
            return ConnectionReport(False, latency_ms=latency, error_class="rate_limited")
        return ConnectionReport(False, latency_ms=latency, error_class="bad_response")
    except (urllib.error.URLError, TimeoutError, OSError):
        latency = int((time.monotonic() - started) * 1000)
        return ConnectionReport(False, latency_ms=latency, error_class="unreachable")
    latency = int((time.monotonic() - started) * 1000)
    try:
        payload = json.loads(body)
        ids = [
            str(entry["id"])
            for entry in payload.get("data", [])
            if isinstance(entry, dict) and entry.get("id")
        ]
    except (ValueError, AttributeError):
        return ConnectionReport(False, latency_ms=latency, error_class="bad_response")
    return ConnectionReport(True, model_ids=ids, latency_ms=latency)
