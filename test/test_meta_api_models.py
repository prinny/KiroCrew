"""Unit tests for the Meta Model API catalogue (meta_api_models).

The catalogue is static-first (design section 2.2, probe P-6): the pinned
app-server's own ``model/list`` never carries Muse ids, so Crew's
catalogue is the only one the picker reads; a ``GET /v1/models`` refresh
merges API-only ids AFTER the static entries and an unreachable API
changes nothing.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kiro_crew.providers.meta_api_models import (
    STATIC_MODELS,
    fetch_models,
    merge_catalogue,
    resolve_model_id,
    static_model,
)

STATIC_IDS = [m.model_id for m in STATIC_MODELS]


def test_resolve_auto_is_the_standard_default_never_contributor():
    assert resolve_model_id("auto") == "muse-spark-1.3"
    assert resolve_model_id("") == "muse-spark-1.3"
    assert resolve_model_id(None) == "muse-spark-1.3"
    assert resolve_model_id("muse-wonder-1.0") == "muse-wonder-1.0"
    assert resolve_model_id("unknown-id") == "unknown-id"


def test_static_catalogue_shape():
    assert STATIC_IDS[0] == "muse-spark-1.3"
    assert set(STATIC_IDS) == {
        "muse-spark-1.3",
        "muse-spark-1.3-contributor",
        "muse-wonder-1.0",
        "muse-ember-1.0",
    }
    for entry in STATIC_MODELS:
        assert entry.context_window == 1_048_576


def test_contributor_is_annotated_and_lacks_max_effort():
    contributor = static_model("muse-spark-1.3-contributor")
    assert contributor is not None and contributor.contributor
    assert "training rights" in contributor.description
    assert "max" not in contributor.effort_levels
    standard = static_model("muse-spark-1.3")
    assert standard is not None and not standard.contributor
    assert "max" in standard.effort_levels


def test_merge_keeps_static_first_and_appends_api_only_ids():
    merged = merge_catalogue(["muse-wonder-1.0", "api-only-model", "muse-spark-1.3"])
    ids = [m.model_id for m in merged]
    assert ids[:4] == STATIC_IDS
    assert ids[-1] == "api-only-model"
    assert len(ids) == 5


def test_merge_with_no_refresh_is_the_static_catalogue():
    assert [m.model_id for m in merge_catalogue(None)] == STATIC_IDS
    assert [m.model_id for m in merge_catalogue([])] == STATIC_IDS


class _ModelsHandler(BaseHTTPRequestHandler):
    status = 200
    body: dict = {"data": [{"id": "muse-spark-1.3"}, {"id": "extra-model"}]}
    seen_auth: str | None = None

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        type(self).seen_auth = self.headers.get("Authorization")
        payload = json.dumps(type(self).body).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):  # silence
        pass


@pytest.fixture()
def models_server():
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_fetch_models_success(models_server):
    _ModelsHandler.status = 200
    _ModelsHandler.body = {"data": [{"id": "muse-spark-1.3"}, {"id": "extra-model"}]}
    report = fetch_models(models_server, "test-key-value")
    assert report.ok
    assert report.model_ids == ["muse-spark-1.3", "extra-model"]
    assert report.error_class == ""
    # The key travels ONLY in the Authorization header.
    assert _ModelsHandler.seen_auth == "Bearer test-key-value"


def test_fetch_models_error_classes(models_server):
    for status, expected in [
        (401, "unauthorised"),
        (403, "unauthorised"),
        (429, "rate_limited"),
        (500, "bad_response"),
    ]:
        _ModelsHandler.status = status
        report = fetch_models(models_server, "test-key-value")
        assert not report.ok
        assert report.error_class == expected
        assert report.model_ids == []


def _fetch_with_timeout(base_url, key, timeout):
    return fetch_models(base_url, key, timeout=timeout)


def test_fetch_models_unreachable():
    report = _fetch_with_timeout("http://127.0.0.1:1/v1", "test-key-value", 1.0)
    assert not report.ok
    assert report.error_class == "unreachable"
