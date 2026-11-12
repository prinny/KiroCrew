"""Test-connection handler for the native Meta Model API provider.

``POST /api/meta-api/test`` runs the same check the provider runs at
session start — resolve the configured credential, then ``GET
/v1/models`` — off the event loop, and answers with the outcome only:
success/failure, the model count and ids, latency, and on failure the
error CLASS. The credential and the upstream response body never cross
this boundary, and nothing about the vault or the key's value is
reported — only that a credential resolved (and from which source).
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from kiro_crew.config.loader import KiroCrewConfig

logger = logging.getLogger(__name__)


async def api_meta_api_test(request: web.Request) -> web.Response:
    """POST /api/meta-api/test — probe the configured Meta Model API."""
    cfg = await asyncio.to_thread(KiroCrewConfig.load)

    def _probe() -> dict:
        from kiro_crew.providers.meta_api import (
            DEFAULT_BASE_URL,
            resolve_meta_api_key,
            test_connection,
        )

        key, source = resolve_meta_api_key(cfg.agent.meta_api_secret)
        if not key:
            return {"ok": False, "error_class": "no_credential"}
        report = test_connection(cfg.agent.meta_api_base_url or DEFAULT_BASE_URL, key)
        payload: dict = {
            "ok": report.ok,
            "credential_source": source,
            "latency_ms": round(report.latency_ms),
        }
        if report.ok:
            payload["model_count"] = len(report.model_ids)
            payload["model_ids"] = list(report.model_ids)
        else:
            payload["error_class"] = report.error_class or "unreachable"
        return payload

    return web.json_response(await asyncio.to_thread(_probe))
