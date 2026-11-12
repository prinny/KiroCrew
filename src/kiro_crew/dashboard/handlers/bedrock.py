"""Test-connection handler for the native Amazon Bedrock provider.

``POST /api/bedrock/test`` runs the same check the provider runs at
session start — resolve credentials through the AWS SDK chain, verify
the identity over STS, count the discoverable models — off the event
loop, and answers with the outcome only: success/failure, the resolved
identity (account id, principal ARN, credential source CLASS), the
Region, the model count, latency, and on failure the error CLASS.
Credential material never crosses this boundary: the announcement is
who the gateway acts as, never what it acts with.
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from kiro_crew.config.loader import KiroCrewConfig

logger = logging.getLogger(__name__)


async def api_bedrock_test(request: web.Request) -> web.Response:
    """POST /api/bedrock/test — probe the configured Bedrock access."""
    cfg = await asyncio.to_thread(KiroCrewConfig.load)

    def _probe() -> dict:
        from kiro_crew.providers.bedrock import test_connection
        from kiro_crew.providers.bedrock_client import (
            ENV_ACCESS_KEY_ID,
            ENV_SECRET_ACCESS_KEY,
            ENV_SESSION_TOKEN,
        )

        report = test_connection(
            region=cfg.agent.bedrock_region,
            profile=cfg.agent.bedrock_profile or None,
            credential_refs={
                ENV_ACCESS_KEY_ID: cfg.agent.bedrock_access_key_id_ref,
                ENV_SECRET_ACCESS_KEY: cfg.agent.bedrock_secret_access_key_ref,
                ENV_SESSION_TOKEN: cfg.agent.bedrock_session_token_ref,
            },
        )
        payload: dict = {
            "ok": report.ok,
            "latency_ms": round(report.latency_ms),
        }
        if report.ok:
            payload["identity_account"] = report.identity_account
            payload["identity_arn"] = report.identity_arn
            payload["identity_source"] = report.identity_source
            payload["region"] = report.region
            payload["model_count"] = report.model_count
        else:
            payload["error_class"] = report.error_class or "unreachable"
        return payload

    return web.json_response(await asyncio.to_thread(_probe))
