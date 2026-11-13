"""Amazon Bedrock access for the adapter: catalogue and Converse turns.

Everything AWS lives here, behind one small class, so the ACP server in
``main.py`` never touches boto3 and the tests can drive the whole prompt
loop off a fake event stream. Credentials are boto3's default chain,
resolved inside this process: profiles (``AWS_PROFILE`` or ``--profile``),
``credential_process``, static keys, session tokens, instance roles. Kiro
Crew hands this adapter no credentials at all -- its registration forbids
credential-shaped env bindings in plaintext and the spawn scrub removes the
AWS secret names -- which is the point of the arrangement: the chain is the
operator's own, resolved by the SDK that owns it.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("kirocrew_bedrock_acp.bedrock")

#: Converse tool names must match ``[a-zA-Z0-9_-]+``; MCP tool names are
#: joined to their server with this separator and sanitised to fit.
TOOL_NAME_SEPARATOR = "__"


def converse_tool_name(server: str, tool: str) -> str:
    raw = f"{server}{TOOL_NAME_SEPARATOR}{tool}"
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in raw)


class CatalogueError(RuntimeError):
    """The model catalogue could not be read and no explicit list was given."""


class BedrockDriver:
    """The boto3 surface the adapter uses, constructed lazily.

    Two clients: the control plane (``bedrock``) for the catalogue and the
    runtime (``bedrock-runtime``) for Converse. Both come from one session,
    so one credential chain and one region govern both.
    """

    def __init__(
        self,
        *,
        profile: Optional[str] = None,
        region: Optional[str] = None,
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._profile = profile
        self._region = region
        self._session_factory = session_factory
        self._session: Any = None
        self._control: Any = None
        self._runtime: Any = None

    def _ensure_session(self) -> Any:
        if self._session is None:
            if self._session_factory is not None:
                self._session = self._session_factory(
                    profile_name=self._profile, region_name=self._region
                )
            else:
                import boto3

                self._session = boto3.Session(profile_name=self._profile, region_name=self._region)
        return self._session

    def control(self) -> Any:
        if self._control is None:
            self._control = self._ensure_session().client("bedrock")
        return self._control

    def runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._ensure_session().client("bedrock-runtime")
        return self._runtime

    # ── Catalogue ──

    def catalogue(self) -> list:
        """The selectable models: ``(value, name)`` pairs for the model option.

        Two control-plane reads joined on the model ARN: foundation models
        (bare ids, offered only where the model supports on-demand
        inference -- a provisioned-throughput-only id cannot be invoked
        bare, and offering it would promise a turn that cannot start) and
        inference profiles (profile ids, the cross-region shape). A profile
        whose base model is in the foundation list is named with it; a
        profile whose base model is not listed still appears, under its own
        name -- a listed profile is a fact about the account, and dropping
        it would hide a model the operator can in fact select. Only ACTIVE
        profiles are offered.

        A model being listed is not a promise it is invocable on this
        account (a listed profile can still answer ResourceNotFound); the
        adapter reports the catalogue as listed and lets the Converse error
        speak, rather than inventing an entitlement check of its own.
        """
        control = self.control()
        models: list = []
        names_by_arn: dict = {}
        try:
            paginator = control.get_paginator("list_foundation_models")
            pages = paginator.paginate(byOutputModality="TEXT")
        except Exception:  # noqa: BLE001 - no paginator: one page, same shape
            pages = [control.list_foundation_models(byOutputModality="TEXT")]
        for page in pages:
            for summary in page.get("modelSummaries", []):
                model_id = summary.get("modelId")
                if not model_id:
                    continue
                if "ON_DEMAND" not in (summary.get("inferenceTypesSupported") or []):
                    continue
                name = summary.get("modelName") or model_id
                models.append((model_id, name))
                arn = summary.get("modelArn")
                if arn:
                    names_by_arn[arn] = name
        try:
            paginator = control.get_paginator("list_inference_profiles")
            profile_pages = paginator.paginate()
        except Exception:  # noqa: BLE001
            profile_pages = [control.list_inference_profiles()]
        for page in profile_pages:
            for profile in page.get("inferenceProfileSummaries", []):
                if profile.get("status") != "ACTIVE":
                    continue
                profile_id = profile.get("inferenceProfileId")
                if not profile_id:
                    continue
                profile_name = profile.get("inferenceProfileName") or profile_id
                base_names = [
                    names_by_arn.get(model.get("modelArn", ""), "")
                    for model in profile.get("models", [])
                ]
                base = next((n for n in base_names if n), "")
                label = f"{profile_name} ({base})" if base else profile_name
                models.append((profile_id, label))
        return models

    # ── Converse ──

    def converse_stream(self, **kwargs: Any) -> Iterable[dict]:
        """One ``ConverseStream`` call, as an iterable of stream events."""
        response = self.runtime().converse_stream(**kwargs)
        return response["stream"]


def run_converse_turn(
    driver: BedrockDriver,
    *,
    model_id: str,
    messages: list,
    tools: list,
    emit_text: Callable[[str], None],
    cancelled: Callable[[], bool],
    max_tokens: int = 8192,
) -> tuple:
    """One Converse request/response, accumulated.

    Returns ``(stop_reason, text, tool_uses, usage)`` where ``tool_uses`` is
    a list of ``{toolUseId, name, input}`` in wire order. Text is emitted
    through ``emit_text`` as it streams, and also returned whole so the
    caller can record the assistant message verbatim.
    """
    kwargs: dict = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if tools:
        kwargs["toolConfig"] = {"tools": [{"toolSpec": spec} for spec in tools]}
    text_parts: list = []
    tool_uses: list = []
    current_tool: Optional[dict] = None
    stop_reason = "end_turn"
    usage: dict = {}
    for event in driver.converse_stream(**kwargs):
        if cancelled():
            return "cancelled", "".join(text_parts), tool_uses, usage
        if "contentBlockStart" in event:
            start = event["contentBlockStart"].get("start", {})
            if "toolUse" in start:
                tool_use = start["toolUse"]
                current_tool = {
                    "toolUseId": tool_use.get("toolUseId", ""),
                    "name": tool_use.get("name", ""),
                    "_input_json": "",
                }
        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                text_parts.append(delta["text"])
                emit_text(delta["text"])
            elif "toolUse" in delta and current_tool is not None:
                current_tool["_input_json"] += delta["toolUse"].get("input", "")
        elif "contentBlockStop" in event:
            if current_tool is not None:
                raw = current_tool.pop("_input_json", "")
                try:
                    current_tool["input"] = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    current_tool["input"] = {}
                tool_uses.append(current_tool)
                current_tool = None
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
        elif "metadata" in event:
            raw_usage = event["metadata"].get("usage", {})
            usage = {
                "inputTokens": int(raw_usage.get("inputTokens", 0)),
                "outputTokens": int(raw_usage.get("outputTokens", 0)),
                "totalTokens": int(raw_usage.get("totalTokens", 0)),
            }
    return stop_reason, "".join(text_parts), tool_uses, usage


def assistant_message(text: str, tool_uses: list) -> dict:
    """The Converse assistant message for one turn's output."""
    content: list = []
    if text:
        content.append({"text": text})
    for tool_use in tool_uses:
        content.append(
            {
                "toolUse": {
                    "toolUseId": tool_use["toolUseId"],
                    "name": tool_use["name"],
                    "input": tool_use.get("input", {}),
                }
            }
        )
    return {"role": "assistant", "content": content}


def tool_results_message(results: list) -> dict:
    """The Converse user message carrying one turn's tool results.

    ``results`` is a list of ``(toolUseId, text, is_error)``.
    """
    content = []
    for tool_use_id, text, is_error in results:
        content.append(
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": text}],
                    "status": "error" if is_error else "success",
                }
            }
        )
    return {"role": "user", "content": content}
