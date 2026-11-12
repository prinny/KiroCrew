"""Native Bedrock provider: Converse/ConverseStream via boto3, in-process.

Crew's third provider value (``agent.provider = "bedrock"``), beside
``acp`` and ``meta_api``. There is no harness and no child process: the
gateway itself holds a boto3 session, resolves credentials through the
AWS SDK's own default chain (design §2 — the full chain, gateway-side,
never a scrubbed child env), announces the resolved identity over STS
at ``start()``, discovers the account's catalogue live, and runs the
turn loop in-process: Converse calls, tool approval through Crew's
pending-approval registry, tool execution through
``bedrock_tools.ToolRouter`` behind that registry, and the full
Converse transcript journaled per session (Converse is stateless; the
journal IS the session).

Declarations (design §5.4): no member/worker machinery (the factory
refuses it structurally), no steer channel (Converse has no mid-call
input), effort not served, compaction is Crew-owned summarisation.
Bedrock's refusal signal is typed — ``content_filtered`` and
``guardrail_intervened`` surface as refusals with their trace — which
is strictly more than a streamed provider can say.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_PERMISSION_REQUEST,
    EVENT_TOOL_RESULT,
    STOP_REASON_CANCELLED,
    AcpEvent,
    TurnUsage,
)
from kiro_crew.providers.base import LLMProvider
from kiro_crew.providers.bedrock_client import (
    AUTH_ERROR_CODES,
    ENV_ACCESS_KEY_ID,
    ENV_SECRET_ACCESS_KEY,
    ENV_SESSION_TOKEN,
    SOURCE_VAULT_TRIPLE,
    BedrockCallError,
    BedrockUnavailableError,
    classify_client_error,
    converse_with_retry,
    make_boto_session,
    resolve_secret_refs,
    verify_identity,
)
from kiro_crew.providers.bedrock_events import ConverseStreamTranslator
from kiro_crew.providers.bedrock_models import (
    BedrockConnectionReport,
    BedrockModel,
    catalogue_model,
    join_catalogue,
    load_snapshot,
    resolve_model_id,
    save_snapshot,
    snapshot_is_fresh,
)
from kiro_crew.providers.bedrock_tools import ToolRouter
from kiro_crew.providers.meta_api_client import AppServerError
from kiro_crew.trust_patterns import GrantEnded

logger = logging.getLogger(__name__)

#: This provider's id in the session layer's provider vocabulary.
PROVIDER_ID = "bedrock"

#: Default per-turn cap on Converse calls that request tools (design
#: §0.1): a runaway tool loop ends the turn with an error completion
#: instead of spending without bound.
DEFAULT_MAX_TOOL_STEPS = 50

#: Output-token ceiling when the catalogue has no window for the model.
DEFAULT_MAX_TOKENS = 4096

#: Approval option ids, shared with the other providers' cards.
DECISION_ACCEPT = "accept"
DECISION_ACCEPT_FOR_SESSION = "acceptForSession"
DECISION_DECLINE = "decline"
DECISION_CANCEL = "cancel"

_CREDENTIAL_ENV_NAMES = (ENV_ACCESS_KEY_ID, ENV_SECRET_ACCESS_KEY, ENV_SESSION_TOKEN)


@dataclass
class BedrockTurnStats:
    """Per-turn billing object for the native Bedrock provider.

    A FRESH instance is installed at the start of every turn (the ABC
    identity contract): the usage object the turn's first event carries
    is the object ``billing_stats()`` returns afterwards. ``cost_usd``
    on the usage is an ESTIMATE from the dated rate table (design Q2);
    AWS billing is the ground truth. ``cost_partial`` is set when any
    call in the turn ran on an unrated family — tokens recorded, cost
    knowingly incomplete.
    """

    usage: TurnUsage = field(default_factory=TurnUsage)
    cost_partial: bool = False
    cost_is_estimate: bool = True


@dataclass
class _McpReport:
    """The provider's MCP session report: declared servers, named."""

    declared: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()

    def declared_not_ready(self) -> tuple[str, ...]:
        return self.failed


class BedrockProvider(LLMProvider):
    """One native Bedrock session: gateway-side boto3, in-process loop."""

    #: A pending approval cannot be withdrawn once answered — Converse
    #: has no withdraw channel; cancel-at-pause denies all pending.
    approval_withdrawable = False

    # -- declarations (design §5.4: every property a deliberate answer) ----

    @property
    def context_provider_type(self) -> str:
        return PROVIDER_ID

    @property
    def context_incarnation(self) -> object:
        return (id(self), self.session_id)

    @property
    def compaction_self_managed(self) -> bool:
        # compact() applies the session layer's summary to the
        # journal-backed transcript in-process; there is no harness
        # compaction to delegate to.
        return True

    @property
    def harness_managed_compaction_backend(self) -> str | None:
        return None

    @property
    def manual_compact_unsupported_backend(self) -> str | None:
        # Manual compact stays marked unsupported for this backend
        # until the explicit-compact parity test passes (Item 2's
        # lift precedent).
        return PROVIDER_ID

    @property
    def supports_steer(self) -> bool:
        # Structural: Converse has no mid-call input channel.
        return False

    @property
    def supports_refusal_steer(self) -> bool:
        return False

    @property
    def last_steer_monotonic(self) -> float:
        # The H15 pair for the supports_steer override above: this
        # provider never steers, so the stamp is permanently 0.0.
        return 0.0

    def background_launch(self) -> tuple[float, str] | None:
        # No process is launched; there is nothing to report.
        return None

    @property
    def reports_turn_usage(self) -> bool:
        return True

    @property
    def mcp_config_hot_reload(self) -> bool:
        return False

    @property
    def mcp_config_wire_reinject(self) -> bool:
        return False

    @property
    def mcp_config_relaunch_reload(self) -> bool:
        return False

    # -- session identity -----------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def served_model(self) -> str:
        return self._served_model

    @property
    def model_pin_refused(self) -> str:
        return self._model_pin_refused

    @property
    def cwd(self) -> str:
        return str(self._work_dir)

    def __init__(
        self,
        work_dir: Path,
        model: str,
        agent: Any = None,
        sandbox_mode: str = "auto",
        session_key: str | None = None,
        channel_id: str | None = None,
        extra_env: dict[str, str] | None = None,
        bedrock_region: str = "",
        bedrock_profile: str | None = None,
        credential_refs: dict[str, str] | None = None,
        guardrail_id: str = "",
        guardrail_version: str = "",
        max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS,
        expected_account_id: str = "",
        monthly_budget_usd: float = 0.0,
        mcp_servers: dict[str, dict] | None = None,
        data_home: Path | None = None,
        session_factory: Any = None,
    ) -> None:
        self._work_dir = Path(work_dir)
        self._requested_model = model
        self._agent = agent
        self._sandbox_mode = sandbox_mode
        self._session_key = session_key
        self._channel_id = channel_id
        self._extra_env = dict(extra_env or {})
        self._region_configured = bedrock_region or ""
        self._profile = bedrock_profile or self._extra_env.get("AWS_PROFILE") or None
        self._credential_refs = dict(credential_refs or {})
        self._guardrail_id = guardrail_id or ""
        self._guardrail_version = guardrail_version or ""
        self._max_tool_steps = max(1, int(max_tool_steps or DEFAULT_MAX_TOOL_STEPS))
        self._expected_account_id = expected_account_id or ""
        self._monthly_budget_usd = float(monthly_budget_usd or 0.0)
        self._mcp_servers_arg = mcp_servers
        self._data_home = Path(data_home) if data_home is not None else Path.home()
        self._session_factory = session_factory or make_boto_session

        #: Crew-minted session id (Converse is stateless; the id is
        #: ours, carried in requestMetadata and the journal header).
        self._session_id = session_key or f"bedrock-{uuid.uuid4().hex[:16]}"
        self._model_pin_refused = ""
        self._session: Any = None
        self._runtime: Any = None
        self._control: Any = None
        self._region = ""
        self._identity_account = ""
        self._identity_arn = ""
        self._identity_source = ""
        self._served_model = ""
        self._catalogue: list[BedrockModel] = []
        self._router: ToolRouter | None = None
        self._mcp_servers: dict[str, dict] = {}
        self._mcp_report = _McpReport()
        self._messages: list[dict[str, Any]] = []
        self._stats = BedrockTurnStats()
        self._usage_seen = False
        self._context_used = 0
        self._context_window = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._approval_counter = 0
        self._turn_open = False
        self._last_stop_reason = ""
        self._cancel_requested = False
        self._compaction_done: asyncio.Event | None = None
        self.startup_notice: tuple[str, str] | None = None

    # -- identity ---------------------------------------------------------

    @property
    def is_bedrock_backend(self) -> bool:
        return True

    # -- credential surfaces ------------------------------------------------

    def _apply_vault_triple(self) -> None:
        """Write configured secret:// credential refs into the gateway
        environment, where the SDK chain's environment member reads
        them (design §2.3).

        Precedence is by construction: the triple is applied ONLY when
        no Crew surface selects a profile for this session, because a
        per-session profile binding outranks the vault triple. A ref
        that is not a secret:// reference is never used as a credential
        (config load refuses it; a hand-edited one fails closed here).
        """
        if self._profile:
            return
        refs = {
            name: self._credential_refs.get(name, "")
            for name in _CREDENTIAL_ENV_NAMES
            if self._credential_refs.get(name, "")
        }
        if not refs:
            return
        resolved = resolve_secret_refs(refs, self._data_home)
        applied = False
        for name, value in resolved.items():
            if value:
                os.environ[name] = value
                applied = True
        if applied:
            self._identity_source = SOURCE_VAULT_TRIPLE

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        if self._session is not None and self.is_alive():
            return
        # Bearer-token exclusion, enforced: the bedrock-runtime service
        # model declares bearer auth, and botocore would prefer a
        # gateway-env AWS_BEARER_TOKEN_BEDROCK* token over the chain —
        # silently making the billed identity a different one from the
        # STS-verified identity. The token is not a v1 credential
        # source for this provider, so its presence refuses the start.
        bearer = next(
            (name for name in os.environ if name.startswith("AWS_BEARER_TOKEN_BEDROCK")),
            None,
        )
        if bearer is not None:
            self.startup_notice = (
                "bedrock_bearer_token_present",
                f"{bearer} is set in the gateway environment. The native "
                "Bedrock provider does not accept the Bedrock bearer token "
                "as a credential source in v1 (it would override the "
                "STS-verified chain identity); unset it to start a "
                "session. No session was started.",
            )
            raise AppServerError(self.startup_notice[1])
        try:
            self._apply_vault_triple()
            session = self._session_factory(self._profile)
        except BedrockUnavailableError as exc:
            self.startup_notice = ("bedrock_boto3_missing", str(exc))
            raise AppServerError(str(exc)) from exc
        try:
            identity = await asyncio.to_thread(verify_identity, session)
        except BedrockCallError as exc:
            self.startup_notice = (
                "bedrock_identity_failed",
                f"Bedrock identity verification failed ({exc.code}): "
                f"{exc.message}. No session was started.",
            )
            raise AppServerError(self.startup_notice[1]) from exc
        self._identity_account = identity.account_id
        self._identity_arn = identity.arn
        if not self._identity_source:
            self._identity_source = identity.source_class
        logger.info(
            "bedrock: acting as %s (account %s) via %s",
            self._identity_arn,
            self._identity_account,
            self._identity_source,
        )
        if self._expected_account_id and self._expected_account_id != self._identity_account:
            self.startup_notice = (
                "bedrock_account_mismatch",
                f"Bedrock resolved account {self._identity_account}, but "
                f"agent.bedrock_expected_account_id pins "
                f"{self._expected_account_id}; no session was started.",
            )
            raise AppServerError(self.startup_notice[1])
        region = self._region_configured or getattr(session, "region_name", None) or ""
        if not region:
            self.startup_notice = (
                "bedrock_region_missing",
                "No Bedrock Region is configured: set agent.bedrock_region "
                "or a Region on the selected AWS profile. No session was "
                "started.",
            )
            raise AppServerError(self.startup_notice[1])
        self._region = str(region)
        if self._monthly_budget_usd > 0:
            spent = self._month_to_date_usd()
            if spent >= self._monthly_budget_usd:
                self.startup_notice = (
                    "bedrock_budget_reached",
                    f"The native Bedrock provider's monthly budget is "
                    f"reached (${spent:.2f} of ${self._monthly_budget_usd:.2f} "
                    "this month); no session was started.",
                )
                raise AppServerError(self.startup_notice[1])
        self._session = session
        self._runtime = session.client("bedrock-runtime", region_name=self._region)
        self._control = session.client("bedrock", region_name=self._region)
        try:
            await self._refresh_catalogue()
        except Exception as exc:  # noqa: BLE001 - classified, fail closed
            classified = classify_client_error(exc)
            self.startup_notice = (
                "bedrock_discovery_failed",
                f"Bedrock model discovery failed ({classified.code}): "
                f"{classified.message}. No session was started.",
            )
            raise AppServerError(self.startup_notice[1]) from exc
        served = resolve_model_id(self._requested_model, self._catalogue)
        if served is None:
            self.startup_notice = (
                "bedrock_no_model",
                "Bedrock discovery returned no servable model for this "
                "account; no session was started.",
            )
            raise AppServerError(self.startup_notice[1])
        pinned = catalogue_model(self._catalogue, served)
        if (
            self._requested_model
            and self._requested_model != "auto"
            and pinned is not None
            and not pinned.served
        ):
            self._model_pin_refused = (
                f"Model {served} is InvokeModel-only and is not served by "
                "the native Bedrock provider in v1"
            )
            self.startup_notice = (
                "bedrock_model_not_served",
                self._model_pin_refused + "; no session was started.",
            )
            raise AppServerError(self.startup_notice[1])
        self._served_model = served
        if pinned is not None:
            self._context_window = pinned.context_window
        self._mcp_servers = self._resolve_mcp_servers()
        self._router = ToolRouter(self._work_dir, self._sandbox_mode, self._mcp_servers)
        await self._router.start()
        self._mcp_report = _McpReport(
            declared=self._router.declared_servers(),
            failed=self._router.failed_servers(),
        )
        self._journal_header()

    async def shutdown(self) -> None:
        if self._router is not None:
            await self._router.close()
            self._router = None
        self._session = None
        self._runtime = None
        self._control = None

    def is_alive(self) -> bool:
        return self._session is not None

    def is_process_alive(self) -> bool:
        return self.is_alive()

    @property
    def process_instance(self) -> str:
        return ""

    @property
    def exit_code(self) -> int | None:
        return None

    def runtime_abort_target(self):  # noqa: ANN202 - mirrors the ABC's type
        return None

    # -- catalogue --------------------------------------------------------------

    def _snapshot_path(self) -> Path:
        return self._data_home / ".kirocrew-bedrock-catalogue.json"

    async def _refresh_catalogue(self) -> None:
        snapshot = load_snapshot(self._snapshot_path())
        if snapshot is not None:
            meta, models = snapshot
            if snapshot_is_fresh(
                meta, region=self._region, identity_account=self._identity_account
            ):
                self._catalogue = models
                return
        models = await asyncio.to_thread(self._discover_models)
        self._catalogue = models
        save_snapshot(
            self._snapshot_path(),
            region=self._region,
            identity_account=self._identity_account,
            models=models,
        )

    def _discover_models(self) -> list[BedrockModel]:
        assert self._control is not None
        foundation: list[dict] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"byOutputModality": "TEXT"}
            if token:
                kwargs["nextToken"] = token
            page = self._control.list_foundation_models(**kwargs)
            foundation.extend(page.get("modelSummaries") or [])
            token = page.get("nextToken")
            if not token:
                break
        profiles: list[dict] = []
        token = None
        while True:
            kwargs = {}
            if token:
                kwargs["nextToken"] = token
            page = self._control.list_inference_profiles(**kwargs)
            profiles.extend(page.get("inferenceProfileSummaries") or [])
            token = page.get("nextToken")
            if not token:
                break
        return join_catalogue(foundation, profiles, region=self._region)

    # -- MCP resolution (mirrors the Meta provider's posture) -------------------

    def _resolve_mcp_servers(self) -> dict[str, dict]:
        if self._mcp_servers_arg is not None:
            return dict(self._mcp_servers_arg)
        from kiro_crew.user_json import load_mcp_servers

        servers: dict[str, dict] = {}
        for path in (
            Path.home() / ".kiro" / "settings" / "mcp.json",
            self._work_dir / ".kiro" / "settings" / "mcp.json",
        ):
            try:
                raw = load_mcp_servers(path)
            except Exception:
                continue
            for name, entry in raw.items():
                if not isinstance(entry, dict) or entry.get("disabled"):
                    continue
                spec: dict[str, Any] = {}
                if entry.get("command"):
                    spec["command"] = entry["command"]
                    if entry.get("args"):
                        spec["args"] = entry["args"]
                    if isinstance(entry.get("env"), dict):
                        spec["env"] = self._resolve_spec_env(entry["env"])
                elif entry.get("url"):
                    spec["url"] = entry["url"]
                if spec:
                    servers[str(name)] = spec
        return servers

    def _resolve_spec_env(self, env: dict) -> dict[str, str]:
        """Resolve secret:// values in an MCP server's env block.

        An unresolved reference drops the KEY (the server then starts
        without it and any failure is named in the session report);
        a plaintext value passes through as the operator wrote it.
        Resolution goes through the provider's own vault helper:
        ``resolve_env_credentials`` would return a reference filed
        under a non-CREDENTIAL_KEYS name as the literal text.
        """
        plain: dict[str, str] = {}
        refs: dict[str, str] = {}
        for key, value in env.items():
            key, value = str(key), str(value)
            if value.startswith("secret://"):
                refs[key] = value
            else:
                plain[key] = value
        resolved = resolve_secret_refs(refs, self._data_home) if refs else {}
        return {**plain, **resolved}

    # -- journal ------------------------------------------------------------------

    def _journal_path(self) -> Path:
        return self._work_dir / ".bedrock-journal.jsonl"

    def _journal_append(self, record: dict) -> None:
        try:
            with self._journal_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.warning("bedrock: could not append to the transcript journal")

    def _journal_header(self) -> None:
        """The journal's header record: identity provenance for the
        session (source CLASS and account — never credential material)."""
        self._journal_append(
            {
                "type": "header",
                "provider": PROVIDER_ID,
                "region": self._region,
                "profile": self._profile or "",
                "identity_account": self._identity_account,
                "identity_source_class": self._identity_source,
            }
        )

    def _journal_message(self, message: dict) -> None:
        self._journal_append({"type": "message", "message": message})

    # -- spend ledger ---------------------------------------------------------------

    def _spend_path(self) -> Path:
        return self._data_home / "bedrock_spend.json"

    def _month_to_date_usd(self) -> float:
        try:
            ledger = json.loads(self._spend_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0.0
        month = time.strftime("%Y-%m")
        return float(ledger.get(month, 0.0) or 0.0)

    def _record_spend(self, amount: float) -> None:
        if amount <= 0:
            return
        month = time.strftime("%Y-%m")
        try:
            ledger = json.loads(self._spend_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ledger = {}
        ledger[month] = float(ledger.get(month, 0.0) or 0.0) + amount
        with contextlib.suppress(OSError):
            self._spend_path().write_text(json.dumps(ledger), encoding="utf-8")

    # -- turns ------------------------------------------------------------------------

    def _system_text(self) -> str:
        agent = self._agent
        for attr in ("system_prompt", "instructions"):
            value = getattr(agent, attr, None)
            if value:
                return str(value)
        return ""

    def _converse_kwargs(self) -> dict[str, Any]:
        row = catalogue_model(self._catalogue, self._served_model)
        # The output ceiling is the overlay's max-output datum for the
        # family — NOT the context window (a model's output cap is a
        # separate, smaller number, and Converse refuses a request
        # above it). Unknown family: the conservative default.
        max_tokens = DEFAULT_MAX_TOKENS
        if row is not None and row.max_output_tokens:
            max_tokens = row.max_output_tokens
        if row is not None and row.context_window:
            max_tokens = min(max_tokens, row.context_window)
        kwargs: dict[str, Any] = {
            "modelId": self._served_model,
            "messages": list(self._messages),
            "inferenceConfig": {"maxTokens": max_tokens},
            "requestMetadata": {
                "kirocrew_provider": PROVIDER_ID,
                "kirocrew_session": self._session_id,
            },
        }
        system_text = self._system_text()
        if system_text:
            kwargs["system"] = [{"text": system_text}]
        if row is None or row.tool_use:
            if self._router is not None:
                kwargs["toolConfig"] = {"tools": self._router.tool_specs()}
        if self._guardrail_id:
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self._guardrail_id,
                "guardrailVersion": self._guardrail_version or "DRAFT",
                "trace": "enabled",
                # Synchronous stream processing: chunks are evaluated
                # before they are delivered, so a guardrail intervention
                # is a property of the stream the translator sees, not
                # of chunks already shown.
                "streamProcessingMode": "sync",
            }
        return kwargs

    async def stream(
        self, message: str, timeout: float | None = None, *, allow_image: bool = True
    ) -> AsyncIterator[AcpEvent]:
        if self._session is None:
            await self.start()
        assert self._runtime is not None and self._router is not None
        self._stats = BedrockTurnStats()
        self._turn_open = True
        self._cancel_requested = False
        user_message = {"role": "user", "content": [{"text": message}]}
        self._messages.append(user_message)
        self._journal_message(user_message)
        deadline = time.monotonic() + timeout if timeout else None
        try:
            async for event in self._run_turn(deadline):
                if event.kind == EVENT_COMPLETE:
                    self._last_stop_reason = event.stop_reason
                yield event
        finally:
            self._record_spend(self._stats.usage.cost_usd)
            self._turn_open = False

    async def _run_turn(self, deadline: float | None) -> AsyncIterator[AcpEvent]:
        steps = 0
        while True:
            if self._cancel_requested:
                yield AcpEvent(
                    kind=EVENT_COMPLETE,
                    stop_reason=STOP_REASON_CANCELLED,
                    synthetic_completion=True,
                    usage=self._stats.usage,
                )
                return
            if deadline is not None and time.monotonic() >= deadline:
                yield AcpEvent(
                    kind=EVENT_COMPLETE,
                    stop_reason="timeout",
                    text="the turn exceeded its time budget and was interrupted",
                    synthetic_completion=True,
                    usage=self._stats.usage,
                )
                return
            translator = ConverseStreamTranslator(self._served_model, self._stats.usage)
            outcome = None
            async for event in self._call_converse(translator, deadline):
                yield event
            outcome = translator.finish_call()
            if not outcome.rated:
                self._stats.cost_partial = True
            self._usage_seen = True
            self._context_used = translator.last_call_input_tokens
            if outcome.error_class:
                async for event in self._fail_turn(outcome):
                    yield event
                return
            if outcome.assistant_blocks:
                assistant_message = {
                    "role": "assistant",
                    "content": outcome.assistant_blocks,
                }
                self._messages.append(assistant_message)
                self._journal_message(assistant_message)
            if outcome.stop_reason == "tool_use":
                steps += 1
                if steps > self._max_tool_steps:
                    yield AcpEvent(
                        kind=EVENT_COMPLETE,
                        stop_reason="error: tool step limit reached",
                        text=(
                            f"the turn ended after {self._max_tool_steps} "
                            "tool steps without a final answer"
                        ),
                        usage=self._stats.usage,
                    )
                    return
                tool_results: list[dict[str, Any]] = []
                for tool_use in outcome.tool_uses:
                    async for ev, result in self._gate_and_execute(tool_use, deadline):
                        if ev is not None:
                            yield ev
                        if result is not None:
                            tool_results.append(result)
                result_message = {"role": "user", "content": tool_results}
                self._messages.append(result_message)
                self._journal_message(result_message)
                continue
            yield AcpEvent(
                kind=EVENT_COMPLETE,
                stop_reason=outcome.crew_stop_reason or "end_turn",
                truncated=outcome.truncated,
                refusal=outcome.refusal,
                usage=self._stats.usage,
            )
            return

    async def _fail_turn(self, outcome: Any) -> AsyncIterator[AcpEvent]:
        """A classified call failure ends the turn as the floor.

        After an auth-class failure the identity is re-verified before
        reporting (design §2.5): a mid-session credential rotation is
        reported as what it is, not as a model refusal.
        """
        code = outcome.error_class or "BedrockError"
        message = outcome.error_message or "the Bedrock call failed"
        if code in AUTH_ERROR_CODES and self._session is not None:
            try:
                identity = await asyncio.to_thread(verify_identity, self._session)
                message += (
                    f" (identity re-verified: account {identity.account_id} "
                    f"via {identity.source_class})"
                )
            except BedrockCallError as exc:
                message += f" (identity re-verification failed: {exc.code})"
        yield AcpEvent(
            kind=EVENT_COMPLETE,
            stop_reason=f"error: {code}",
            text=message,
            usage=self._stats.usage,
        )

    async def _call_converse(
        self, translator: ConverseStreamTranslator, deadline: float | None
    ) -> AsyncIterator[AcpEvent]:
        row = catalogue_model(self._catalogue, self._served_model)
        streaming = row.streaming if row is not None else True
        kwargs = self._converse_kwargs()
        runtime = self._runtime
        assert runtime is not None
        if not streaming:
            try:
                response = await asyncio.to_thread(
                    converse_with_retry, lambda: runtime.converse(**kwargs)
                )
            except BedrockCallError as exc:
                translator.note_error(exc.code, exc.message)
                return
            for event in translator.feed_response(response):
                yield event
            return
        try:
            response = await asyncio.to_thread(
                converse_with_retry, lambda: runtime.converse_stream(**kwargs)
            )
        except BedrockCallError as exc:
            translator.note_error(exc.code, exc.message)
            return
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stream_holder: dict[str, Any] = {}

        def pump() -> None:
            try:
                for raw_event in response.get("stream") or []:
                    if self._cancel_requested:
                        break
                    for event in translator.feed(raw_event):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # noqa: BLE001 - classified into the outcome
                classified = classify_client_error(exc)
                stream_holder["error"] = classified
                translator.note_error(classified.code, classified.message)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        pump_task = asyncio.create_task(asyncio.to_thread(pump))
        try:
            while True:
                wait = 0.25
                if deadline is not None:
                    wait = max(0.0, min(wait, deadline - time.monotonic()))
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=wait)
                except TimeoutError:
                    if self._cancel_requested:
                        break
                    continue
                if event is None:
                    break
                yield event
        finally:
            # The pump may be parked in a socket read on a cancelled
            # stream; give it a brief moment, then abandon the thread
            # (it ends with the process) rather than hang the turn.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(pump_task), timeout=2.0)
        if translator.turn_open and not self._cancel_requested:
            for event in translator.stream_died():
                yield event

    async def _gate_and_execute(
        self, tool_use: Any, deadline: float | None
    ) -> AsyncIterator[tuple[AcpEvent | None, dict | None]]:
        """One tool call through the approval registry, then execution.

        Rule 1 (design §5.2): execution happens ONLY here, after the
        pending-approval future resolves approved. A denial appends a
        toolResult with status ``error`` — visible to the model on its
        next call, never silently dropped.
        """
        self._approval_counter += 1
        request_id = f"bedrock-{self._approval_counter}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        tool_kind = {"shell": "execute", "write_file": "edit"}.get(tool_use.name, "other")
        yield (
            AcpEvent(
                kind=EVENT_PERMISSION_REQUEST,
                request_id=request_id,
                tool_call_id=tool_use.tool_use_id,
                title=tool_use.name,
                tool_kind=tool_kind,
                tool_input=json.dumps(tool_use.input, default=str),
                options=[
                    {"optionId": DECISION_ACCEPT, "name": "Accept", "kind": "allow_once"},
                    {
                        "optionId": DECISION_ACCEPT_FOR_SESSION,
                        "name": "Accept for session",
                        "kind": "allow_once",
                    },
                    {"optionId": DECISION_DECLINE, "name": "Decline", "kind": "reject_once"},
                    {"optionId": DECISION_CANCEL, "name": "Cancel", "kind": "reject_once"},
                ],
            ),
            None,
        )
        approved = False
        try:
            while not future.done():
                if self._cancel_requested:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                try:
                    approved = await asyncio.wait_for(asyncio.shield(future), timeout=0.25)
                    break
                except TimeoutError:
                    continue
            else:
                approved = bool(future.result())
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
        if not approved:
            yield (
                None,
                {
                    "toolResult": {
                        "toolUseId": tool_use.tool_use_id,
                        "content": [{"text": "the user declined this tool call"}],
                        "status": "error",
                    }
                },
            )
            return
        assert self._router is not None
        text, is_error = await self._router.execute(tool_use.name, tool_use.input)
        yield (
            AcpEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id=tool_use.tool_use_id,
                title=tool_use.name,
                tool_output=text,
            ),
            {
                "toolResult": {
                    "toolUseId": tool_use.tool_use_id,
                    "content": [{"text": text}],
                    "status": "error" if is_error else "success",
                }
            },
        )

    # -- approvals -----------------------------------------------------------------

    async def approve_tool(
        self,
        request_id: str | int,
        *,
        always: bool = False,
        deadline: Any = None,
    ) -> bool:
        if deadline is not None and deadline.ended(wall=time.time(), mono=time.monotonic()):
            raise GrantEnded("the approval grant ended before it was answered")
        future = self._pending.get(str(request_id))
        if future is None or future.done():
            return False
        # ``always`` degrades to one-shot at this provider (design
        # §5.1): Converse holds no session grant surface, so a session
        # grant is honoured for this call and asked again next time.
        future.set_result(True)
        return True

    async def reject_tool(self, request_id: str | int) -> None:
        future = self._pending.get(str(request_id))
        if future is not None and not future.done():
            future.set_result(False)

    # -- turn lifecycle (the watchdog's view) -------------------------------------------

    def has_active_turn(self) -> bool:
        # A turn paused on an approval is still a turn in flight: the
        # ABC default (False) would report it as idle.
        return self._turn_open

    def has_unfinished_turn(self) -> bool:
        return self._turn_open

    async def wait_turn_done(self, timeout: float) -> str:
        deadline = time.monotonic() + max(0.0, timeout)
        while self._turn_open and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return self._last_stop_reason

    # -- cancel / compact --------------------------------------------------------------

    async def cancel(self, *, wait_ack_timeout: float = 0.0) -> Any:
        if not self._turn_open:
            return "no_turn"
        self._cancel_requested = True
        for future in list(self._pending.values()):
            if not future.done():
                future.set_result(False)
        return "acked"

    async def compact(self, context: str = "") -> None:
        """Crew-owned summarisation: the journal-backed transcript is
        replaced by the summary the session layer supplies."""
        self._compaction_done = asyncio.Event()
        if context:
            self._messages = [
                {
                    "role": "user",
                    "content": [{"text": f"[Conversation summary so far:\n{context}\n]"}],
                }
            ]
        self._compaction_done.set()

    async def wait_for_compaction(self, timeout: float = 120.0) -> dict:
        if self._compaction_done is None:
            return {"type": "timeout"}
        try:
            await asyncio.wait_for(self._compaction_done.wait(), timeout=timeout)
        except TimeoutError:
            return {"type": "timeout"}
        return {"type": "completed"}

    # -- context / billing / models -------------------------------------------------------

    def context_usage_pct(self) -> float:
        if not self._usage_seen or not self._context_window:
            return 0.0
        return min(100.0, 100.0 * self._context_used / self._context_window)

    def context_usage_unknown(self) -> bool:
        return not self._usage_seen

    def context_window_tokens(self) -> int:
        return self._context_window

    def context_used_tokens(self) -> int:
        return self._context_used

    def billing_stats(self) -> object | None:
        return self._stats

    def available_models(self) -> list[dict[str, str]]:
        return [model.picker_row() for model in self._catalogue]

    async def maybe_refresh_available_models(self, catalog_ids: list[str]) -> list[dict[str, str]]:
        if not self._catalogue and self._session is not None:
            await self._refresh_catalogue()
        return self.available_models()

    def get_valid_effort_levels(self) -> list[str]:
        return []

    def supports_effort(self) -> bool:
        return False

    async def change_effort(self, level: str) -> bool:
        return False

    async def clear_effort(self) -> bool | None:
        return None

    # -- MCP report ----------------------------------------------------------------------------

    def mcp_session_report(self):  # noqa: ANN202 - the ABC's Protocol type
        if not self._mcp_servers:
            return None
        return self._mcp_report

    async def wait_for_declared_mcp(self, timeout: float) -> tuple[str, ...]:
        return self._mcp_report.declared_not_ready()

    # -- cleanup -----------------------------------------------------------------------------------

    async def cleanup_session(self, session_id: str) -> None:
        # The journal is a file in the session work dir; deleting it is
        # filesystem-only and safe with no process to stop (the ABC's
        # contract for this hook).
        with contextlib.suppress(OSError):
            self._journal_path().unlink(missing_ok=True)


def test_connection(
    *,
    region: str = "",
    profile: str | None = None,
    credential_refs: dict[str, str] | None = None,
) -> BedrockConnectionReport:
    """The read-only connection check behind the dashboard route and
    the doctor row: STS identity + discovery counts, error CLASS only,
    never credential material."""
    started = time.monotonic()
    probe = BedrockProvider(
        work_dir=Path.cwd(),
        model="auto",
        bedrock_region=region,
        bedrock_profile=profile,
        credential_refs=credential_refs,
    )

    def latency() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        probe._apply_vault_triple()
        session = make_boto_session(probe._profile)
        identity = verify_identity(session)
    except BedrockUnavailableError:
        return BedrockConnectionReport(False, latency_ms=latency(), error_class="bad_response")
    except BedrockCallError as exc:
        error_class = (
            "no_credentials"
            if exc.code == "NoCredentialsError"
            else "unauthorised" if exc.code in AUTH_ERROR_CODES else "unreachable"
        )
        return BedrockConnectionReport(False, latency_ms=latency(), error_class=error_class)
    resolved_region = region or getattr(session, "region_name", None) or ""
    if not resolved_region:
        return BedrockConnectionReport(
            False,
            latency_ms=latency(),
            error_class="bad_response",
            identity_account=identity.account_id,
            identity_arn=identity.arn,
            identity_source=identity.source_class,
        )
    try:
        control = session.client("bedrock", region_name=str(resolved_region))
        page = control.list_foundation_models(byOutputModality="TEXT")
        model_ids = [
            str(summary.get("modelId") or "") for summary in page.get("modelSummaries") or []
        ]
    except Exception as exc:  # noqa: BLE001 - classified below
        code = classify_client_error(exc).code
        return BedrockConnectionReport(
            False,
            latency_ms=latency(),
            error_class="unauthorised" if code in AUTH_ERROR_CODES else "unreachable",
            identity_account=identity.account_id,
            identity_arn=identity.arn,
            identity_source=identity.source_class,
            region=str(resolved_region),
        )
    return BedrockConnectionReport(
        True,
        model_ids=model_ids,
        latency_ms=latency(),
        identity_account=identity.account_id,
        identity_arn=identity.arn,
        identity_source=identity.source_class,
        region=str(resolved_region),
    )
