"""MetaApiProvider — the native Meta Model API provider (``agent.provider =
"meta_api"``).

Sessions run on the Codex app-server (the agent loop) with its model
provider pointed at the Meta Model API's Responses endpoint, supervised
through :mod:`kiro_crew.providers.meta_api_client`. The provider's
identity is positive (``is_meta_api_backend``), its capability set is the
reduced set declared in ``docs/system-specs/modules/providers.md``
(harness-parity H14: every property below is a deliberate answer, never an
inherited default left unexamined), and its per-turn billing is metered
API money computed from the API's own usage counts by
:mod:`kiro_crew.providers.meta_api_rates`.

Permission-gate parity is the §2.4 protocol, enforced in
:meth:`start`: configuration forced at spawn, read back and compared,
and the parity probe suite (``test_meta_api_integration.py``, P-1) stands
in CI against the pinned binary so an approval-semantics change fails a
pin bump rather than a user. The declared limit stands: there is no
per-call Crew PreToolUse hook inside the app-server process; tool calls
are gated by the forced approval path and the loop's sandbox, and MCP
tools routed through Crew's gateway keep the gateway's enforcement.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator

from kiro_crew.acp.types import EVENT_COMPLETE, AcpEvent, TurnUsage
from kiro_crew.providers.base import CancelOutcome, LLMProvider
from kiro_crew.providers.meta_api_client import (
    API_KEY_ENV,
    AppServerClient,
    AppServerError,
    codex_binary,
    map_sandbox_mode,
)
from kiro_crew.providers.meta_api_events import (
    DECISION_ACCEPT,
    DECISION_ACCEPT_FOR_SESSION,
    DECISION_DECLINE,
    TurnTranslator,
)
from kiro_crew.providers.meta_api_models import (
    ConnectionReport,
    fetch_models,
    merge_catalogue,
    resolve_model_id,
    static_model,
)
from kiro_crew.trust_patterns import GrantEnded

logger = logging.getLogger("kiro_crew.providers.meta_api")

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_SECRET_REF = "secret://MODEL_API_KEY"


class MetaApiTurnStats:
    """Per-turn billing object; a FRESH instance is installed at every
    turn start (the ABC's identity contract for ``billing_stats()``)."""

    def __init__(self) -> None:
        self.usage = TurnUsage()

    def to_turn_usage(self) -> TurnUsage:
        return self.usage

    @property
    def credits(self) -> float:
        return 0.0


class _McpReport:
    """This session's MCP registration report (the SessionMcpReport shape).

    Built from the app-server's ``mcpServer/startupStatus/updated``
    notifications: a server that failed to start is named, never silently
    absent.
    """

    def __init__(self, declared: tuple[str, ...]) -> None:
        self._declared = declared
        self._status: dict[str, str] = {}
        self._errors: dict[str, str] = {}

    def record_event(
        self, kind: str, server_name: str, error: str = "", *, fanout_no_owner: bool = False
    ) -> bool:
        if kind == "ready":
            self._status[server_name] = "ready"
        elif kind in ("failed", "error"):
            self._status[server_name] = "failed"
            if error:
                self._errors[server_name] = error
        return True

    def note_startup_status(self, params: dict) -> None:
        name = str(params.get("name") or params.get("serverName") or "")
        if not name:
            return
        status = str(params.get("status") or "")
        if status in ("ready", "completed", "connected"):
            self.record_event("ready", name)
        elif status in ("failed", "error"):
            self.record_event("failed", name, str(params.get("error") or ""))

    def payload(self) -> dict | None:
        if not self._declared:
            return None
        return {
            "servers": {
                name: {"status": self._status.get(name, "pending")} for name in self._declared
            }
        }

    def problem_summary(self, *, include_reasons: bool = True) -> str:
        bad = self.declared_not_ready()
        if not bad:
            return ""
        return "MCP servers not ready: " + ", ".join(bad)

    def declared_not_ready(self) -> tuple[str, ...]:
        return tuple(name for name in self._declared if self._status.get(name) != "ready")

    def declared_never_reported(self) -> tuple[str, ...]:
        return tuple(name for name in self._declared if name not in self._status)


class MetaApiProvider(LLMProvider):
    """Native provider: Muse models via the app-server loop on the Meta API."""

    reports_effort_support: bool = True

    def __init__(
        self,
        work_dir: str | Path | None = None,
        model: str | None = None,
        agent: str | None = None,
        sandbox_mode: str = "auto",
        session_key: str | None = None,
        channel_id: str | None = None,
        extra_env: dict[str, str] | None = None,
        effort_per_model: dict[str, str] | None = None,
        unbound_effort: str = "",
        inherited_effort: str = "",
        api_base_url: str = "",
        credential_ref: str = DEFAULT_SECRET_REF,
        monthly_budget_usd: float = 0.0,
        mcp_servers: dict[str, dict] | None = None,
        data_home: str | Path | None = None,
        **_kwargs: object,
    ) -> None:
        self._work_dir = Path(work_dir) if work_dir else Path.cwd()
        self._requested_model = model or ""
        self._agent = agent
        self._sandbox_mode = sandbox_mode
        self._session_key = session_key
        self._channel_id = channel_id
        self._extra_env = dict(extra_env or {})
        self._effort = (
            (effort_per_model or {}).get(self._requested_model, "")
            or unbound_effort
            or inherited_effort
            or ""
        )
        self._base_url = (api_base_url or DEFAULT_BASE_URL).rstrip("/")
        self._credential_ref = credential_ref
        self._monthly_budget_usd = float(monthly_budget_usd or 0.0)
        # None means "resolve at start from the operator's mcp.json files"
        # (the same global + workspace files the ACP path reads); a mapping
        # — even an empty one — is used exactly as given.
        self._mcp_servers_arg = mcp_servers
        self._mcp_servers: dict[str, dict] = dict(mcp_servers or {})
        self._data_home = Path(data_home) if data_home else None

        self._client: AppServerClient | None = None
        self._catalogue = merge_catalogue(None)
        self._thread_id = ""
        self._turn_id = ""
        self._served_model = ""
        self._model_pin_refused = ""
        self._translator: TurnTranslator | None = None
        self._stats = MetaApiTurnStats()
        self._event_queue: asyncio.Queue[AcpEvent | None] | None = None
        self._context_used = 0
        self._context_window = 0
        self._usage_seen = False
        self._mcp_report = _McpReport(tuple(sorted(self._mcp_servers)))
        self._compaction_done: asyncio.Event | None = None
        self._pending_replay = ""
        self._started_at = 0.0

    # -- identity and capability declarations (H14) ---------------------------

    @property
    def is_meta_api_backend(self) -> bool:
        return True

    @property
    def context_provider_type(self) -> str:
        return "meta_api"

    @property
    def context_incarnation(self) -> object:
        return (id(self), self._thread_id or self.session_id)

    @property
    def compaction_self_managed(self) -> bool:
        return True

    @property
    def harness_managed_compaction_backend(self) -> str | None:
        return "meta_api"

    @property
    def manual_compact_unsupported_backend(self) -> str | None:
        # The explicit-compact path is parity-tested against the pinned
        # binary (test_meta_api_integration.py, P-compact); it is served.
        return None

    @property
    def supports_steer(self) -> bool:
        # Off in v1: the protocol's turn/steer is enabled only when the
        # P-7 parity test passes on the pinned version.
        return False

    @property
    def supports_refusal_steer(self) -> bool:
        return False

    @property
    def last_steer_monotonic(self) -> float:
        # The H15 pair for the supports_steer override above: this provider
        # never steers, so the stamp is permanently "never" (0.0).
        return 0.0

    def background_launch(self) -> tuple[float, str] | None:
        # The pinned protocol's turn boundary is total: no loop work
        # outlives turn/completed, so there is nothing to report.
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

    # -- session identity -------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._thread_id

    @property
    def served_model(self) -> str:
        return self._served_model

    @property
    def model_pin_refused(self) -> str:
        return self._model_pin_refused

    @property
    def cwd(self) -> str:
        return str(self._work_dir)

    # -- credentials -------------------------------------------------------------

    async def _resolve_api_key(self) -> tuple[str, str]:
        return await asyncio.to_thread(resolve_meta_api_key, self._credential_ref)

    # -- budget guard -------------------------------------------------------------

    def _spend_ledger_path(self) -> Path | None:
        if self._data_home is None:
            return None
        return self._data_home / "meta_api_spend.json"

    def _month_to_date_usd(self) -> float:
        path = self._spend_ledger_path()
        if path is None:
            return 0.0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0.0
        month = time.strftime("%Y-%m")
        return float(data.get(month, 0.0))

    def _record_spend(self, cost_usd: float) -> None:
        path = self._spend_ledger_path()
        if path is None or cost_usd <= 0:
            return
        month = time.strftime("%Y-%m")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[month] = round(float(data.get(month, 0.0)) + cost_usd, 6)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            logger.warning("meta_api: could not persist the spend ledger", exc_info=True)

    def _resolve_mcp_servers(self) -> dict[str, dict]:
        """The MCP servers this session passes to the loop at spawn.

        From the operator's own files — the global ``~/.kiro/settings/
        mcp.json`` and the workspace ``.kiro/settings/mcp.json``, the same
        files the ACP path reads — translated to the app-server's shape.
        Crew-managed gateway servers are NOT routed into native sessions
        in v1 (declared in providers.md); a server whose entry is disabled
        or shapeless is skipped, and a server that then fails to start is
        named by :meth:`mcp_session_report`.
        """
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
                        spec["env"] = entry["env"]
                elif entry.get("url"):
                    spec["url"] = entry["url"]
                if spec:
                    servers[str(name)] = spec
        return servers

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        if self._client is not None and self._client.is_alive():
            return
        binary = codex_binary()
        if binary is None:
            self.startup_notice = (
                "meta_api_codex_missing",
                "The native Meta provider needs the `codex` binary on PATH "
                "(install @openai/codex); no session was started.",
            )
            raise AppServerError(self.startup_notice[1])
        api_key, source = await self._resolve_api_key()
        if not api_key:
            self.startup_notice = (
                "meta_api_key_missing",
                "No Meta Model API key is available: store one under "
                "Settings > Secrets and reference it from "
                "agent.meta_api_secret (secret://MODEL_API_KEY), or set "
                "MODEL_API_KEY in the environment. No session was started.",
            )
            raise AppServerError(self.startup_notice[1])
        logger.info("meta_api: API key supplied by %s", source)
        if self._monthly_budget_usd > 0:
            spent = self._month_to_date_usd()
            if spent >= self._monthly_budget_usd:
                self.startup_notice = (
                    "meta_api_budget_reached",
                    f"The native Meta provider's monthly budget is reached "
                    f"(${spent:.2f} of ${self._monthly_budget_usd:.2f} this "
                    "month); no session was started.",
                )
                raise AppServerError(self.startup_notice[1])
        self._served_model = resolve_model_id(self._requested_model)
        self._mcp_servers = self._resolve_mcp_servers()
        self._mcp_report = _McpReport(tuple(sorted(self._mcp_servers)))
        codex_home = str(self._work_dir / ".meta-api-codex-home")
        self._client = AppServerClient(
            binary=binary,
            base_url=self._base_url,
            api_key=api_key,
            model=self._served_model,
            cwd=str(self._work_dir),
            sandbox_mode=map_sandbox_mode(self._sandbox_mode),
            codex_home=codex_home,
            extra_env=self._extra_env,
            mcp_servers=self._mcp_servers,
        )
        self._client.set_frame_handler(self._on_frame)
        self._client.set_exit_handler(self._on_process_exit)
        try:
            await self._client.start()
        except BaseException:
            await self._client.close()
            self._client = None
            raise
        self._started_at = time.monotonic()
        await self._refresh_catalogue(api_key)

    async def _refresh_catalogue(self, api_key: str) -> None:
        report = await asyncio.to_thread(fetch_models, self._base_url, api_key)
        ids = report.model_ids if report.ok else None
        self._catalogue = merge_catalogue(ids)

    async def shutdown(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()
        self._thread_id = ""

    def is_alive(self) -> bool:
        return self._client is not None and self._client.is_alive()

    def is_process_alive(self) -> bool:
        return self.is_alive()

    @property
    def process_instance(self) -> str:
        return self._client.process_instance if self._client else ""

    @property
    def exit_code(self) -> int | None:
        return self._client.exit_code if self._client else None

    def runtime_abort_target(self):  # noqa: ANN202 - mirrors the ABC's type
        from kiro_crew.mcp_gateway.abort import RuntimeAbortTarget

        if self._client is None:
            return None
        return RuntimeAbortTarget.build(self._client.pid, None)

    # -- frames -------------------------------------------------------------------

    def _on_process_exit(self) -> None:
        translator = self._translator
        queue = self._event_queue
        if translator is None or queue is None:
            return
        for event in translator.process_died():
            queue.put_nowait(event)

    async def _on_frame(self, frame: dict) -> None:
        method = frame.get("method")
        if method == "mcpServer/startupStatus/updated":
            self._mcp_report.note_startup_status(frame.get("params") or {})
            return
        if self._compaction_done is not None and not self._compaction_done.is_set():
            params = frame.get("params") or {}
            item = params.get("item") or {}
            if method == "thread/compacted" or (
                method == "item/completed" and item.get("type") == "contextCompaction"
            ):
                self._compaction_done.set()
        if method == "thread/tokenUsage/updated":
            usage = ((frame.get("params") or {}).get("tokenUsage") or {}).get("last") or {}
            if usage:
                self._usage_seen = True
                self._context_used = int(usage.get("totalTokens") or 0)
                model = static_model(self._served_model)
                self._context_window = model.context_window if model else self._context_window
        translator = self._translator
        queue = self._event_queue
        if translator is None or queue is None:
            # Server requests outside a turn still must not hang the loop:
            # answer an approval request with a refusal-by-default error.
            if "id" in frame and isinstance(method, str) and "requestApproval" in method:
                if self._client is not None:
                    await self._client.respond(frame["id"], {"decision": DECISION_DECLINE})
            return
        for event in translator.feed(frame):
            queue.put_nowait(event)
        if method == "thread/compacted" and self._compaction_done is not None:
            self._compaction_done.set()

    # -- transcript journal ---------------------------------------------------

    def _journal_path(self) -> Path:
        return self._work_dir / ".meta-api-journal.jsonl"

    def _journal_append(self, text: str) -> None:
        try:
            with self._journal_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"role": "user", "text": text}) + "\n")
        except OSError:
            logger.warning("meta_api: could not append to the transcript journal")

    def _journal_replay_text(self) -> str:
        """The journaled user messages as a context preamble, bounded.

        Used only when the loop's own thread record is gone: the design
        does not depend on the API (or the loop) keeping state for us.
        Tool side effects are never replayed — only the user's own words.
        """
        try:
            lines = self._journal_path().read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""
        texts: list[str] = []
        for line in lines[-50:]:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("role") == "user" and entry.get("text"):
                texts.append(str(entry["text"]))
        if not texts:
            return ""
        joined = "\n\n".join(texts)
        return (
            "[Prior conversation — this session's thread was rebuilt; the "
            "user's earlier messages were:\n" + joined[-32_000:] + "\n]"
        )

    # -- turns ----------------------------------------------------------------------

    async def _ensure_thread(self) -> str:
        assert self._client is not None
        if self._thread_id:
            try:
                await self._client.request(
                    "thread/resume",
                    {
                        "threadId": self._thread_id,
                        "model": self._served_model,
                        "cwd": str(self._work_dir),
                        "approvalPolicy": "on-request",
                        "sandbox": map_sandbox_mode(self._sandbox_mode),
                    },
                )
                return self._thread_id
            except AppServerError:
                logger.info(
                    "meta_api: thread resume failed; starting a fresh thread "
                    "and replaying the transcript journal",
                    exc_info=True,
                )
                self._thread_id = ""
                self._pending_replay = self._journal_replay_text()
        result = await self._client.request(
            "thread/start",
            {
                "model": self._served_model,
                "cwd": str(self._work_dir),
                "approvalPolicy": "on-request",
                "sandbox": map_sandbox_mode(self._sandbox_mode),
            },
        )
        thread = result.get("thread") or {}
        self._thread_id = str(thread.get("id") or result.get("threadId") or "")
        if not self._thread_id:
            raise AppServerError("the app-server did not return a thread id")
        return self._thread_id

    async def stream(
        self, message: str, timeout: float | None = None, *, allow_image: bool = True
    ) -> AsyncIterator[AcpEvent]:
        if self._client is None:
            await self.start()
        assert self._client is not None
        thread_id = await self._ensure_thread()
        self._journal_append(message)
        if self._pending_replay:
            message = self._pending_replay + "\n\n" + message
            self._pending_replay = ""
        self._stats = MetaApiTurnStats()
        translator = TurnTranslator(model_id=self._served_model)
        translator.usage = self._stats.usage
        self._translator = translator
        queue: asyncio.Queue[AcpEvent | None] = asyncio.Queue()
        self._event_queue = queue
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": message}],
        }
        if self._effort:
            params["effort"] = self._effort
        try:
            started = await self._client.request("turn/start", params)
        except AppServerError as exc:
            self._translator = None
            self._event_queue = None
            yield AcpEvent(
                kind=EVENT_COMPLETE,
                stop_reason="error",
                text=f"the turn could not start: {exc}",
                usage=self._stats.usage,
            )
            return
        self._turn_id = str((started.get("turn") or {}).get("id") or "")
        deadline = time.monotonic() + timeout if timeout else None
        try:
            while True:
                wait = None
                if deadline is not None:
                    wait = max(0.0, deadline - time.monotonic())
                    if wait == 0.0:
                        await self.cancel(wait_ack_timeout=2.0)
                        yield AcpEvent(
                            kind=EVENT_COMPLETE,
                            stop_reason="timeout",
                            text="the turn exceeded its time budget and was interrupted",
                            usage=self._stats.usage,
                            synthetic_completion=True,
                        )
                        return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=wait)
                except TimeoutError:
                    continue
                if event is None:
                    break
                yield event
                if event.kind == EVENT_COMPLETE:
                    break
        finally:
            self._record_spend(self._stats.usage.cost_usd)
            self._translator = None
            self._event_queue = None

    # -- approvals -------------------------------------------------------------------

    async def approve_tool(
        self,
        request_id: str | int,
        *,
        always: bool = False,
        deadline: Any = None,
    ) -> bool:
        if deadline is not None and deadline.ended(wall=time.time(), mono=time.monotonic()):
            raise GrantEnded("the approval grant ended before it was answered")
        translator = self._translator
        client = self._client
        if translator is None or client is None:
            return False
        pending = translator.take_pending(request_id)
        if pending is None:
            return False
        decision = DECISION_ACCEPT_FOR_SESSION if always else DECISION_ACCEPT
        await client.respond(pending.request_id, {"decision": decision})
        return True

    async def reject_tool(self, request_id: str | int) -> None:
        translator = self._translator
        client = self._client
        if translator is None or client is None:
            return
        pending = translator.take_pending(request_id)
        if pending is None:
            return
        await client.respond(pending.request_id, {"decision": DECISION_DECLINE})

    # -- cancel / compact ---------------------------------------------------------------

    async def cancel(self, *, wait_ack_timeout: float = 0.0) -> CancelOutcome:
        client = self._client
        translator = self._translator
        if client is None or translator is None or not translator.turn_open:
            return "no_turn"
        try:
            await client.request(
                "turn/interrupt",
                {"threadId": self._thread_id, "turnId": self._turn_id},
            )
        except AppServerError:
            return "error"
        return "acked"

    async def compact(self, context: str = "") -> None:
        if self._client is None or not self._thread_id:
            return
        self._compaction_done = asyncio.Event()
        await self._client.request("thread/compact/start", {"threadId": self._thread_id})

    async def wait_for_compaction(self, timeout: float = 120.0) -> dict:
        if self._compaction_done is None:
            return {"type": "timeout"}
        try:
            await asyncio.wait_for(self._compaction_done.wait(), timeout=timeout)
        except TimeoutError:
            return {"type": "timeout"}
        return {"type": "completed"}

    # -- context / billing / models ------------------------------------------------------

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
        catalogue = getattr(self, "_catalogue", None) or merge_catalogue(None)
        return [model.picker_row() for model in catalogue]

    async def maybe_refresh_available_models(self, catalog_ids: list[str]) -> list[dict[str, str]]:
        return self.available_models()

    def get_valid_effort_levels(self) -> list[str]:
        model = static_model(self._served_model)
        return list(model.effort_levels) if model else []

    def supports_effort(self) -> bool:
        return bool(self.get_valid_effort_levels())

    async def change_effort(self, level: str) -> bool:
        if level not in self.get_valid_effort_levels():
            return False
        self._effort = level
        return True

    async def clear_effort(self) -> bool | None:
        self._effort = ""
        return True

    # -- MCP report -------------------------------------------------------------------------

    def mcp_session_report(self):  # noqa: ANN202 - the ABC's Protocol type
        if not self._mcp_servers:
            return None
        return self._mcp_report

    async def wait_for_declared_mcp(self, timeout: float) -> tuple[str, ...]:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            not_ready = self._mcp_report.declared_not_ready()
            if not not_ready or time.monotonic() >= deadline:
                return not_ready
            await asyncio.sleep(0.05)

    # -- cleanup --------------------------------------------------------------------------------

    async def cleanup_session(self, session_id: str) -> None:
        # The local journal is the session work dir's codex home; deleting
        # it is filesystem-only and safe with the process dead, per the ABC.
        journal = self._work_dir / ".meta-api-codex-home"
        if journal.is_dir():
            import shutil

            shutil.rmtree(journal, ignore_errors=True)
        with contextlib.suppress(OSError):
            self._journal_path().unlink(missing_ok=True)
        if self._client is not None and session_id:
            with contextlib.suppress(Exception):
                await self._client.request("thread/delete", {"threadId": session_id})


def resolve_meta_api_key(credential_ref: str) -> tuple[str, str]:
    """(key, source) or ("", "") for a configured credential reference.

    Precedence: the ``secret://`` reference through the vault, then the
    ``MODEL_API_KEY`` environment variable, then fail closed. A plaintext
    value in the reference field is never used as a key (it is refused at
    config write; a hand-edited one fails closed here too). The source is
    a human description of WHERE the key came from — the value itself is
    never logged or returned by anything but this function's first item.
    Shared by the provider, the dashboard test-connection route and the
    doctor row so all three resolve identically.
    """
    ref = (credential_ref or "").strip()
    if ref.startswith("secret://"):
        from kiro_crew.config.loader_credentials import resolve_env_credentials

        resolved = resolve_env_credentials({API_KEY_ENV: ref})
        key = resolved.get(API_KEY_ENV, "")
        if key:
            return key, "the configured secret:// reference"
    elif ref:
        logger.error(
            "meta_api: agent.meta_api_secret is not a secret:// reference; "
            "refusing to use a plaintext value as an API key"
        )
        return "", ""
    env_key = os.environ.get(API_KEY_ENV, "")
    if env_key:
        return env_key, "the MODEL_API_KEY environment variable"
    return "", ""


def test_connection(base_url: str, api_key: str) -> ConnectionReport:
    """The read-only connection check behind the dashboard route and the
    doctor row: GET /v1/models, error CLASS only."""
    return fetch_models(base_url.rstrip("/"), api_key)
