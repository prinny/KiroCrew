"""Projects the agent spec onto the ``muse-acp`` adapter -- behind a closed gate.

The channel exists and is measured: muse-acp accepts the ``session/new``
``mcpServers`` array and forwards it to Muse (its ``initialize`` advertises
MCP over HTTP, and the adapter's own documentation states that client-supplied
MCP servers are forwarded to the Muse host). The array's MECHANICS are the
ones the other wire-face mirrors already own: the same shared translation
(:func:`kiro_crew.acp.session_mcp.session_mcp_projection`, no new translator),
the same placement (:func:`~kiro_crew.providers.mirrors.opencode.place_single_binary_array`),
the same one-owner rule for the pooled broker stubs.

What is muse's own is the GATE in front of the channel. Crew does not pass its
MCP servers into a muse session by default. The reason is a property of the
harness, not a missing transport: MCP tools are not covered by Muse's
write/shell disable flags and the servers run unsandboxed, so a server mounted
into a muse session holds powers the session's approval posture does not
govern -- in Muse's read-only and plan modes, where its own file and shell
tools are disabled, a mounted server can still write files, run commands and
use the network. Crew's answer is that the projection is WITHHELD unless the
operator opens the gate for the gateway process by setting
:data:`MUSE_MCP_OPT_IN_ENV`. The withhold is total -- the spec's servers and
the pooled broker stubs alike, because a stub is the same server under the
same name -- and it is this module's, so the declaration and the wire cannot
drift apart: the session's array is empty because the projection says so, in
one place, with the reason attached.
"""

from __future__ import annotations

import os
from typing import Any, Collection, Mapping

from kiro_crew.acp.session_mcp import session_mcp_projection
from kiro_crew.acp.session_mcp_env_refs import keep_env_references
from kiro_crew.agent_sdk.backends import ACP_BACKEND_MUSE
from kiro_crew.providers.mirrors.base import (
    RESOURCES_NO_READER,
    AgentConfigMirror,
    Concern,
)
from kiro_crew.providers.mirrors.base import Disposition as _D
from kiro_crew.providers.mirrors.base import Ruling, SessionProjection
from kiro_crew.providers.mirrors.opencode import place_single_binary_array

__all__ = ["MUSE_MCP_OPT_IN_ENV", "MuseMirror", "muse_mcp_opted_in", "muse_projection"]

#: The environment variable that opens this backend's MCP gate for the
#: gateway process. Off by default: its absence, or any value outside
#: :data:`_OPT_IN_VALUES`, leaves the projection withheld.
MUSE_MCP_OPT_IN_ENV = "KIROCREW_MUSE_MCP"

_OPT_IN_VALUES = frozenset({"1", "true", "yes", "on"})


def muse_mcp_opted_in() -> bool:
    """Whether the operator has opened the muse MCP gate for this process."""
    return os.environ.get(MUSE_MCP_OPT_IN_ENV, "").strip().lower() in _OPT_IN_VALUES


def muse_projection(
    agent: str | None,
    *,
    stub_server_names: Collection[str] = (),
    stub_elements: Collection[Mapping[str, Any]] = (),
    work_dir: object = None,
    session_key: str = "",
    channel_id: str = "",
    session_token: str = "",
) -> SessionProjection:
    """The whole muse array -- or, by default, the withheld empty one.

    The spec is parsed either way, from the same single parse the other
    mirrors use, because the parse's obligations do not lapse with the
    withhold: the session report, the unresolved-ref guard and the member
    append all read this projection's answers, and an unparsed spec would
    answer them with silence. What the gate changes is only the array.

    With the gate closed the array is empty and every server the parse would
    have mounted is named in ``unhonoured_servers`` -- the spec's servers and
    the pooled broker stubs alike -- so no caller-side append can re-add one
    behind the projection's back.

    With the gate open the array is placed by the shared single-binary
    placement, and every server the spec narrows per tool is withheld WHOLE,
    Crew's own control plane included, with ``denied_tools`` empty: no
    per-call deny channel is verified for this harness, and a mounted MCP
    server's calls are the ones this backend's posture exists to govern, so a
    narrowed server is not kept mounted on the strength of the approval
    prompt alone.

    Blocking (parses the agent spec once), so callers run it off the event
    loop.
    """
    projection = session_mcp_projection(
        agent,
        stub_server_names=stub_server_names,
        work_dir=work_dir,  # type: ignore[arg-type]
    )
    # Muse keeps session state of its own, so a resolved reference must not
    # ride its array (session_mcp_env_refs): an env value the spec spells as
    # a reference stays a reference and the server resolves it from the
    # environment Crew gives the session.
    projection = projection._replace(
        servers=keep_env_references(projection.servers, projection.agent_spec)
    )
    narrowed = frozenset(server for server, _tool in projection.disabled_tools)
    if not muse_mcp_opted_in():
        withheld = frozenset(
            element.get("name")
            for element in projection.servers
            if isinstance(element, dict) and element.get("name")
        ) | frozenset(stub_server_names)
        return SessionProjection(
            params={"mcpServers": []},
            disabled_servers=projection.disabled_servers,
            restricted_servers=narrowed,
            unhonoured_servers=withheld,
            zero_tools=projection.zero_tools,
            derived_spec_snapshot=projection.derived_spec_snapshot,
        )
    out = place_single_binary_array(
        projection,
        label="muse",
        unhonoured=narrowed,
        stub_elements=stub_elements,
        session_key=session_key,
        channel_id=channel_id,
        session_token=session_token,
    )
    return SessionProjection(
        params={"mcpServers": out},
        disabled_servers=projection.disabled_servers,
        restricted_servers=narrowed,
        unhonoured_servers=narrowed,
        zero_tools=projection.zero_tools,
        derived_spec_snapshot=projection.derived_spec_snapshot,
    )


class MuseMirror(AgentConfigMirror):
    """Projects the agent spec onto the ``muse-acp`` adapter."""

    backend = ACP_BACKEND_MUSE

    def rulings(self) -> Mapping[Concern, Ruling]:
        return {
            Concern.MCP_SERVERS: Ruling(
                _D.TRANSLATED,
                "by the session/new + session/load mcpServers array, translated "
                "by acp.session_mcp.session_mcp_projection with NO new "
                "translator -- and withheld in full unless the operator opens "
                "the gate (KIROCREW_MUSE_MCP) for the gateway process. The "
                "channel is real and measured (the adapter forwards "
                "client-supplied servers to the Muse host), so the withhold "
                "is a decision rather than an absence: MCP tools are not "
                "covered by Muse's write/shell disable flags and the servers "
                "run unsandboxed, so in the modes where Muse's own tools are "
                "disabled a mounted server can still write files, run "
                "commands and use the network. With the gate open, a server "
                "the spec narrows per tool is withheld whole, as on goose: no "
                "per-call deny channel is verified for this harness",
            ),
            Concern.TOOL_ALLOWLIST: Ruling(
                _D.TRANSLATED,
                "into the allowlist that decides which servers enter the array "
                "when the gate is open -- the shared translation's own rule, "
                "not a second one here. With the gate closed it decides "
                "nothing, because the array is empty either way",
            ),
            Concern.DENIED_TOOLS: Ruling(
                _D.TRANSLATED,
                "by withholding the server it narrows, Crew's own control "
                "plane included, and declared as per_tool_deny=whole-server "
                "on the projection record. The adapter does send MCP calls "
                "through session/request_permission under the approval mode "
                "Crew arms, but a refusal there is Crew trusting the "
                "adapter's own framing of a call from an unsandboxed server; "
                "withholding the server is the only form that does not depend "
                "on it",
            ),
            Concern.MODEL: Ruling(
                _D.DELIVERED,
                "as a session config option, from the model select the "
                "adapter advertises on session/new -- not from this array",
            ),
            Concern.MODEL_ALLOWLIST: Ruling(
                _D.WITHHELD,
                "the harness advertises its own model vocabulary on "
                "session/new, so a Crew-side list would be a second answer to "
                "a question the session already answers",
            ),
            Concern.AUTO_APPROVE: Ruling(
                _D.WITHHELD,
                "this harness's approval posture is a session config option "
                "Crew's routing owns, and the only value Crew puts on the "
                "wire is promptUnmatched. An auto-approve projection would "
                "have to name allowAll, which suppresses the permission "
                "frames the host gate is reached through",
            ),
            Concern.PERMISSION_MODE: Ruling(
                _D.WITHHELD,
                "not projected as spec data. The approval mode Crew requires "
                "travels as MUSE_APPROVAL_MODE in the child's environment "
                "and as the session/set_config_option write Crew's routing "
                "performs and verifies; putting it here as well would give "
                "one guarantee two owners",
            ),
            Concern.PROMPT: Ruling(
                _D.WITHHELD,
                "no definition is projected in any shape, so there is "
                "nothing for a prompt to ride on; the session's instructions "
                "arrive as the turn's own text",
            ),
            Concern.RESOURCES: Ruling(_D.WITHHELD, RESOURCES_NO_READER),
            Concern.HOOKS: Ruling(
                _D.WITHHELD,
                "the session/new element set has no hooks field and this "
                "backend is not in ACP_BACKENDS_CREW_FIRES_SPEC_HOOKS, so a "
                "saved hooks block is reported as a projection gap rather "
                "than claimed. Crew does not fire them itself for the reason "
                "that set's own record gives for claude and codex: a "
                "PreToolUse hook runs where a call arrives as a permission "
                "request, and this harness approves the calls its own "
                "permission rules match without asking, so a hook would be "
                "skipped on exactly those calls",
            ),
            Concern.TOOLS_SETTINGS: Ruling(
                _D.WITHHELD,
                "the session/new element set has no toolsSettings field and "
                "Crew feeds this backend's permission requests no spec "
                "allowlist, so the field is reported as a projection gap "
                "rather than claimed. Its subagent.availableAgents is the "
                "exception: Crew's spawn gate reads it from the parent's "
                "spec on every backend (subagent_parts/agent_policy.py)",
            ),
        }

    def session_params(
        self,
        agent: str | None,
        *,
        stub_server_names: Collection[str] = (),
        work_dir: object = None,
        session_key: str = "",
        channel_id: str = "",
        session_token: str = "",
        **kwargs: object,
    ) -> dict[str, object]:
        """The wire face: the ``mcpServers`` array for this muse session.

        ``permission_surface_owned`` is accepted and IGNORED (it arrives in
        ``kwargs``), which is the documented behaviour for a mirror outside
        claude's class. That flag exists because claude's permission surface
        is a file Crew may not own; muse has no such file in play: its
        routing is a session config option Crew writes and reads back before
        the first prompt.

        ``stub_elements`` are the shared gateway's broker stubs for this
        session, which the caller holds and this mirror places -- or
        withholds, with the gate closed -- so one rule covers both halves of
        the array. See :func:`muse_projection`.

        Blocking -- it reads the agent spec. The caller warms this on the
        muse spawn path and serves the shared ``session/new`` call site from
        that cache (harness-parity H13).
        """
        stubs = kwargs.get("stub_elements") or ()
        return self.session_projection(
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stubs if isinstance(stubs, (list, tuple)) else (),
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
        ).params

    def session_projection(
        self,
        agent: str | None,
        *,
        stub_server_names: Collection[str] = (),
        stub_elements: Collection[Mapping[str, Any]] = (),
        work_dir: object = None,
        session_key: str = "",
        channel_id: str = "",
        session_token: str = "",
        **kwargs: object,
    ) -> SessionProjection:
        """The structured face: :func:`muse_projection`, with ``kwargs``
        ignored as :meth:`session_params` documents. ``denied_tools`` is
        empty on this backend by decision, not by default -- see the
        ``disabledTools`` ruling."""
        del kwargs
        return muse_projection(
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stub_elements,
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
        )
