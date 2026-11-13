"""Projects the agent spec onto an operator-registered custom ACP backend.

The generic mirror, and the reason it can be generic: a custom backend takes
its MCP servers the way ACP itself defines -- the ``session/new`` ``mcpServers``
array -- so the projection is the ONE shared translation
(:func:`kiro_crew.acp.session_mcp.session_mcp_projection`, no new translator)
placed by the same single-binary placement the sibling harnesses share
(:func:`kiro_crew.providers.mirrors.opencode.place_single_binary_array`).
What is declared rather than measured is stated as such in the rulings: Crew
has not verified this adapter's per-call permission channel, so a server the
spec narrows per tool is withheld WHOLE -- the only form of the restriction
no adapter behaviour can undo -- and the projection record declares
``per_tool_deny=whole-server`` to match.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping

from kiro_crew.acp.session_mcp import session_mcp_projection
from kiro_crew.providers.mirrors.base import (
    TOOLS_SETTINGS_GATE_CHANNEL,
    AgentConfigMirror,
    Concern,
)
from kiro_crew.providers.mirrors.base import Disposition as _D
from kiro_crew.providers.mirrors.base import (
    Ruling,
    SessionProjection,
)
from kiro_crew.providers.mirrors.opencode import place_single_binary_array

__all__ = ["CustomMirror", "custom_projection"]


def custom_projection(
    backend: str,
    agent: str | None,
    *,
    stub_server_names: Collection[str] = (),
    stub_elements: Collection[Mapping[str, Any]] = (),
    work_dir: object = None,
    session_key: str = "",
    channel_id: str = "",
    session_token: str = "",
) -> SessionProjection:
    """The whole array for one custom-backend session: spec translation AND pooled stubs.

    Placed by the shared single-binary placement, so one withhold rule covers
    both halves of the array (a stub appended after the projection withheld
    that name would un-withhold it, and the stub is the unrestricted server).
    Every server the spec narrows per tool is withheld whole: Crew has not
    measured this adapter's per-call channel, and a narrowed server mounted
    on an unmeasured channel is a restriction Crew cannot show it honours.

    Blocking (parses the agent spec once), so callers run it off the event
    loop -- the launch warms it on the backend's own spawn path.
    """
    projection = session_mcp_projection(
        agent,
        stub_server_names=stub_server_names,
        work_dir=work_dir,  # type: ignore[arg-type]
    )
    narrowed = frozenset(server for server, _tool in projection.disabled_tools)
    out = place_single_binary_array(
        projection,
        label=backend,
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


class CustomMirror(AgentConfigMirror):
    """Projects the agent spec onto one registered custom backend.

    One instance per backend id (the registry constructs it with the id), so
    unlike the shipped mirrors this class carries no fixed ``backend`` class
    attribute: the id is the registration's, not the code's.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def rulings(self) -> Mapping[Concern, Ruling]:
        return {
            Concern.MCP_SERVERS: Ruling(
                _D.DELIVERED,
                "the session/new + session/load mcpServers array, translated by "
                "acp.session_mcp.session_mcp_servers with NO new translator and "
                "placed by the shared single-binary placement. The channel is "
                "ACP's own: an adapter that speaks ACP v1 accepts the array by "
                "construction of the protocol, which is the only claim a generic "
                "mirror may make -- Crew has not measured this adapter's "
                "round trip, and the ruling does not pretend otherwise",
            ),
            Concern.TOOL_ALLOWLIST: Ruling(
                _D.TRANSLATED,
                "into the allowlist that decides which servers enter the array -- "
                "the shared translation's own rule, not a second one here",
            ),
            Concern.DENIED_TOOLS: Ruling(
                _D.TRANSLATED,
                "by withholding the server it narrows, Crew's own control plane "
                "included, and declared as per_tool_deny=whole-server on the "
                "projection record. A per-call refusal would need this adapter's "
                "permission channel measured the way a shipped harness's is; "
                "until then, withholding the server is the only form of the "
                "restriction no adapter behaviour can undo",
            ),
            Concern.AUTO_APPROVE: Ruling(
                _D.WITHHELD,
                "the spec's auto-approve is not projected. Whether this backend "
                "asks is its registration's permission posture, set by the "
                "operator who registered it; spec data Crew carries must not "
                "widen an adapter's asking on its own",
            ),
            Concern.MODEL: Ruling(
                _D.DELIVERED,
                "as a session config option, when the registration names the "
                "option that carries it -- the adapter advertises its own model "
                "vocabulary on session/new, and the option is the channel ACP "
                "provides for choosing from it",
            ),
            Concern.MODEL_ALLOWLIST: Ruling(
                _D.WITHHELD,
                "the adapter advertises its own model vocabulary on session/new, "
                "so a Crew-side list would be a second answer to a question the "
                "session already answers",
            ),
            Concern.PERMISSION_MODE: Ruling(
                _D.WITHHELD,
                "not projected as spec data. The posture Crew requires travels "
                "through the registration's own permission mechanism -- a session "
                "config option the client sets and verifies, or a seed the "
                "launch writes -- never as a second copy riding the spec",
            ),
            Concern.PROMPT: Ruling(
                _D.WITHHELD,
                "no definition is projected in any shape, so there is nothing "
                "for a prompt to ride on; the session's instructions arrive as "
                "the turn's own text",
            ),
            Concern.RESOURCES: Ruling(
                _D.WITHHELD,
                "no channel is advertised for them, and the array carries "
                "servers rather than resources",
            ),
            Concern.HOOKS: Ruling(
                _D.WITHHELD,
                "the spec's hooks block is not projected, and Crew does not fire "
                "spec hooks for a custom backend: whether this adapter's tool "
                "calls all arrive as permission requests is unmeasured, and a "
                "PreToolUse hook Crew cannot show runs is not a control",
            ),
            Concern.TOOLS_SETTINGS: Ruling(
                _D.NO_CHANNEL,
                "kiro-cli's per-tool settings (shell denyByDefault and "
                "allowedCommands, the write allowlist, deniedPaths, "
                "web_fetch.blocked) have no field on the ACP session/new element "
                "set, and Crew feeds this session's permission requests no spec "
                "allowlist, so the field is applied nowhere. Its "
                "subagent.availableAgents is the exception: Crew's spawn gate "
                "reads it from the parent's spec on every backend "
                "(subagent_parts/agent_policy.py), so no notice names it",
                channel=TOOLS_SETTINGS_GATE_CHANNEL,
            ),
        }

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
        """The structured face: the array, plus what the client must enforce itself.

        ``permission_surface_owned`` is accepted and ignored (it arrives in
        ``kwargs``), which is the documented behaviour for a mirror outside
        claude's class: a custom backend's permission surface is its own
        process, not a file Crew may or may not own, and its asking posture
        is established by its registration's routing, not by this projection.
        """
        return custom_projection(
            self.backend,
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stub_elements,
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
        )

    def session_params(
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
    ) -> dict[str, Any]:
        """The wire face: the ``mcpServers`` array for this custom session."""
        return dict(
            self.session_projection(
                agent,
                stub_server_names=stub_server_names,
                stub_elements=stub_elements,
                work_dir=work_dir,
                session_key=session_key,
                channel_id=channel_id,
                session_token=session_token,
            ).params
        )
