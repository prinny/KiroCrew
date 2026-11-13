"""Custom ACP backend registration: the generic mechanism, seam by seam.

Every test registers through the same call the config loader makes
(``config.sections.normalize_custom_backends``) or through the registry with
the production policy wiring, so what is pinned here is the behaviour an
operator's ``agent.custom_backends`` block actually gets -- never a test-only
registration path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kiro_crew.agent_sdk import backend_install
from kiro_crew.agent_sdk import backends as backends_mod
from kiro_crew.agent_sdk import capabilities, custom_backends, host_auth, tool_gate
from kiro_crew.agent_sdk.backends import Routing


def _policy() -> dict:
    """The production policy wiring, assembled the way the loader assembles it."""
    from kiro_crew.agent_sdk.backends import ACP_BACKENDS_KNOWN, POLICY_ID_BY_BACKEND
    from kiro_crew.config import sections_agent_env

    return {
        "reserved_ids": frozenset(ACP_BACKENDS_KNOWN) | frozenset(POLICY_ID_BY_BACKEND.values()),
        "name_refusal": sections_agent_env.agent_env_refusal,
        "reserved_name_refusal": sections_agent_env.agent_env_reserved_refusal,
        "is_credential_name": lambda name: bool(
            sections_agent_env.AGENT_ENV_CREDENTIAL_CLASS.search(name)
        ),
        "scrub_removes": sections_agent_env.agent_env_scrub_removes,
    }


def _register(raw) -> tuple:
    return custom_backends.register_custom_backends(raw, **_policy())


@pytest.fixture(autouse=True)
def _clean_registry():
    custom_backends.clear_custom_backends()
    backend_install.clear_probe_cache()
    yield
    for backend_id in custom_backends.custom_backend_ids():
        backends_mod.unregister_custom_selectable_backend(backend_id)
    custom_backends.clear_custom_backends()
    backend_install.clear_probe_cache()


BEDROCK = {
    "label": "Bedrock (ACP adapter)",
    "command": "kirocrew-bedrock-acp",
    "args": ["--region", "us-west-2"],
    "install_hint": "pip install kirocrew-bedrock-acp",
    "env": {"AWS_PROFILE": "default"},
    "permission": {
        "kind": "session_config",
        "option_id": "approval_mode",
        "value": "promptUnmatched",
    },
    "mcp": "array",
    "credentials": {"posture": "adapter_chain", "reads": ["~/.aws"]},
    "capabilities": {"model_option_id": "model", "report_usage": True},
}


# ── Registration and validation ──


def test_valid_entry_registers_with_its_declared_facts():
    errors = _register({"bedrock": BEDROCK})
    assert errors == ()
    spec = custom_backends.spec_for("bedrock")
    assert spec is not None
    assert spec.command == "kirocrew-bedrock-acp"
    assert spec.args == ("--region", "us-west-2")
    assert spec.env_map == {"AWS_PROFILE": "default"}
    assert spec.protocol_version == 1
    assert custom_backends.custom_backend_ids() == ("bedrock",)
    assert custom_backends.is_custom_backend("bedrock")
    assert not custom_backends.is_custom_backend("claude")


def test_reregistration_replaces_the_table():
    assert _register({"bedrock": BEDROCK}) == ()
    errors = _register({"other-agent": {"command": "other-acp"}})
    assert errors == ()
    assert custom_backends.spec_for("bedrock") is None
    assert custom_backends.custom_backend_ids() == ("other-agent",)


@pytest.mark.parametrize("bad_id", ["Bedrock", "1st", "-lead", "a" * 33, "has space", ""])
def test_id_grammar_is_refused(bad_id):
    errors = _register({bad_id: {"command": "x-acp"}})
    assert errors and bad_id not in custom_backends.custom_backend_ids()


def test_builtin_id_cannot_be_redeclared():
    errors = _register({"claude": {"command": "x-acp"}})
    assert errors and "claude" in errors[0]
    assert not custom_backends.is_custom_backend("claude")


def test_missing_command_is_refused_and_names_the_key():
    errors = _register({"bedrock": {"label": "no command"}})
    assert errors and "'command'" in errors[0]
    assert not custom_backends.is_custom_backend("bedrock")


def test_one_bad_entry_does_not_block_a_valid_sibling():
    errors = _register({"bedrock": BEDROCK, "broken": {"args": []}})
    assert errors and len(errors) == 1
    assert custom_backends.is_custom_backend("bedrock")
    assert not custom_backends.is_custom_backend("broken")


# ── The env rules (the security core) ──


def test_env_name_the_scrub_removes_is_refused():
    entry = dict(BEDROCK, env={"AWS_SECRET_ACCESS_KEY": "secret://aws-key"})
    errors = _register({"bedrock": entry})
    assert errors and "AWS_SECRET_ACCESS_KEY" in errors[0]
    assert not custom_backends.is_custom_backend("bedrock")


def test_env_credential_name_with_plaintext_value_is_refused():
    entry = dict(BEDROCK, env={"MY_API_KEY": "plaintext-value"})
    errors = _register({"bedrock": entry})
    assert errors and "MY_API_KEY" in errors[0]


def test_env_credential_name_with_secret_reference_is_accepted():
    entry = dict(BEDROCK, env={"MY_API_KEY": "secret://my-key"})
    assert _register({"bedrock": entry}) == ()
    assert custom_backends.spec_for("bedrock").env_map == {"MY_API_KEY": "secret://my-key"}


def test_env_crew_owned_name_is_refused():
    entry = dict(BEDROCK, env={"PATH": "/tmp"})
    errors = _register({"bedrock": entry})
    assert errors and "PATH" in errors[0]


def test_env_name_outside_the_grammar_is_refused():
    entry = dict(BEDROCK, env={"1BAD": "x"})
    errors = _register({"bedrock": entry})
    assert errors and "1BAD" in errors[0]


# Regression (validator finding F1): the credential-shaped branch used to
# return accept for a secret:// value BEFORE the Crew-owned refusal was
# consulted, so a Crew-owned credential-shaped name registered as a
# reference and spawn resolved it into the child env. Design §3 rule 4
# refuses Crew-owned names with no value-shape exception.
_CREW_OWNED_CREDENTIAL_NAMES = ("KIROCREW_API_KEY", "PYTHON_API_TOKEN", "KIRO_API_TOKEN")


@pytest.mark.parametrize("name", _CREW_OWNED_CREDENTIAL_NAMES)
def test_env_crew_owned_credential_name_refused_as_secret_reference(name):
    entry = dict(BEDROCK, env={name: "secret://x"})
    errors = _register({"bedrock": entry})
    assert errors and name in errors[0]
    assert not custom_backends.is_custom_backend("bedrock")


@pytest.mark.parametrize("name", _CREW_OWNED_CREDENTIAL_NAMES)
def test_env_crew_owned_credential_name_refused_as_plaintext(name):
    entry = dict(BEDROCK, env={name: "plaintext-value"})
    errors = _register({"bedrock": entry})
    assert errors and name in errors[0]
    assert not custom_backends.is_custom_backend("bedrock")


def test_env_crew_owned_name_refused_as_secret_reference():
    # PATH is not credential-shaped; its refusal already applied to every
    # shape, and still does after the ordering fix.
    entry = dict(BEDROCK, env={"PATH": "secret://x"})
    errors = _register({"bedrock": entry})
    assert errors and "PATH" in errors[0]


# Design §3's stated consequence -- "AWS credentials can never be bound
# through this surface in a form the adapter could be handed by Crew" --
# covers AWS_ACCESS_KEY_ID too: the scrub sets refuse the AWS_SECRET /
# AWS_SESSION family, and the key-id name is refused here for both shapes.
def test_env_aws_access_key_id_refused_as_secret_reference():
    entry = dict(BEDROCK, env={"AWS_ACCESS_KEY_ID": "secret://x"})
    errors = _register({"bedrock": entry})
    assert errors and "AWS_ACCESS_KEY_ID" in errors[0]
    assert not custom_backends.is_custom_backend("bedrock")


def test_env_aws_access_key_id_refused_as_plaintext():
    entry = dict(BEDROCK, env={"AWS_ACCESS_KEY_ID": "plaintext-value"})
    errors = _register({"bedrock": entry})
    assert errors and "AWS_ACCESS_KEY_ID" in errors[0]
    assert not custom_backends.is_custom_backend("bedrock")


def test_env_non_crew_owned_credential_name_still_accepted_as_secret_reference():
    # Guard against over-refusal: a credential-shaped name that is neither
    # Crew-owned/reserved nor scrub-listed is still accepted as secret://.
    entry = dict(BEDROCK, env={"VENDOR_API_TOKEN": "secret://x"})
    assert _register({"bedrock": entry}) == ()
    assert custom_backends.spec_for("bedrock").env_map == {"VENDOR_API_TOKEN": "secret://x"}


# ── Permission declarations ──


def test_session_config_without_option_and_value_is_refused():
    entry = dict(BEDROCK, permission={"kind": "session_config"})
    errors = _register({"bedrock": entry})
    assert errors and "permission" in errors[0]


def test_verify_on_a_non_seeded_kind_is_refused():
    entry = dict(
        BEDROCK,
        permission={
            "kind": "asks",
            "verify": {"source": "config_option", "id": "approval_mode"},
        },
    )
    errors = _register({"bedrock": entry})
    assert errors


def test_verify_with_an_unimplemented_source_is_refused():
    entry = dict(
        BEDROCK,
        permission={
            "kind": "seeded",
            "value": "ask",
            "verify": {"source": "mode", "id": "mode"},
        },
    )
    errors = _register({"bedrock": entry})
    assert errors and "config_option" in errors[0]


def test_credential_leaf_escaping_home_is_refused():
    entry = dict(
        BEDROCK,
        credentials={"posture": "masked", "leaves": ["../outside"]},
    )
    errors = _register({"bedrock": entry})
    assert errors and "leaves" in errors[0]


# ── Routing and the tool gate ──


def test_session_config_routing_is_enforced_and_answers_its_option():
    assert _register({"bedrock": BEDROCK}) == ()
    assert backends_mod.routing_for("bedrock") is Routing.SESSION_CONFIG
    assert tool_gate.is_enforced("bedrock")
    assert tool_gate.permission_config_for("bedrock") == ("approval_mode", "promptUnmatched")
    assert tool_gate.label_for("bedrock") == "Bedrock (ACP adapter)"


def test_seeded_with_verify_is_verified_routing():
    entry = dict(
        BEDROCK,
        permission={
            "kind": "seeded",
            "value": "promptUnmatched",
            "seed_env": {"EXAMPLE_APPROVAL_MODE": "promptUnmatched"},
            "verify": {"source": "config_option", "id": "approval_mode"},
        },
    )
    assert _register({"bedrock": entry}) == ()
    assert backends_mod.routing_for("bedrock") is Routing.VERIFIED_SEEDED_SETTINGS
    assert tool_gate.is_enforced("bedrock")
    assert tool_gate.permission_setting_for("bedrock") == ("approval_mode", "promptUnmatched")
    assert tool_gate.seeded_setting_issue("bedrock", "promptUnmatched") == ""
    assert tool_gate.seeded_setting_issue("bedrock", "auto") != ""
    assert tool_gate.seeded_setting_issue("bedrock", None) != ""


def test_asks_is_declared_and_unenforced():
    entry = dict(BEDROCK, permission={"kind": "asks"})
    assert _register({"bedrock": entry}) == ()
    assert backends_mod.routing_for("bedrock") is Routing.SEEDED_SETTINGS
    assert not tool_gate.is_enforced("bedrock")


def test_exposed_leaf_is_contained_by_the_mask():
    entry = dict(
        BEDROCK,
        credentials={
            "posture": "masked",
            "leaves": [".aws/credentials"],
            "exposed_leaves": [".aws/config", ".ssh/id_rsa"],
        },
    )
    assert _register({"bedrock": entry}) == ()
    home = Path.home()
    hidden = (str(home / ".aws"),)
    exposed = tool_gate.adapter_expose_files("bedrock", hidden)
    # .aws/config sits under the mask and is re-exposed; .ssh/id_rsa sits
    # under nothing this mask hides, so the gate drops it rather than
    # honouring the declaration.
    assert str(home / ".aws" / "config") in exposed
    assert str(home / ".ssh" / "id_rsa") not in exposed


# ── Selectability and construction ──


def test_loader_normalize_registers_and_withdraws_selectability():
    from kiro_crew.config import sections

    loaded = sections.normalize_custom_backends({"bedrock": BEDROCK})
    assert set(loaded) == {"bedrock"}
    assert "bedrock" in backends_mod.selectable_backends()
    assert "bedrock" in backends_mod.registered_backends()

    loaded = sections.normalize_custom_backends({})
    assert loaded == {}
    assert "bedrock" not in backends_mod.selectable_backends()
    assert not custom_backends.is_custom_backend("bedrock")


def test_a_config_may_select_the_backend_it_registers():
    """Registration runs before the backend choice in the same document is read.

    Otherwise a config whose ``acp_backend`` names the custom backend its own
    ``custom_backends`` block declares would have that choice degraded to the
    floor for not being selectable yet -- the registration would succeed and
    the selection silently not.
    """
    from kiro_crew.config.loader_builders import _build_agent_config

    agent = _build_agent_config({"acp_backend": "bedrock", "custom_backends": {"bedrock": BEDROCK}})
    assert agent.acp_backend == "bedrock"
    assert set(agent.custom_backends) == {"bedrock"}


def test_registrar_refuses_an_unregistered_id():
    with pytest.raises(ValueError):
        backends_mod.register_custom_selectable_backend("never-registered")


def test_registrar_cannot_unregister_a_builtin():
    with pytest.raises(ValueError):
        backends_mod.unregister_custom_selectable_backend("claude")


def test_provider_construction_accepts_a_custom_backend():
    from kiro_crew.providers.acp import AcpProvider

    assert _register({"bedrock": BEDROCK}) == ()
    provider = AcpProvider(work_dir=None, acp_backend="bedrock")
    assert provider is not None
    with pytest.raises(ValueError):
        AcpProvider(work_dir=None, acp_backend="no-such-backend")


def test_provider_label_is_the_custom_id_itself():
    from kiro_crew.agent_sdk.drivers.acp import provider_label_for_backend
    from kiro_crew.providers.acp import provider_label

    assert _register({"bedrock": BEDROCK}) == ()
    assert provider_label_for_backend("bedrock") == "bedrock"
    assert provider_label(object()) == "acp"  # a non-provider still defaults


# ── Host auth, install probe, capabilities, mirror ──


def test_adapter_chain_declaration_is_ambient_and_fences_nothing():
    assert _register({"bedrock": BEDROCK}) == ()
    declaration = host_auth.declaration_for("bedrock")
    assert declaration.entitlement_source == host_auth.ENTITLEMENT_AMBIENT_ENVIRONMENT
    assert declaration.credential_leaves == ()
    assert declaration.adapter_own_leaves == ()
    assert not declaration.host_logout_retires_children


def test_masked_declaration_fences_and_spares_its_own_leaves():
    entry = dict(
        BEDROCK,
        credentials={"posture": "masked", "leaves": [".example/credentials"]},
    )
    assert _register({"bedrock": entry}) == ()
    declaration = host_auth.declaration_for("bedrock")
    assert declaration.entitlement_source == host_auth.ENTITLEMENT_OWN_CREDENTIAL_FILE
    assert declaration.credential_leaves == (".example/credentials",)
    assert declaration.adapter_own_leaves == (".example/credentials",)


def test_install_probe_answers_from_the_registration():
    entry = dict(BEDROCK, command="sh", install_hint="install sh")
    assert _register({"bedrock": entry}) == ()
    state = backend_install.probe_backend("bedrock")
    assert state.installed == backend_install.INSTALLED

    custom_backends.clear_custom_backends()
    backend_install.clear_probe_cache()
    entry = dict(BEDROCK, command="no-such-binary-xyz-123", install_hint="install the thing")
    assert _register({"bedrock": entry}) == ()
    state = backend_install.probe_backend("bedrock")
    assert state.installed == backend_install.MISSING
    assert state.install_command == "install the thing"
    assert state.missing_components == ("no-such-binary-xyz-123",)


def test_capabilities_are_the_declaration():
    assert _register({"bedrock": BEDROCK}) == ()
    caps = capabilities.capabilities_for("bedrock")
    assert caps.acp_client_spawnable
    assert caps.resolves_model_from_advertised_list
    assert not caps.effort_via_config_option
    assert caps.model_id_namespace == "bedrock"
    unknown = capabilities.capabilities_for("never-registered")
    assert not unknown.acp_client_spawnable


def test_mirror_projection_for_array_and_withheld():
    from kiro_crew.providers.mirrors import has_mirror, mirror_for, projection_for
    from kiro_crew.providers.mirrors.custom import CustomMirror
    from kiro_crew.providers.mirrors.registry import PerToolDeny, ProjectionKind

    assert _register({"bedrock": BEDROCK}) == ()
    mirror = mirror_for("bedrock")
    assert isinstance(mirror, CustomMirror)
    assert has_mirror("bedrock")
    projection = projection_for("bedrock")
    assert projection.kind is ProjectionKind.MIRROR
    assert projection.per_tool_deny is PerToolDeny.WHOLE_SERVER
    rulings = mirror.rulings()
    assert set(rulings) == set(mirror.rulings().keys())
    from kiro_crew.providers.mirrors.base import Concern

    assert set(rulings) == set(Concern)

    custom_backends.clear_custom_backends()
    entry = dict(BEDROCK, mcp="withheld")
    assert _register({"bedrock": entry}) == ()
    assert mirror_for("bedrock") is None
    assert not has_mirror("bedrock")
    assert projection_for("bedrock").kind is ProjectionKind.NO_CHANNEL


def test_process_markers_include_the_declared_command():
    from kiro_crew.agent_sdk.backends import agent_process_markers

    entry = dict(BEDROCK, command="/opt/tools/kirocrew-bedrock-acp")
    assert _register({"bedrock": entry}) == ()
    assert "kirocrew-bedrock-acp" in agent_process_markers()


# ── Spawn ──


class _FakeSession:
    def __init__(self):
        self._extra_env: dict = {}
        self.prepared = False

    async def _prepare_session_mcp(self) -> None:
        self.prepared = True


def _spawn_context(session):
    from kiro_crew.acp.harness.base import SpawnContext

    return SpawnContext(
        agent="",
        work_dir=None,
        model=None,
        environ={},
        home=Path.home(),
        session=session,
    )


async def _plan_for(entry):
    from kiro_crew.acp.harness import process_adapter_for

    assert _register({"bedrock": entry}) == ()
    adapter = process_adapter_for("bedrock")
    assert adapter is not None
    session = _FakeSession()
    plan = await adapter.resolve_spawn(_spawn_context(session))
    return adapter, session, plan


@pytest.mark.asyncio
async def test_spawn_plan_carries_the_declared_argv():
    entry = dict(
        BEDROCK,
        command=sys.executable,
        args=["-m", "fake_acp"],
        permission={"kind": "asks"},
        credentials={"posture": "adapter_chain"},
    )
    adapter, session, plan = await _plan_for(entry)
    assert plan.argv == [sys.executable, "-m", "fake_acp"]
    assert plan.extra_hidden_dirs == ()
    assert session.prepared  # mcp: "array" warms the session array


@pytest.mark.asyncio
async def test_spawn_env_carries_bindings_and_seed_only():
    entry = dict(
        BEDROCK,
        command=sys.executable,
        env={"AWS_PROFILE": "default"},
        permission={
            "kind": "seeded",
            "value": "ask",
            "seed_env": {"EXAMPLE_MODE": "ask"},
        },
        credentials={"posture": "adapter_chain"},
    )
    adapter, _session, _plan = await _plan_for(entry)
    env: dict = {}
    adapter.apply_spawn_env(env)
    assert env == {"AWS_PROFILE": "default", "EXAMPLE_MODE": "ask"}


@pytest.mark.asyncio
async def test_unresolvable_secret_reference_fails_the_spawn_closed(monkeypatch):
    import kiro_crew.secrets as secrets_mod
    from kiro_crew.acp.transport_errors import AcpError

    class _EmptyVault:
        def __init__(self, *args, **kwargs):
            pass

        def get_many(self, names):
            return {name: None for name in names}

    monkeypatch.setattr(secrets_mod, "SecretVault", _EmptyVault)
    missing_ref = "secret:/" + "/missing-key"
    entry = dict(
        BEDROCK,
        command=sys.executable,
        env={"MY_API_KEY": missing_ref},
        permission={"kind": "asks"},
        credentials={"posture": "adapter_chain"},
    )
    assert _register({"bedrock": entry}) == ()
    from kiro_crew.acp.harness import process_adapter_for

    adapter = process_adapter_for("bedrock")
    with pytest.raises(AcpError) as excinfo:
        await adapter.resolve_spawn(_spawn_context(_FakeSession()))
    assert "MY_API_KEY" in str(excinfo.value)
    assert "missing-key" not in str(excinfo.value)


def test_process_adapter_for_unknown_id_is_none():
    from kiro_crew.acp.harness import process_adapter_for

    assert process_adapter_for("never-registered") is None


# ── Backend card ──


def test_card_is_the_declaration_with_unmeasured_lines_beyond_it():
    from kiro_crew.agent_sdk import backend_cards
    from kiro_crew.config import sections

    sections.normalize_custom_backends({"bedrock": BEDROCK})
    card = backend_cards.card_for("bedrock")
    lines = {line.id: line for line in card.capabilities}
    # The declaration answers these three lines, and the card reads them
    # from it: the MCP array carries Crew's tools, the named model option
    # switches models, and no effort option was declared.
    assert lines["crew_tools"].available and lines["crew_tools"].measured
    assert lines["model_switch"].available and lines["model_switch"].measured
    assert not lines["reasoning_effort"].available
    assert lines["reasoning_effort"].measured
    # Lines the declaration does not reach are unmeasured, never a measured
    # absence: Crew has run nothing against an agent it does not ship.
    for line_id in ("mid_turn_steer", "manual_compact", "member_thread_tools"):
        assert not lines[line_id].available
        assert not lines[line_id].measured
        assert lines[line_id].unmeasured_reason == "declared_not_measured"
    details = {line.id: line for line in card.details}
    assert not details["member_saved_agent"].measured
    # The graded lines come from the same dynamic answers as every seam.
    assert card.tool_approval == backends_mod.routing_for("bedrock").value
    assert card.offered_by_build
    assert card.policy_id == "bedrock"
    # An id nobody registered keeps the fail-closed measured reading.
    unknown = backend_cards.card_for("never-registered")
    assert all(line.measured for line in unknown.capabilities)
    assert not unknown.offered_by_build


# ── Generality: a second consumer through config only ──


ECHO_AGENT = {
    "label": "Echo (generic ACP agent)",
    "command": "echo-acp",
    "install_hint": "pip install echo-acp",
    "permission": {
        "kind": "seeded",
        "value": "ask",
        "seed_env": {"ECHO_MODE": "ask"},
        "verify": {"source": "config_option", "id": "mode"},
    },
    "mcp": "withheld",
    "credentials": {"posture": "masked", "leaves": [".echo-acp/credentials"]},
    "capabilities": {},
}


def test_a_second_consumer_registers_through_the_same_surface():
    """A different family, posture and permission kind, config only.

    Both registrations land in one call, and every seam answers each id
    from its own declaration -- the mechanism is shared, no core code
    names either consumer.
    """
    from kiro_crew.config import sections
    from kiro_crew.providers.mirrors import mirror_for

    sections.normalize_custom_backends({"bedrock": BEDROCK, "echo-agent": ECHO_AGENT})
    assert set(custom_backends.custom_backend_ids()) == {"bedrock", "echo-agent"}
    assert "bedrock" in backends_mod.selectable_backends()
    assert "echo-agent" in backends_mod.selectable_backends()
    # Permission kinds map to different routings per declaration.
    assert backends_mod.routing_for("bedrock") is Routing.SESSION_CONFIG
    assert backends_mod.routing_for("echo-agent") is Routing.VERIFIED_SEEDED_SETTINGS
    assert tool_gate.label_for("echo-agent") == "Echo (generic ACP agent)"
    assert tool_gate.permission_setting_for("echo-agent") == ("mode", "ask")
    # Credential postures differ: ambient chain vs a fenced own leaf.
    bedrock_decl = host_auth.declaration_for("bedrock")
    echo_decl = host_auth.declaration_for("echo-agent")
    assert bedrock_decl.entitlement_source == host_auth.ENTITLEMENT_AMBIENT_ENVIRONMENT
    assert echo_decl.entitlement_source == host_auth.ENTITLEMENT_OWN_CREDENTIAL_FILE
    assert echo_decl.credential_leaves == (".echo-acp/credentials",)
    # MCP delivery differs: a mirror for the array, none for withheld.
    assert mirror_for("bedrock") is not None
    assert mirror_for("echo-agent") is None
    # Declared capabilities differ, and the cards read the same split.
    assert capabilities.capabilities_for("bedrock").resolves_model_from_advertised_list
    assert not capabilities.capabilities_for("echo-agent").resolves_model_from_advertised_list
    from kiro_crew.agent_sdk import backend_cards

    bedrock_lines = {line.id: line for line in backend_cards.card_for("bedrock").capabilities}
    echo_lines = {line.id: line for line in backend_cards.card_for("echo-agent").capabilities}
    assert bedrock_lines["crew_tools"].available
    assert not echo_lines["crew_tools"].available and echo_lines["crew_tools"].measured
    assert not echo_lines["model_switch"].available and echo_lines["model_switch"].measured


@pytest.mark.asyncio
async def test_second_consumer_spawn_plan_matches_its_declaration(monkeypatch):
    """The generic launch plans the second consumer from its declaration.

    Same generic adapter as the first consumer, a different id: the argv
    is the declared command/args, the masked posture carries the
    preflight mask into the plan, the seeded posture's seed env is what
    apply_spawn_env puts on the child, and MCP withheld means the
    session's MCP array is never warmed. The sandbox floor itself is
    stubbed (the harness-contract precedent): it is a host fact, not a
    fact of the declaration this test pins.
    """
    from kiro_crew.acp import launch as launch_mod
    from kiro_crew.acp.harness import process_adapter_for

    entry = dict(ECHO_AGENT, command=sys.executable, args=["-m", "echo_acp"])
    assert _register({"echo-agent": entry}) == ()
    adapter = process_adapter_for("echo-agent")
    assert adapter is not None
    hidden = (str(Path.home() / ".echo-acp"),)
    monkeypatch.setattr(launch_mod, "_sandbox_preflight", lambda backend, mode: hidden)
    session = _FakeSession()
    plan = await adapter.resolve_spawn(_spawn_context(session))
    assert plan.argv == [sys.executable, "-m", "echo_acp"]
    assert plan.stderr_label == "Echo (generic ACP agent)"
    assert plan.extra_hidden_dirs == hidden
    assert not session.prepared  # mcp: "withheld" warms no session MCP array
    env: dict = {}
    adapter.apply_spawn_env(env)
    assert env == {"ECHO_MODE": "ask"}  # the declared permission seed env
