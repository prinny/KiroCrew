"""Operator-registered custom ACP backends: the dynamic registry.

A backend Crew can spawn but this build does not ship is declared by the
operator in ``agent.custom_backends`` (config surface) and lands here, parsed
and validated, as one :class:`CustomBackendSpec` per id. Every per-backend
surface that used to need a code row per harness -- routing, the tool gate's
labels and credential leaves, the host-auth declaration, the install probe,
the backend card, the MCP projection -- answers for a custom id by consulting
this registry at the seam where it already special-cases a backend. There is
no second dispatch architecture: one dynamic fallback per existing seam.

This module is a LEAF, like ``backends_launch``: it imports nothing from
``kiro_crew.acp``, ``kiro_crew.config`` or ``kiro_crew.platform`` at module
scope, because the credential floor and the config loader both read it while
their own modules are still being constructed. Host policy that lives outside
this leaf -- the reserved id sets, the ``agent.env`` name rules, the spawn
scrub -- is therefore PASSED IN by the caller of
:func:`register_custom_backends` (the config loader), never imported.

Registration is fail-closed and key-only in its errors: a malformed entry is
refused with a message naming the backend id and the offending KEY, never a
value, and the entry is not registered. Valid siblings still register.
Re-registration replaces the whole table atomically, so an id removed from
the config stops being a backend on the next load.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Mapping, Optional, Tuple

__all__ = [
    "MCP_ARRAY",
    "MCP_WITHHELD",
    "PERMISSION_ASKS",
    "PERMISSION_SEEDED",
    "PERMISSION_SESSION_CONFIG",
    "POSTURE_ADAPTER_CHAIN",
    "POSTURE_MASKED",
    "SECRET_REF_PREFIX",
    "CustomBackendSpec",
    "clear_custom_backends",
    "custom_backend_ids",
    "is_custom_backend",
    "register_custom_backends",
    "spec_for",
    "validate_custom_backends",
]

#: An env value that names a vault entry instead of carrying a secret.
SECRET_REF_PREFIX = "secret://"

#: The id grammar. Lowercase, starts with a letter, at most 32 characters:
#: the same shape every shipped backend id already has, so a custom id is
#: spellable everywhere a built-in one is (policy files, config, dashboards).
_ID_GRAMMAR = re.compile(r"[a-z][a-z0-9-]{0,31}")

_ENV_NAME_GRAMMAR = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Permission postures an operator may declare.
PERMISSION_SESSION_CONFIG = "session_config"
PERMISSION_SEEDED = "seeded"
PERMISSION_ASKS = "asks"
_PERMISSION_KINDS = frozenset({PERMISSION_SESSION_CONFIG, PERMISSION_SEEDED, PERMISSION_ASKS})

#: MCP delivery an operator may declare.
MCP_ARRAY = "array"
MCP_WITHHELD = "withheld"

#: Credential postures an operator may declare.
POSTURE_MASKED = "masked"
POSTURE_ADAPTER_CHAIN = "adapter_chain"

#: The AWS access key id name. The spawn scrub already refuses the
#: ``AWS_SECRET`` / ``AWS_SESSION`` family, but this name is in neither set
#: and is credential-shaped, so without an explicit refusal it could be
#: bound as a ``secret://`` reference. Design §3 states the consequence
#: this surface must hold by construction: "AWS credentials can never be
#: bound through this surface in a form the adapter could be handed by
#: Crew" -- so this name is refused for literal and ``secret://`` values
#: alike, like every other AWS credential name.
_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"

#: The one verify source generic code implements today: read the named ACP
#: config option's current value back off ``session/new`` and compare it to
#: the required value, through ``tool_gate.seeded_setting_issue``.
VERIFY_CONFIG_OPTION = "config_option"


@dataclass(frozen=True)
class CustomBackendSpec:
    """One operator-declared backend, parsed and validated.

    Frozen, and every field is data: consumers read this from inside floors
    and registries that must not reach a resolver, the config, or the
    filesystem. ``env`` values may carry ``secret://`` references; resolving
    them is the spawn path's job, at spawn time, fail-closed.
    """

    backend_id: str
    label: str
    command: str
    args: Tuple[str, ...] = ()
    install_hint: str = ""
    protocol_version: int | str = 1
    env: Tuple[Tuple[str, str], ...] = ()
    permission_kind: str = PERMISSION_ASKS
    permission_option_id: str = ""
    permission_value: str = ""
    permission_seed_env: Tuple[Tuple[str, str], ...] = ()
    permission_verify_id: str = ""
    mcp: str = MCP_ARRAY
    credentials_posture: str = POSTURE_MASKED
    credentials_reads: Tuple[str, ...] = ()
    credentials_leaves: Tuple[str, ...] = ()
    credentials_exposed_leaves: Tuple[str, ...] = ()
    credentials_home_override_env: str = ""
    sign_in_hint: str = ""
    model_option_id: str = ""
    effort_option_id: str = ""
    report_usage: bool = False

    @property
    def env_map(self) -> dict:
        """The declared env bindings as a plain dict (values unresolved)."""
        return dict(self.env)

    @property
    def seed_env_map(self) -> dict:
        """The permission seed bindings as a plain dict (values unresolved)."""
        return dict(self.permission_seed_env)


_SPECS: dict = {}


def spec_for(backend: str) -> Optional[CustomBackendSpec]:
    """The spec registered for *backend*, or ``None`` when it is not custom."""
    if not isinstance(backend, str):
        return None
    return _SPECS.get(backend)


def is_custom_backend(backend: str) -> bool:
    """Whether *backend* is an operator-registered custom backend id."""
    return spec_for(backend) is not None


def custom_backend_ids() -> Tuple[str, ...]:
    """Every registered custom backend id, sorted."""
    return tuple(sorted(_SPECS))


def clear_custom_backends() -> None:
    """Drop every registration. For tests; production re-registers by load."""
    _SPECS.clear()


# ── Validation ──
#
# Every helper returns a refusal phrase or ``None``. A phrase names the key
# that failed and never a value, matching the ``agent.env`` rule's discipline
# (``config.sections_agent_env._agent_env_entry_refusal``), whose per-entry
# decision this surface deliberately shares rather than restating.


def _leaf_refusal(leaf: object) -> Optional[str]:
    """Why *leaf* is not a usable home-relative credential leaf, or ``None``.

    The floor's spelling: POSIX separators, relative to the home directory,
    no ``..`` -- a declaration may name a file under the operator's home, never
    a path out of it.
    """
    if not isinstance(leaf, str) or not leaf.strip():
        return "it must be a non-empty home-relative path"
    if "\x00" in leaf:
        return "it contains a NUL"
    path = PurePosixPath(leaf)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return "it must be a home-relative path with no '..' segment"
    return None


def _string_list(raw: object) -> Optional[Tuple[str, ...]]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for item in raw:
        if not isinstance(item, str) or "\x00" in item:
            return None
        out.append(item)
    return tuple(out)


def _env_refusal(
    name: object,
    value: object,
    *,
    name_refusal: Callable[[str], Optional[str]],
    is_credential_name: Callable[[str], bool],
    scrub_removes: Callable[[str], bool],
    reserved_name_refusal: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[str]:
    """Why one env binding is refused, or ``None`` when it is accepted.

    The ``agent.env`` rule, shared, applied in the design's order (§3):

    1. grammar and value shape;
    2. a name the spawn scrub removes is refused, for literal and
       ``secret://`` values alike (rule 2);
    3. ``AWS_ACCESS_KEY_ID`` is refused for both shapes (see
       ``_AWS_ACCESS_KEY_ID``: §3's consequence is that AWS credentials
       can never be bound through this surface in a form the adapter
       could be handed by Crew);
    4. Crew-owned / reserved names are refused, for literal and
       ``secret://`` values alike (rule 4) -- this refusal is consulted
       BEFORE the credential-shaped rule below, so a credential-shaped
       name that is also Crew-owned is never accepted as a reference;
    5. a credential-shaped name is refused for plaintext and accepted
       only as a ``secret://`` vault reference (rule 3, the deepseek
       provider-env precedent), and then only when rules 2-4 did not
       refuse the name.

    Plaintext under a credential-shaped name is refused, so a custom
    backend's config can never become a second, unguarded place a secret
    is stored.
    """
    if not isinstance(name, str) or not _ENV_NAME_GRAMMAR.fullmatch(name):
        return "the name is not an environment-variable name ([A-Za-z_][A-Za-z0-9_]*)"
    if not isinstance(value, str) or "\x00" in value:
        return "the value must be a string with no NUL"
    try:
        os.fsencode(value)
    except UnicodeEncodeError:
        return "the value cannot be encoded for a process environment"
    if scrub_removes(name):
        return "the agent environment scrub removes that name, so the binding would never reach the child"
    if name.upper() == _AWS_ACCESS_KEY_ID:
        return (
            "AWS credentials can never be bound through this surface in a form "
            "the adapter could be handed by Crew, so that name is refused for "
            "literal and secret:// values alike"
        )
    if reserved_name_refusal is not None:
        reserved_reason = reserved_name_refusal(name)
    else:
        # Legacy wiring passes the full ``agent.env`` refusal, whose
        # credential-class phrase ("the name looks like a credential...")
        # is the one refusal this surface re-decides by value shape in
        # step 5; every other phrase it returns is a Crew-owned /
        # reserved-name refusal and applies here, to every value shape.
        full_reason = name_refusal(name)
        reserved_reason = (
            None
            if (full_reason is not None and "looks like a credential" in full_reason)
            else full_reason
        )
    if reserved_reason is not None:
        return reserved_reason
    if is_credential_name(name):
        if value.startswith(SECRET_REF_PREFIX) and len(value) > len(SECRET_REF_PREFIX):
            return None
        return (
            "the name looks like a credential; give its value as a secret:// "
            "reference to a vault entry, never as plaintext"
        )
    return None


def _env_map_refusal(
    raw: object,
    *,
    name_refusal: Callable[[str], Optional[str]],
    is_credential_name: Callable[[str], bool],
    scrub_removes: Callable[[str], bool],
    reserved_name_refusal: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[Optional[Tuple[Tuple[str, str], ...]], Optional[str]]:
    """Parse one env mapping, or refuse it whole with the first reason."""
    if raw is None:
        return (), None
    if not isinstance(raw, Mapping):
        return None, "it must be an object of NAME: value strings"
    out = []
    for name, value in raw.items():
        reason = _env_refusal(
            name,
            value,
            name_refusal=name_refusal,
            is_credential_name=is_credential_name,
            scrub_removes=scrub_removes,
            reserved_name_refusal=reserved_name_refusal,
        )
        if reason is not None:
            return None, f"env name {name!r}: {reason}"
        out.append((name, value))
    return tuple(out), None


def _parse_spec(
    backend_id: str,
    raw: object,
    *,
    name_refusal: Callable[[str], Optional[str]],
    is_credential_name: Callable[[str], bool],
    scrub_removes: Callable[[str], bool],
    reserved_name_refusal: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[Optional[CustomBackendSpec], Optional[str]]:
    """One config entry -> one spec, or the refusal that names the bad key."""
    if not isinstance(raw, Mapping):
        return None, "the entry must be an object"

    def _str(key: str, default: str = "") -> Tuple[Optional[str], Optional[str]]:
        value = raw.get(key, default)
        if not isinstance(value, str) or "\x00" in value:
            return None, f"{key!r} must be a string with no NUL"
        return value, None

    label, err = _str("label", backend_id)
    if err is not None or label is None:
        return None, err or "'label' must be a string with no NUL"
    command, err = _str("command")
    if err is not None or command is None:
        return None, err or "'command' must be a string with no NUL"
    if not command.strip():
        return None, "'command' must name the program to spawn"
    args = _string_list(raw.get("args"))
    if args is None:
        return None, "'args' must be a list of strings"
    install_hint, err = _str("install_hint")
    if err is not None or install_hint is None:
        return None, err or "'install_hint' must be a string with no NUL"
    protocol_version = raw.get("protocol_version", 1)
    if isinstance(protocol_version, bool) or not isinstance(protocol_version, (int, str)):
        return None, "'protocol_version' must be an integer or a string"
    if isinstance(protocol_version, str) and not protocol_version.strip():
        return None, "'protocol_version' must not be empty"

    env, err = _env_map_refusal(
        raw.get("env"),
        name_refusal=name_refusal,
        is_credential_name=is_credential_name,
        scrub_removes=scrub_removes,
        reserved_name_refusal=reserved_name_refusal,
    )
    if err is not None or env is None:
        return None, err or "'env' must be an object mapping names to values"

    permission = raw.get("permission") or {}
    if not isinstance(permission, Mapping):
        return None, "'permission' must be an object"
    kind = permission.get("kind", PERMISSION_ASKS)
    if kind not in _PERMISSION_KINDS:
        return None, "'permission.kind' must be 'session_config', 'seeded' or 'asks'"
    option_id = permission.get("option_id", "")
    perm_value = permission.get("value", "")
    if not isinstance(option_id, str) or not isinstance(perm_value, str):
        return None, "'permission.option_id' and 'permission.value' must be strings"
    if kind == PERMISSION_SESSION_CONFIG and (not option_id or not perm_value):
        return None, (
            "'permission.kind' is 'session_config' but 'permission.option_id' "
            "and 'permission.value' are not both named"
        )
    seed_env, err = _env_map_refusal(
        permission.get("seed_env"),
        name_refusal=name_refusal,
        is_credential_name=is_credential_name,
        scrub_removes=scrub_removes,
        reserved_name_refusal=reserved_name_refusal,
    )
    if err is not None or seed_env is None:
        return None, (err or "'permission.seed_env' must be an object").replace(
            "env name", "permission.seed_env name", 1
        )
    verify_id = ""
    verify = permission.get("verify") or {}
    if verify:
        if not isinstance(verify, Mapping):
            return None, "'permission.verify' must be an object"
        if kind != PERMISSION_SEEDED:
            return None, "'permission.verify' is only meaningful with 'permission.kind' 'seeded'"
        source = verify.get("source", VERIFY_CONFIG_OPTION)
        if source != VERIFY_CONFIG_OPTION:
            return None, (
                "'permission.verify.source' must be 'config_option' for a custom "
                "backend; no other read-back is implemented generically"
            )
        verify_id = verify.get("id", "")
        if not isinstance(verify_id, str) or not verify_id:
            return None, "'permission.verify.id' must name the config option to read back"
        if not perm_value:
            return None, "'permission.value' must name the value the read-back must carry"

    mcp = raw.get("mcp", MCP_ARRAY)
    if mcp not in (MCP_ARRAY, MCP_WITHHELD):
        return None, "'mcp' must be 'array' or 'withheld'"

    credentials = raw.get("credentials") or {}
    if not isinstance(credentials, Mapping):
        return None, "'credentials' must be an object"
    posture = credentials.get("posture", POSTURE_MASKED)
    if posture not in (POSTURE_MASKED, POSTURE_ADAPTER_CHAIN):
        return None, "'credentials.posture' must be 'masked' or 'adapter_chain'"
    leaves_by_key = {}
    for key in ("reads", "leaves", "exposed_leaves"):
        values = _string_list(credentials.get(key))
        if values is None:
            return None, f"'credentials.{key}' must be a list of home-relative paths"
        for leaf in values:
            reason = _leaf_refusal(leaf.removeprefix("~/"))
            if reason is not None:
                return None, f"'credentials.{key}' entry {leaf!r}: {reason}"
        leaves_by_key[key] = tuple(v.removeprefix("~/") for v in values)
    home_override_env = credentials.get("home_override_env", "")
    if not isinstance(home_override_env, str) or (
        home_override_env and not _ENV_NAME_GRAMMAR.fullmatch(home_override_env)
    ):
        return None, "'credentials.home_override_env' must be an environment-variable name or empty"
    sign_in_hint = credentials.get("sign_in_hint", "")
    if not isinstance(sign_in_hint, str) or "\x00" in sign_in_hint:
        return None, "'credentials.sign_in_hint' must be a string with no NUL"

    capabilities = raw.get("capabilities") or {}
    if not isinstance(capabilities, Mapping):
        return None, "'capabilities' must be an object"
    model_option_id = capabilities.get("model_option_id", "")
    effort_option_id = capabilities.get("effort_option_id", "")
    if not isinstance(model_option_id, str) or not isinstance(effort_option_id, str):
        return (
            None,
            "'capabilities.model_option_id' and 'capabilities.effort_option_id' must be strings",
        )
    report_usage = capabilities.get("report_usage", False)
    if not isinstance(report_usage, bool):
        return None, "'capabilities.report_usage' must be a boolean"

    return (
        CustomBackendSpec(
            backend_id=backend_id,
            label=label or backend_id,
            command=command,
            args=args,
            install_hint=install_hint,
            protocol_version=protocol_version,
            env=env,
            permission_kind=kind,
            permission_option_id=option_id,
            permission_value=perm_value,
            permission_seed_env=seed_env,
            permission_verify_id=verify_id,
            mcp=mcp,
            credentials_posture=posture,
            credentials_reads=leaves_by_key["reads"],
            credentials_leaves=leaves_by_key["leaves"],
            credentials_exposed_leaves=leaves_by_key["exposed_leaves"],
            credentials_home_override_env=home_override_env,
            sign_in_hint=sign_in_hint,
            model_option_id=model_option_id,
            effort_option_id=effort_option_id,
            report_usage=report_usage,
        ),
        None,
    )


def _validate_all(
    raw: object,
    *,
    reserved_ids: frozenset,
    name_refusal: Callable[[str], Optional[str]],
    is_credential_name: Callable[[str], bool],
    scrub_removes: Callable[[str], bool],
    reserved_name_refusal: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[dict, Tuple[str, ...]]:
    """Parse every entry in *raw*: the specs that validate, and the refusals.

    Shared by registration (which stores the specs) and the config write
    floor (which only needs the refusals), so what a save turns away and what
    a load drops cannot disagree.
    """
    if raw is None:
        return {}, ()
    if not isinstance(raw, Mapping):
        return {}, (
            "agent.custom_backends must be an object mapping a backend id to its declaration",
        )
    errors = []
    specs = {}
    for backend_id, entry in raw.items():
        if not isinstance(backend_id, str) or not _ID_GRAMMAR.fullmatch(backend_id):
            errors.append(
                f"agent.custom_backends key {backend_id!r}: a backend id must match "
                "[a-z][a-z0-9-]{0,31}"
            )
            continue
        if backend_id in reserved_ids:
            errors.append(
                f"agent.custom_backends {backend_id!r}: that id is already a built-in "
                "backend or policy id and cannot be redeclared"
            )
            continue
        spec, err = _parse_spec(
            backend_id,
            entry,
            name_refusal=name_refusal,
            is_credential_name=is_credential_name,
            scrub_removes=scrub_removes,
            reserved_name_refusal=reserved_name_refusal,
        )
        if err is not None:
            errors.append(f"agent.custom_backends {backend_id!r}: {err}")
            continue
        specs[backend_id] = spec
    return specs, tuple(errors)


def validate_custom_backends(
    raw: object,
    *,
    reserved_ids: frozenset,
    name_refusal: Callable[[str], Optional[str]],
    is_credential_name: Callable[[str], bool],
    scrub_removes: Callable[[str], bool],
    reserved_name_refusal: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[str, ...]:
    """The refusals in *raw*, without registering anything.

    For the config publish floor (``config.loader._refuse_unpublishable``),
    which every config writer reaches, so an entry registration would refuse
    is turned away when it is saved rather than dropped at the next load.
    """
    _specs, errors = _validate_all(
        raw,
        reserved_ids=reserved_ids,
        name_refusal=name_refusal,
        is_credential_name=is_credential_name,
        scrub_removes=scrub_removes,
        reserved_name_refusal=reserved_name_refusal,
    )
    return errors


def register_custom_backends(
    raw: object,
    *,
    reserved_ids: frozenset,
    name_refusal: Callable[[str], Optional[str]],
    is_credential_name: Callable[[str], bool],
    scrub_removes: Callable[[str], bool],
    reserved_name_refusal: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[str, ...]:
    """Validate *raw* (the ``agent.custom_backends`` value) and register it.

    Replaces the whole table atomically: ids absent from *raw* stop being
    backends. Returns operator-readable errors, one per refused entry or
    structural problem, each naming the id and the offending key -- never a
    value. An empty tuple means every entry registered.

    *reserved_ids* are the ids config may not claim: every built-in backend
    id and every policy id, so a custom id can never shadow a shipped
    harness or a governance spelling. The three callables carry the host's
    env policy (see the module docstring); this leaf imports none of it.
    """
    specs, errors = _validate_all(
        raw,
        reserved_ids=reserved_ids,
        name_refusal=name_refusal,
        is_credential_name=is_credential_name,
        scrub_removes=scrub_removes,
        reserved_name_refusal=reserved_name_refusal,
    )
    _SPECS.clear()
    _SPECS.update(specs)
    return errors
