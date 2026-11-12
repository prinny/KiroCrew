"""muse-acp, launched one process per session by ``AcpClient``.

The adapter is a Node stdio shim published on npm as ``@brokkai/muse-acp``; the
shim execs a native binary that speaks ACP v1 on one side and drives
``muse serve`` (the Muse server protocol) on the other. Launching it is four
facts, all here: where its entry script is (the shared Node-adapter ladder,
cached for the gateway's life), the ``muse`` CLI the adapter is pointed at
through ``MUSE_CLI``, the approval-mode seed the session is born under, and
the per-session MCP array the session must have resolved first.

Two components, and the split is load-bearing for the install probe: the
adapter is what Crew spawns and the ``muse`` CLI is what the adapter spawns,
and either can be absent on its own. Neither is bundled with the other -- the
adapter never ships a ``muse`` binary -- so a missing CLI is a dead backend
named here, not an opaque failure inside the adapter.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from kiro_crew import platform_compat
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.harness.base import ProcessAdapter, SpawnContext, SpawnPlan
from kiro_crew.acp.transport_errors import AcpError
from kiro_crew.agent_sdk import tool_gate as acp_tool_gate
from kiro_crew.agent_sdk.backends import (
    ACP_BACKEND_MUSE,
    ACP_BACKEND_PROCESS_NAMES,
)
from kiro_crew.env import describe_search_path

# The client's logger name: these lines have always been filed under it.
logger = logging.getLogger("kiro_crew.acp.client")

__all__ = [
    "MUSE_ACP_BIN",
    "MUSE_ACP_NPM_PKG",
    "MUSE_ACP_VERSION",
    "MUSE_BIN",
    "MUSE_CLI_INSTALL_COMMAND",
    "MUSE_INSTALL_COMMAND",
    "MuseLaunch",
]

MUSE_ACP_BIN = ACP_BACKEND_PROCESS_NAMES[ACP_BACKEND_MUSE]
# On-disk name of the Muse backend CLI. The adapter spawns it as
# ``muse serve``; it does not search beyond PATH and its installer's own
# location, so Crew resolves it and names it explicitly through ``MUSE_CLI``
# (the env var the adapter itself reads), the way the claude launch names
# ``CLAUDE_CODE_EXECUTABLE``.
MUSE_BIN = "muse"
# npm package that provides the muse-acp binary. Deliberately NOT read out of
# ``ACP_BACKEND_NODE_ADAPTER_PACKAGES``: that table names the adapters Crew
# launches as ``node <package>/dist/index.js``, and this one's entry is a
# ``bin`` shim that execs a native binary, so it has no row there.
MUSE_ACP_NPM_PKG = "@brokkai/muse-acp"
# The adapter release this backend is wired and measured against. The install
# command pins it exactly, the way the packaging pin does: an unpinned
# ``npm i -g`` would let a later release move the ``approval_mode`` surface
# this backend's routing rests on without anything saying so.
MUSE_ACP_VERSION = "0.21.0"
# What installs the adapter half, from the pin.
MUSE_INSTALL_COMMAND = f"npm i -g {MUSE_ACP_NPM_PKG}@{MUSE_ACP_VERSION}"
# What installs the CLI half: Muse's own installer, the command the
# adapter's own documentation names.
MUSE_CLI_INSTALL_COMMAND = "curl -fsSL https://dev.meta.ai/install.sh | sh"
# Entry script relative to the installed package directory (its package.json
# "bin" field). Locates a copy under a project ``node_modules``.
_MUSE_ACP_PKG_ENTRY = Path(MUSE_ACP_NPM_PKG, "bin", "muse-acp.cjs")
# The completeness check for a project-local copy. The shared Node-adapter
# check asks for the adapter's ACP SDK import; this adapter is a native
# binary with no Node dependencies, so the thing a copy must hold to be
# runnable is its ``native`` directory -- the platform binaries the shim
# execs. A copy without it would die at spawn, after the ladder accepted it.
_MUSE_ACP_DEP_MARKER = Path("@brokkai", "muse-acp", "native")
# The adapter's own variable for the approval posture it forces on every
# ``muse serve`` it starts, resolved above the host's saved default.
_ENV_MUSE_APPROVAL_MODE = "MUSE_APPROVAL_MODE"
# The adapter's own variable naming the ``muse`` executable it spawns.
_ENV_MUSE_CLI = "MUSE_CLI"
# The operator override for the adapter's entry script, spelled the way the
# other adapters spell theirs.
_ENV_MUSE_ACP_BIN = "MUSE_ACP_BIN"

# Cache the PATH with the resolution result. A failed resolve is cached too, so
# recomputing PATH at the error site could report directories that were never searched.
_muse_acp_argv_cache: tuple[list[str] | None, str] | object = launch_mod._UNRESOLVED

_muse_bin_cache: tuple[str | None, str] | object = launch_mod._UNRESOLVED


def _resolve_vendored_muse_acp(pkg_dir: Path | None = None) -> str | None:
    """Return the path to a vendored muse-acp entry script, or None.

    The muse spelling of the ONE shared check,
    :func:`~kiro_crew.acp.launch._vendored_adapter_entry`:
    ``<root>/@brokkai/muse-acp/bin/muse-acp.cjs`` under each candidate
    ``node_modules`` root, accepted only when the package's ``native``
    directory is reachable from the entry's real location. *pkg_dir* is
    threaded through so tests can inject a fake package layout.
    """
    return launch_mod._vendored_adapter_entry(
        _MUSE_ACP_PKG_ENTRY, _MUSE_ACP_DEP_MARKER, pkg_dir=pkg_dir
    )


def _resolve_muse_acp_bin() -> tuple[list[str] | None, str]:
    """Find the muse-acp entry script and the PATH searched for it.

    The shared ladder (:func:`~kiro_crew.acp.launch._resolve_node_adapter_argv`)
    with this adapter's three parameters; ``MUSE_ACP_BIN`` is the override.
    """
    return launch_mod._resolve_node_adapter_argv(
        bin_name=MUSE_ACP_BIN,
        override_env=_ENV_MUSE_ACP_BIN,
        vendored_entry=_resolve_vendored_muse_acp,
    )


def _resolve_muse_bin() -> tuple[str | None, str]:
    """Find the ``muse`` CLI executable and the PATH searched for it.

    The plain-binary ladder's shape. The override rung is the ADAPTER'S
    variable: an operator who told muse-acp which ``muse`` to run has made
    the choice this resolver exists to honour, and naming exactly that binary
    back through the same variable keeps the choice rather than losing it.
    """
    # Read through the client at call time: its unit tests rebind
    # ``kiro_crew.acp.client.subprocess_mod`` under ``_mise_which``, and the client's
    # ``augmented_path`` for the directories searched.
    from kiro_crew.acp.client import _mise_which, augmented_path

    search_path = augmented_path(os.environ.get("PATH", ""))

    override = os.environ.get(_ENV_MUSE_CLI)
    if override:
        if platform_compat.is_executable_file(override):
            return launch_mod._normalize_exe_casing(override) or override, search_path
        on_path = shutil.which(override, path=search_path)
        if on_path:
            return launch_mod._normalize_exe_casing(on_path) or on_path, search_path

    mise_resolved = _mise_which(MUSE_BIN)
    if mise_resolved:
        return mise_resolved, search_path

    on_path = shutil.which(MUSE_BIN, path=search_path)
    if on_path:
        return launch_mod._normalize_exe_casing(on_path) or on_path, search_path

    return None, search_path


class MuseLaunch(ProcessAdapter):
    """muse-acp: a Node-shim adapter over a native binary, seeded before it starts."""

    backend = ACP_BACKEND_MUSE

    async def resolve_spawn(self, ctx: SpawnContext) -> SpawnPlan:
        """Resolve both components, warm the session's MCP array, then mask and seed.

        Both halves are resolved here rather than left to the adapter: the
        adapter is a dead backend without a ``muse`` to spawn, and naming the
        absent half (with the command that installs it) is the install probe's
        answer arrived early, at the spawn that would otherwise fail inside
        the adapter with no such naming.
        """
        global _muse_acp_argv_cache, _muse_bin_cache  # noqa: PLW0603
        session = ctx.session
        assert session is not None, "muse-acp is launched for one session"
        cached: tuple[list[str] | None, str] | object = _muse_acp_argv_cache
        if cached is launch_mod._UNRESOLVED:
            # Fenced on the resolution generation -- see ``acp.launch._resolution_generation``.
            epoch = launch_mod._resolution_epoch(self.backend)
            cached = await asyncio.to_thread(_resolve_muse_acp_bin)
            if launch_mod._resolution_epoch(self.backend) == epoch:
                _muse_acp_argv_cache = cached
        muse_argv, acp_search_path = cached if isinstance(cached, tuple) else (None, "")
        if not isinstance(muse_argv, list) or not muse_argv:
            raise AcpError(
                f"{MUSE_ACP_BIN} not found "
                f"({describe_search_path(acp_search_path)}). Install it with "
                f"'{MUSE_INSTALL_COMMAND}' (or add it as a project "
                f"dependency), or set {_ENV_MUSE_ACP_BIN} to its entry script."
            )
        cached_muse: tuple[str | None, str] | object = _muse_bin_cache
        if cached_muse is launch_mod._UNRESOLVED:
            epoch = launch_mod._resolution_epoch(self.backend)
            cached_muse = await asyncio.to_thread(_resolve_muse_bin)
            if launch_mod._resolution_epoch(self.backend) == epoch:
                _muse_bin_cache = cached_muse
        muse_bin, muse_search_path = (
            cached_muse if isinstance(cached_muse, tuple) else (None, "")
        )
        if not muse_bin:
            raise AcpError(
                f"{MUSE_BIN} not found "
                f"({describe_search_path(muse_search_path)}). The "
                f"{MUSE_ACP_BIN} adapter spawns it as '{MUSE_BIN} serve' and "
                f"bundles no copy of its own. Install it with "
                f"'{MUSE_CLI_INSTALL_COMMAND}', or set {_ENV_MUSE_CLI} to "
                f"the executable."
            )
        argv: list[str] = muse_argv
        # Translate the agent spec into this session's MCP array HERE, for the
        # reason the other adapters do it in their own launches: the
        # translation reads disk, and doing it at the shared session/new call
        # site would put an executor hop and a new failure mode on EVERY
        # backend's construction path, kiro-cli included (harness-parity H13).
        # What the translation yields for this backend is decided by the
        # mirror, not here -- see ``providers.mirrors.muse``.
        await session._prepare_session_mcp()
        # The same refuse-then-mask preflight the other enforced hosts run,
        # keyed on the routing question rather than on this harness's identity:
        # it is ENFORCED, so the OS credential mask is the compensating control
        # for the passive reads ACP v1 cannot make it ask about.
        hidden = await launch_mod._run_preflight_bounded(
            launch_mod._sandbox_preflight, self.backend, ctx.sandbox_mode
        )
        # The routing seed. The adapter resolves ``MUSE_APPROVAL_MODE`` above
        # the host's saved Muse default, so seeding it here means the session
        # is BORN in the mode Crew requires; the SESSION_CONFIG write at
        # session start then confirms a state rather than creating one, and
        # its read-back is what the gate's verdict rests on. Written onto the
        # session's own environment overlay so every child the session starts
        # carries it, and written unconditionally: an inherited weaker posture
        # is exactly what the seed exists to override.
        _option_id, required_mode = acp_tool_gate.permission_config_for(self.backend)
        session._extra_env = {
            **session._extra_env,
            _ENV_MUSE_APPROVAL_MODE: required_mode,
        }
        return SpawnPlan(
            argv=argv,
            spawn_label=launch_mod._adapter_spawn_label(
                argv,
                MUSE_ACP_BIN,
                pkg_entry=_MUSE_ACP_PKG_ENTRY,
                override_env=_ENV_MUSE_ACP_BIN,
            ),
            stderr_label=launch_mod._adapter_spawn_label(
                argv,
                "muse-acp",
                pkg_entry=_MUSE_ACP_PKG_ENTRY,
                override_env=_ENV_MUSE_ACP_BIN,
            ),
            extra_hidden_dirs=hidden,
            extra_expose_files=acp_tool_gate.adapter_expose_files(self.backend, hidden),
        )

    async def prepare_spawn_env(
        self,
        env: dict[str, str],
        *,
        to_thread: Callable[..., Awaitable[Any]],
    ) -> None:
        """:meth:`apply_spawn_env` through the guarded hop: its lookup runs ``mise``."""
        await to_thread(self.apply_spawn_env, env)

    def apply_spawn_env(self, env: dict[str, str], *, spawned_binary: str | None = None) -> None:
        """Point the adapter at the resolved ``muse`` binary, unless one is named.

        The adapter searches PATH and its installer's location itself, which
        in a non-login daemon context can miss an installed ``muse``; naming
        the resolved absolute path makes the spawn deterministic. Only set
        when unset, so an operator override always wins.
        """
        if env.get(_ENV_MUSE_CLI):
            return
        cached: tuple[str | None, str] | object = _muse_bin_cache
        muse_bin = cached[0] if isinstance(cached, tuple) else None
        if not muse_bin:
            muse_bin, _searched = _resolve_muse_bin()
        if muse_bin:
            env[_ENV_MUSE_CLI] = muse_bin
        else:
            logger.warning(
                "%s not found on PATH; the %s adapter will fail to start its "
                "host. Set %s.",
                MUSE_BIN,
                MUSE_ACP_BIN,
                _ENV_MUSE_CLI,
            )
