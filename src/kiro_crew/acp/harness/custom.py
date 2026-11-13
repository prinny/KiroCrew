"""An operator-registered custom ACP backend, launched one process per session.

The generic launch: every fact it needs is in the backend's registration
(``agent_sdk.custom_backends``), so this one adapter serves every custom
backend there will ever be, and adding an external ACP agent to Kiro Crew
never adds a file here again. What it does is the shape every per-process
adapter shares, with each per-backend fact read off the spec instead of a
code table: resolve the declared command, warm the session's MCP array when
the declaration takes one, run the refuse-then-mask preflight when the
backend's routing is enforced, and put the declared env bindings -- with
``secret://`` references resolved from the vault, fail-closed -- on the
child's environment and nowhere else.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.harness.base import ProcessAdapter, SpawnContext, SpawnPlan
from kiro_crew.acp.transport_errors import AcpError
from kiro_crew.agent_sdk import custom_backends
from kiro_crew.agent_sdk import tool_gate as acp_tool_gate

logger = logging.getLogger("kiro_crew.acp.client")

__all__ = ["CustomLaunch"]


def _resolve_command(spec: custom_backends.CustomBackendSpec) -> str:
    """The declared command as an executable path, or an ``AcpError`` naming the remedy.

    An absolute (or path-carrying) command is tested as a file; a bare name is
    searched on PATH, the same resolution the install probe performs, so the
    probe's verdict and the spawn's cannot disagree.
    """
    command = spec.command
    if os.path.sep in command or (os.path.altsep and os.path.altsep in command):
        if os.path.isfile(command) and os.access(command, os.X_OK):
            return command
    else:
        resolved = shutil.which(command)
        if resolved:
            return resolved
    remedy = spec.install_hint or f"install {command!r} and put it on PATH"
    raise AcpError(f"{command} not found for custom backend {spec.backend_id!r}. {remedy}.")


def _resolve_bindings(spec: custom_backends.CustomBackendSpec) -> dict:
    """The child's env bindings, with ``secret://`` values resolved from the vault.

    Both the ``env`` bindings and the permission seed bindings land here: the
    seed is how a seeded-posture session is BORN in its required mode (the
    muse adapter's ``MUSE_APPROVAL_MODE`` shape), and the bindings are how
    the operator points the adapter at its own configuration. A reference
    the vault cannot answer fails the spawn closed, naming the env NAME and
    the backend -- never the reference's target or any value.
    """
    bindings = {**spec.env_map, **spec.seed_env_map}
    refs = {
        name: value[len(custom_backends.SECRET_REF_PREFIX) :]
        for name, value in bindings.items()
        if value.startswith(custom_backends.SECRET_REF_PREFIX)
    }
    if not refs:
        return bindings
    # Deferred: the vault lives beside the config loader, and this module is
    # imported on the client's construction path, where a module-scope edge
    # on the loader would be a cycle.
    from kiro_crew.config.loader_credentials import env_path
    from kiro_crew.secrets import SecretVault

    try:
        fetched = SecretVault(env_path().parent).get_many(list(refs.values()))
    except Exception:  # noqa: BLE001 - any vault fault fails closed
        fetched = None
    for name, ref in refs.items():
        secret = fetched.get(ref) if fetched else None
        if secret is None:
            raise AcpError(
                f"custom backend {spec.backend_id!r}: env name {name} names a "
                "secret:// reference the vault cannot resolve; the backend was "
                "not started"
            )
        bindings[name] = secret.reveal()
    return bindings


class CustomLaunch(ProcessAdapter):
    """Spawn one registered custom backend for one session.

    One instance per spawn (``process_adapter_for`` constructs it with the
    backend id), so the resolved bindings computed at plan time are the ones
    the same launch puts on the child's environment.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self._resolved_env: dict | None = None

    def _spec(self) -> custom_backends.CustomBackendSpec:
        spec = custom_backends.spec_for(self.backend)
        if spec is None:
            raise AcpError(
                f"backend {self.backend!r} is not a registered custom backend; "
                "it cannot be launched"
            )
        return spec

    async def resolve_spawn(self, ctx: SpawnContext) -> SpawnPlan:
        """Resolve the command, warm the MCP array, mask and seed, then plan."""
        spec = self._spec()
        session = ctx.session
        assert session is not None, "a custom backend is launched for one session"
        resolved = _resolve_command(spec)
        argv = [resolved, *spec.args]
        if spec.mcp == custom_backends.MCP_ARRAY:
            # The translation reads disk; doing it here, on this backend's own
            # spawn path, keeps the shared session/new call site free of a hop
            # every backend would pay for (harness-parity H13).
            await session._prepare_session_mcp()
        # The same refuse-then-mask preflight every enforced host runs, keyed
        # on the routing question rather than on this harness's identity. For
        # an unenforced declaration it returns an empty mask, so the spawn
        # arguments stay byte-identical to an unmasked spawn.
        hidden = await launch_mod._run_preflight_bounded(
            launch_mod._sandbox_preflight, self.backend, ctx.sandbox_mode
        )
        # Resolve bindings off the loop: a vault read is file IO.
        self._resolved_env = await asyncio.to_thread(_resolve_bindings, spec)
        label = Path(resolved).name
        return SpawnPlan(
            argv=argv,
            spawn_label=label,
            stderr_label=spec.label or label,
            extra_hidden_dirs=hidden,
            extra_expose_files=acp_tool_gate.adapter_expose_files(self.backend, hidden),
        )

    def apply_spawn_env(self, env: dict[str, str], *, spawned_binary: str | None = None) -> None:
        """Put this launch's resolved bindings on the child's environment.

        Only the child's: the bindings are the adapter's own configuration,
        and a resolved secret among them must not ride the session's overlay
        into every tool the session later runs. Resolved at plan time; the
        synchronous fallback covers a caller that assembles an environment
        without a plan.
        """
        if self._resolved_env is None:
            self._resolved_env = _resolve_bindings(self._spec())
        env.update(self._resolved_env)
