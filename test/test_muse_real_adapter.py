"""The muse-acp adapter, driven for real: the handshake this backend rests on.

The measurements the Muse backend's registry claims stand on are the
adapter's wire answers: it speaks ACP protocol version 1, it names itself
``muse-acp``, and the release doing so is the pinned one. Those are asserted
here against the real adapter rather than a double, in the Real Adapter
Contract Tests lane (``KIROCREW_E2E_REQUIRE=1`` makes an absent adapter a
red lane rather than a skip; see ``test/real_adapter_gate.py``).

The session-level measurements -- the ``approval_mode`` force and read-back,
the answered prompt, the refused tool call -- are recorded as frame captures
under ``test/fixtures/acp_frames/muse/`` and replayed by
``test/test_acp_frame_replay.py``. They are not re-driven here: a live
session needs a Muse sign-in the lane does not hold, and a contract that
only a credentialed host can run is a contract the lane would skip forever.
What the lane CAN hold every adapter release to is the handshake, which is
also where a release that moved the surface this backend routes on shows up
first.
"""

from __future__ import annotations

import json
import subprocess
import threading

import pytest

from kiro_crew.acp.harness import muse as muse_mod
from real_adapter_gate import (
    MEASURED_MUSE_ACP_VERSION,
    MUSE_ACP_PACKAGE,
    require_real_adapter,
)

_INSTALL = f"npm i -g {MUSE_ACP_PACKAGE}@{MEASURED_MUSE_ACP_VERSION}"
_CLI_INSTALL = muse_mod.MUSE_CLI_INSTALL_COMMAND


def _resolved_pair() -> tuple[list[str] | None, str | None]:
    """The adapter argv and the muse executable Crew would spawn, or ``None``.

    Crew's own resolvers, so the lane's environment is honoured exactly as a
    session honours it.
    """
    argv, _search = muse_mod._resolve_muse_acp_bin()
    muse_bin, _search = muse_mod._resolve_muse_bin()
    return argv, muse_bin


def _require_muse() -> list[str]:
    argv, muse_bin = _resolved_pair()
    require_real_adapter(argv, what=MUSE_ACP_PACKAGE, install=_INSTALL)
    require_real_adapter(muse_bin, what="muse (the Muse)", install=_CLI_INSTALL)
    assert argv is not None
    return argv


def _initialize(argv: list[str]) -> dict:
    """Spawn the real adapter and return its ``initialize`` result."""
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    result: dict = {}

    def _drive() -> None:
        proc.stdin.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": 1,
                            "clientCapabilities": {
                                "fs": {"readTextFile": False, "writeTextFile": False},
                                "terminal": False,
                            },
                            "clientInfo": {"name": "kiro_crew", "version": "0"},
                        },
                    }
                )
                + "\n"
            ).encode()
        )
        proc.stdin.flush()
        for raw in proc.stdout:
            msg = json.loads(raw)
            if msg.get("id") == 1:
                result.update(msg.get("result") or {})
                return

    thread = threading.Thread(target=_drive, daemon=True)
    thread.start()
    thread.join(timeout=60)
    proc.terminate()
    return result


@pytest.mark.real_adapter
def test_real_muse_acp_is_the_pinned_release() -> None:
    """The adapter Crew resolves is the release the measurements name.

    The version is the binary's own answer on the wire, not its manifest:
    the same proof the lane's assert step takes from the other adapters.
    """
    argv = _require_muse()
    result = _initialize(argv)
    info = result.get("agentInfo") or {}
    assert info.get("version") == MEASURED_MUSE_ACP_VERSION, (
        f"muse-acp {info.get('version')!r} resolved, {MEASURED_MUSE_ACP_VERSION} pinned"
    )


@pytest.mark.real_adapter
def test_real_muse_acp_handshake_answers_the_acp_dialect() -> None:
    """The handshake facts the backend's wiring rests on, off the real wire.

    Protocol 1 (the literal ``PROTOCOL_VERSION_MUSE`` declares), the agent
    name the frame corpus expects, session load support, and the MCP
    capability shape whose stdio half ACP's schema cannot advertise.
    """
    argv = _require_muse()
    result = _initialize(argv)
    assert result.get("protocolVersion") == 1
    info = result.get("agentInfo") or {}
    assert info.get("name") == "muse-acp"
    caps = result.get("agentCapabilities") or {}
    assert caps.get("loadSession") is True
    mcp = caps.get("mcpCapabilities") or {}
    assert mcp.get("http") is True
    assert mcp.get("sse") is False
