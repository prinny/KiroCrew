"""The Codex app-server JSON-RPC client for the native Meta provider.

This is the replaceable seam of the provider (design §1.5/§2.2): everything
that knows the loop is a separate child process speaking app-server
JSON-RPC over stdio lives here; ``MetaApiProvider`` speaks only this
module's interface.

The protocol is pinned, because OpenAI labels ``codex app-server``
[experimental] and states it is not supported for production workloads:

- The binary version is recorded (:data:`PINNED_CODEX_VERSION`).
- At :meth:`AppServerClient.start`, the binary's own schema export is
  fingerprinted and compared with :data:`PINNED_SCHEMA_FINGERPRINT`; a
  mismatch refuses the session loudly instead of letting a moved protocol
  drift silently under the translator.
- The provider configuration Crew forces (Meta base URL, env key, wire
  API, no OpenAI auth, on-request approvals, sandbox mode) is written at
  SPAWN time through ``-c`` overrides, so the post-``initialize``
  ``config/read`` read-back in :meth:`verify_forced_config` compares the
  values that actually govern every session this process serves.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger("kiro_crew.providers.meta_api")

#: The app-server release this provider is pinned to. Recorded frames,
#: fixtures, and the fingerprint below were all taken from this binary.
PINNED_CODEX_VERSION = "0.167.2"

#: sha256 over the method vocabulary of the pinned binary's schema export
#: (ClientRequest + ServerRequest + ServerNotification), computed by
#: :func:`schema_fingerprint` from that export. A different binary whose
#: protocol surface moved produces a different fingerprint and is refused.
PINNED_SCHEMA_FINGERPRINT = "3fb928ca4851240fe4db05106677c9d011800492207ae9f70d32dfd5f3a58bd5"

#: The env var the app-server reads the Meta API key from, named by the
#: forced ``env_key`` config. The key itself is only ever placed in the
#: child process environment under this name — never on a command line,
#: never in a config file.
API_KEY_ENV = "MODEL_API_KEY"

PROVIDER_ID = "meta"
PROVIDER_NAME = "Meta Model API"

FrameHandler = Callable[[dict], Coroutine[Any, Any, None]]


class AppServerError(RuntimeError):
    """The app-server refused a request or the session cannot start."""


def codex_binary() -> str | None:
    """The ``codex`` executable on PATH, or None when it is not installed."""
    return shutil.which("codex")


def schema_fingerprint(schema_dir: Path) -> str:
    """Fingerprint a ``generate-json-schema`` export directory.

    The fingerprint covers the protocol's method vocabulary: every method
    name the client may send, the server may request, and the server may
    notify. It deliberately ignores type details — a payload-shape change
    is caught by the recorded fixtures; a method-vocabulary change is the
    drift that would silently break the translator's dispatch.
    """
    methods: set[str] = set()
    for name in ("ClientRequest.json", "ServerRequest.json", "ServerNotification.json"):
        path = schema_dir / name
        if not path.is_file():
            raise AppServerError(f"schema export is missing {name}")
        text = path.read_text(encoding="utf-8")
        methods.update(re.findall(r'"enum":\s*\[\s*"([a-z][a-zA-Z0-9/._-]+)"\s*\]', text))
        methods.update(re.findall(r'"method":\s*{\s*"enum":\s*\[([^\]]+)\]', text))
    digest = hashlib.sha256()
    for method in sorted(methods):
        digest.update(method.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def map_sandbox_mode(crew_mode: str) -> str:
    """Map Crew's ``agent.sandbox`` vocabulary onto the loop's sandbox modes.

    The loop's sandbox is the confinement for native sessions. ``auto``
    and ``strict`` both land on the standard working tier; only an
    explicit operator choice widens or narrows it.
    """
    mode = (crew_mode or "auto").strip().lower()
    if mode in ("read_only", "readonly", "read-only"):
        return "read-only"
    if mode in ("off", "none", "disabled", "full", "danger-full-access"):
        return "danger-full-access"
    return "workspace-write"


class AppServerClient:
    """One app-server child process and its JSON-RPC plumbing."""

    def __init__(
        self,
        *,
        binary: str,
        base_url: str,
        api_key: str,
        model: str,
        cwd: str,
        sandbox_mode: str,
        codex_home: str,
        extra_env: dict[str, str] | None = None,
        mcp_servers: dict[str, dict] | None = None,
    ) -> None:
        self._binary = binary
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._cwd = cwd
        self._sandbox_mode = sandbox_mode
        self._codex_home = codex_home
        self._extra_env = dict(extra_env or {})
        self._mcp_servers = dict(mcp_servers or {})
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._futures: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._on_frame: FrameHandler | None = None
        self._on_exit: Callable[[], None] | None = None
        self._spawn_token = ""
        self.exit_code: int | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def process_instance(self) -> str:
        return self._spawn_token

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def set_frame_handler(self, handler: FrameHandler) -> None:
        self._on_frame = handler

    def set_exit_handler(self, handler: Callable[[], None]) -> None:
        """Called once when the child's stdout closes (the process ended or
        the transport broke), so an open turn can be terminated synthetically
        instead of hanging."""
        self._on_exit = handler

    def _spawn_args(self) -> list[str]:
        def q(value: str) -> str:
            return json.dumps(value)

        args = [
            self._binary,
            "app-server",
            "-c",
            f"model_provider={q(PROVIDER_ID)}",
            "-c",
            f"model_providers.{PROVIDER_ID}.name={q(PROVIDER_NAME)}",
            "-c",
            f"model_providers.{PROVIDER_ID}.base_url={q(self._base_url)}",
            "-c",
            f"model_providers.{PROVIDER_ID}.env_key={q(API_KEY_ENV)}",
            "-c",
            f"model_providers.{PROVIDER_ID}.wire_api={q('responses')}",
            "-c",
            f"model_providers.{PROVIDER_ID}.requires_openai_auth=false",
            "-c",
            f"model={q(self._model)}",
            "-c",
            'approval_policy="on-request"',
            "-c",
            f"sandbox_mode={q(self._sandbox_mode)}",
        ]
        for name, spec in sorted(self._mcp_servers.items()):
            prefix = f"mcp_servers.{name}"
            if spec.get("command"):
                args += ["-c", f"{prefix}.command={q(str(spec['command']))}"]
            if spec.get("args"):
                args += ["-c", f"{prefix}.args={json.dumps([str(a) for a in spec['args']])}"]
            if spec.get("url"):
                args += ["-c", f"{prefix}.url={q(str(spec['url']))}"]
            env = spec.get("env") or {}
            if env:
                table = "{" + ", ".join(f"{k} = {q(str(v))}" for k, v in sorted(env.items())) + "}"
                args += ["-c", f"{prefix}.env={table}"]
        return args

    async def check_fingerprint(self) -> None:
        """Refuse to start when the binary's protocol surface has moved."""
        with tempfile.TemporaryDirectory(prefix="meta-api-schema-") as tmp:
            proc = await asyncio.create_subprocess_exec(
                self._binary,
                "app-server",
                "generate-json-schema",
                "--out",
                tmp,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise AppServerError(
                    "the codex app-server could not export its protocol schema: "
                    + (stderr or b"").decode("utf-8", "replace")[:300]
                )
            fingerprint = schema_fingerprint(Path(tmp))
        if fingerprint != PINNED_SCHEMA_FINGERPRINT:
            raise AppServerError(
                "the codex app-server protocol does not match the pinned "
                f"{PINNED_CODEX_VERSION} surface (fingerprint mismatch); "
                "refusing to start a native session on an unverified protocol"
            )

    async def start(self) -> None:
        await self.check_fingerprint()
        env = {
            "PATH": _path_env(),
            "HOME": str(Path.home()),
            "CODEX_HOME": self._codex_home,
            API_KEY_ENV: self._api_key,
        }
        env.update(self._extra_env)
        Path(self._codex_home).mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            *self._spawn_args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=self._cwd,
        )
        self._spawn_token = f"meta-api-{self._proc.pid}-{id(self):x}"
        self._reader_task = asyncio.create_task(self._read_loop())
        result = await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "kiro-crew",
                    "title": "Kiro Crew (native Meta provider)",
                    "version": "1",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        logger.debug("meta_api app-server initialized: %s", result.get("userAgent", ""))
        await self.notify("initialized", None)
        await self.verify_forced_config()

    async def verify_forced_config(self) -> None:
        """§2.4 step 2: read the effective config back and compare every
        forced key against the value written at spawn. A mismatch refuses
        the session before any prompt is sent."""
        result = await self.request("config/read", {"includeLayers": False})
        config = result.get("config") or {}
        providers = config.get("model_providers") or {}
        mine = providers.get(PROVIDER_ID) or {}
        expected = {
            "approval_policy": "on-request",
            "sandbox_mode": self._sandbox_mode,
            "model_provider": PROVIDER_ID,
        }
        mismatches = [
            f"{key}: expected {want!r}, read back {config.get(key)!r}"
            for key, want in expected.items()
            if config.get(key) != want
        ]
        provider_expected = {
            "base_url": self._base_url,
            "env_key": API_KEY_ENV,
            "wire_api": "responses",
            "requires_openai_auth": False,
        }
        mismatches += [
            f"model_providers.{PROVIDER_ID}.{key}: expected {want!r}, read back {mine.get(key)!r}"
            for key, want in provider_expected.items()
            if mine.get(key) != want
        ]
        if mismatches:
            raise AppServerError(
                "the app-server's effective configuration does not match the "
                "forced configuration; refusing the session before any prompt: "
                + "; ".join(mismatches)
            )

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        self.exit_code = proc.returncode if proc is not None else None
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        for future in self._futures.values():
            if not future.done():
                future.set_exception(AppServerError("the app-server process closed"))
        self._futures.clear()

    # -- JSON-RPC plumbing -------------------------------------------------

    async def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(frame, dict):
                    continue
                self._dispatch(frame)
        except asyncio.CancelledError:
            raise
        except Exception:  # the reader must never take the provider down
            logger.exception("meta_api app-server reader failed")
        finally:
            for future in self._futures.values():
                if not future.done():
                    future.set_exception(AppServerError("the app-server process exited"))
            self._futures.clear()
            if self._proc is not None:
                self.exit_code = self._proc.returncode
            if self._on_exit is not None:
                self._on_exit()

    def _dispatch(self, frame: dict) -> None:
        frame_id = frame.get("id")
        if frame_id is not None and ("result" in frame or "error" in frame):
            future = self._futures.pop(frame_id, None)
            if future is not None and not future.done():
                if "error" in frame:
                    error = frame["error"] or {}
                    future.set_exception(
                        AppServerError(
                            f"{error.get('message', 'app-server error')} "
                            f"(code {error.get('code')})"
                        )
                    )
                else:
                    future.set_result(frame.get("result") or {})
            return
        if self._on_frame is not None:
            asyncio.get_running_loop().create_task(self._on_frame(frame))

    async def _send(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise AppServerError("the app-server process is not running")
        proc.stdin.write((json.dumps(payload) + "\n").encode())
        await proc.stdin.drain()

    async def request(self, method: str, params: dict | None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._futures[request_id] = future
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        result = await future
        return result if isinstance(result, dict) else {}

    async def notify(self, method: str, params: dict | None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def respond(self, request_id: str | int, result: dict) -> None:
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def respond_error(self, request_id: str | int, message: str) -> None:
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": message},
            }
        )


def _path_env() -> str:
    import os

    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
