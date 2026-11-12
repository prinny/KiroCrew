# Muse provider — a selectable ACP harness

## What this backend is

`agent.acp_backend = "muse"` runs a Kiro Crew session on Meta's Muse coding
agent. The harness speaks the Agent Client Protocol (ACP) v1 to Kiro Crew
through a third-party adapter, `@brokkai/muse-acp` (Apache-2.0), which
translates ACP onto Muse's own server protocol (MSP): the adapter spawns
`muse serve` and drives it per session. Muse's models — the catalogue the
host's sign-in is entitled to — are what the session runs on.

The adapter release this backend is wired against is **0.21.0**, driving
**Muse 1.4.13**. The version is pinned exactly in
`packaging/acp-adapters/package.json` (Dependabot-tracked) and in
`MUSE_ACP_VERSION` in `acp/harness/muse.py`; `test/test_installer_pins.py`
holds the two in step.

## The two components

Like the Pi backend, this one is two programs, and either can be absent on
its own:

| Component | What it is | Install |
| --- | --- | --- |
| `muse-acp` | The ACP adapter Crew spawns (a Node shim over a native binary) | `npm i -g @brokkai/muse-acp@0.21.0` |
| `muse` | The Muse the adapter spawns (`muse serve`) | `curl -fsSL https://dev.meta.ai/install.sh \| sh` |

`agent_sdk.backend_install._probe_muse` probes for both and names the absent
one with its own install command. Resolution follows the shared Node-adapter
ladder for the adapter (project-local `node_modules` first, then the global
install; `MUSE_ACP_BIN` overrides) and the plain-binary ladder for the CLI
(`MUSE_CLI` — the adapter's own variable — is honoured first, then PATH).
Crew names the resolved CLI to the adapter through `MUSE_CLI` at spawn, so a
non-login gateway process finds the same binary an interactive shell would.

## Selecting it

```json
{ "agent": { "acp_backend": "muse" } }
```

The id is a member of the baseline selectable set. Selectable is not the same
as usable: a machine without both components, or without a Muse sign-in,
cannot run a session, and the install probe and the sign-in declaration say
which is missing.

## Approval routing — forced and verified

MSP `session/start` accepts an approval mode and reports the effective mode
back. The adapter exposes that posture on the ACP wire as the `approval_mode`
session config option, and this backend's routing is
`Routing.SESSION_CONFIG` on it:

- At spawn, the launch seeds `MUSE_APPROVAL_MODE=promptUnmatched` into the
  session's environment. The adapter resolves that variable above the host's
  saved Muse default, so the session is born in the required mode.
- At session start, Crew verifies the `approval_mode` option was advertised
  on `session/new`, writes `promptUnmatched` through
  `session/set_config_option`, and refuses the session if the option is
  missing or the write is rejected — before the first prompt.

`promptUnmatched` is the value that routes: every tool call Muse's own
permission rules do not match is sent through `session/request_permission`,
which is where Crew's gate decides it. The alternatives are why the mode is
Crew's rather than inherited: `allowAll` approves without asking, `onRequest`
defers to whatever posture the operator's Muse profile saved, and
`denyUnmatched` refuses unmatched calls inside Muse without asking, which
would hide them from the gate rather than route them through it. (The
inherited mode is not invisible — the adapter reports it as the option's
current value on `session/new` — but inheriting it is not a posture Crew can
stand behind, so it never does.)

The adapter's own `auto_review` option stays **off**. Approvals on a Crew
session are answered by Crew's gate alone; a second reviewer inside the
adapter would compete with it.

One residual is shared with the other session-config harnesses and stated
plainly: calls Muse's own permission rules *match* (an operator's allow
rules in their Muse settings) run without asking. The OS credential mask
that covers every enforced harness's child — everything on the read-gate
floor except Muse's own credential file — is the compensating control for
the passive reads in that gap.

## MCP posture — withheld by default

**Crew does not pass its MCP servers into muse sessions by default.** The
adapter can carry them (it forwards client-supplied MCP servers to the Muse
host), so this is a decision, not a missing transport, and it is implemented
in exactly one place: `providers/mirrors/muse.py` projects an empty
`mcpServers` array — the agent spec's servers and the pooled broker stubs
alike — unless the operator opens the gate.

The reason is a property of the harness: MCP tools are **not** covered by
Muse's write/shell disable flags, and MCP servers run unsandboxed. In
Muse's read-only and plan modes, where its own file and shell tools are
disabled, a mounted MCP server can still write files, run commands and use
the network. A Crew session's tools are therefore kept out of that surface
unless the operator explicitly accepts it.

The opt-in is the environment variable **`KIROCREW_MUSE_MCP`** (values `1`,
`true`, `yes`, `on`) on the gateway process. It is off by default, and there
is no per-agent or per-session form. With the gate open, the projection is
the shared single-binary translation, and a server the spec narrows per tool
is withheld whole — no per-call deny channel is verified for this harness.

Servers an operator configures in Muse itself (Muse's own MCP
configuration) are Muse's, not Crew's: this posture governs what Crew
hands the session, and says nothing about what the operator's own Muse
setup mounts.

## Transport

Crew spawns the **adapter over stdio**, which is ACP v1's only stable
transport and the shape every per-process backend here uses. The adapter in
turn owns its `muse serve` child over stdio (plus a second, write/shell-
disabled host when a session is in Muse's read-only or plan mode).

Muse 1.4.13 also ships `muse serve --listen`, a multi-client socket daemon
mode. It is **not wired**: nothing on the ACP side consumes it, session
control stays one host per session through the adapter, and multiplexing
sessions onto a shared daemon would change the teardown and identity
properties the per-process shape gives for free. The stdio-only statement
applies to Crew's ACP spawn, not to MSP — the backend records the choice
here so a future adapter release built on `--listen` can be adopted without
redesigning this backend.

## Models and effort

Muse advertises no model list in the ACP `models` block; its models arrive
on the `model` session config option, and that option is the channel a switch
travels down (`ACP_BACKENDS_MODEL_VIA_CONFIG_OPTION`). The advertised ids
are Muse's own and live in their own model-registry namespace (`muse`), so
one harness's list can never overwrite another's in the picker. Reasoning
effort travels on the `reasoning_effort` config option
(`ACP_BACKENDS_EFFORT_VIA_CONFIG_OPTION`). Turn usage is reported as token
counts on the `session/prompt` result (`ACP_BACKENDS_REPORT_USAGE`).

## Authentication

Sign-in happens **outside** the product, in the operator's own Muse
install: an interactive `muse` sign-in (which stores its credential at
`~/.config/muse/auth.json`), or `META_API_KEY` in the gateway's environment
for pay-as-you-go access. Crew declares the credential file in
`agent_sdk/host_auth.py` so the read gate fences it and the spawn's
credential mask spares it for the `muse` process itself. There is no Crew
sign-in flow for this backend, deliberately: the harness owns its identity.

## Capability limits, stated

Everything below is a non-membership, which in this codebase is the
fail-closed default rather than an oversight:

- **No steering.** MSP has a steer surface, but the adapter's ACP surface
  for it is not measured, so no steer set claims it.
- **`/compact` is not offered.** The adapter advertises a `compact` command
  and MSP has a compact operation, but compaction driven through Crew is
  not measured, so `ACP_BACKENDS_COMPACT` does not claim it.
- **No member or worker sessions.** Enrolled-member dispatch, the crewmate
  worker kit and the member panel are not claimed; a member session on this
  backend is refused before spawn.
- **No side-chat read-only execution.** The derived read-only spec shape is
  proven for other harnesses, not for this one's pre-approval surface, so
  `ACP_BACKENDS_SIDE_READONLY` does not claim it.
- **Spec hooks are not fired by Crew** for this backend, and its sessions do
  not honour a spec's zero-tool ban: both rest on every call arriving as a
  permission request, and calls Muse's own rules match do not arrive.
- **Sessions are per-process** through `AcpClient`, like Claude and Pi —
  not multiplexed on a shared runtime.
