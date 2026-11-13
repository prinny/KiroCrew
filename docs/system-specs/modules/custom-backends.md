# Custom backends — operator-registered ACP harnesses

## Boundary

A harness Kiro Crew can spawn but this build does not ship is declared by
the operator in `agent.custom_backends` and registered at config load into
`agent_sdk/custom_backends.py`, a leaf registry holding one frozen
`CustomBackendSpec` per id. Registration is the ONLY way a custom id comes
to exist: `AcpProvider` construction accepts an id that is in
`ACP_BACKENDS_KNOWN` or in the registry, and refuses everything else, so a
typo'd or unregistered value fails at construction rather than falling
through to kiro-cli.

Nothing about a custom backend is per-provider code. Every surface that
used to take a code row per harness answers for a custom id from the
registry, at the seam where it already special-cases a backend:

| Surface | Owner | Custom answer |
|---|---|---|
| Selectability | `agent_sdk/backends.py` | `register_custom_selectable_backend`, called by the loader's normalize; applies `register_selectable_backend`'s two guards in dynamic form (registered, and routing not `UNVERIFIED`). Removal unregisters. |
| Routing | `agent_sdk/backends_routing.py` | `routing_for` answers from the declared permission kind: `session_config` → `SESSION_CONFIG`; `seeded` with a verify block → `VERIFIED_SEEDED_SETTINGS`; `seeded` without, and `asks`, → `SEEDED_SETTINGS`. `permission_config_for` / `permission_setting_for` answer from the same declaration. |
| Tool gate | `agent_sdk/tool_gate.py` | `label_for` reads the declared label. The credential mask needs no custom code: `is_enforced` follows `routing_for`. `adapter_expose_files` unions the declared `exposed_leaves` into the re-exposure loop, where the existing containment check drops any leaf under nothing the mask hides. A masked backend's own leaves are spared through its synthesised host-auth declaration. |
| Host auth | `agent_sdk/host_auth.py` | `declaration_for` synthesises a declaration from the registration. `adapter_chain` declares no leaves and the `ambient_environment` entitlement source (the adapter resolves its own provider chain); `masked` declares its leaves as both fenced and adapter-own, with the `own_credential_file` source. |
| Install probe | `agent_sdk/backend_install.py` | One generic probe: the declared `command` resolved as the spawn resolves it; the remedy is the declared `install_hint`. |
| Backend card | `agent_sdk/backend_cards.py` | No custom code: `card_for` is total, reads routing from `routing_for` and offered-ness from the selectable registry, and answers fail-closed on every undeclared capability line. |
| Capabilities | `agent_sdk/capabilities.py` | `capabilities_for` builds the record from the declaration: spawnable, model namespace = the backend id (`backends_models.model_registry_namespace`), model/effort surfaces exactly when the registration names their config options. Everything undeclared is withheld. |
| Spawn | `acp/harness/custom.py` | One `CustomLaunch(ProcessAdapter)` for every custom id; `process_adapter_for` returns it when no static adapter matches and the id is registered. |
| Handshake | `acp/client_start.py` | `_initialize_params` uses the declared `protocol_version` for a custom id instead of the kiro date-stamped default. |
| Permission routing | `acp/client_permissions.py` | `session_config` backends are armed and verified by the existing generic path (`_apply_session_permission_routing`). A verified `seeded` backend's read-back is `_verify_custom_routing`: the declared option's `currentValue` off the session response, judged by `tool_gate.seeded_setting_issue`, on both the new and load paths. |
| Model switching | `acp/client_model_selection.py` | The membership sites read `ACP_BACKENDS_MODEL_VIA_CONFIG_OPTION` first and the registry second; the push addresses the declared option id. |
| MCP | `providers/mirrors/custom.py`, `registry.py` | `CustomMirror` (constructed with the id) projects through the shared translation and the single-binary placement, withholding a per-tool-narrowed server whole; `projection_for` answers `MIRROR`/`WHOLE_SERVER` for `mcp: "array"` and `NO_CHANNEL` for `mcp: "withheld"`. The client's `_session_mcp_servers` gate and the three `session/new`/`session/load` splices treat a custom id like a shipped array harness, gated on `_is_custom`. |
| Provider label | `providers/acp.py`, `agent_sdk/drivers/acp.py` | A custom id labels itself, so its sessions never persist as another harness's. |
| Process reclaim | `agent_sdk/backends_launch.py` | `agent_process_markers` includes the basename of the declared command. |

## The config surface

`agent.custom_backends` is a plain `dict` field on the agent section
(beside `acp_failover`, whose plain-JSON shape is the precedent), stored
raw by `sections.normalize_custom_backends`, which also performs
registration and the selectable sync on every load, and returns only the
entries that registered. The write path refuses what load would drop, in
two places that share the one validation: `sections.validate_custom_backends`
for `config set`, and the publish floor
(`loader_document._refuse_custom_backends_entries`) for whole-document
writes, scoped like its siblings to what the write changes. The key is one
of `loader_builders._HARNESS_CHOICE_KEYS`, so a full-document save writes
the operator's block back from disk verbatim.

Validation lives in the registry (`register_custom_backends`), with host
policy passed in by the caller — the reserved ids (every built-in backend
id and policy id) and the `agent.env` name rules — because the registry is
a leaf and imports none of it. Errors name the id and the offending key,
never a value; a refused entry is not registered and valid siblings are.

### Env rules

A binding's name must satisfy the `agent.env` rule
(`config/sections_agent_env.agent_env_refusal`: grammar, Crew-owned names,
home-relocation names) and must not be a name the spawn scrub removes
(`agent_env_scrub_removes`) — a binding the scrub would silently eat is
refused instead. One extension, following the provider-env precedent: a
credential-shaped name is accepted only with a `secret://` value. At
spawn, `CustomLaunch` resolves references from the vault and fails closed
when one cannot be answered; resolved bindings are placed on the adapter
child's environment only, never on the session overlay.

## Permission postures

Three declared kinds (see the table above for their routing). No new
`Routing` member exists: each kind is a mechanism generic code already
implements, which is what makes the declaration a routing establishment
rather than a promise. A `verify` block is only meaningful for `seeded`,
and its only implemented source is `config_option`.

## Credential postures

`masked` is the default and the enforced shape: the mask machinery is
Crew's, and the declaration only names which of the backend's own leaves
are fenced/spared and which masked files are re-exposed (contained by the
mask check, as for every backend). `adapter_chain` is unenforced by
declaration: the adapter resolves its provider's chain from the host, and
`credentials.reads` is audit text for the card — it grants nothing, and
the spec does not pretend a fence exists where there is none.

## What registration does not cover

A harness whose differences are behaviour — a gate extension injected into
its process, a bespoke read-back child, launch files written per session —
still takes a code harness (the pi/deepseek/strands shape). Registration
expresses facts: a command, bindings, an option name, leaves. It refuses
nothing on that ground; it simply cannot express behaviour, and the
boundary is stated here rather than discovered per attempt.

## First consumer

`packages/kirocrew-bedrock-acp` is a standalone ACP adapter driving
Bedrock's Converse API through boto3's default chain, registered with the
`bedrock` block documented in `docs/configuration.md` ("Custom backends").
It is the worked example that the registration carries a real external
agent end to end — catalogue, model option, permission option, MCP tools —
with zero Bedrock-specific code in the core tree.
