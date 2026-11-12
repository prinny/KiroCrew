# This fork admits a second native provider: `agent.provider = "bedrock"` for Amazon Bedrock

Decided by: the fork maintainer (@prinny)
Date: 2026-11-12

## Decision

This fork deliberately diverges from upstream's "Other providers" boundary
a second time: `agent.provider` admits a third value, `bedrock`, selecting
a first-party native provider that calls Amazon Bedrock's Converse API
directly instead of driving a harness over ACP. This entry SUPERSEDES the
"exactly one place / no third provider value" wording of
[2026-11-12-a-native-meta-model-api-provider.md](2026-11-12-a-native-meta-model-api-provider.md):
that decision's substance stands (the `meta_api` value, its bar, its
retirement clause), and its exclusivity sentence is replaced by this one.
The boundary text in `oss-fork-boundaries.md` is amended in the same
change to name both native providers.

The Bedrock provider's admission bar is the same one the Meta provider
met, plus the credential requirement that motivated it:

- The ACP provider remains the default and the floor; a harness is still
  never selected at `agent.provider`.
- Every tool call routes through Crew's permission machinery: the
  provider's own pending-approval registry gates execution, and a denial
  is returned to the model as an error tool result.
- The capability set is declared, reduced, and documented in
  `docs/system-specs/modules/providers.md` (`BedrockProvider`).
- Per-turn usage fills the same record the ACP provider fills; costs are
  estimates from a dated rate table, labelled as such, with AWS billing
  as the ground truth.
- **Credential resolution is the AWS SDK's full default chain,
  gateway-side** — shared files, named profiles (bound per session),
  `credential_process` helpers, SSO, static keys, session tokens,
  environment, instance metadata. A provider restricted to one auth
  mechanism (a bearer API key, or environment variables alone) does not
  meet this bar and is not an acceptable substitute; the resolved
  identity is announced over STS at session start.

## Why

- The maintainer directed that Bedrock be integrated natively in the
  fork rather than reached through a harness: routing the whole
  Bedrock catalogue through an ACP backend leaves every non-Claude
  family dependent on that harness's own provider support, and the one
  in-tree harness that speaks Bedrock natively accepts a single
  bearer-key auth mechanism.
- Three architectures were audited against the code before building
  (fork-native provider; a standalone ACP adapter; a local
  OpenAI-compatible router fronting Bedrock). External-adapter
  registration is code-only in this tree — no configuration surface
  points Crew at a custom backend — so the adapter route still needs a
  fork change and permanently splits maintenance across two codebases.
  The router route works (the full chain lives in the router process)
  but reintroduces harness dependence and loses typed refusals,
  guardrail traces and per-turn usage fidelity in translation. The
  fork-native route is the only one that keeps all three.

## Evidence

- `src/kiro_crew/providers/bedrock.py` and its sibling modules
  (`bedrock_client.py`, `bedrock_models.py`, `bedrock_events.py`,
  `bedrock_rates.py`, `bedrock_tools.py`) — the provider, its boto3
  seam, its live-discovery catalogue, its event translation, its rate
  table and its gateway-side tool executor.
- `docs/system-specs/modules/providers.md` (`BedrockProvider`) — the
  declared capability set and the credential-chain contract.
- `docs/system-specs/oss-fork-boundaries.md` — the superseding boundary
  amendment this decision records.
