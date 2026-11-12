# This fork admits one native provider: `agent.provider = "meta_api"` for the Meta Model API

Decided by: the fork maintainer (@prinny)
Date: 2026-11-12

## Decision

This fork deliberately diverges from upstream's "Other providers" boundary
in exactly one place: `agent.provider` admits a second value, `meta_api`,
selecting a first-party native provider (`MetaApiProvider`,
`src/kiro_crew/providers/meta_api.py`) that calls the Meta Model API
directly instead of driving a harness over ACP. The ACP provider remains
the default and the floor; no third provider value is admitted by this
decision, and a harness is still never selected at `agent.provider` —
harnesses remain at `agent.acp_backend`. The divergence, its limits, and
its retirement condition (upstream accepting
`rfc-pluggable-model-providers`) are recorded in
`docs/system-specs/oss-fork-boundaries.md` ("Fork extension — one native
provider"), and the provider's capability set and declared limits are
specified in `docs/system-specs/modules/providers.md`.

## Why

- The maintainer directed full native provider support for the Meta Model
  API in this fork (2026-11-12), overruling the earlier recommendation to
  keep Muse access behind ACP harnesses only: a native provider bills per
  token at published API rates with the cost on every turn, resolves its
  model catalogue independently of any harness, and is not coupled to a
  harness's release cadence.
- The divergence is kept narrow on purpose: one value, one provider class
  behind the existing `LLMProvider` seam, the same per-turn usage record,
  the same permission-gate admission bar (approval mode forced at spawn
  and verified by read-back before the first prompt), and a declared
  reduced capability set instead of silent gaps.
- The loop the provider supervises (the Codex app-server) is labelled
  [experimental] by its own vendor and is not supported for production
  workloads; the provider therefore pins the binary, refuses to start on
  a protocol schema-fingerprint mismatch, and carries recorded fixtures
  so a pin bump re-validates the translation.

## Evidence

- https://github.com/prinny/KiroCrew/issues/4 — the fork's route decision
  record for Muse support, where the native-provider route was taken up.
- `docs/system-specs/oss-fork-boundaries.md` — the boundary amendment
  itself, landed in the same change as this entry.
- `docs/system-specs/modules/harness-parity.md` — the H2 row as amended:
  the two-value enum with `"acp"` first and default, pinned by
  `test_harness_parity.py::test_provider_enum_is_acp_only`.
