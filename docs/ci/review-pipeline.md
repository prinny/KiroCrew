# Review pipeline

Dependency-bump pull requests merge on evidence, not on summaries. Every
claim in a validation review must be backed by captured test output, and
the pipeline exists so that backing is produced — and checked — by
machines rather than by prose.

## How it fits together

- A **private validation pipeline** (maintainer-run, not part of this
  repository) executes the validation for a pull request: it installs the
  bumped tree in CI mode, runs the security probes for the advisories in
  scope against the old and new versions and against this project's own
  configuration, runs the affected test scope with controls, and
  generates the evidence.
- The **evidence** lives on this repository's **wiki**: one page per
  validated PR head, titled with the repository, PR number and head SHA.
  Each page carries a verdict table and, per claim, the captured output
  the claim rests on.
- The pipeline posts a compact **evidence comment** on the PR: a verdict
  table whose rows link to the per-claim sections of the evidence page.
  Both the comment and the page carry a verification marker — an opaque
  run reference, the head SHA they cover, a clean / not-clean result, and
  a digest of the content.
- The **`review-gate` check** (`.github/workflows/review-gate.yml`,
  `scripts/review-pipeline/check-evidence.mjs`) runs on every pull
  request and fails unless the evidence comment exists for the PR's
  current head, its marker verifies, and the linked evidence page
  resolves and carries a matching marker for the same head. It is a
  required status check on `main`.

## Running it as a maintainer

1. Dispatch the validation pipeline for the PR (target repository, PR
   number, package, and the from/to versions of the bump).
2. Wait for the evidence comment to appear on the PR. The `review-gate`
   check turns green once the comment and the evidence page verify
   against the PR's current head.
3. If a PR predates the workflow or its check needs re-evaluating (the
   head has not changed, so no pull-request event fires), run the
   `review-gate` workflow manually from the Actions tab with the PR
   number; the evaluation is recorded as a check run on the PR's head
   commit.

Any push that changes the PR head invalidates the previous evidence: the
marker covers one exact head SHA, so validation must be re-run for the
new head before the check passes again.

<!-- review-gate negative test: trivial non-bump change; no validation evidence exists for this head. -->
