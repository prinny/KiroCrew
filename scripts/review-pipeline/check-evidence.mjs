/**
 * check-evidence.mjs — the public half of the review pipeline.
 *
 * Verifies opaque facts only, for one pull request:
 *   1. A pipeline evidence comment exists on the PR, posted by the
 *      account the pipeline itself posts as (see PIPELINE_AUTHOR).
 *      Marker comments from any other author are ignored: on a public
 *      repository anyone can comment, so an unverified author proves
 *      nothing.
 *   2. Its verification marker names this PR's current head SHA and a
 *      "clean" result, and the marker's digest matches the comment body.
 *   3. The comment links an evidence page, and that page exists and
 *      carries a verification marker for the same head SHA and result,
 *      with a matching digest.
 * The marker format is: run id, head SHA, result, content digest — the
 * check learns nothing beyond those facts.
 *
 * Modes:
 *   - Pull-request event: HEAD_SHA is supplied by the workflow; the job's
 *     own exit code is the check result.
 *   - Manual dispatch (EMIT_CHECK=true): the evaluation is additionally
 *     recorded as a check run on the PR's head commit, so it counts
 *     toward branch protection for PRs that predate the workflow.
 *
 * Env: GH_TOKEN, TARGET_REPO, PR_NUMBER, HEAD_SHA (optional),
 *      EMIT_CHECK ('true'|'false').
 */
import { createHash } from 'node:crypto';

const token = process.env.GH_TOKEN;
const repo = process.env.TARGET_REPO;
const prNumber = process.env.PR_NUMBER;
const emitCheck = process.env.EMIT_CHECK === 'true';
if (!token || !repo || !prNumber) {
  console.error('check-evidence: GH_TOKEN, TARGET_REPO and PR_NUMBER are required');
  process.exit(2);
}
const headers = {
  Authorization: `Bearer ${token}`,
  'Content-Type': 'application/json',
  'User-Agent': 'review-gate',
};
async function api(path, opts = {}) {
  const res = await fetch(`https://api.github.com${path}`, { headers, ...opts });
  return res;
}

const MARKER_RE = /<!-- pipeline-verification: run=(\d+) head=([0-9a-f]{40}) result=(clean|not-clean)(?: wiki=([A-Za-z0-9._-]+))? digest=([0-9a-f]{64}) -->/;
const COMMENT_MARKER = '<!-- kirocrew-validation-evidence -->';
// The account the validation pipeline posts its evidence comments as.
// A marker comment counts only when this account authored it.
const PIPELINE_AUTHOR = 'prinny';

function digestOf(body) {
  return createHash('sha256')
    .update(body.split('\n').filter((l) => !l.includes('pipeline-verification:')).join('\n'))
    .digest('hex');
}
function checkBody(body, expectedHead, label) {
  const problems = [];
  const m = MARKER_RE.exec(body || '');
  if (!m) {
    problems.push(`${label}: no verification marker found`);
    return { problems, marker: null };
  }
  const [, run, head, result, wiki, digest] = m;
  if (head !== expectedHead) problems.push(`${label}: marker covers head ${head}, not the current head ${expectedHead}`);
  if (result !== 'clean') problems.push(`${label}: pipeline result is "${result}"`);
  if (digestOf(body) !== digest) problems.push(`${label}: content digest does not match the marker`);
  return { problems, marker: { run, head, result, wiki, digest } };
}

const problems = [];
let headSha = process.env.HEAD_SHA || '';
{
  const res = await api(`/repos/${repo}/pulls/${prNumber}`);
  if (!res.ok) {
    console.error(`check-evidence: cannot read PR #${prNumber}: ${res.status}`);
    process.exit(2);
  }
  const pr = await res.json();
  if (headSha && pr.head.sha !== headSha) {
    console.log(`check-evidence: head moved since the triggering event (${headSha} -> ${pr.head.sha}); evaluating the current head`);
  }
  headSha = pr.head.sha;
}

let comment = null;
for (let page = 1; page <= 10 && !comment; page++) {
  const res = await api(`/repos/${repo}/issues/${prNumber}/comments?per_page=100&page=${page}`);
  if (!res.ok) {
    console.error(`check-evidence: cannot list comments: ${res.status}`);
    process.exit(2);
  }
  const comments = await res.json();
  if (comments.length === 0) break;
  comment = comments.find((c) => c.body && c.body.includes(COMMENT_MARKER)
    && c.user && c.user.login === PIPELINE_AUTHOR) || null;
}

let marker = null;
if (!comment) {
  problems.push('no pipeline evidence comment on this PR');
} else {
  const checked = checkBody(comment.body, headSha, 'evidence comment');
  problems.push(...checked.problems);
  marker = checked.marker;
  if (marker && !marker.wiki) problems.push('evidence comment: marker names no evidence page');
  if (marker && marker.wiki) {
    if (!comment.body.includes(`/wiki/${marker.wiki}`)) {
      problems.push('evidence comment: does not link the evidence page named in its marker');
    }
    const rawUrl = `https://raw.githubusercontent.com/wiki/${repo}/${marker.wiki}.md`;
    const res = await fetch(rawUrl, { headers: { 'User-Agent': 'review-gate' } });
    if (!res.ok) {
      problems.push(`evidence page ${marker.wiki}: does not resolve (${res.status})`);
    } else {
      const pageBody = await res.text();
      const pageChecked = checkBody(pageBody, headSha, 'evidence page');
      problems.push(...pageChecked.problems);
    }
  }
}

const ok = problems.length === 0;
for (const p of problems) console.log(`review-gate: FAIL — ${p}`);
if (ok) console.log(`review-gate: evidence verified for ${repo}#${prNumber} at ${headSha} (pipeline run ${marker.run})`);

if (emitCheck) {
  const res = await api(`/repos/${repo}/check-runs`, {
    method: 'POST',
    body: JSON.stringify({
      name: 'review-gate',
      head_sha: headSha,
      status: 'completed',
      conclusion: ok ? 'success' : 'failure',
      output: {
        title: ok ? 'Validation evidence verified' : 'Validation evidence missing or not clean',
        summary: ok
          ? `Pipeline evidence for head ${headSha} verified (run ${marker.run}).`
          : problems.map((p) => `- ${p}`).join('\n'),
      },
    }),
  });
  if (!res.ok) {
    console.error(`check-evidence: could not record check run: ${res.status} ${await res.text()}`);
    process.exit(2);
  }
  console.log('review-gate: check run recorded on the PR head commit');
}
process.exit(ok ? 0 : 1);
