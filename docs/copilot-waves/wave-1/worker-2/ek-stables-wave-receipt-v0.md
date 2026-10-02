# EKingston s-table evolution — Wave 1 receipt v0

## Worker, time, and scope

- Worker: Worker 2; lane: Copilot; assignment: wave-receipt.
- Topic: EKingston s-table evolution: v1 secant fits across beam-n6/n10/n15/n20 and alt regions.
- Actual collection checkpoints: **2026-10-02 07:56:43 and 07:57:07 America/Chicago (CDT, UTC−05:00)**, read from the session clock. These are collection checkpoints, not per-PR completion timestamps or elapsed-time measurements.
- Authorization and deliverables: [task PR #444](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/444). Its read-only GitHub metadata showed `draft: true`, base `main`, and `merged: false` during collection.

## What was done

Worker 2 posted the initial checklist before task investigation, inspected read-only GitHub PR metadata and file diffs, and authored this receipt and the companion [Worker 2 STATUS](STATUS-WORKER2-v0.md). This is documentation of evidence and gaps, not implementation or validation of numerical fits.

| Claim | Verdict | Read-only GitHub evidence / limitation |
| --- | --- | --- |
| An s-table timeline assignment exists. | VERIFIED | [PR #293 file diff](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/293/files), `docs/agent-openers/o0491-ek-stables-timeline.md`, lines 1–12, explicitly says “draft — work stub, not finished work.” |
| An s-table claim-verification assignment exists with suggested Copilot lane. | VERIFIED | [PR #413 file diff](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/413/files), `docs/agent-openers/o1331-ek-stables-verify-claims.md`, lines 1–12, is also a work stub, not completed verification. |
| v1 secant fits were completed for beam-n6. | UNVERIFIED | No numerical artifact or completed result was established by the inspected evidence. |
| v1 secant fits were completed for beam-n10. | UNVERIFIED | Same evidence limitation; no coefficients, inputs, or acceptance evidence established. |
| v1 secant fits were completed for beam-n15. | UNVERIFIED | Same evidence limitation; no coefficients, inputs, or acceptance evidence established. |
| v1 secant fits were completed for beam-n20. | UNVERIFIED | Same evidence limitation; no coefficients, inputs, or acceptance evidence established. |
| Alternate-region fits and their region identities were established. | UNVERIFIED | No canonical region list, fit artifacts, or comparison evidence established. |
| Historical fit completion times and implementing workers/lanes are known. | UNVERIFIED | Assignment stubs do not establish execution, authorship of fits, or completion times. |

GitHub code searches scoped to this repository for `"secant"` and `"beam-n6"` each returned zero matches. These limited search results are not proof that artifacts do not exist. The inspected stubs establish assignments only, not a v1 evolution history.

## What remains open

1. Locate canonical, revision-pinned v1 s-tables and input data for all four beam sizes and identify the alternate regions.
2. Obtain read-only evidence of coefficients, secant endpoints, method/units, fit quality, and acceptance criteria; do not compute or run experiments under this assignment.
3. Establish the version-to-version changes, actual fit authors/workers and lanes, and source-backed completion times. No per-PR seconds are inferred here.
4. Human review and any subsequent authorization remain outside this receipt. Draft status is not readiness or merge approval.

## Boundaries and cost

Only the two assigned Markdown paths are deliverables. No source/config/CI/tests/builds/installs/deploys, merges, ready-for-review toggles, purchases, paid add-ons, QPU, render/FIRE/RLS/LOCK, or provider actions were initiated by Worker 2. An [existing provider-bot comment on #444](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/444#issuecomment-5952821053) was read as GitHub evidence only; its external links were not followed. Independent automation activity and its cost are UNVERIFIED.

Gates remain exact direct Shawn phrases; none are authorized here. No phrase is invented or paraphrased as approval. Stop on any purchase or paid-access requirement. Rook remains separate. Copilot own-lane cost: **UNKNOWN** (not measured); no Codex/Claude costs are combined or attributed.

## STATUS receipt

```text
STATUS: DOCS RECEIPT COMPLETE; SCIENTIFIC CLAIMS UNVERIFIED
WAVE: 1
WORKER: 2
LANE: Copilot
COLLECTED_AMERICA_CHICAGO: 2026-10-02 07:56:43 / 07:57:07 CDT (UTC-05:00)
PR: #444; observed DRAFT against main; no merge or readiness authorization
DELIVERABLES: ek-stables-wave-receipt-v0.md; STATUS-WORKER2-v0.md
OPEN: canonical v1 artifacts, alt-region identities, fit validation, history/authorship, human review
GATES: exact direct Shawn phrases only; none authorized here
ROOK: separate
OWN_LANE_COST: UNKNOWN; Copilot only; not measured
```
