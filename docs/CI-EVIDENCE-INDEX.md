# CI evidence index

**Snapshot: 2026-09-30.** This index separates documented local checks from
CI evidence. It is an evidence map, not a report of passing runs.

## Current evidence

No CI workflow definition or CI run receipt is tracked in this repository.
Accordingly, this index reports no CI pass/fail status, pass rate, lane yield,
or measured result.

| Area | Documented check | What it establishes | CI evidence |
| --- | --- | --- | --- |
| Bottleneck corpus | [`python3 search.py --verify`](../bottlenecks/README.md), run from `bottlenecks/` | Checks entry-file hashes against `index.json` only; it does not verify the index or embeddings against the manifest | No linked workflow run or receipt |
| Erdős simulator lanes | [`python -m unittest -v test_run_all`](../research/quantum-erdos-sequences/README.md), run from `research/quantum-erdos-sequences/` | Runs the documented local runner tests; this is not a CI run or a hardware check | No linked workflow run or receipt |
| Bounded Aer exercise | [`python run_all.py --lane 123`](../research/quantum-erdos-sequences/README.md), run from `research/quantum-erdos-sequences/` | Produces local run evidence for a selected lane when dependencies are installed; it does not establish hardware execution, open-problem solutions, or quantum advantage | No linked workflow run or receipt |

The commands above are references to repository documentation, not evidence
that they were run for this snapshot. Proposed gates are not configured
workflows. Local run receipts and generated result files are not CI receipts.

## Recording future CI evidence

To make a CI claim, link a specific run and identify its commit SHA, workflow
and job, command, outcome, and relevant artifact or receipt. State the scope of
the check and any counts exactly as reported by that run; do not extrapolate a
lane yield or pass rate from this index. Label simulator evidence `LOCAL_SIM`.
It is not `REAL_QPU` evidence, quantum-advantage evidence, or a financial yield.

This documentation-only index does not authorize QPU submission, secrets,
settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work. The
[DeepNet master](https://agenci-main.github.io/deepnet-chat/) remains an
external documentation link only.
