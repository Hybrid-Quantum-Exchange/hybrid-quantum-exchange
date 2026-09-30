# CI evidence index

**Snapshot: 2026-09-30.** This page indexes documentation and evidence
requirements. It is not a CI dashboard, and it does not report passing runs,
yield, fidelity, or deployment status.

## Current inventory

No CI workflow, workflow run, artifact, or signed receipt is tracked in this
repository. The entries below are **proposed gates**, not completed checks.

| Gate | Repository documentation | Receipt required before claiming a pass |
| --- | --- | --- |
| Corpus entry integrity | [`bottlenecks/README.md`](../bottlenecks/README.md) documents `search.py --verify` | A linked run with the command, outcome, and artifact; compare the index and embeddings with manifest hashes separately |
| Erdős simulator lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents `run_all.py` and per-run receipts | A linked run and generated results showing attempted, executed, and classical-check counts |
| Grover verification artifacts | [`research/grover-verification/evidence/README.md`](../research/grover-verification/evidence/README.md) describes historical artifacts | A linked, reproducible receipt with source and artifact hashes |

Until those receipts are linked, report no CI pass rate or lane yield from
this index. Simulator results remain `LOCAL_SIM`, not hardware evidence.

## Boundaries

This index does not authorize QPU work, Settings, billing/IAM, invites,
secrets, DeepNet runtime/schema changes, or FIRE work. The [DeepNet master
endpoint](https://agenci-main.github.io/deepnet-chat/) is an external
documentation link only. See the [organization operations boundary](ORG-OPS.md)
and [honesty / QPU hold](HONESTY-QPU-HOLD.md).
