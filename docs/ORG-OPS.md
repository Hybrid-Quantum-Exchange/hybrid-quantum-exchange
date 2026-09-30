# Public operations and CI evidence index

**Snapshot: 2026-09-30.** This is a documentation index, not an operational
dashboard, live service status, or evidence that a check has run.

## Scope and posture

- [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external
  documentation link only. This repository does not define or verify its API,
  runtime, schema, availability, or an integration.
- **QPU Hold:** no QPU submission or unlock is authorized. This documentation
  does not authorize settings, billing/IAM, invites, secrets, DeepNet
  runtime/schema, or FIRE work.
- Do not report yields, fidelity, speedup, or quantum advantage without
  linked, reproducible evidence and appropriate baselines. No such result is
  asserted by this index.

See [public status](../STATUS.md), [documentation progress](../PROGRESS.md),
[honesty / QPU hold](HONESTY-QPU-HOLD.md), and
[research status](../RESEARCH-STATUS.md) for the detailed public boundaries.

## CI and evidence

There are no CI run receipts indexed here. The checks listed in
[research status](../RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results)
are **proposed gates**, not passing checks or measured results.

| Area | Documented check | Evidence needed before reporting an outcome |
| --- | --- | --- |
| Bottlenecks corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) documents `search.py --verify` for entry hashes against `index.json` | Link the CI run, command, outcome, and artifact. The command does not verify `index.json` or embeddings against the manifest; check those hashes separately. |
| Aer lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents `run_all.py` and generated results | Link the CI run and generated results, and report attempted, executed, and classically checked counts only as recorded by that run. Simulator outcomes are `LOCAL_SIM`, not hardware evidence. |

Until receipts are linked, do not infer CI pass rates, lane yields, or
hardware results. A proposed gate or a generated artifact without its run
context is not a CI receipt.
