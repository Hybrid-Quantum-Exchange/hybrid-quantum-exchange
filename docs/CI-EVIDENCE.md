# CI evidence index

**Snapshot: 2026-09-30.** This is an index of evidence requirements, not a list
of passing workflows or measured results. No CI workflow receipt is currently
tracked in this repository.

| Check area | Documentation pointer | Receipt required before a stronger claim |
| --- | --- | --- |
| Bottleneck corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) | A linked run with the command, outcome, artifact, and separate manifest-hash comparison |
| Erdős simulator lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) | A linked run with generated results and attempted, executed, and classical-check counts |
| DeepNet master | [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) | No runtime, schema, availability, integration, QPU, yield, or fidelity claim is established by the link |

Until receipts are linked, do not report a CI pass rate, lane yield, fidelity,
quantum advantage, or hardware result. Simulator output remains `LOCAL_SIM` or
`CLOUD_SIM`, never `REAL_QPU`.
