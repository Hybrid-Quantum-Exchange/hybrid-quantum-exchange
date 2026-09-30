# CI evidence index

**Snapshot: 2026-09-30.** This index describes candidate checks and the evidence
needed to support stronger claims; it is not a CI results page. No CI workflow or
run receipt is tracked in this repository. Accordingly, this index reports no
passing checks, pass rate, lane yield, fidelity, or measured result.

| Candidate check | Existing method documentation | Evidence not currently linked here |
| --- | --- | --- |
| Bottlenecks corpus integrity | [`bottlenecks/README.md`](../bottlenecks/README.md) documents `search.py --verify` for entry-file hashes against `index.json`, plus separate comparisons of the index and embeddings with manifest hashes. | A run receipt with the commands, outcomes, and artifacts. The entry check alone does not verify all corpus artifacts. |
| Aer lane verdicts | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents the local runner and its per-run receipts. | A linked run and generated receipts/results with attempted, executed, and classical-check counts from that run. No counts are claimed here. |

These are proposed evidence gates, not checks known to have run in CI. Aer
results are `LOCAL_SIM`, not `REAL_QPU`, hardware evidence, or speedup. The
external [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is a
documentation link only and is not evidence of an integration.
