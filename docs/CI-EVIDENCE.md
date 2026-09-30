# CI evidence index (public)

**Snapshot: 2026-09-30.** This index records documentation and evidence gaps.
It is not a workflow run, pass report, yield measurement, or deployment
approval. No CI workflow or run receipt is linked here yet.

| Check surface | Documented command or artifact | Evidence still needed |
| --- | --- | --- |
| Bottleneck entry integrity | [`search.py --verify`](../bottlenecks/README.md) checks entries against `index.json` | A linked run receipt and separate manifest comparisons for the index and embeddings |
| Erdős simulator lanes | [`run_all.py`](../research/quantum-erdos-sequences/README.md) generates `RESULTS.json` and `RESULTS.md` | A linked run with attempted, executed, and classical-check counts |
| Public documentation boundary | [ORG-OPS boundary](ORG-OPS.md), [honesty hold](HONESTY-QPU-HOLD.md) | Review evidence that remains docs-only; no runtime or hardware claim follows |

Until receipts are linked, do not report CI pass rates, lane yields, hardware
results, fidelity, or quantum advantage. Simulator evidence remains
`LOCAL_SIM`; QPU submission remains locked.

