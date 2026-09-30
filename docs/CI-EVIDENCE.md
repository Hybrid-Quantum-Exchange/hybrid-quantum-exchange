# CI evidence index

**Snapshot: 2026-09-30.** This index distinguishes repository documentation from observed CI evidence. It is not a CI run report.

## Inventory

No CI workflow configuration or linked CI run receipt is present in this repository. Therefore, this index makes no claim of a passing CI run, lane yield, fidelity, hardware execution, or measured performance.

| Area | What is documented | Evidence not present here |
| --- | --- | --- |
| Bottleneck corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) describes the entry-hash check against `index.json` | Linked CI run, artifact, and separate verification of index and embeddings against manifest hashes |
| Erdős/Aer lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) describes `run_all.py` and generated results | Linked CI run and generated results for that run; attempted, executed, and classically checked counts |

These are documentation of possible checks, not receipts that the checks ran. Any future evidence claim should link to its run and artifacts and distinguish `LOCAL_SIM` from hardware results. See [Research status](../RESEARCH-STATUS.md) and [Honesty / QPU Hold](HONESTY-QPU-HOLD.md).
