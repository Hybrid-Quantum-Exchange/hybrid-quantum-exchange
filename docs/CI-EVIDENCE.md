# CI evidence index

**Snapshot: 2026-09-30.** This index distinguishes proposed checks from recorded run evidence. It is not a CI dashboard or a report of passing runs.

No CI workflow or CI run receipt is tracked in this repository. Consequently, this index reports no CI pass rate, lane yield, or measured performance.

| Area | Documented check or source | Evidence not currently linked |
| --- | --- | --- |
| Bottlenecks corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) describes `search.py --verify`, which checks entry hashes against `index.json`. | A run receipt with command, outcome, and artifact; separate verification of the index and embeddings against `manifest.json` hashes. |
| Erdős simulator lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) describes `run_all.py` and generated results. | A linked run receipt and its results, including attempted, executed, and classical-check counts from that run. |

These are proposed evidence targets, not passing CI gates. Aer and other simulator results are `LOCAL_SIM`; they are not QPU results, fidelity measurements, hardware speedup, or evidence of quantum advantage.
