# CI evidence index

**Snapshot: 2026-09-30.** This index records where evidence would be linked; it
does not claim that a workflow or run has passed.

| Area | Documentation currently available | Evidence still required |
| --- | --- | --- |
| Bottlenecks corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) documents `search.py --verify` | A linked run receipt and separate manifest/index/embedding hash verification |
| Erdős simulator lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents `run_all.py` and result artifacts | A linked run receipt with attempted, executed, and classical-check counts |
| QPU or paid cloud | None | Explicit unlock and `REAL_QPU` receipts; no such work is authorized by this repository |

There is currently no CI workflow or run receipt tracked here. `LOCAL_SIM`
results are simulator evidence only, not hardware evidence, speedup, yield, or
fidelity measurements. Do not infer a pass rate, lane yield, or financial
return from this index.
