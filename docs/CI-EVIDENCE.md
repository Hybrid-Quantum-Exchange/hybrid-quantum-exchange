# CI evidence index

**Snapshot: 2026-09-30.** This index separates documented commands from actual CI
receipts. It is not a CI dashboard.

| Check | Documentation | Receipt status |
| --- | --- | --- |
| Bottleneck entry verification | [`bottlenecks/README.md`](../bottlenecks/README.md) documents `search.py --verify` | No linked CI receipt in this repository |
| Aer lane execution | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents `run_all.py` and result artifacts | No linked CI receipt in this repository |
| Documentation links and posture | README, `PROGRESS.md`, `STATUS.md`, and `docs/` posture pages | Reviewed as documentation; not a workflow result |

No CI pass rate, lane yield, fidelity, speedup, hardware result, or financial return
is claimed here. Simulator output remains `LOCAL_SIM`; it is not `REAL_QPU` evidence.
Until a dated run receipt and artifact are linked, proposed gates remain proposals.

The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external
documentation link only. This index does not define, test, or verify its runtime,
schema, availability, or integration.
