# CI evidence index

**Snapshot: 2026-09-30.** This index distinguishes repository documentation
from execution evidence. No CI workflow definition or CI run receipt is tracked
in this repository at this snapshot. The gates below are proposed checks, not
configured or passing workflows.

| Area | Existing documentation | Evidence needed before claiming a pass |
| --- | --- | --- |
| Bottlenecks corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) describes entry-hash verification against `index.json` and separate manifest-hash checks | A linked run with command, outcome, and retained artifact; do not infer complete manifest integrity from the entry check alone |
| Erdős simulator lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents local execution, focused tests, and per-run receipts | A linked run and its receipt/results, including attempted, executed, and classical-check counts from that run |

Until run receipts are linked, this index reports **no CI pass rate, lane yield,
or measured performance**. Simulator output is `LOCAL_SIM`, not hardware
evidence, fidelity evidence, or quantum advantage. Historical self-reports in
the lane documentation are not current CI receipts.

See [public status](../STATUS.md), [documentation progress](../PROGRESS.md),
[research status](../RESEARCH-STATUS.md), and [honesty / QPU Hold](HONESTY-QPU-HOLD.md).
