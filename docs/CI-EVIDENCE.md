# CI evidence index (public)

This index distinguishes documentation of a check from a receipt that proves
the check ran. It is not a CI dashboard, pass-rate report, or yield report.

| Area | Public pointer | Receipt status |
| --- | --- | --- |
| Bottleneck entry hashes | [`bottlenecks/README.md`](../bottlenecks/README.md) | No CI receipt is tracked here |
| Aer simulator lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) | No current run receipt is tracked here |
| Public posture | [`HONESTY-QPU-HOLD.md`](HONESTY-QPU-HOLD.md) | Documentation only; not a workflow result |

Until a dated command, outcome, and artifact are linked, do not report a CI
pass rate, lane yield, hardware result, fidelity, or financial yield. Any
simulator result remains `LOCAL_SIM`, not `REAL_QPU`.

See [organization operations](ORG-OPS.md) for the documentation-only boundary
and [DeepNet master](https://agenci-main.github.io/deepnet-chat/) for the
external reference link. The DeepNet link does not establish CI, runtime,
schema, or integration evidence.
