# CI evidence index (public)

This is an index of verification work that could be recorded; it is not a CI
dashboard. This repository currently publishes no linked CI run receipts.
Nothing in this page establishes a pass rate, yield, fidelity, hardware result,
or quantum advantage.

| Check | Documented command or artifact | Receipt still needed |
| --- | --- | --- |
| Bottleneck entry integrity | `bottlenecks/search.py --verify` and [`bottlenecks/README.md`](../bottlenecks/README.md) | A linked run with command, outcome, and artifact; manifest and embedding hashes require separate comparison |
| Aer lane verification | [`research/quantum-erdos-sequences/run_all.py`](../research/quantum-erdos-sequences/run_all.py) and generated results | A linked run with attempted, executed, and classical-check counts |
| Documentation links | README, status, progress, and this index | A reviewer check; link presence is not runtime validation |

Any future receipt should identify the commit, command, outcome, and retained
artifact. Aer and classical results remain `LOCAL_SIM` or classical evidence,
not `REAL_QPU` evidence.
