# CI evidence index

**Snapshot: 2026-09-30.** This page indexes evidence and gaps; it is not a CI
dashboard or a record of passing runs. No CI workflow or CI run receipt is
tracked in this repository at this snapshot. Do not infer a pass rate, lane
yield, or hardware result from this index.

| Area | Existing public documentation | Evidence needed before making a stronger claim |
| --- | --- | --- |
| Bottlenecks corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) documents `search.py --verify` for entry hashes against `index.json`. | A linked run with command, outcome, and artifact. Check `index.json` and `embeddings.json` separately against manifest hashes before claiming full corpus integrity. |
| Erdős simulator lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents the local `run_all.py` runner and per-run receipts. | A linked run and its receipts, with attempted, executed, and classical-check counts taken from that run. Label results `LOCAL_SIM`; they are not hardware evidence or proof of an open problem. |
| Hardware / QPU | [`HONESTY-QPU-HOLD.md`](HONESTY-QPU-HOLD.md) records the documentation-only hold. | No hardware execution or unlock is claimed. This documentation task does not authorize QPU work. |

Local simulator receipts describe a particular local execution; they are not
CI receipts, signed audits, or evidence of hardware performance. Do not invent
yield, return, fidelity, speedup, or quantum-advantage values.
