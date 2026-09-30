# CI evidence index

**Snapshot: 2026-09-30.** This repository has no tracked CI workflow or linked CI run receipt for the checks below. The table is an index of possible evidence and existing documentation, **not** a list of passing CI checks. No CI pass rate, lane yield, fidelity, or hardware result can be inferred from it.

| Surface | Existing public material | CI receipt status / missing evidence |
| --- | --- | --- |
| Bottlenecks integrity | [Corpus verification guide](../bottlenecks/README.md) describes entry hashes checked against `index.json` | No linked CI run. A receipt would need the command, outcome, and artifact identifiers; index and embeddings must also be compared separately against `manifest.json` hashes. These checks do not establish scientific correctness. |
| Erdős simulator lanes | [Runner evidence guide](../research/quantum-erdos-sequences/README.md) describes local `.runs/` receipts and `RESULTS.json` / `RESULTS.md` | No linked CI run or current aggregate receipt here. A run link and artifacts with attempted, executed, and verdict counts would be needed before reporting current results. Self-reported demo verdicts are not independent verification. |
| Grover known-target study | [Historical evidence note](../research/grover-verification/evidence/README.md) describes selected local simulator artifacts | Not CI evidence or a complete portable receipt. Historical artifacts do not establish a new execution, independent replication, hardware run, or quantum advantage. |

For the research snapshot see [research status](../RESEARCH-STATUS.md); for the documentation-only boundary see [honesty / QPU hold](HONESTY-QPU-HOLD.md). The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external reference only, not a CI run, integration receipt, or QPU endpoint verified here.
