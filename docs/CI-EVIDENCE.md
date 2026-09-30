# CI evidence index (public)

**Snapshot: 2026-09-30.** This index distinguishes documentation of possible checks from run evidence. It is not a live status feed or a claim of research validation. See [public status](../STATUS.md), [honesty / QPU hold](HONESTY-QPU-HOLD.md), and [operator boundaries](ORG-OPS.md).

| Scope | Existing reference | Evidence status |
| --- | --- | --- |
| PR code scanning | [CodeQL run for PR #136](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/actions/runs/36725556109) (completed successfully on 2026-09-30, for that PR's head only) | Code scanning only; not a test of this PR, corpus accuracy, research results, or hardware. |
| Bottlenecks entry hashes | [`search.py --verify` guide](../bottlenecks/README.md) | Proposed CI gate; no linked CI run receipt for the corpus here. The command compares entry files with `index.json` only; compare index and embeddings separately against `manifest.json` hashes. |
| Aer lane execution | [Runner and receipt guide](../research/quantum-erdos-sequences/README.md) | Proposed CI gate; no linked CI run receipt for current lane outcomes here. Historical self-reports are not revalidated totals. |

For a future evidence entry, link the immutable run or artifact, commit and command, date, environment/dependency versions, scope of files or lanes, exit status and failure details. For Aer, include the run's manifest, per-lane receipts and logs, and aggregate results; distinguish attempted from executed lanes, self-reported demo verdicts from independent classical review, and simulator results from hardware. For corpus checks, link both entry-hash results and separately checked manifest digests, explaining that hashes do not establish scientific correctness.

Until such receipts exist, do not report a research CI pass rate, lane yield, fidelity, quantum advantage, or hardware result from this index. A green code-scanning job does not supply that evidence. The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is a reference link only, not a CI target or verified integration.
