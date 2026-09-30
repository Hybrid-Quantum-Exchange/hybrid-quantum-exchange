# CI and evidence index (public)

**Snapshot: 2026-09-30.** This is an index of published documentation and artifacts, not a CI dashboard. No workflow definition or linked CI run receipt is tracked in this repository; none of the items below is a claimed passing CI gate.

| Surface | Available here | What is not established |
| --- | --- | --- |
| [Bottlenecks](../bottlenecks/README.md) | Entry-hash verification is documented; `search.py --verify` compares entries with `index.json` | A linked CI run or independent scientific review. Index and embeddings must be compared separately with hashes in `manifest.json`; the manifest itself is not independently authenticated |
| [Erdős simulator lanes](../research/quantum-erdos-sequences/README.md) | Runner, focused tests, and local per-run receipt format are documented | A linked CI run, independently reviewed verdicts, reproducible full-corpus totals, or hardware results. Generated `.runs/` receipts are local and not published here |
| [Grover known-target study](../research/grover-verification/evidence/README.md) | Selected historical simulator outputs and circuit exports are published | Complete portable historical run receipts, independent replication, T9 acceptance, or quantum advantage |

To support a future CI claim, link the workflow and specific run, record the revision, command, outcome, and relevant artifacts, and distinguish local simulator evidence (`LOCAL_SIM`) from any separately authorized hardware evidence. Until then, do not report a CI pass rate or infer lane yields from this index.

The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external documentation reference only; this index does not verify its availability or integration. The [honesty / QPU hold](HONESTY-QPU-HOLD.md) remains in force: this page authorizes no QPU submission, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work.
