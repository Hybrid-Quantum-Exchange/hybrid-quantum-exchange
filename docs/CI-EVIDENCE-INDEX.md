# CI and research evidence index

**Documentation snapshot: 2026-09-30.** No CI workflow or linked CI run receipt
is tracked in this repository. The checks below are documentation or proposed
gates, **not passing CI jobs**. This index does not claim that a check ran for
this change or that historical results were reproduced.

| Surface | Available public material | Missing evidence / limit |
| --- | --- | --- |
| Bottlenecks corpus | [Method and integrity instructions](../bottlenecks/README.md), [`search.py`](../bottlenecks/search.py), [`manifest.json`](../bottlenecks/manifest.json) | `search.py --verify` compares entry hashes with `index.json` only. Separately compare index and embeddings hashes with the manifest; none of these checks establish scientific correctness. No linked CI run receipt. |
| Erdős Aer lanes | [Runner and receipt format](../research/quantum-erdos-sequences/README.md), [`run_all.py`](../research/quantum-erdos-sequences/run_all.py), [runner tests](../research/quantum-erdos-sequences/test_run_all.py) | Run receipts live in ignored local `.runs/`; latest `RESULTS` copies are ignored. Historical aggregates predate the hardened runner and are not revalidated totals. No linked CI run or independently reviewed lane verdicts. |
| Grover known-target study | [Study protocol and limitations](../research/grover-verification/README.md), [selected historical simulator artifacts](../research/grover-verification/evidence/README.md), [recovery checkpoint](../research/VERA-CHECKPOINT-20260920.md) | Published artifacts are a sanitized subset, not complete portable receipts; original manifests and receipts remain private. Matched-seed replay is not independent replication, hardware evidence or T9 acceptance. |

## What would qualify as a CI receipt

A future claim about a passing gate needs a link to the exact workflow run,
revision, command, outcome and retained artifacts, including failures. For a
corpus-integrity claim, record entry verification **and** the separate manifest
hash comparisons. For lane outcomes, link the generated per-run receipts and
report attempted, executed and self-reported demo verdicts separately; those
verdicts are not independent classical review. See the [proposed gates in
research status](../RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results).

No CI pass rate, simulator yield, fidelity, financial yield, quantum advantage
or `REAL_QPU` outcome follows from this index. The [QPU hold](HONESTY-QPU-HOLD.md)
and [public org operations boundary](ORG-OPS.md) apply. [DeepNet
master](https://agenci-main.github.io/deepnet-chat/) is an external reference,
not evidence of a connected runtime.
