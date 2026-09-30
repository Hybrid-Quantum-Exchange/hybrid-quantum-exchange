# CI evidence ledger (public)

**Documentation snapshot: 2026-09-30.** This page indexes evidence available in this repository; it is not a CI run, a live Actions status feed, or a record of measured yields. No CI workflow or linked CI run receipt is tracked here. A local command or historical report is not a passing CI gate.

| Candidate check | What can be inspected here | CI evidence status |
| --- | --- | --- |
| Bottleneck entry integrity | [`search.py --verify`](../bottlenecks/README.md#integrity-checks) checks entry file hashes against `index.json`; compare `index.json` and `embeddings.json` hashes separately with `manifest.json`. | No linked run, logs, or artifact. Checked-in hashes show internal consistency only, not independent authentication or scientific review. |
| Erdős runner regression tests | [`test_run_all.py`](../research/quantum-erdos-sequences/README.md#run-evidence-and-interpretation) can test the receipt runner without running the corpus. | No linked CI run or test report. |
| Erdős simulator lanes | [`run_all.py`](../research/quantum-erdos-sequences/README.md#run-evidence-and-interpretation) writes per-run manifests, lane receipts and aggregate results under ignored `.runs/`; top-level `RESULTS.json` / `RESULTS.md` are latest-run convenience copies. | No linked current CI run or published per-run receipt here. Historical aggregates in the lane guide predate the current parser and are not revalidated. |
| Grover verification study | [Selected historical artifacts](../research/grover-verification/evidence/README.md) are published with stated omissions. | Not a complete portable receipt, new run, CI pass, or independent replication. |

## Bar for a future CI claim

For each claimed gate, link the workflow run and immutable revision, record the command and UTC run time, report exit status, and retain logs and artifact identifiers or hashes. For simulator claims, link the **same run's** manifest, per-lane receipts, and aggregate results; distinguish selected, attempted, cleanly executed, valid-protocol, and self-reported-pass counts, including failures and timeouts. Do not equate `verified_against_classical` with independent verification: the runner documents it as an alias for `self_reported_demo_pass`. Do not turn a historical aggregate, proposed check, or unlinked local result into a CI success rate.

This index authorizes no runs or operational changes. Simulator results are `LOCAL_SIM`, not `REAL_QPU`, hardware speedup, open-problem proofs, exchange operation, or financial yields. See [honesty / QPU hold](HONESTY-QPU-HOLD.md) and [public status](../STATUS.md).
