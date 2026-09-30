# CI evidence index

**Snapshot: 2026-09-30.** This is an index of documented checks and missing
evidence, not a CI result report. No CI workflow or run receipt is tracked in
this repository. The checks below are proposed gates, not passing CI checks.

| Proposed gate | Existing documentation (not a receipt) | Evidence still needed |
| --- | --- | --- |
| Bottleneck corpus integrity | [Bottlenecks guide](../bottlenecks/README.md) documents `search.py --verify` for entry hashes against `index.json`. | A linked CI run with the command, outcome, and artifact. Compare index and embedding hashes with the manifest separately before claiming full corpus integrity. |
| Aer lane verdicts | [Erdős simulator guide](../research/quantum-erdos-sequences/README.md) documents `run_all.py` and generated `RESULTS.json` / `RESULTS.md`. | A linked CI run and generated results showing attempted, executed, and classical-check counts for that run. |

Until receipts are linked, report no CI pass rate or lane yield from this
index. Simulator verdicts are `LOCAL_SIM`, not hardware evidence or speedup.
Do not invent yields, fidelity numbers, or other measured outcomes.

For a useful receipt, record the workflow/run link, commit, command, outcome,
and relevant artifact links. Do not include secrets or credentials in receipts.
