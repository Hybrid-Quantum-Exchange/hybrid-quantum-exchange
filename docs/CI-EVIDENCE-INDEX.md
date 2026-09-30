# CI evidence index

**Snapshot: 2026-09-30.** This is an index of where evidence could be found,
not a CI result, live status feed, or run receipt.

## Automation and available evidence

This checkout contains no project-owned CI workflow definitions or checked-in
CI run receipts. GitHub Actions lists platform-managed [CodeQL](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/actions/workflows/github-code-scanning/codeql)
and [Dependency Graph](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/actions/workflows/dependabot/update-graph)
workflow pages. Their existence does not establish that repository tests ran
or passed. Check the linked pages for any current run and inspect its commit,
conclusion, and artifacts before citing it.

| Check or evidence | Documented source | What it establishes / does not establish |
| --- | --- | --- |
| Erdős runner unit tests | [`test_run_all.py`](../research/quantum-erdos-sequences/test_run_all.py); run from `research/quantum-erdos-sequences/` with `python -m unittest -v test_run_all.py` | A local test command is documented; no CI run or result is recorded here. |
| Bottleneck entry hashes | [`bottlenecks/README.md`](../bottlenecks/README.md); `python3 search.py --verify` from `bottlenecks/` | Checks entry files against `index.json` only; it does not verify the index or embeddings against manifest hashes. |
| Bottleneck index and embeddings hashes | From the repository root, run `sha256sum bottlenecks/index.json bottlenecks/embeddings.json` and compare with `index_sha256` and `embeddings_sha256` in [`manifest.json`](../bottlenecks/manifest.json). | A local comparison checks consistency with the checked-in manifest, not independent authenticity or scientific correctness. |
| Erdős simulator run receipts | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents `run_all.py` and per-run receipts. | Receipts are local run records; this index contains no run receipt or aggregate outcome. Runs are `LOCAL_SIM`, not hardware evidence. |

## Evidence bar

When reporting a CI result, link the exact workflow run and commit, identify
the check and conclusion, and link any relevant artifact or receipt. Do not
turn the existence of a workflow, a local command, or historical self-reports
into a claim that CI passed. Do not infer yield, fidelity, quantum advantage,
or hardware execution from simulator output.

This index authorizes no QPU work, secrets, settings, billing/IAM, invites,
DeepNet runtime/schema, or FIRE changes. The [DeepNet master](https://agenci-main.github.io/deepnet-chat/)
is a documentation link only.
