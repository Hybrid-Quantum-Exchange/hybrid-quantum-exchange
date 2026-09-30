# CI evidence index (public)

**Snapshot: 2026-09-30.** This is an index of existing documentation and *proposed* CI evidence gates, not a record of passing CI or measured scientific outcomes. No project-specific workflow or run receipt for these gates is checked into this repository. GitHub-hosted agent/code-scanning checks, if present, are not receipts for the research gates below.

| Proposed gate | Existing source and check | Receipt needed before claiming a CI result | Current evidence |
| --- | --- | --- | --- |
| Corpus consistency | [Bottlenecks guide](../bottlenecks/README.md): `search.py --verify` compares entries against `index.json`; compare `index.json` and `embeddings.json` hashes separately with [`manifest.json`](../bottlenecks/manifest.json) | Linked run with revision, commands, exit statuses, and both hash comparisons | Procedure documented; no linked CI receipt |
| Simulator runner tests | [Erdős guide](../research/quantum-erdos-sequences/README.md): `python -m unittest -v test_run_all.py` | Linked run with revision, environment, command, and outcome | Procedure documented; no linked CI receipt |
| Simulator lanes | [Erdős guide](../research/quantum-erdos-sequences/README.md): bounded `run_all.py --lane` or full corpus | Linked run with selected lanes, runner manifest, lane receipts, logs, `RESULTS.json` / `RESULTS.md`, and separate execution, protocol, hypothesis, and reviewer statuses | Receipt format documented; no linked CI receipt |

To update this index, link the actual workflow run and retained artifacts, state the tested revision and scope, and distinguish a failed, incomplete, or missing run from a pass. `.runs/` receipts are ignored by Git and the top-level `RESULTS.*` files are latest-run convenience copies; documentation of their format is **not** evidence that a particular CI run occurred. Do not compute a pass rate from historical self-reports.

These checks would establish only their stated software or corpus consistency outcomes. They do not independently authenticate the corpus, validate scientific conclusions, solve Erdős problems, verify DeepNet behavior, or demonstrate `REAL_QPU` execution, fidelity, financial yield, or quantum advantage. See [public status](../STATUS.md), [honesty / QPU hold](HONESTY-QPU-HOLD.md), and [ORG-OPS boundary](ORG-OPS.md).
