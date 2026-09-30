# Claim and evidence guide

Use the narrowest claim supported by the public artifact. The [README](../README.md) maps the repository, [progress](../PROGRESS.md) lists reproducible checks, and [research status](../RESEARCH-STATUS.md) distinguishes public artifacts from private reports.

| Artifact | What it supports | What it does not support |
| --- | --- | --- |
| [Bottlenecks corpus](../bottlenecks/README.md) | Source-grounded research leads and reproducible file hashes | Peer review, exhaustive literature coverage, or scientific correctness of embeddings |
| [Erdős simulator lanes](../research/quantum-erdos-sequences/README.md) | Finite circuit demonstrations checked against classical answers on ideal Aer simulation | Solving the named open problems, hardware results, or quantum speedup |
| Static site | A public description of intended direction | A deployed exchange, live trading, or proof that described workflows are implemented |
| Separately maintained projects | Context only | Results independently verified by this repository |

When quoting runner totals, name the run and its conditions: the [lane README](../research/quantum-erdos-sequences/README.md#totals) describes one run and notes unseeded-shot variability and substitute properties. When quoting corpus search results, note that `search.py --top` is keyword matching, not vector search. For corpus integrity, `search.py --verify` checks entry hashes against `index.json`; compare the index and embeddings with `manifest.json` independently.

No result published here establishes a production system, hardware experiment, or advantage over a classical baseline.
