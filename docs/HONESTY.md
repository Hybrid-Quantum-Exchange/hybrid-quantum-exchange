# Honesty and evidence

This repository publishes research notes, local simulation scripts, and a product UI shell. [Progress](../PROGRESS.md) and [research status](../RESEARCH-STATUS.md) describe what is publicly visible; neither is a substitute for independent validation.

| Label | Meaning in public documentation |
| --- | --- |
| Research notes | Source-grounded starting points, not peer-reviewed conclusions. See the [corpus limitations](../bottlenecks/README.md). |
| `LOCAL_SIM` | Local simulator evidence only. The [Erdős lanes](../research/quantum-erdos-sequences/README.md) use ideal, noiseless Qiskit Aer and finite classical checks. |
| `CLOUD_SIM` | Cloud simulator evidence, if accompanied by reproducible run records; none is presented here. |
| `REAL_QPU` | Hardware evidence requiring attributable run records and a classical baseline; none is presented here. |

The Erdős scripts do not solve the named open problems: many lanes use substitute finite properties, some reported passes vary with unseeded shots, and a passing circuit is not evidence of speedup. Read the [lane totals and limitations](../research/quantum-erdos-sequences/README.md#totals) before quoting them.

The bottlenecks index and embeddings support lookup and similarity, not scientific correctness. `search.py --top` is keyword matching, not semantic search; `--verify` checks entry hashes against `index.json`, not the index or embeddings against the manifest. See [search and integrity guidance](../bottlenecks/README.md#searching) and the [manifest](../bottlenecks/manifest.json).

No private-project outcomes, production exchange functionality, or hardware results should be inferred from this public repository. Claims of advantage require a documented workload, reproducible measurements, and a comparable classical baseline. The [vault map](VAULT.md) explains the public/private boundary.
