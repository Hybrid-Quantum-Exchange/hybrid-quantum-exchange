# Vault map (public-safe)

Hybrid-Quantum-Exchange separates **public research** from **private control**.

| Repo | Visibility | What you get here |
| --- | --- | --- |
| This repo (`hybrid-quantum-exchange`) | public | Bottlenecks corpus, Erdős/Aer `LOCAL_SIM` lanes, product shell |
| `oeisdata` (`sequences`) | public | OEIS content fork for sequence lookup |
| `quantum-worker-pilot` | private | Classical hybrid QAOA pilot |
| `quantum-project-ledger` | private | Posture, progress, security rules (no secrets in git) |

Hardware / paid cloud QPU work is gated. Default: **submit locked**. Labels: `LOCAL_SIM` · `CLOUD_SIM` · `REAL_QPU`.

## Public evidence boundaries

- The [bottlenecks corpus](../bottlenecks/README.md) is a set of source-grounded research notes, not peer-reviewed findings. Its keyword lookup is not semantic search; entry-hash verification alone does not authenticate the index or embeddings against their manifest.
- The [Erdős lanes](../research/quantum-erdos-sequences/README.md) run on an ideal local simulator. Many use substitute finite properties; a PASS does not solve the named open problem or establish speedup over a classical baseline.
- This repository publishes a [product UI shell](../README.md), not evidence of a live exchange. Private pilot and provider statements in [research status](../RESEARCH-STATUS.md) are dated reports, not publicly verifiable run receipts.

Use `LOCAL_SIM` only for local simulator evidence; do not relabel it as `CLOUD_SIM` or `REAL_QPU`. Hardware claims require their own independently checkable receipts and a classical comparison for any speedup claim.
