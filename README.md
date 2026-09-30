# Hybrid Quantum Exchange

**Public research and a product shell** for quantum-computing and crypto-exchange software. The published research is exploratory, not a demonstration of quantum advantage or a live exchange.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

## What this repository is
Public research surface and product shell:

| Path | Contents | Evidence label |
| --- | --- | --- |
| `bottlenecks/` | 999 unsolved bottlenecks across science/tech — **source-grounded notes, not peer review** | research notes |
| `research/quantum-erdos-sequences/` | 999 Erdős-linked Qiskit **Aer** simulator lanes | `LOCAL_SIM` |
| Site (`index.html`, `app.js`) | Product / exchange UI shell, not an operating exchange | n/a |

## Public record
- [Progress](PROGRESS.md): what is published and what remains unverified.
- [Research status](RESEARCH-STATUS.md): evidence labels by track.
- [Honesty and evidence](docs/HONESTY.md): how to read results and claims.
- [Vault map](docs/VAULT.md): public versus private boundaries.

## Sister repositories
- **Private control plane:** `quantum-project-ledger` (posture, progress, security — no secrets)
- **Private classical pilot:** `quantum-worker-pilot` (QAOA hybrid assign simulator — `LOCAL_SIM`)
- **Public OEIS fork:** `oeisdata` branch `sequences` (optional sequence lookup)
- **DeepNet master:** [deepnet-chat](https://agenci-main.github.io/deepnet-chat/) (external project; not evidence for this repository's research claims)

## Honesty bar
- The published Erdős lanes run on an ideal local simulator, not hardware; they do not resolve open Erdős problems.
- No speedup / quantum-advantage claims without reproducible evidence and a classical baseline.
- No public hardware-run receipts are included here. Do not interpret a simulation label as `REAL_QPU`; see [honesty and evidence](docs/HONESTY.md).

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
