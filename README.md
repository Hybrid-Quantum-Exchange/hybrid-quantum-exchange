# Hybrid Quantum Exchange

**Building quantum-computing and crypto-exchange software** — with an honesty-first research vault for **human advancement via hybrid quantum computing**.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

**Project status:** This repository contains a public research corpus, local simulator
examples, and a product UI shell—not a live exchange or a demonstrated QPU service.
See [STATUS.md](STATUS.md) for the evidence and
[CI honesty](docs/ci-honesty.md) for what is (and is not) checked.

**DeepNet master endpoint:** [https://agenci-main.github.io/deepnet-chat/](https://agenci-main.github.io/deepnet-chat/)
(external project link; not an API or runtime integration in this repository).

## What this repository is
Public research surface and product shell:

| Path | Contents | Evidence label |
| --- | --- | --- |
| `bottlenecks/` | 999 unsolved bottlenecks across science/tech — **source-grounded notes, not peer review** | research notes |
| `research/quantum-erdos-sequences/` | 999 Erdős-linked Qiskit **Aer** simulator lanes | `LOCAL_SIM` |
| Site (`index.html`, `app.js`) | Product / exchange UI shell | n/a |

## Sister repositories
- **Private control plane:** `quantum-project-ledger` (posture, progress, security — no secrets)
- **Private classical pilot:** `quantum-worker-pilot` (QAOA hybrid assign simulator — `LOCAL_SIM`)
- **Public OEIS fork:** `oeisdata` branch `sequences` (optional sequence lookup)

## Honesty bar
- Classical client → cloud API → queued QPU or simulator. There is **no SSH shell into a QPU**.
- No speedup / quantum-advantage claims without reproducible evidence and a classical baseline.
- Hardware runs require explicit unlock + receipts (`REAL_QPU`). Default posture: **submit locked**.

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
