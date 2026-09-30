# Hybrid Quantum Exchange

**Building quantum-computing and crypto-exchange software** — with an honesty-first research vault for **human advancement via hybrid quantum computing**.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

## Research and paper posture

This repository contains public research notes, simulator exercises, and a product shell. It is not a peer-reviewed paper, does not claim a published research result, and does not claim quantum advantage or hardware speedup. Any future paper or hardware result must be reproducible, source-grounded, and compared with a classical baseline.

The DeepNet master documentation is available at [agenci-main.github.io/deepnet-chat](https://agenci-main.github.io/deepnet-chat/). This is a documentation link only; this repository does not import or run DeepNet.

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
- No secrets, billing/IAM, invitations, or runtime/schema integration are part of this documentation update.

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
