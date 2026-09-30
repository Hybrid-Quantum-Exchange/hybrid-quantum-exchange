# Hybrid Quantum Exchange

**Building quantum-computing and crypto-exchange software** — with an honesty-first research vault for **human advancement via hybrid quantum computing**.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

## What this repository is
Public research surface and product shell:

| Path | Contents | Evidence label |
| --- | --- | --- |
| [`bottlenecks/`](bottlenecks/README.md) | 999 unsolved bottlenecks across science/tech — **source-grounded notes, not peer review** | research notes |
| [`research/quantum-erdos-sequences/`](research/quantum-erdos-sequences/README.md) | 999 Erdős-themed Qiskit **Aer** simulator lanes; many use substitute finite properties rather than an OEIS sequence | `LOCAL_SIM` |
| Site (`index.html`, `app.js`) | Product / exchange UI shell | n/a |

Public [progress snapshot](PROGRESS.md), [research status](RESEARCH-STATUS.md), and [evidence boundaries](docs/VAULT.md) describe what is and is not demonstrated here.

## Sister repositories
- **Private control plane:** `quantum-project-ledger` (posture, progress, security — no secrets)
- **Private classical pilot:** `quantum-worker-pilot` (QAOA hybrid assign simulator — `LOCAL_SIM`)
- **Public OEIS fork:** `oeisdata` branch `sequences` (optional sequence lookup)
- **DeepNet master:** [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) (external project; not an integration or result of this repository)

## Honesty bar
- Classical client → cloud API → queued QPU or simulator. There is **no SSH shell into a QPU**.
- No speedup / quantum-advantage claims without reproducible evidence and a classical baseline.
- Hardware runs require explicit unlock + receipts (`REAL_QPU`). Default posture: **submit locked**.
- Simulator demonstrations are not evidence of hardware runs, solved Erdős problems, or a deployed exchange. See the [lane limitations](research/quantum-erdos-sequences/README.md) and [public status](RESEARCH-STATUS.md).

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
