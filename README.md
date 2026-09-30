# Hybrid Quantum Exchange

**A public research surface and product shell for quantum-computing and crypto-exchange software.** Published research here is not evidence of a production exchange or quantum advantage.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

## Start here
- [Public status and limitations](STATUS.md) — what is published, what is simulated, and what is not established.
- [Research status](RESEARCH-STATUS.md) — dated research snapshot.
- [Bottlenecks guide](bottlenecks/README.md) and [Erdős simulator guide](research/quantum-erdos-sequences/README.md) — methods, verification, and caveats.
- [Vault map](docs/VAULT.md) — public vs. private surfaces.
- [DeepNet master endpoint](https://agenci-main.github.io/deepnet-chat/) — external link; this repository does not define or verify a DeepNet API or integration.

## What this repository is
Public research surface and product shell:

| Path | Contents | Evidence label |
| --- | --- | --- |
| `bottlenecks/` | 999 unsolved bottlenecks across science/tech — **source-grounded notes, not peer review** | research notes |
| `research/quantum-erdos-sequences/` | 999 Erdős-related Qiskit **Aer** simulator lanes; finite demonstrations, not solutions to open problems | `LOCAL_SIM` |
| Site (`index.html`, `app.js`, `styles.css`) | Product / exchange UI shell; not evidence of a live exchange | n/a |

## Sister repositories
- **Private control plane:** `quantum-project-ledger` (posture, progress, security — no secrets)
- **Private classical pilot:** `quantum-worker-pilot` (QAOA hybrid assign simulator — `LOCAL_SIM`)
- **Public OEIS fork:** `oeisdata` branch `sequences` (optional sequence lookup)

## Honesty bar
- A cloud API can queue QPU or simulator jobs; this repository does not establish an active QPU service. There is **no SSH shell into a QPU**.
- No speedup / quantum-advantage claims without reproducible evidence and a classical baseline.
- Hardware runs require explicit unlock + receipts (`REAL_QPU`). Default posture: **submit locked**.

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
