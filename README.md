# Hybrid Quantum Exchange

**Public research and a product UI shell** for hybrid quantum-computing and exchange concepts. Published simulator experiments are not evidence of hardware performance or a live exchange.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

**DeepNet master:** [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) (related project; this repository does not contain its implementation).

## What this repository is
Public research surface and product shell; see [public progress](PROGRESS.md), [research status](RESEARCH-STATUS.md), and the [public-safe vault map](docs/VAULT.md) for scope and limitations.

| Path | Contents | Evidence label |
| --- | --- | --- |
| [`bottlenecks/`](bottlenecks/README.md) | 999 research notes across science/tech; source coverage is thematic, not 999 systematic reviews | research notes |
| [`research/quantum-erdos-sequences/`](research/quantum-erdos-sequences/README.md) | 999 small Qiskit **Aer** simulator lanes; finite checks, not solutions to the named open problems | `LOCAL_SIM` |
| Site (`index.html`, `app.js`) | Product / exchange UI shell; not a live trading service | UI only |

## Sister repositories
- **Private control plane:** `quantum-project-ledger` (posture, progress, security — no secrets)
- **Private classical pilot:** `quantum-worker-pilot` (QAOA hybrid assign simulator — `LOCAL_SIM`)
- **Public OEIS fork:** `oeisdata` branch `sequences` (optional sequence lookup)

## Honesty bar
- Simulator verdicts compare small finite properties to classical checks; they do not solve the named open problems or establish speedup.
- The public site is a presentation shell, not proof of operational exchange infrastructure.
- No speedup / quantum-advantage claims without reproducible evidence and a classical baseline. See [evidence guidance](docs/VAULT.md).

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
