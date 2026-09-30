# Hybrid Quantum Exchange

Public research notes and a product-facing shell for hybrid quantum and exchange work. The published simulator examples are demonstrations, not evidence of quantum advantage or an operating exchange.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

## What this repository is
Public research surface and product shell:

| Path | Contents | Evidence label |
| --- | --- | --- |
| [`bottlenecks/`](bottlenecks/README.md) | 999 science/tech bottleneck entries — **source-grounded notes, not peer review** | research notes |
| [`research/quantum-erdos-sequences/`](research/quantum-erdos-sequences/README.md) | 999 Erdős-associated Qiskit **Aer** simulator lanes; many use substitute finite properties | `LOCAL_SIM` |
| Site (`index.html`, `app.js`) | Product / exchange presentation shell, not evidence of a live service | n/a |

## Public documentation
- [Progress](PROGRESS.md) — dated, evidence-linked publication milestones and open limitations
- [Status](STATUS.md) — current public-scope snapshot; [research detail](RESEARCH-STATUS.md)
- [Honesty and evidence](docs/HONESTY.md) — what the published results can and cannot establish
- [Vault map](docs/VAULT.md) — public vs. private boundaries

## Sister repositories
- **Private control plane:** `quantum-project-ledger` (posture, progress, security — no secrets)
- **Private classical pilot:** `quantum-worker-pilot` (QAOA hybrid assign simulator — `LOCAL_SIM`)
- **Public OEIS fork:** `oeisdata` branch `sequences` (optional sequence lookup)
- **DeepNet master:** [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) (separate project; not a component of this repository)

## Honesty bar
- Published circuits here are ideal local simulator runs, with classical checks; no hardware measurements or speedup are established.
- Corpus entries are research starting points, not independent systematic reviews or validated scientific conclusions.
- See [honesty and evidence](docs/HONESTY.md) for verification boundaries and caveats.

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
