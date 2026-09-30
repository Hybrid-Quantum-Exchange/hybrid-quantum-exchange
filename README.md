# Hybrid Quantum Exchange

**A public research surface and product shell for quantum-computing and crypto-exchange software.** Published research here is not evidence of a production exchange or quantum advantage.

Org: [github.com/Hybrid-Quantum-Exchange](https://github.com/Hybrid-Quantum-Exchange) · Site: [hybrid-quantum-exchange.vercel.app](https://hybrid-quantum-exchange.vercel.app)

## Start here
- [Public status and limitations](STATUS.md) — what is published, what is simulated, and what is not established.
- [Documentation progress](PROGRESS.md) — what is documented and what evidence is still missing.
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

For the dated public status snapshot, see [Research status](RESEARCH-STATUS.md); for the public/private boundary, see the [Vault map](docs/VAULT.md). The status snapshot is not a live hardware-status feed.

## Sister repositories
- **Private control plane:** `quantum-project-ledger` (posture, progress, security — no secrets)
- **Private classical pilot:** `quantum-worker-pilot` (QAOA hybrid assign simulator — `LOCAL_SIM`)
- **Public OEIS fork:** `oeisdata` branch `sequences` (optional sequence lookup)
- **DeepNet master:** [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) (external project; link only, not an integration claim)

## Operator endpoint note
- **DeepNet master endpoint:** [agenci-main.github.io/deepnet-chat](https://agenci-main.github.io/deepnet-chat/) — the operator-facing documentation link for this project.
- This repository does not configure or call a DeepNet API. The page, its availability, and any endpoint behavior are unverified here; the link is not evidence of a running integration or QPU access.

## Operator entry point
Use the [DeepNet Chat master face](https://agenci-main.github.io/deepnet-chat/) as the operator entry point. Do not invent yields or present estimates as measured results. **QPU: Hold** — no hardware submissions without explicit unlock and receipts. **Copilot: docs/CI only**; do not use it to change Settings, billing, QPU, or runtime.

## Honesty bar
- A cloud API can queue QPU or simulator jobs; this repository does not establish an active QPU service. There is **no SSH shell into a QPU**.
- No speedup / quantum-advantage claims without reproducible evidence and a classical baseline.
- Hardware runs require explicit unlock + receipts (`REAL_QPU`). Default posture: **submit locked**.
- For the dated public track-by-track status, see [Research status](RESEARCH-STATUS.md); for the boundary between public research and private control, see the [vault map](docs/VAULT.md). `LOCAL_SIM` results are not hardware results.

## QPU Hold — 2026-09-30
This repository is on **QPU Hold**: no QPU work is being requested or authorized. This is a documentation-only posture note; the DeepNet master is [text-only documentation](https://agenci-main.github.io/deepnet-chat/).

Hard locks for this hold: **NO QPU, NO Settings, NO billing/IAM, NO invites, NO secrets, NO DeepNet runtime/schema, NO FIRE**. Progress and status remain documentation-only, with no `REAL_QPU` submission or unlock.

## Operator reference
- DeepNet Chat master face: https://agenci-main.github.io/deepnet-chat/
- Do not invent yields or present projections as realized returns.
- **QPU Hold:** do not submit hardware jobs or imply hardware results while submission is locked.
- **Copilot = docs/CI only:** do not use it to change Settings, billing, QPU, or runtime.

## Compact-500 posture
Compact-500 is a classical-first paper/simulation effort, not a live financial-yield platform. Paper or simulation outcomes are not verified hardware measurements or financial yields; no live yields, returns, realized profits, quantum advantage, or hardware authorization are implied. This repository's exchange UI is a product shell, not evidence of live execution or earnings.

## License
See `LICENSE`. Third-party research trees keep their upstream attributions.
