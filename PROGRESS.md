# Public documentation progress

**Snapshot: 2026-09-30.** This page tracks what is documented in this repository, not live service uptime, operational readiness, or work in private repositories. See [public status](STATUS.md) for scope and limitations and [research status](RESEARCH-STATUS.md) for the research snapshot.

| Surface | Documented here | Evidence still needed for stronger claims |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and a documented entry-hash check | Independent scientific review; `search.py --verify` checks entries against `index.json` only, so index and embeddings need separate comparison with the manifest hashes |
| [Erdős lanes](research/quantum-erdos-sequences/README.md) | 999 finite Qiskit Aer simulator exercises (`LOCAL_SIM`) | Linked run receipts for current outcomes; no open-problem solution or hardware result follows from simulator runs |
| [Site shell](index.html) | Public-facing product and exchange UI | Evidence of a working exchange or production backend |
| [DeepNet master](https://agenci-main.github.io/deepnet-chat/) | External documentation link only | No integration, endpoint behavior, schema, or availability is verified here |
| [ORG-OPS boundary](docs/ORG-OPS-BOUNDARY.md) | Documentation-only coordination scope and evidence boundaries | Explicit authorization and reproducible receipts for any stronger operational or hardware claim |

The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) applies throughout: no QPU submission or unlock, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work is authorized by this documentation. No `REAL_QPU` results, quantum advantage, or live exchange are claimed.
