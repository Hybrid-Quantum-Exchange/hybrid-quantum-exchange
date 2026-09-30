# Public documentation progress

**Snapshot: 2026-09-30.** This page tracks what is documented in this repository, not live service uptime, operational readiness, or work in private repositories. See [public status](STATUS.md) for scope and limitations and [research status](RESEARCH-STATUS.md) for the research snapshot.

| Surface | Documented here | Evidence still needed for stronger claims |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and a documented entry-hash check | Independent scientific review; `search.py --verify` checks entries against `index.json` only, so index and embeddings need separate comparison with the manifest hashes |
| [Erdős lanes](research/quantum-erdos-sequences/README.md) | 999 finite Qiskit Aer simulator exercises (`LOCAL_SIM`) | Linked run receipts for current outcomes; no open-problem solution or hardware result follows from simulator runs |
| [Grover known-target study](research/grover-verification/README.md) | Separate ideal-simulator study with [selected historical artifacts](research/grover-verification/evidence/README.md) | Complete portable receipts and independent review; matched-seed replay does not establish independent replication |
| [Site shell](index.html) | Public-facing product and exchange UI | Evidence of a working exchange or production backend |
| [CI evidence index](docs/CI-EVIDENCE-INDEX.md) | Inventory of published artifacts and proposed checks; no CI workflow or linked CI run receipt tracked here | Linked workflow runs, outcomes and retained artifacts before reporting passing gates or rates |
| [Public org operations boundary](docs/ORG-OPS.md) | Navigation and documentation review rules | Not a record of account operations or permission to change private systems |
| [DeepNet master](https://agenci-main.github.io/deepnet-chat/) | External documentation link only | No integration, endpoint behavior, schema, or availability is verified here |

The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) applies throughout: no QPU submission or unlock, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work is authorized by this documentation. No `REAL_QPU` results, quantum advantage, or live exchange are claimed.
