# Public documentation progress

**Snapshot: 2026-09-30.** This page tracks what is documented in this repository, not live service uptime, operational readiness, or work in private repositories. See [public status](STATUS.md) for scope and limitations, [research status](RESEARCH-STATUS.md) for the research snapshot, and the [CI evidence ledger](docs/CI-EVIDENCE.md) for the distinction between candidate checks and run receipts.

| Surface | Documented here | Evidence still needed for stronger claims |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and a documented entry-hash check | Independent scientific review; `search.py --verify` checks entries against `index.json` only, so index and embeddings need separate comparison with the manifest hashes |
| [Erdős lanes](research/quantum-erdos-sequences/README.md) | 999 finite Qiskit Aer simulator exercises (`LOCAL_SIM`) with self-reported demo verdicts | Linked run receipts for current outcomes and independent review for stronger claims; no open-problem solution or hardware result follows from simulator runs |
| [CI evidence](docs/CI-EVIDENCE.md) | Candidate checks and evidence requirements, not a passing workflow | Linked CI run, revision, logs, and artifacts for each claimed gate |
| [Site shell](index.html) | Public-facing product and exchange UI | Evidence of a working exchange or production backend |
| [DeepNet master](https://agenci-main.github.io/deepnet-chat/) | External documentation link only | No integration, endpoint behavior, schema, or availability is verified here |

The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) applies throughout: no QPU submission or unlock, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work is authorized by this documentation. No `REAL_QPU` results, quantum advantage, or live exchange are claimed.
