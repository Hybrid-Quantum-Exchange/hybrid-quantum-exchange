# Public documentation progress

**Snapshot: 2026-09-30.** This page tracks what is documented in this repository, not live service uptime, operational readiness, or work in private repositories. See [public status](STATUS.md) for scope and limitations and [research status](RESEARCH-STATUS.md) for the research snapshot.

| Surface | Documented here | Evidence still needed for stronger claims |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and a documented entry-hash check | Independent scientific review; `search.py --verify` checks entries against `index.json` only, so index and embeddings need separate comparison with the manifest hashes |
| [Erdős lanes](research/quantum-erdos-sequences/README.md) | 999 finite Qiskit Aer simulator exercises (`LOCAL_SIM`); the runner documents per-run local receipts | Linked receipts for any *reported* outcomes and independent review; a local receipt alone does not prove an open-problem solution or hardware result |
| [Site shell](index.html) | Public-facing product and exchange UI | Evidence of a working exchange or production backend |
| [DeepNet master](https://agenci-main.github.io/deepnet-chat/) | External context link only | No integration, endpoint behavior, schema, or availability is verified here |

This table indexes documentation, not completed CI gates: [research status](RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results) lists proposed gates but no linked CI run or artifact. A reproducible report would identify the exact source revision, command, environment, result, and linked receipt; for the corpus, entry hashes and the index/embeddings manifest checks are distinct. Do not report a pass rate from a procedure alone.

The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) applies throughout: no QPU submission or unlock, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work is authorized by this documentation. No `REAL_QPU` results, quantum advantage, financial returns, or live exchange are claimed.
