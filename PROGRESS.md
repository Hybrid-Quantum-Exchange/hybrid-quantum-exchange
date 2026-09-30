# Public documentation progress

**Snapshot: 2026-09-30.** This page tracks what is documented in this repository, not live service uptime, operational readiness, or work in private repositories. See [public status](STATUS.md) for scope and limitations and [research status](RESEARCH-STATUS.md) for the research snapshot.

| Surface | Documented here | Evidence still needed for stronger claims |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and a documented entry-hash check | Independent scientific review; `search.py --verify` checks entries against `index.json` only, so index and embeddings need separate comparison with the manifest hashes |
| [Erdős lanes](research/quantum-erdos-sequences/README.md) | 999 finite Qiskit Aer simulator exercises (`LOCAL_SIM`) and a runner that writes local receipts | Linked receipts for current outcomes and independent review; historical totals predate the revised runner and are not revalidated |
| [Site shell](index.html) | Public-facing product and exchange UI | Evidence of a working exchange or production backend |
| [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) | External contextual link only | No integration, endpoint behavior, schema, or availability is verified here |

## How to check the published evidence

- For corpus consistency, follow the [bottlenecks integrity checks](bottlenecks/README.md#integrity-checks): compare entry hashes with the index and independently compare index and embeddings digests with `manifest.json`. These checks do not authenticate the manifest or review the science.
- For simulator claims, follow the [Erdős run-evidence guide](research/quantum-erdos-sequences/README.md#run-evidence-and-interpretation). Local receipts are generated per run and ignored by Git; no linked CI run or current full-corpus verdict is established by this page. A lane's self-reported pass is not independent verification.
- For the dated research snapshot and proposed (not passing) CI gates, see [research status](RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results).

The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) applies throughout: no QPU submission or unlock, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work is authorized by this documentation. No `REAL_QPU` results, quantum advantage, realized financial yield, or live exchange are claimed.
