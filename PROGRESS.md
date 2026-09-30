# Public documentation progress

**Snapshot: 2026-09-30.** This page tracks what is documented in this repository, not live service uptime, operational readiness, or work in private repositories. See [public status](STATUS.md) for scope and limitations, [research status](RESEARCH-STATUS.md) for the research snapshot and CI evidence index, and the [ORG-OPS map](docs/ORG-OPS.md) for public-safe repository boundaries.

## Burn-wave U documentation refresh

This refresh is documentation-only. The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external reference link, not an integration, runtime/schema, or availability claim. The [ORG-OPS map](docs/ORG-OPS.md) describes public/private boundaries without exposing credentials or operational settings.

QPU Hold remains in effect: no QPU, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work. No hardware run, yield, fidelity, speedup, or quantum-advantage result is claimed. The CI evidence index in [research status](RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results) remains an index of proposed gates; it is not a CI receipt or pass report.

| Surface | Documented here | Evidence still needed for stronger claims |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and a documented entry-hash check | Independent scientific review; `search.py --verify` checks entries against `index.json` only, so index and embeddings need separate comparison with the manifest hashes |
| [Erdős lanes](research/quantum-erdos-sequences/README.md) | 999 finite Qiskit Aer simulator exercises (`LOCAL_SIM`) | Linked run receipts for current outcomes; no open-problem solution or hardware result follows from simulator runs |
| [Site shell](index.html) | Public-facing product and exchange UI | Evidence of a working exchange or production backend |
| [DeepNet master](https://agenci-main.github.io/deepnet-chat/) | External documentation link only | No integration, endpoint behavior, schema, or availability is verified here |
| [CI evidence index](RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results) | Proposed gates and existing documentation references | No tracked CI workflow/run receipt or measured pass rate |

The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) applies throughout: no QPU submission or unlock, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work is authorized by this documentation. No `REAL_QPU` results, quantum advantage, or live exchange are claimed.
