# Public documentation progress

**Snapshot: 2026-09-30.** This page tracks what is documented in this repository, not live service uptime, operational readiness, or work in private repositories. See [public status](STATUS.md) for scope and limitations and [research status](RESEARCH-STATUS.md) for the research snapshot.

| Surface | Documented here | Evidence still needed for stronger claims |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and a documented entry-hash check | Independent scientific review; `search.py --verify` checks entries against `index.json` only, so index and embeddings need separate comparison with the manifest hashes |
| [Erdős lanes](research/quantum-erdos-sequences/README.md) | 999 finite Qiskit Aer simulator exercises (`LOCAL_SIM`) | Linked run receipts for current outcomes; no open-problem solution or hardware result follows from simulator runs |
| [Site shell](index.html) | Public-facing product and exchange UI | Evidence of a working exchange or production backend |
| [DeepNet master](https://agenci-main.github.io/deepnet-chat/) | External documentation link only | No integration, endpoint behavior, schema, or availability is verified here |
| [Organization operations](docs/ORG-OPS.md) | Public-safe pointers and hold boundaries | No private operational details or authorization is published |
| [CI evidence index](docs/CI-EVIDENCE.md) | Proposed gates and receipt status | No CI pass rate, lane yield, or hardware result is claimed |

## CI evidence still outstanding

There is no tracked CI workflow or linked CI run receipt in this repository. The following are **proposed documentation checks**, not passing gates:

- **Corpus integrity:** Record the outcome of `python3 search.py --verify` from `bottlenecks/`, then compare the SHA-256 digests of `index.json` and `embeddings.json` with the corresponding fields in `manifest.json`. The entry check alone does not verify the index or embeddings; matching stored hashes does not establish scientific correctness or independently authenticate the manifest.
- **Simulator runner:** Record the outcome of `python -m unittest -v test_run_all.py` from `research/quantum-erdos-sequences/`. For any bounded Aer lane run, link the run's manifest, lane receipt, logs, and aggregate results, and distinguish attempted lanes, clean executions, valid verdicts, and self-reported demo passes. Generated `.runs/` receipts and `RESULTS.json` / `RESULTS.md` are ignored by Git; their presence on one machine is not a published CI result. A test pass or simulator verdict is not independent mathematical review or `REAL_QPU` evidence.

Before claiming a CI result, link a specific run with its revision, commands, outcomes, and retained evidence. No pass rate or lane yield is established by this page.

## Evidence receipt stub (no results recorded)

No CI or experiment run receipts are linked from this progress snapshot. The entries below are placeholders, not evidence; no yield or fidelity numbers are reported.

| Receipt | Link / artifact | Status | Claims supported |
| --- | --- | --- | --- |
| CI run | Not provided | No receipt linked | None |
| QPU run | Not provided | Submission locked; no run claimed | None |
| Yield / fidelity result | Not provided | No receipt linked | None |

The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) applies throughout: no QPU submission or unlock, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work is authorized by this documentation. No `REAL_QPU` results, quantum advantage, or live exchange are claimed.
