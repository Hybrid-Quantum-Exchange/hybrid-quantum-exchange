# CI evidence index

**Snapshot: 2026-09-30 08:25 UTC.** This index records externally visible CI
receipts and their scope. It is not a service-status page, yield report, or
hardware result.

## Green receipt

| Workflow | Receipt | Scope |
| --- | --- | --- |
| CodeQL | [PR #108 run 36721739926](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/actions/runs/36721739926) — `success` | Static analysis for the pull-request revision |

This green CodeQL receipt does not certify the bottleneck corpus, Aer lanes,
the exchange UI, DeepNet, QPU access, or production readiness.

## Evidence still needed

| Area | Documented check | Stronger claim requires |
| --- | --- | --- |
| Bottlenecks corpus | `search.py --verify` checks entry hashes against `index.json` | A linked run with command, outcome, artifact, and separate manifest/embedding hash verification |
| Aer simulator lanes | `run_all.py` and generated `RESULTS.json` / `RESULTS.md` | A linked run with selected lanes, counts, classical checks, and artifacts |
| Hardware | Submission remains held; published work is `LOCAL_SIM` | Explicit unlock and `REAL_QPU` receipts; no fidelity, yield, or speedup is claimed here |

Do not derive a CI pass rate, lane yield, fidelity, quantum advantage, or
financial return from this index. The [DeepNet master](https://agenci-main.github.io/deepnet-chat/)
is an external documentation link only; no runtime, schema, or integration is
claimed.
