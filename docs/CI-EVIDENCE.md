# CI evidence index

**Snapshot: 2026-09-30.** This is an index of the evidence that would be
needed for stronger claims. It is not a CI dashboard, a run receipt, or a
yield report.

## Current record

No CI workflow or CI run receipt is tracked in this repository. The documented
checks below are proposed gates only:

| Area | Proposed check | Evidence required before claiming a pass |
| --- | --- | --- |
| Bottlenecks corpus | Run `search.py --verify` and compare the index and embeddings with the manifest hashes separately | Linked run, command, outcome, artifact, and manifest comparison |
| Aer lanes | Run `run_all.py` and retain the generated results and per-run receipts | Linked run, selected lanes, attempted/executed counts, classical-check counts, and artifacts |
| Documentation posture | Review the public status, honesty, and organization-operations boundaries | Dated review link; this page alone is not a CI receipt |

Until those receipts are linked, do not report a CI pass rate, lane yield,
fidelity, speedup, quantum advantage, production readiness, or financial
return. `LOCAL_SIM` is simulator evidence only; it is not `REAL_QPU` evidence.

## Scope boundary

This index is documentation-only. It does not authorize QPU submission,
settings, billing/IAM, invites, secrets, DeepNet runtime/schema changes, or
FIRE work. The [DeepNet master](https://agenci-main.github.io/deepnet-chat/)
is an external documentation reference, not a verified integration.

See [public status](../STATUS.md), [documentation progress](../PROGRESS.md),
and [honesty / QPU hold](HONESTY-QPU-HOLD.md).
