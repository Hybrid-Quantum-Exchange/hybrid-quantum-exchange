# Public operator map

**Snapshot: 2026-09-30.** This is a documentation map, not an operational runbook, access grant, or live service status.

## Operator entry and boundaries

| Surface | Publicly established | Operator boundary |
| --- | --- | --- |
| DeepNet master | [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) is an external documentation reference. | Link only; this repository does not establish its availability, behavior, schema, or integration. |
| This repository | Public research notes, `LOCAL_SIM` exercises, and a product UI shell are documented. | Limit this task to documentation. The UI is not evidence of a live exchange or production backend. |
| Simulation and hardware | Aer exercises are labeled `LOCAL_SIM`; no `REAL_QPU` result is claimed. | QPU submission is on hold. Do not unlock, submit jobs, or describe simulation as hardware evidence. |
| CI evidence | [`RESEARCH-STATUS.md`](../RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results) indexes proposed gates and evidence still needed. | The index is not a run receipt or a CI pass-rate report. Report outcomes only when linked receipts support them. |

## Claim and change rules

- Report measured results only with reproducible evidence and the relevant baseline. No yield, fidelity, return, or quantum-advantage value is established by this map.
- Keep `LOCAL_SIM`, `CLOUD_SIM`, and `REAL_QPU` distinct; none implies another.
- Keep changes to documentation, including CI evidence indexes; do not alter CI configuration. Do not change settings, billing/IAM, invitations, secrets, QPU access, DeepNet runtime/schema, or FIRE.
- Preserve the [public/private boundary](VAULT.md). This public map does not expose or authorize private operational work.

See [public status](../STATUS.md), [documentation progress](../PROGRESS.md), and the [honesty / QPU hold](HONESTY-QPU-HOLD.md) for related limits and evidence gaps.
