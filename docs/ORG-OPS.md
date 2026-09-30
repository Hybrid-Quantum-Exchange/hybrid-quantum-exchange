# Public org operations boundary

**Documentation snapshot: 2026-09-30.** This is a public navigation and review
guide, not an operational runbook or authorization to change an account.

| Surface | Public reference | Boundary |
| --- | --- | --- |
| This repository | [README](../README.md), [status](../STATUS.md), [progress](../PROGRESS.md) | Research notes, simulator exercises and a UI shell; not a live exchange |
| Research evidence | [CI evidence index](CI-EVIDENCE-INDEX.md), [research status](../RESEARCH-STATUS.md) | Local artifacts and proposed checks are not CI run receipts or hardware results |
| Public/private split | [Vault map](VAULT.md) | Private control and pilot records are not reproduced here |
| DeepNet master | [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) | External reference only; no verified integration, availability or API contract |

## Documentation review boundary

For public updates, link the source artifact and its limitations, distinguish
`LOCAL_SIM` from `REAL_QPU`, and leave missing run receipts explicitly missing.
Do not report pass rates, yields, fidelity, returns or hardware readiness from
an unlinked result. A source file, a proposed CI gate or a historical artifact
subset is not a passing CI run.

The [honesty / QPU hold](HONESTY-QPU-HOLD.md) remains in force. This guide grants
no QPU submissions or unlock, settings changes, billing/IAM work, invites,
secret handling, DeepNet runtime/schema changes or FIRE work. Account and
private-repository operations are outside this public documentation scope.
