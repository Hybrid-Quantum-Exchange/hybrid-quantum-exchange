# ORG-OPS index (public-safe)

This index routes readers to the repository's public operational posture and evidence notes. It is documentation only, not an operational runbook, authorization, or proof of service readiness.

| Topic | Reference | Scope |
| --- | --- | --- |
| Repository overview | [README](../README.md) | Public research and product-shell description |
| Current public status | [STATUS](../STATUS.md) | Repository scope and limitations; not live service or hardware status |
| Documentation progress | [PROGRESS](../PROGRESS.md) | Documented surfaces and evidence still needed |
| Research snapshot and CI evidence index | [RESEARCH-STATUS](../RESEARCH-STATUS.md) | Dated research posture and proposed CI gates; no CI run receipts are tracked here |
| Honesty and QPU hold | [HONESTY-QPU-HOLD](HONESTY-QPU-HOLD.md) | Evidence standards and documentation-only restrictions |
| Public/private boundary | [Vault map](VAULT.md) | Public repository contents versus separate private repositories |

## Operating boundary

The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external documentation reference only. Its availability, behavior, schema, and integration are not verified here.

**Documentation-only / QPU Hold:** no QPU submissions or unlocks, secrets, Settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work. Do not report yields, fidelity, quantum advantage, or hardware outcomes without reproducible supporting evidence. Simulator and classical results are not `REAL_QPU` evidence.

## Evidence boundary

The CI section in [RESEARCH-STATUS](../RESEARCH-STATUS.md) is an index of proposed gates, not CI results. No CI workflow or run receipt is tracked in this repository; do not infer pass rates, lane yields, or measured performance from this index.
