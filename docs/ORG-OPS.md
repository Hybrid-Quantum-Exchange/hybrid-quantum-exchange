# Public ORG-OPS boundary

This is a public documentation map, **not** an operational runbook or an authorization to change accounts, deploy services, or submit jobs. For the dated research posture see [research status](../RESEARCH-STATUS.md); for the public/private split see the [vault map](VAULT.md).

| Surface | Public reference | Boundary |
| --- | --- | --- |
| Research notes | [Bottlenecks guide](../bottlenecks/README.md) | Source-grounded notes, not peer-reviewed findings |
| Simulation | [Erdős simulator guide](../research/quantum-erdos-sequences/README.md) | `LOCAL_SIM` demonstrations, not hardware or solutions to open problems |
| Product shell | [Public status](../STATUS.md) | UI is not evidence of a live exchange |
| External operator reference | [DeepNet master](https://agenci-main.github.io/deepnet-chat/) | Link only; no verified availability, integration, runtime, or schema |
| Evidence review | [CI evidence index](CI-EVIDENCE.md) | Proposed checks and receipt requirements, not a CI pass claim |

## Documentation review boundary

- Keep changes to public documentation and CI evidence indexing; distinguish proposed checks from linked run receipts.
- Label simulator evidence `LOCAL_SIM`. Do not report hardware results, quantum advantage, yields, fidelity, or returns without applicable independently reviewable evidence; no such measurements are established by this index.
- Keep the [honesty / QPU hold](HONESTY-QPU-HOLD.md): no QPU work, secrets, settings, billing/IAM, invites, DeepNet runtime/schema, or FIRE work. A documentation change cannot unlock submissions or authorize operational work.
