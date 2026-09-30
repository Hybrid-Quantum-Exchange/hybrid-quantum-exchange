# Public organization and operations map

This is a public documentation index, not an operational control plane. It
records where the public HQE materials point and what those links do **not**
authorize.

## Endpoint map

| Surface | Link | Public status |
| --- | --- | --- |
| DeepNet master | [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) | External documentation reference only; no API, schema, runtime, availability, or integration is defined here |
| HQE research surface | [Repository root](../) | Public notes, finite `LOCAL_SIM` exercises, and a product shell |
| HQE status index | [STATUS.md](../STATUS.md) | Documentation snapshot, not live service or hardware status |
| HQE CI evidence index | [RESEARCH-STATUS.md](../RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results) | Proposed gates and evidence needed; no CI receipt is claimed |

## Operating boundary

- Documentation and CI indexing only; no QPU submission or unlock.
- No Settings, billing/IAM, invites, secrets, DeepNet runtime/schema, or FIRE
  work is authorized by this map.
- Do not invent yield, fidelity, speedup, or quantum-advantage numbers.
- `LOCAL_SIM` is not `REAL_QPU`; any stronger claim requires reproducible
  receipts and a classical baseline.

See [Honesty / QPU Hold](HONESTY-QPU-HOLD.md) for the governing public
boundary and [Vault map](VAULT.md) for public/private repository separation.
