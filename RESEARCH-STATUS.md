# Research status (public)

**As of 2026-09-30.** Detailed posture lives in the private `quantum-project-ledger`.

**QPU Hold:** Documentation-only. No QPU, Settings, billing/IAM, invites, secrets,
DeepNet runtime/schema, or FIRE work is authorized. The DeepNet master is
[text-only documentation](https://agenci-main.github.io/deepnet-chat/); no
`REAL_QPU` submission or unlock is planned.

| Track | Status | Label |
| --- | --- | --- |
| Bottlenecks corpus (999) | Published in `bottlenecks/` | research notes |
| Erdős quantum sequences | Published in `research/quantum-erdos-sequences/` | `LOCAL_SIM` (Aer) |
| Classical hybrid pilot | Private repo `quantum-worker-pilot` | `LOCAL_SIM` |
| IBM Open Plan | Account handshake done; **job submit locked** | no `REAL_QPU` yet |
| OEIS `sequences` branch | Public fork `oeisdata` | lookup library |
| DeepNet master | [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/) | external project; link only, not an integration claim |

## CI gates and receipts

[The CI evidence index](docs/CI-EVIDENCE.md) documents candidate checks and the
receipts needed to support results. No CI workflow or run receipt is tracked in
this repository, so no CI pass rate or lane yield is reported. Simulator
verdicts are `LOCAL_SIM`, not hardware evidence or speedup.

Do not interpret Aer or classical pilot results as hardware speedup.

**Documentation note (2026-09-30):** The [DeepNet master endpoint](https://agenci-main.github.io/deepnet-chat/) is listed in the README as a reference only. This does not change the research status above or indicate an active integration, QPU run, or hardware speedup.
