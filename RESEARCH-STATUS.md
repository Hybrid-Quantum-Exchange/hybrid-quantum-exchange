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

## CI gates and receipts (scoped index)
The [CI evidence index](docs/CI-EVIDENCE.md) records one green CodeQL receipt
for static analysis of the pull-request revision. It is not a corpus,
simulator, hardware, or production-readiness receipt.

| Proposed gate | Existing documentation (not a receipt) | Evidence needed |
| --- | --- | --- |
| Corpus integrity | [`bottlenecks/README.md`](bottlenecks/README.md) documents `search.py --verify` for entry hashes against `index.json` | No linked corpus run; verify manifest hashes separately before claiming full corpus integrity |
| Aer lane verdicts | [`research/quantum-erdos-sequences/README.md`](research/quantum-erdos-sequences/README.md) documents `run_all.py` and generated `RESULTS.json` / `RESULTS.md` | No linked lane run with attempted, executed, and classical-check counts |

Until those receipts are linked, report no CI pass rate or lane yield from
this index. Simulator verdicts are `LOCAL_SIM`, not hardware evidence or
speedup.

Do not interpret Aer or classical pilot results as hardware speedup.

**Documentation note (2026-09-30):** The [DeepNet master endpoint](https://agenci-main.github.io/deepnet-chat/) is listed in the README as a reference only. This does not change the research status above or indicate an active integration, QPU run, or hardware speedup.
