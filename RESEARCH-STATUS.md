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

## CI gates and receipts (index, not run results)

No CI workflow or run receipt is tracked in this repository. The checks below are **proposed gates**, not passing CI checks or measured yields.

| Proposed gate | Existing documentation or artifacts (not a CI receipt) | Evidence needed |
| --- | --- | --- |
| Corpus integrity | [`bottlenecks/README.md`](bottlenecks/README.md) documents `search.py --verify` for entry hashes against `index.json` | Linked CI run with command, outcome, and artifact; verify manifest hashes separately before claiming full corpus integrity |
| Aer lane verdicts | [`research/quantum-erdos-sequences/README.md`](research/quantum-erdos-sequences/README.md) documents `run_all.py` and generated `RESULTS.json` / `RESULTS.md` | Linked CI run and its generated results, with attempted, executed, and classical-check counts from that run |
| Grover known-target demonstrator | [Historical simulator artifacts](research/grover-verification/evidence/README.md) contain selected per-arm outputs and circuit exports, not complete portable run receipts | Linked CI run and its artifacts from the public source; the historical private manifests and receipts are not published here |

Until such CI receipts are linked, report no CI pass rate or lane yield from this index. Historical artifacts are not evidence that a CI gate passed. Simulator verdicts are `LOCAL_SIM`, not hardware evidence or speedup.

Do not interpret Aer or classical pilot results as hardware speedup.

**Documentation note (2026-09-30):** The [DeepNet master endpoint](https://agenci-main.github.io/deepnet-chat/) is listed in the README as a reference only. This does not change the research status above or indicate an active integration, QPU run, or hardware speedup.
