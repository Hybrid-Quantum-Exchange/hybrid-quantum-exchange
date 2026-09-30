# CI evidence index

**Documentation snapshot: 2026-09-30.** This repository does not contain a CI workflow or linked CI run receipts for the checks below. This is an index of documentation and evidence still needed, not a record of passing checks. For overall posture, see [public status](../STATUS.md) and the [honesty / QPU hold](HONESTY-QPU-HOLD.md).

| Candidate check | What is documented here | What a CI claim would require |
| --- | --- | --- |
| Bottlenecks corpus integrity | [`search.py --verify`](../bottlenecks/README.md#integrity-checks) compares entry files with `index.json`; separate index and embeddings hash comparison with `manifest.json` is documented there. | A linked run identifying revision, commands, exit status, and relevant outputs; do not call entry-only verification a full manifest check. |
| Erdős simulator runner | [Runner and receipt format](../research/quantum-erdos-sequences/README.md#run-evidence-and-interpretation) describes local `.runs/` receipts and latest-run result copies. | A linked CI run identifying revision, environment, executed selection, outcome, and preserved receipts/results. Self-reported demo verdicts are not independent review. |
| Grover historical simulator evidence | [Selected local artifacts](../research/grover-verification/evidence/README.md) are published separately from CI. | A linked CI run and complete attributable artifacts before claiming CI verification; the selected historical files are not CI receipts. |

No CI pass rate, lane yield, or fidelity measurement follows from these links. Local simulation and historical artifacts do not establish hardware execution, quantum advantage, or live exchange behavior. The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external documentation reference only, not CI evidence or a verified integration.
