# CI evidence index

**Snapshot: 2026-09-30.** This index records documentation and evidence
requirements. It is not a CI dashboard, a pass report, or a yield report.

| Surface | Documented check | Evidence status | Needed before a stronger claim |
| --- | --- | --- | --- |
| Bottlenecks | `bottlenecks/search.py --verify` checks entries against `index.json` | Procedure documented; no receipt linked | Dated CI command output and artifact; compare index and embeddings with `bottlenecks/manifest.json` separately |
| Erdős lanes | `research/quantum-erdos-sequences/run_all.py` and generated results | Procedure documented; no current receipt linked | Dated run receipt and `RESULTS` artifacts with attempted, executed, and classical-check counts |
| DeepNet master | Public link only | No integration or runtime evidence | None is requested during this documentation-only hold |
| QPU | Submission remains locked | No `REAL_QPU` evidence | Explicit unlock and receipts; not authorized by this issue |

Simulator output is `LOCAL_SIM`, not hardware evidence. Do not infer a CI
pass rate, yield, fidelity, return, speedup, or quantum advantage from this
index.

Related: [operator map](ORG-OPS.md), [public progress](../PROGRESS.md), and
[research status](../RESEARCH-STATUS.md).
