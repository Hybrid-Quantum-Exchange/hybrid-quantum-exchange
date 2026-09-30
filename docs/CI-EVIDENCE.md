# CI evidence index

**Snapshot: 2026-09-30.** This index distinguishes documented checks from actual CI evidence. It is not a CI run report or a claim that any gate passed.

| Area | Evidence currently documented | What is not established |
| --- | --- | --- |
| Repository CI | No workflow or CI run receipt is tracked in this repository. | A CI pass, failure, coverage result, or run rate. |
| Bottlenecks corpus | [`bottlenecks/README.md`](../bottlenecks/README.md) documents `search.py --verify` for entry hashes against `index.json`. | Full corpus integrity; `index.json` and `embeddings.json` require separate comparison with manifest hashes. |
| Erdős lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) documents finite Aer simulator checks and generated result files. | Any current run outcome, lane yield, open-problem solution, hardware result, or quantum advantage. |
| DeepNet master | [External documentation link](https://agenci-main.github.io/deepnet-chat/). | Integration, endpoint behavior, schema, availability, or CI coverage. |

The documented checks are not receipts. Proposed gates are not passing CI; no run counts, yields, fidelity values, or results are inferred here. Simulator work is `LOCAL_SIM`, not hardware evidence.

For the broader public evidence and limitations, see [PROGRESS](../PROGRESS.md), [STATUS](../STATUS.md), and [Research status](../RESEARCH-STATUS.md).
