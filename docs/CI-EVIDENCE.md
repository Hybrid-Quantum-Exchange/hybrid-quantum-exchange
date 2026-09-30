# CI evidence index

**Snapshot: 2026-09-30.** This index separates documented verification procedures from CI receipts. No CI workflow or CI run receipt is tracked in this repository. The entries below are references to documented checks, not evidence that a check was run or passed.

| Area | Documented check or output | Evidence currently linked here | Needed before claiming a CI result |
| --- | --- | --- | --- |
| Bottlenecks corpus | [`search.py --verify`](../bottlenecks/README.md) checks entry hashes against [`index.json`](../bottlenecks/index.json) | None | Link the CI run, command, outcome, and artifact. Compare index and embeddings against [`manifest.json`](../bottlenecks/manifest.json) separately before claiming full corpus integrity. |
| Erdős simulator lanes | [`run_all.py`](../research/quantum-erdos-sequences/README.md) documents generating `RESULTS.json` and `RESULTS.md` | None | Link the CI run and generated outputs, with attempted, executed, and classical-check counts from that run. |

Until receipts are linked, report no CI pass rate, lane yield, fidelity, or hardware result from this index. Simulator results are `LOCAL_SIM`; they are not `REAL_QPU` evidence or proof of quantum advantage.

The [DeepNet master documentation](https://agenci-main.github.io/deepnet-chat/) is an external documentation reference only and is not CI evidence or proof of an integration.

See the [ORG-OPS guide](ORG-OPS.md) and [Honesty / QPU Hold](HONESTY-QPU-HOLD.md) for the documentation-only operating boundaries.
