# CI evidence index

This is an index of documentation and evidence expectations, not a CI dashboard.
No workflow run receipt is claimed by this repository unless a linked receipt
identifies the command, outcome, and artifact.

| Area | Public reference | Receipt status |
| --- | --- | --- |
| Bottleneck entry hashes | [`bottlenecks/README.md`](../bottlenecks/README.md) | No linked CI receipt; `search.py --verify` covers entries against `index.json` only |
| Bottleneck index and embeddings | [`bottlenecks/manifest.json`](../bottlenecks/manifest.json) | No separate manifest-hash comparison receipt |
| Erdős/Aer lanes | [`research/quantum-erdos-sequences/README.md`](../research/quantum-erdos-sequences/README.md) | No linked run receipt or current `RESULTS.json`/`RESULTS.md` artifact |
| DeepNet master | [External documentation](https://agenci-main.github.io/deepnet-chat/) | No integration, runtime, schema, availability, or CI evidence claimed |

Until receipts are linked, do not report CI pass rates, lane yields, fidelity,
hardware results, or quantum advantage. Proposed checks and simulator output
remain documentation or `LOCAL_SIM` evidence only.
