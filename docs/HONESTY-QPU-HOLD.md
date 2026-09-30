# Honesty / QPU Hold

The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external context link, not a verified integration, QPU execution, or evidence of quantum advantage. This repository does not specify its endpoint behavior, availability, or schema.

**QPU Hold:** Hardware / paid cloud QPU submission remains locked. Published Aer and classical pilot work is `LOCAL_SIM`, not `REAL_QPU`. `CLOUD_SIM` would still be simulation, not hardware execution. Do not claim a hardware run, speedup, or quantum advantage without explicit unlock, run receipts, reproducible evidence, and a classical baseline.

## Claims and evidence

| Claim | What is available here | What is not established |
| --- | --- | --- |
| Corpus integrity | [Entry verification](../bottlenecks/README.md#integrity-checks) and published hashes | `search.py --verify` does not check index or embeddings against the manifest; matching published hashes does not independently validate the research |
| Simulator outcomes | [Erdős runner](../research/quantum-erdos-sequences/README.md#run-evidence-and-interpretation) can write local per-run receipts | A runner description is not a linked result, independent mathematical review, hardware result, or speedup |
| Exchange and returns | [Site shell](../index.html) | A live backend, executed trades, profits, or realized yields |

Do not turn a proposed CI gate into a passing check or an estimate into a measurement. Link the precise revision, run, outcome, and artifact before reporting results; a self-reported simulator verdict should remain labeled as such. See [Research status](../RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results) for the proposed gates and missing CI receipts.

This documentation-only hold does not authorize settings, billing/IAM, invites, secrets, DeepNet runtime/schema, or FIRE work. See [Vault map](VAULT.md), [Research status](../RESEARCH-STATUS.md), and [Documentation progress](../PROGRESS.md) for the public posture and outstanding evidence.
