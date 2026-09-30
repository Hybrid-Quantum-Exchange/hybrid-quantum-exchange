# Public status and limitations

This is a description of the repository, **not a live service or hardware status page**. For the dated research snapshot, see [RESEARCH-STATUS.md](RESEARCH-STATUS.md); for documentation progress and missing evidence, see [PROGRESS.md](PROGRESS.md).

| Surface | What is documented | What it does not establish |
| --- | --- | --- |
| [Bottlenecks](bottlenecks/README.md) | 999 source-grounded research notes and stored embeddings | Peer review, scientific correctness, or 999 independent literature reviews |
| [Erdős simulator lanes](research/quantum-erdos-sequences/README.md) | 999 small, finite Qiskit Aer exercises checked against classical answers (`LOCAL_SIM`) | Solutions to open Erdős problems, quantum advantage, or hardware results |
| [Site shell](index.html) | Public product and exchange UI | A live trading service, operational backend, or production readiness |
| [Private pilot and control plane](docs/VAULT.md) | Separate private repositories are described in the vault map | Publicly reproducible pilot results or a public operational control plane |

**Evidence labels:** `LOCAL_SIM` denotes local simulation, not QPU execution; `CLOUD_SIM` denotes cloud simulation, not QPU execution; `REAL_QPU` would require an explicitly unlocked hardware submission and receipts. The [honesty note](docs/HONESTY-QPU-HOLD.md) explains why none of these labels alone proves quantum advantage.

**Hardware posture:** QPU job submission is documented as locked in the [dated research snapshot](RESEARCH-STATUS.md); no `REAL_QPU` results are claimed here. Simulation is not hardware evidence. Hardware or paid cloud runs require explicit unlock and receipts.

**DeepNet:** The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external context link. Its availability, behavior, schema, and integration with this repository are not verified or specified here.

**Results and operations:** The UI is not proof of a deployed exchange or realized returns. [Research status](RESEARCH-STATUS.md#ci-gates-and-receipts-index-not-run-results) lists proposed CI gates, not passing checks or measured yields; local receipt generation is not a linked CI result. The [honesty / QPU hold](docs/HONESTY-QPU-HOLD.md) records the documentation-only boundary; this page does not authorize QPU, settings, billing/IAM, invites, secrets, DeepNet runtime/schema, or FIRE work.
