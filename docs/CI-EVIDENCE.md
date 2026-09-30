# CI evidence index

**Snapshot: 2026-09-30.** This is an index of evidence documented in this
repository, not a live CI dashboard. No workflow definition or CI run receipt
is tracked in this checkout. The checks below are documented commands, not
reported CI passes; this page makes no claim about pass rates, lane yields,
fidelity, hardware execution, or quantum advantage.

| Check | Documented command and scope | CI evidence in this checkout | Evidence needed for a result claim |
| --- | --- | --- | --- |
| Bottleneck corpus | `python3 search.py --verify` checks entry hashes against `index.json`; it does not verify the index or embeddings against manifest hashes. See [bottlenecks guide](../bottlenecks/README.md). | No CI run receipt linked or tracked. | Link the run, commit, command, outcome, and artifact; separately compare `index.json` and `embeddings.json` with `manifest.json`. |
| Erdős Aer lane | `python run_all.py --lane 123` is a bounded local simulator run. See [Erdős guide](../research/quantum-erdos-sequences/README.md). | No CI run receipt linked or tracked. | Link the run and its receipts, with attempted, executed, and classical-check counts from that run. Label it `LOCAL_SIM`; it is not hardware or independent mathematical verification. |
| Erdős runner tests | `python -m unittest -v test_run_all` runs the documented local runner tests. See [Erdős guide](../research/quantum-erdos-sequences/README.md). | No CI run receipt linked or tracked. | Link the CI run, commit, command, outcome, and relevant logs or artifacts. |

Record evidence only when a run URL and commit identify the actual execution.
Keep failed or incomplete runs distinguishable from passes, and do not turn
proposed checks or local results into CI claims.

The [DeepNet master](https://agenci-main.github.io/deepnet-chat/) remains an
external documentation link only; this index documents no integration or
runtime behavior. The [honesty / QPU hold](HONESTY-QPU-HOLD.md) remains in
force: no QPU, secrets, settings, billing/IAM, invites, DeepNet runtime/schema,
or FIRE work is authorized.
