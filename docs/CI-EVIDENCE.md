# CI evidence index

**Snapshot: 2026-09-30.** This is an index of documented checks and evidence
gaps, not a list of passing workflows.

| Area | Documented check | Receipt status |
| --- | --- | --- |
| Bottleneck corpus | `search.py --verify` checks entry hashes against `index.json` | No linked CI receipt; manifest and embedding hashes still need separate verification |
| Erdős simulator lanes | `run_all.py` and generated `RESULTS.json` / `RESULTS.md` are documented | No current linked run receipt in this repository |
| QPU work | Submission is held and labeled `REAL_QPU` only after an explicit unlock | No QPU run, fidelity, yield, or speedup is claimed |

Do not derive a pass rate, yield, fidelity, or quantum-advantage claim from
this index. A future receipt must identify the command, outcome, artifact, and
scope of the run.
