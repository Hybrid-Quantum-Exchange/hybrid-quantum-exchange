# Research status (public)

**As of 2026-09-20.** Detailed posture lives in the private `quantum-project-ledger`.

| Track | Status | Label |
| --- | --- | --- |
| Bottlenecks corpus (999) | Published in `bottlenecks/` | research notes |
| Erdős quantum sequences | Published in `research/quantum-erdos-sequences/` | `LOCAL_SIM` (Aer) |
| Classical hybrid pilot | Private repo `quantum-worker-pilot` | `LOCAL_SIM` |
| IBM Open Plan | Account handshake done; **job submit locked** | no `REAL_QPU` yet |
| OEIS `sequences` branch | Public fork `oeisdata` | lookup library |

Do not interpret Aer or classical pilot results as hardware speedup.

## CI gates and receipts (this repository)

| Check | Evidence available | Gate / receipt status |
| --- | --- | --- |
| Bottlenecks integrity | [`manifest.json`](bottlenecks/manifest.json) records index and embedding hashes; [`search.py --verify`](bottlenecks/README.md#searching) checks entry hashes against `index.json` only. | Local verification available; no CI gate configured here. Index and embedding hashes require separate checks against the manifest. |
| Erdős simulator lanes | [README totals and limitations](research/quantum-erdos-sequences/README.md#totals) describe a prior Aer run and how to regenerate `RESULTS.json` / `RESULTS.md`. | Reported simulation evidence, not a CI receipt; no CI gate configured here. |
| Hardware execution | [Honesty bar](README.md#honesty-bar) requires explicit unlock and `REAL_QPU` receipts. | Planned gate only; no hardware receipt in this repository. |

No CI pass rate, hardware yield, or quantum speedup is claimed by this index.
