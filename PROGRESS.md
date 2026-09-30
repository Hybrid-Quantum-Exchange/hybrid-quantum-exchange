# Public progress

This page tracks artifacts visible **in this repository**, not milestones reported by private projects. For evidence labels and caveats, see [honesty guidance](docs/HONESTY.md) and [research status](RESEARCH-STATUS.md).

| Area | Public artifact | Current boundary |
| --- | --- | --- |
| Bottlenecks | [999 entries, index, embeddings and manifest](bottlenecks/README.md) | Source-grounded starting points; thematic coverage rather than independent reviews of every entry. Keyword search is not semantic search. |
| Erdős exercises | [999 simulator lane scripts and runner](research/quantum-erdos-sequences/README.md) | Finite classical cross-checks on ideal Aer simulation; no open problems solved. Some verdicts vary with unseeded shots. |
| Site | `index.html`, `styles.css`, `app.js` | Static concept page; no exchange backend or live transactions in this repository. |

## How to check the published work

- From `bottlenecks/`, run `python3 search.py --verify` to check entry hashes against `index.json`. Check the index and embeddings against the hashes in `manifest.json` separately; `--verify` does not check those files.
- From `research/quantum-erdos-sequences/`, follow the [runner instructions](research/quantum-erdos-sequences/README.md) to generate fresh results. The README records one run and explains its limitations; a new run may differ.
- Read [research status](RESEARCH-STATUS.md) for the public research snapshot and [vault map](docs/VAULT.md) for the separation between public and private work.
