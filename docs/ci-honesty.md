# CI honesty

As documented here, this repository has **no tracked GitHub Actions workflow**
under `.github/workflows/`. A documentation PR therefore has no repository
workflow that automatically verifies its claims or gates merging. An absent
check is not a passing check; do not advertise CI coverage, security scanning,
QPU validation, or deployment verification on the basis of this repository.

The research trees provide local commands, not CI evidence:

- `bottlenecks/README.md` documents `python3 search.py --verify` from
  `bottlenecks/`. It verifies entry hashes against `index.json`; it does not
  validate `index.json` or `embeddings.json` against `manifest.json`.
- `research/quantum-erdos-sequences/README.md` documents
  `python3 run_all.py` from that directory after installing its pinned
  requirements. It runs ideal Aer simulations, not QPU jobs; lane verdicts
  can vary with unseeded shots and are not a CI guarantee.

For documentation-only changes, review the diff and check links and claims
against the repository before merging. If checks are added later, describe
their actual scope and report their results separately from simulation and
hardware evidence. The [DeepNet master endpoint](https://agenci-main.github.io/deepnet-chat/)
is an external project link, not a CI target, API contract, or runtime
integration here.
