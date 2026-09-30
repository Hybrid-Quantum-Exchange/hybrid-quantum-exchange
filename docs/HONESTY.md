# Honesty and evidence

This page describes claims supportable by the public files in this repository. The [status](../STATUS.md) and [progress](../PROGRESS.md) pages distinguish published evidence from private or aspirational work.

## Research notes are not validation

The [bottlenecks corpus](../bottlenecks/README.md) contains 999 concise source-grounded notes. Sources are thematic and sometimes approximate; entries are not peer-reviewed findings or 999 independent systematic literature reviews. The vectors represent text similarity, not scientific truth. See the [manifest](../bottlenecks/manifest.json) for methodology and limitations.

To check entry file hashes against the index, run `python3 search.py --verify` from `bottlenecks/`. This **does not** authenticate the index or embeddings against the manifest. Check their SHA-256 values against `index_sha256` and `embeddings_sha256` in `bottlenecks/manifest.json` separately if auditing those artifacts. Hash matches show file consistency, not source quality or correctness.

## Simulator results are not hardware or mathematical results

The [Erdős lane documentation](../research/quantum-erdos-sequences/README.md) describes 999 small circuits on the ideal Qiskit Aer simulator, each comparing a finite property against a classical computation. Its recorded 993/999 PASS count is from one run, not a guaranteed rerun result: unseeded measurement shots can flip some verdicts. Many lanes substitute a finite property where no genuine OEIS link exists, and six explicitly test an unrelated placeholder. None solves an open Erdős problem or demonstrates quantum advantage or real-hardware performance. Consult the lane README for reproducibility instructions and the exact limitations.

## Product and external links

The [site source](../index.html) is a product-facing presentation, not proof of deployed exchange operations. [DeepNet master](https://agenci-main.github.io/deepnet-chat/) is an external sister-project link, not evidence of an integration or functionality in this repository. Private work listed in the [vault map](VAULT.md) has no public verification here.
