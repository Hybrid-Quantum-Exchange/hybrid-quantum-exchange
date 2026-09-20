# 999 Unsolved Bottlenecks in Human Advancement

A research corpus of 999 bottleneck entries across 15 domains of science and
technology, each entry describing one unsolved problem, why it blocks progress,
and the current state of the field.

**Research date: 2026-09.** These are concise **source-grounded research
notes, not peer-reviewed conclusions**. Every entry links 1–2 public sources
found via web search at research time. Source coverage was **thematic and
domain-level, not 999 independent systematic literature reviews** — some
entries rest on a single source, and a few sources are approximate topic
matches rather than dedicated treatments. Treat entries as starting points
for research, not as settled verdicts.

## Layout

| File | Contents |
|---|---|
| `bottleneck_001.json` … `bottleneck_999.json` | The corpus. Schema: `id`, `domain`, `title`, `bottleneck`, `why_it_blocks`, `current_state`, `sources` (list of `{title, url}`) |
| `index.json` | Per-entry SHA-256, domain, title, word count, stored embedding norm |
| `embeddings.json` | 999 × 384 L2-normalized vectors (rounded to 6 dp) + model metadata |
| `manifest.json` | Corpus manifest: counts, model pin, hashes, methodology, limitations |
| `search.py` | Lookup / keyword search / integrity verification (see below) |
| `embed.py` | Regenerate `embeddings.json` deterministically from the pinned model |
| `requirements-embeddings.txt` | Pinned Python dependencies for `embed.py` |

## Searching

```bash
python3 search.py --id 42          # print entry 42, verify its hash
python3 search.py --top "fusion"   # keyword search (see warning below)
python3 search.py --verify         # verify all 999 file hashes against index.json
```

> **Important:** `search.py --top` is **keyword-based token matching, not
> semantic embedding search.** It is labeled as such in its output. The
> stored vectors in `embeddings.json` exist so you can run your own semantic
> queries — but embedding a query requires the pinned model.

## Verifiable embeddings

Vectors were produced with `BAAI/bge-small-en-v1.5` at pinned commit
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`:

- tokenizer: the revision's `tokenizer.json`, max 128 tokens
- inference: `onnx/model.onnx` via onnxruntime (CPU)
- pooling: attention-mask mean pooling, then L2 normalization
- embedded text: `title + ". " + bottleneck`

Near-duplicate concepts were removed by replacing one side of every pair
scoring above **0.88 cosine similarity**, until zero pairs remained above the
threshold (final: 0 pairs). To regenerate or audit the vectors:

```bash
pip install -r requirements-embeddings.txt
python3 embed.py
```

`embed.py` downloads the exact pinned revision, regenerates all 999 vectors
deterministically, and prints the model name, revision, and SHA-256 of
`onnx/model.onnx`.

## Domains and quotas

fundamental_physics 85 · cosmology_multiverse 60 · energy 85 ·
compute_semiconductors 65 · artificial_intelligence 60 · medicine_cures 95 ·
health_longevity 80 · biotech_genetics 65 · neuroscience 60 · space 70 ·
materials 65 · climate_environment 60 · xr_interface 60 ·
food_water_agriculture 45 · coordination 44
