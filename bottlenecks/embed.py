#!/usr/bin/env python3
"""Regenerate embeddings.json deterministically with the pinned model.

Downloads BAAI/bge-small-en-v1.5 at the exact pinned commit revision, then
embeds every bottleneck entry with the documented method:

  - tokenizer: the revision's tokenizer.json (HuggingFace `tokenizers` lib)
  - text: entry "title" + ". " + entry "bottleneck"
  - truncation: 128 tokens max
  - inference: onnx/model.onnx via onnxruntime (CPU)
  - pooling: attention-mask mean pooling
  - normalization: L2

Output: bottlenecks/embeddings.json (384-dim, L2-normalized, rounded to 6 dp).
Prints model name, pinned revision, and SHA-256 of onnx/model.onnx.
"""
import hashlib, json, os, sys

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MAX_LEN = 128
DIM = 384

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REV,
        allow_patterns=["onnx/model.onnx", "tokenizer.json"],
    )
    onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
    tok_path = os.path.join(model_dir, "tokenizer.json")

    h = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    model_hash = h.hexdigest()

    print("model:   ", MODEL_NAME)
    print("revision:", MODEL_REV)
    print("onnx sha256:", model_hash)

    tok = Tokenizer.from_file(tok_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def batch_embed(batch):
        enc = tok.encode_batch(batch)
        n = len(batch)
        ids_a = np.zeros((n, MAX_LEN), dtype=np.int64)
        mask = np.zeros((n, MAX_LEN), dtype=np.int64)
        for r, e in enumerate(enc):
            t = e.ids[:MAX_LEN]
            ids_a[r, :len(t)] = t
            mask[r, :len(t)] = 1
        tt = np.zeros((n, MAX_LEN), dtype=np.int64)
        out = sess.run(None, {"input_ids": ids_a, "attention_mask": mask,
                              "token_type_ids": tt})[0]
        m = mask[:, :, None].astype(np.float32)
        pooled = (out * m).sum(1) / m.sum(1).clip(min=1e-9)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return pooled / norm

    texts, ids = [], []
    for i in range(1, 1000):
        with open(os.path.join(HERE, f"bottleneck_{i:03d}.json")) as f:
            e = json.load(f)
        texts.append(e["title"].strip() + ". " + e["bottleneck"].strip())
        ids.append(i)

    B = 64
    vecs = np.vstack([batch_embed(texts[i:i + B]) for i in range(0, len(texts), B)])
    assert vecs.shape == (999, DIM), vecs.shape

    out = {
        "meta": {
            "model": MODEL_NAME,
            "revision": MODEL_REV,
            "dimension": DIM,
            "method": "pinned tokenizer.json, max 128 tokens, onnx/model.onnx CPU inference, attention-mask mean pooling, L2 normalization",
            "text_embedded": "title + '. ' + bottleneck",
            "dedup_threshold_cosine": 0.88,
        },
        "vectors": {str(i): [round(float(x), 6) for x in vecs[j]]
                    for j, i in enumerate(ids)},
    }
    out_path = os.path.join(HERE, "embeddings.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print("wrote", out_path, f"({vecs.shape[0]} x {vecs.shape[1]})")
    print("note: index.json embedding_norm values are unaffected (vectors are L2-normalized)")

if __name__ == "__main__":
    sys.exit(main())
