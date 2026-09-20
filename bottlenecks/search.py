#!/usr/bin/env python3
"""Search and verify the 999-bottleneck research corpus.

Commands:
  search.py --id 42          print entry 42 and verify its SHA-256 against index.json
  search.py --top "phrase"   KEYWORD search (token overlap, NOT semantic embedding
                             search); verifies each hit's file hash against
                             index.json and the stored embedding's L2 norm
  search.py --verify         hash all 999 entry files against index.json

This tool deliberately does NOT perform semantic/embedding search: --top is a
plain keyword matcher. True semantic queries require embedding the query with
the pinned model via embed.py and comparing against embeddings.json yourself.
"""
import argparse, hashlib, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def load_index():
    with open(os.path.join(HERE, "index.json")) as f:
        return json.load(f)

def load_entry(eid):
    with open(os.path.join(HERE, f"bottleneck_{eid:03d}.json"), "rb") as f:
        raw = f.read()
    return json.loads(raw), raw

def check_hash(eid, raw, index):
    want = next(e["sha256"] for e in index["entries"] if e["id"] == eid)
    got = hashlib.sha256(raw).hexdigest()
    return want == got, want, got

def cmd_id(eid):
    index = load_index()
    if not 1 <= eid <= 999:
        sys.exit(f"error: id must be 1..999, got {eid}")
    entry, raw = load_entry(eid)
    ok, want, got = check_hash(eid, raw, index)
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    print(f"\nSHA-256 {'MATCHES' if ok else 'MISMATCH!'} index.json")
    if not ok:
        print(f"  index: {want}\n  file:  {got}")
        sys.exit(1)

def cmd_top(phrase, n):
    index = load_index()
    emb = json.load(open(os.path.join(HERE, "embeddings.json")))
    qtokens = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) > 2]
    if not qtokens:
        sys.exit("error: query has no usable tokens")
    scored = []
    for meta in index["entries"]:
        entry, raw = load_entry(meta["id"])
        text = (entry["title"] + " " + entry["bottleneck"]).lower()
        score = sum(text.count(q) for q in qtokens)
        if score:
            scored.append((score, meta["id"], entry, raw))
    scored.sort(key=lambda s: (-s[0], s[1]))
    print(f"KEYWORD search (token overlap; NOT a semantic embedding search) for: {phrase!r}\n")
    for score, eid, entry, raw in scored[:n]:
        ok, want, got = check_hash(eid, raw, index)
        vec = emb["vectors"][str(eid)]
        norm = math.sqrt(sum(x * x for x in vec))
        norm_ok = abs(norm - 1.0) < 1e-3
        print(f"[{eid}] score={score}  hash={'OK' if ok else 'MISMATCH'}  embedding_norm={norm:.6f} ({'OK' if norm_ok else 'BAD'})")
        print(f"  {entry['domain']} -- {entry['title']}")
        if not ok:
            print(f"  index: {want}\n  file:  {got}")
    if not scored:
        print("no keyword matches")

def cmd_verify():
    index = load_index()
    bad = []
    for meta in index["entries"]:
        eid = meta["id"]
        _, raw = load_entry(eid)
        ok, _, _ = check_hash(eid, raw, index)
        if not ok:
            bad.append(eid)
    if bad:
        print(f"VERIFY FAILED: {len(bad)} of 999 hashes mismatch: {bad[:20]}")
        sys.exit(1)
    print("VERIFY OK: all 999 file hashes match index.json")

def main():
    ap = argparse.ArgumentParser(description="Search/verify the bottlenecks corpus")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, metavar="N")
    g.add_argument("--top", metavar="PHRASE")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=5, help="hits for --top (default 5)")
    a = ap.parse_args()
    if a.id is not None:
        cmd_id(a.id)
    elif a.top is not None:
        cmd_top(a.top, a.n)
    else:
        cmd_verify()

if __name__ == "__main__":
    main()
