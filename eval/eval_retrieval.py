"""
eval/eval_retrieval.py
----------------------
Offline retrieval evaluation for Celestia: compares chunking strategies and
retrieval settings on a fixed question set, without calling any LLM.

For every question we know which zodiac sign (and which field) holds the
answer, so we can check whether the retriever brings back the right chunk.

Metrics
  Hit@1  - the top retrieved chunk belongs to the correct sign
  Hit@3  - the correct sign appears in the top 3 chunks
  MRR@3  - mean reciprocal rank of the first correct chunk (1, 1/2, 1/3 or 0)
  Field@3 - (field-level chunking only) the exact field is in the top 3

Run:
    python eval/eval_retrieval.py
Results are printed and written to eval/RESULTS.md.
"""

import re
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = (ROOT / "zodiac_data.txt").read_text(encoding="utf-8")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # same model as build_index.py


# ── 1. Parse the knowledge base into {sign: {field: value}} ────────────────
def parse_signs(text):
    signs = {}
    for block in text.split("---"):
        fields = {}
        for line in block.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        if "Sign" in fields:
            signs[fields["Sign"]] = fields
    return signs


SIGNS = parse_signs(DATA)


# ── 2. Chunking strategies; every chunk carries its sign (and field) ───────
def chunk_whole_sign():
    """Current Celestia approach: one chunk per sign (split on '---')."""
    return [{"text": block.strip(), "sign": name, "field": None}
            for name, block in zip(SIGNS, [b for b in DATA.split("---") if "Sign:" in b])]


def chunk_field_level():
    """One chunk per field, prefixed with the sign so it stays self-contained."""
    chunks = []
    for name, fields in SIGNS.items():
        for key, value in fields.items():
            if key != "Sign":
                chunks.append({"text": f"{name} - {key}: {value}", "sign": name, "field": key})
    return chunks


def chunk_fixed_size(size=300, overlap=50):
    """Naive fixed-size character chunks with overlap (sign boundaries ignored)."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"Sign: (\w+)", DATA)]

    def sign_at(pos):  # the sign whose section contains this character position
        current = starts[0][1]
        for start, name in starts:
            if start <= pos:
                current = name
        return current

    chunks, step = [], size - overlap
    for i in range(0, len(DATA), step):
        text = DATA[i:i + size].strip()
        if text and text != "---":
            chunks.append({"text": text, "sign": sign_at(i + len(text) // 2), "field": None})
    return chunks


# ── 3. Test questions generated from the data, so the answer is known ──────
def build_questions():
    questions = []
    for name, f in SIGNS.items():
        questions += [
            # named-sign questions: the user mentions the sign
            {"q": f"What careers suit a {name}?", "sign": name, "field": "Career"},
            {"q": f"What are the weaknesses of {name} people?", "sign": name, "field": "Weaknesses"},
            {"q": f"Who is the best match for {name} in love?", "sign": name, "field": "Best Matches"},
            # reverse-lookup questions: the sign is NOT named, only a fact about it
            {"q": f"Which zodiac sign is symbolised by {f['Symbol'].lower()}?", "sign": name, "field": "Symbol"},
            {"q": f"Which sign is ruled by {f['Ruling Planet']} and has the element {f['Element']}?",
             "sign": name, "field": "Ruling Planet"},
        ]
    return questions


QUESTIONS = build_questions()


# ── 4. Retrieval: exact cosine search (FAISS) or MMR re-ranking ────────────
def build_index(model, chunks):
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
    index.add(np.asarray(vectors, dtype="float32"))
    return index, vectors


def search(index, vectors, query_vec, k, mode):
    if mode == "similarity":
        _, ids = index.search(query_vec[None, :], k)
        return list(ids[0])
    # MMR: balance relevance to the query against diversity among picked chunks
    _, cand = index.search(query_vec[None, :], min(20, index.ntotal))
    cand, picked, lam = list(cand[0]), [], 0.5
    while cand and len(picked) < k:
        def score(i):
            relevance = float(vectors[i] @ query_vec)
            redundancy = max((float(vectors[i] @ vectors[j]) for j in picked), default=0.0)
            return lam * relevance - (1 - lam) * redundancy
        best = max(cand, key=score)
        picked.append(best)
        cand.remove(best)
    return picked


def evaluate(model, chunks, mode, k=3):
    index, vectors = build_index(model, chunks)
    q_vecs = model.encode([q["q"] for q in QUESTIONS], normalize_embeddings=True)
    hit1 = hitk = rr = field_hits = 0
    by_type = {"named": [0, 0], "reverse": [0, 0]}
    t0 = time.perf_counter()
    for q, vec in zip(QUESTIONS, q_vecs):
        ids = search(index, vectors, np.asarray(vec, dtype="float32"), k, mode)
        signs = [chunks[i]["sign"] for i in ids]
        fields = [chunks[i]["field"] for i in ids]
        hit1 += signs[0] == q["sign"]
        hitk += q["sign"] in signs
        rr += next((1 / (r + 1) for r, s in enumerate(signs) if s == q["sign"]), 0)
        field_hits += any(s == q["sign"] and f == q["field"] for s, f in zip(signs, fields))
        kind = "reverse" if q["field"] in ("Symbol", "Ruling Planet") else "named"
        by_type[kind][0] += signs[0] == q["sign"]
        by_type[kind][1] += 1
    n = len(QUESTIONS)
    return {
        "chunks": len(chunks),
        "hit1": hit1 / n, "hitk": hitk / n, "mrr": rr / n,
        "field": field_hits / n if chunks[0]["field"] else None,
        "named1": by_type["named"][0] / by_type["named"][1],
        "reverse1": by_type["reverse"][0] / by_type["reverse"][1],
        "ms_per_query": (time.perf_counter() - t0) * 1000 / n,
    }


def main():
    model = SentenceTransformer(MODEL_NAME)
    strategies = {
        "Whole sign (current)": chunk_whole_sign(),
        "Field-level": chunk_field_level(),
        "Fixed 300 chars / 50 overlap": chunk_fixed_size(300, 50),
        "Fixed 800 chars / 100 overlap": chunk_fixed_size(800, 100),
    }
    rows = []
    for name, chunks in strategies.items():
        for mode in ("similarity", "mmr"):
            r = evaluate(model, chunks, mode)
            rows.append((name, mode, r))
            print(f"{name:32s} {mode:10s} chunks={r['chunks']:3d} Hit@1={r['hit1']:.2f} "
                  f"Hit@3={r['hitk']:.2f} MRR@3={r['mrr']:.2f}")

    def pct(x):
        return "-" if x is None else f"{x * 100:.0f}%"

    lines = [
        "# Retrieval evaluation",
        "",
        f"Embedding model: `{MODEL_NAME}` | FAISS exact cosine search | k = 3 | "
        f"{len(QUESTIONS)} test questions ({len(SIGNS)} signs x 5: 3 name the sign, "
        "2 are reverse lookups that only describe it).",
        "",
        "| Chunking | Retrieval | Chunks | Hit@1 | Hit@3 | MRR@3 | Field@3 | Hit@1 named | Hit@1 reverse |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, mode, r in rows:
        lines.append(f"| {name} | {mode} | {r['chunks']} | {pct(r['hit1'])} | {pct(r['hitk'])} | "
                     f"{r['mrr']:.2f} | {pct(r['field'])} | {pct(r['named1'])} | {pct(r['reverse1'])} |")
    (ROOT / "eval" / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote eval/RESULTS.md")


if __name__ == "__main__":
    main()
