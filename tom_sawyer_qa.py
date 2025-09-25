#!/usr/bin/env python3
# tom_sawyer_offline_qa.py
# Offline, dependency-free TF-IDF QA over a local Tom Sawyer text file.

import os, re, math, argparse, gzip, pickle, time, pathlib, textwrap
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# ---------- Defaults ----------
DEFAULT_WORDS_PER_CHUNK = 180
DEFAULT_TOPK = 5
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".tom_sawyer_offline")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl.gz")
META_PATH = os.path.join(CACHE_DIR, "meta.pkl")

# ---------- Tokenization ----------
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]

def sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

# ---------- File I/O ----------
def read_local_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    # If it's a Gutenberg file, trim boilerplate if markers present
    start = txt.find("THE ADVENTURES OF TOM SAWYER")
    end = txt.find("End of the Project Gutenberg EBook")
    if start != -1 and end != -1 and end > start:
        txt = txt[start:end]
    return txt

def ensure_cache_dir():
    pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Chunking ----------
def chunk_text(text: str, words_per_chunk: int) -> List[str]:
    # Split into paragraphs then re-wrap to ~N words per chunk to keep context coherent
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    tokens: List[str] = []
    for p in paras:
        tokens.extend(p.split())
    chunks, cur = [], []
    for w in tokens:
        cur.append(w)
        if len(cur) >= words_per_chunk:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ---------- TF-IDF ----------
def build_tfidf(chunks: List[str]):
    """
    Returns:
      idf: Dict[str,float]
      chunk_vectors: List[Dict[str,float]]  (sparse tf-idf vectors)
      vocab: Set[str]
    """
    N = len(chunks)
    # Document frequencies
    df = Counter()
    tokenized_chunks: List[List[str]] = []
    for ch in chunks:
        toks = tokenize(ch)
        tokenized_chunks.append(toks)
        df.update(set(toks))

    # Smooth idf
    idf: Dict[str, float] = {}
    for term, d in df.items():
        idf[term] = math.log((N + 1) / (d + 1)) + 1.0

    # Build normalized tf-idf vectors per chunk
    chunk_vectors: List[Dict[str, float]] = []
    for toks in tokenized_chunks:
        tf = Counter(toks)
        vec: Dict[str, float] = {}
        # log-scaled tf-idf (ok for literary text)
        for t, c in tf.items():
            w = (1 + math.log(c)) * idf[t]
            vec[t] = w
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for t in vec:
            vec[t] /= norm
        chunk_vectors.append(vec)

    vocab = set(idf.keys())
    return idf, chunk_vectors, vocab

def vectorize_query(q: str, idf: Dict[str,float]) -> Dict[str,float]:
    toks = tokenize(q)
    tf = Counter(toks)
    vec: Dict[str, float] = {}
    for t, c in tf.items():
        if t in idf:
            vec[t] = (1 + math.log(c)) * idf[t]
    # L2 normalize
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    for t in vec:
        vec[t] /= norm
    return vec

def cosine_sparse(a: Dict[str,float], b: Dict[str,float]) -> float:
    # dot over intersection
    if len(a) > len(b):
        a, b = b, a
    return sum(a[t]*b.get(t, 0.0) for t in a)

# ---------- Index persistence ----------
def save_index(chunks: List[str], idf, chunk_vectors, words_per_chunk: int, source_path: str):
    ensure_cache_dir()
    with gzip.open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "idf": idf,
            "chunk_vectors": chunk_vectors,
            "words_per_chunk": words_per_chunk,
            "source_path": os.path.abspath(source_path),
            "built_at": time.time()
        }, f)
    with open(META_PATH, "wb") as m:
        pickle.dump({"mtime": os.path.getmtime(source_path)}, m)

def load_index(source_path: str, words_per_chunk: int):
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None
    try:
        with open(META_PATH, "rb") as m:
            meta = pickle.load(m)
        src_mtime = os.path.getmtime(source_path)
        if abs(meta.get("mtime", 0) - src_mtime) > 1e-6:
            return None  # source changed, rebuild

        with gzip.open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        if (data.get("source_path") != os.path.abspath(source_path) or
            data.get("words_per_chunk") != words_per_chunk):
            return None
        return data
    except Exception:
        return None

# ---------- Retrieval & Answer ----------
def retrieve(query: str, idf, chunk_vectors, chunks: List[str], topk: int):
    qvec = vectorize_query(query, idf)
    scores = []
    for i, cvec in enumerate(chunk_vectors):
        s = cosine_sparse(qvec, cvec)
        scores.append((s, i))
    scores.sort(reverse=True)
    hits = [(idx, score, chunks[idx]) for score, idx in scores[:topk] if score > 0]
    if not hits:
        # Fall back to naive keyword scan if no TF-IDF overlap
        hits = naive_keyword_fallback(query, chunks, topk)
    return hits

def naive_keyword_fallback(query: str, chunks: List[str], topk: int):
    kws = tokenize(query)
    scored = []
    for i, ch in enumerate(chunks):
        low = ch.lower()
        score = sum(low.count(k) for k in kws)
        if score > 0:
            scored.append((score, i))
    scored.sort(reverse=True)
    return [(i, float(s), chunks[i]) for s, i in scored[:topk]]

def concise_extractive_answer(question: str, hits: List[Tuple[int,float,str]], max_sents: int = 3) -> str:
    if not hits:
        return "I couldn’t find a relevant passage in the local text."
    idx, _, txt = hits[0]
    sents = sentences(txt)
    kws = set(tokenize(question))
    selected = []
    # choose sentences that hit most keywords
    scored = []
    for s in sents:
        toks = set(tokenize(s))
        overlap = len(kws & toks)
        scored.append((overlap, s))
    scored.sort(reverse=True, key=lambda x: (x[0], len(x[1]) * -1))
    for ov, s in scored:
        if s not in selected and s.strip():
            selected.append(s.strip())
        if len(selected) >= max_sents:
            break
    blurb = " ".join(selected) if selected else " ".join(sents[:2])
    return f"{blurb} (from chunk #{idx})"

def fmt_preview(text: str, width=100, maxlen=280) -> str:
    t = text.strip().replace("\n", " ")
    if len(t) > maxlen:
        t = t[:maxlen] + "…"
    return "\n     ".join(textwrap.wrap(t, width=width))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Offline TF-IDF QA for Tom Sawyer (local file).")
    ap.add_argument("--file", required=True, help="Path to local tom_sawyer.txt")
    ap.add_argument("--words", type=int, default=DEFAULT_WORDS_PER_CHUNK, help="Words per chunk (default 180)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Top K supporting chunks (default 5)")
    ap.add_argument("--reindex", action="store_true", help="Force rebuild of index")
    ap.add_argument("question", nargs="+", help="Your question about the book")
    args = ap.parse_args()

    source_path = args.file
    question = " ".join(args.question).strip()

    ensure_cache_dir()

    data = None
    if not args.reindex:
        data = load_index(source_path, args.words)

    if data is None:
        print("Building index (first run or source changed)…")
        text = read_local_text(source_path)
        chunks = chunk_text(text, args.words)
        idf, chunk_vectors, _ = build_tfidf(chunks)
        save_index(chunks, idf, chunk_vectors, args.words, source_path)
    else:
        chunks = data["chunks"]
        idf = data["idf"]
        chunk_vectors = data["chunk_vectors"]

    # Retrieve
    hits = retrieve(question, idf, chunk_vectors, chunks, args.topk)

    print("\nTop supporting passages:")
    if not hits:
        print("  (no hits)")
    else:
        for rank, (idx, score, ch) in enumerate(hits, 1):
            print(f"{rank}. chunk #{idx}  score={score:.3f}")
            print(f"     {fmt_preview(ch)}\n")

    print("Answer:")
    print(concise_extractive_answer(question, hits))

if __name__ == "__main__":
    main()

