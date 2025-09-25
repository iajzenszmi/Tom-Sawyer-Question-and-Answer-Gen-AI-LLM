# Tom-Sawyer-Question-and-Answer-Gen-AI-LLM
Tom Sawyer Question and Answer generative Artificial Intelligence Large Language Model 
Skip to content
Chat history
You said:
An llm generative Ai to answer questions on mark Twain's tom sawyer
ChatGPT said:

Awesome idea! Here’s a lightweight, local RAG (retrieval-augmented) question-answering tool for The Adventures of Tom Sawyer that works from the public-domain text. It’s a single Python file (CLI) that:

    downloads & caches the Gutenberg text,

    builds a TF-IDF vector index (no heavyweight dependencies),

    retrieves the most relevant passages to your question,

    optionally uses Ollama (if installed) for a short, generative answer grounded in those passages,

    otherwise falls back to a concise extractive answer.

1) Quick start

python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn requests

# (Optional for better answers) Install Ollama + a model, e.g.:
# https://ollama.com/download
# ollama pull llama3.1

# Run it:
python tom_sawyer_qa.py "Why do Tom and Huck go to the graveyard?"

    If Ollama is running and a model is available, you’ll get a grounded, generative answer. If not, you’ll still get top passages + a concise extractive answer.

2) The code (save as tom_sawyer_qa.py)

#!/usr/bin/env python3
import os, sys, json, re, subprocess, shutil, math, pathlib, time
from typing import List, Tuple
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/74/pg74.txt"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".tom_sawyer_qa")
RAW_PATH = os.path.join(CACHE_DIR, "tom_sawyer.txt")
INDEX_PATH = os.path.join(CACHE_DIR, "index.npz")
META_PATH = os.path.join(CACHE_DIR, "meta.json")

# ---- Utilities ----
def ensure_cache_dir():
    pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def download_text():
    ensure_cache_dir()
    if os.path.exists(RAW_PATH) and os.path.getsize(RAW_PATH) > 100_000:
        return RAW_PATH
    print("Downloading Tom Sawyer text from Project Gutenberg…")
    r = requests.get(GUTENBERG_URL, timeout=30)
    r.raise_for_status()
    text = r.text
    # Trim Gutenberg header/footer
    start = text.find("THE ADVENTURES OF TOM SAWYER")
    end = text.find("End of the Project Gutenberg EBook")
    if start != -1 and end != -1:
        text = text[start:end]
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return RAW_PATH

def chunk_text(text: str, words_per_chunk: int = 180) -> List[str]:
    # simple paragraph split, then re-chunk to ~N words
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    tokens = []
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

def build_or_load_index() -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
    ensure_cache_dir()
    text_path = download_text()
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            data = np.load(INDEX_PATH, allow_pickle=True)
            vectorizer = TfidfVectorizer(vocabulary=data["vocab"].tolist())
            X = data["X"]
            with open(META_PATH, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if meta.get("n_chunks") == len(chunks):
                return vectorizer, X, chunks
        except Exception:
            pass

    print("Building TF-IDF index…")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        max_df=0.9,
        min_df=2
    )
    X = vectorizer.fit_transform(chunks).astype(np.float32)

    np.savez_compressed(INDEX_PATH, X=X, vocab=np.array(list(vectorizer.vocabulary_.keys())))
    with open(META_PATH, "w", encoding="utf-8") as mf:
        json.dump({"n_chunks": len(chunks), "ts": time.time()}, mf)
    return vectorizer, X, chunks

def retrieve(query: str, vectorizer: TfidfVectorizer, X: np.ndarray, chunks: List[str], k: int = 5):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).flatten()
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[int(i)]) for i in idx]

def has_ollama() -> bool:
    return shutil.which("ollama") is not None

def ask_ollama(prompt: str, model: str = "llama3.1") -> str:
    try:
        # Streamed or non-streamed; keep it simple here:
        res = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=120
        )
        if res.returncode == 0:
            return res.stdout.strip()
        return ""
    except Exception:
        return ""

def make_prompt(question: str, passages: List[Tuple[int,float,str]]) -> str:
    ctx = "\n\n".join([f"[Chunk #{i} | score={score:.3f}]\n{txt}" for i,score,txt in passages])
    guard = (
        "You are a helpful literary assistant answering ONLY from the provided passages "
        "from Mark Twain's *The Adventures of Tom Sawyer*. If the answer is not present, say you don't know. "
        "Cite chunk numbers you used. Be concise (3–6 sentences)."
    )
    return f"{guard}\n\nPassages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    
def extractive_answer(question: str, passages: List[Tuple[int,float,str]]) -> str:
    # Minimal heuristic: choose the top passage, return 2-3 sentences that include query keywords
    top_idx, _, top_txt = passages[0]
    sentences = re.split(r'(?<=[.!?])\s+', top_txt)
    # pick up to 3 sentences that contain any keyword from the query
    kws = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z']+", question)]
    selected = []
    for s in sentences:
        low = s.lower()
        if any(k in low for k in kws):
            selected.append(s)
        if len(selected) >= 3:
            break
    if not selected:
        selected = sentences[:2]
    cited = f"(from chunk #{top_idx})"
    return " ".join(selected).strip() + " " + cited

def main():
    if len(sys.argv) < 2:
        print("Usage: python tom_sawyer_qa.py \"Your question about Tom Sawyer\"")
        sys.exit(1)
    question = sys.argv[1]

    vectorizer, X, chunks = build_or_load_index()
    hits = retrieve(question, vectorizer, X, chunks, k=5)

    print("\nTop supporting passages:")
    for i, (idx, score, txt) in enumerate(hits, 1):
        preview = (txt[:240] + "…") if len(txt) > 240 else txt
        print(f"{i}. chunk #{idx}  score={score:.3f}\n   {preview}\n")

    if has_ollama():
        prompt = make_prompt(question, hits)
        print("Using Ollama for a grounded answer…")
        answer = ask_ollama(prompt)
        if answer:
            print("\nAnswer (LLM, grounded):\n" + answer)
            return

    # Fallback extractive
    print("\nAnswer (extractive):")
    print(extractive_answer(question, hits))

if __name__ == "__main__":
    main()

3) What you get

    Grounded answers: The LLM sees only the most relevant passages, so responses stick to Twain’s text.

    Citations: Chunk numbers printed with scores; the LLM is asked to cite chunks it used.

    Offline-friendly: Works without an LLM (you still get useful passages + a short answer).

4) Example questions to try

python tom_sawyer_qa.py "How does Tom get out of whitewashing the fence?"
python tom_sawyer_qa.py "Why do Tom and Huck visit the graveyard at midnight?"
python tom_sawyer_qa.py "Who is Becky Thatcher and what happens between her and Tom?"
python tom_sawyer_qa.py "What role does Injun Joe play in the story?"
python tom_sawyer_qa.py "How are the cave scenes resolved?"

5) Optional upgrades

    Better retrieval: swap TF-IDF for embeddings (e.g., sentence-transformers + FAISS).

    UI: wrap with Textual (TUI) or a tiny FastAPI/Gradio app (only if you want a UI; otherwise CLI avoids the whole uvicorn thing).

    Stricter grounding: add a “do not answer outside context” system prompt and limit generation length.

If you want, I can give you an embeddings+FAISS version or a no-internet zip with everything set up.
You said:
alhost:~$ pip install numpy --force
Collecting numpy
  Using cached numpy-2.2.6.tar.gz (20.3 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [23 lines of output]
      + /home/userland/.venv/bin/python3 /tmp/pip-install-_5oo7uws/numpy_00ea879b654e46b28bef7e7e1c4c9c78/vendored-meson/meson/meson.py setup /tmp/pip-install-_5oo7uws/numpy_00ea879b654e46b28bef7e7e1c4c9c78 /tmp/pip-install-_5oo7uws/numpy_00ea879b654e46b28bef7e7e1c4c9c78/.mesonpy-583ide2b -Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-install-_5oo7uws/numpy_00ea879b654e46b28bef7e7e1c4c9c78/.mesonpy-583ide2b/meson-python-native-file.ini
      The Meson build system
      Version: 1.5.2
      Source dir: /tmp/pip-install-_5oo7uws/numpy_00ea879b654e46b28bef7e7e1c4c9c78
      Build dir: /tmp/pip-install-_5oo7uws/numpy_00ea879b654e46b28bef7e7e1c4c9c78/.mesonpy-583ide2b
      Build type: native build
      Project name: NumPy
      Project version: 2.2.6
      C compiler for the host machine: cc (gcc 11.4.0 "cc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0")
      C linker for the host machine: cc ld.bfd 2.38
      C++ compiler for the host machine: c++ (gcc 11.4.0 "c++ (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0")
      C++ linker for the host machine: c++ ld.bfd 2.38
      Cython compiler for the host machine: cython (cython 3.1.4)
      Host machine cpu family: arm
      Host machine cpu: armv7l
      Program python found: YES (/home/userland/.venv/bin/python3)
      Did not find pkg-config by name 'pkg-config'
      Found pkg-config: NO
      Run-time dependency python found: NO (tried pkgconfig, pkgconfig and sysconfig)

      ../meson.build:41:12: ERROR: Python dependency not found

      A full log can be found at /tmp/pip-install-_5oo7uws/numpy_00ea879b654e46b28bef7e7e1c4c9c78/.mesonpy-583ide2b/meson-logs/meson-log.txt
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
(.venv) userland@localhost:~$ pip install sysconfig
ERROR: Could not find a version that satisfies the requirement sysconfig (from versions: none)
ERROR: No matching distribution found for sysconfig
(.venv) userland@localhost:~$
ChatGPT said:

You’re on 32-bit ARM (armv7l) inside UserLAnd, and pip is trying to build NumPy 2.2.6 from source with Meson. The build fails because the Python dev files and pkg-config aren’t present, so Meson can’t find the “Python dependency.” Also, sysconfig is part of the Python standard library—it’s not a pip package, so pip install sysconfig will always fail.

Here are 3 reliable ways to fix it—pick one.
✅ Option A (easiest): Use the Ubuntu package

This avoids compiling NumPy on armv7.

sudo apt update
sudo apt install -y python3-numpy

If you want to use it inside a venv, recreate your venv with system packages visible:

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import numpy as np; print(np.__version__)"

✅ Option B: Install build deps, then pin to a 1.x wheel

NumPy 2.x is heavier to build and often has no wheels for armv7. NumPy 1.26.4 is the last 1.x and usually works better on 32-bit.

sudo apt update
sudo apt install -y build-essential python3-dev pkg-config \
                    libopenblas-dev gfortran
# (optional but helpful) ninja-build

# In your venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# Prefer an older wheel that’s more likely to exist for armv7:
pip install "numpy==1.26.4"

    If a wheel isn’t available and it still tries to compile, you at least have the headers and BLAS so the build can succeed.

✅ Option C: Stay with system Python (no venv) just for NumPy

For quickest progress:

sudo apt install -y python3-numpy
python3 -c "import numpy as np; print(np.__version__)"

You can still run your Tom Sawyer QA script with the system Python.
Why the error happened (plain English)

    Meson says: Run-time dependency python found: NO → it can’t locate Python’s development info via pkg-config/sysconfig because python3-dev and pkg-config aren’t installed.

    sysconfig is built into Python; don’t pip install it.

Quick sanity checks

Run these and share outputs if anything still breaks:

python3 -c "import sys, platform, sysconfig; print(sys.version); print(platform.machine()); print(sysconfig.get_config_var('INCLUDEPY'))"
pkg-config --version || echo "no pkg-config"
python3-config --includes || echo "no python3-dev"

If you tell me which option you chose and the exact errors (if any), I’ll adjust the commands for your setup.
You said:
read tom_sawyer.txt from local file without going online
ChatGPT said:

Got it — if you already have tom_sawyer.txt on your local system, you don’t need the script to download from Project Gutenberg. You just need to tell the QA script to read that file instead of fetching online.

Here’s how you can adapt the Python code I gave you earlier:
1) Put the file in place

Let’s say your local copy is saved as:

/home/userland/tom_sawyer.txt

2) Minimal change in the script

Replace the download_text() function with a version that simply opens your local file:

LOCAL_PATH = "/home/userland/tom_sawyer.txt"  # adjust path

def load_local_text():
    if not os.path.exists(LOCAL_PATH):
        raise FileNotFoundError(f"Cannot find {LOCAL_PATH}")
    with open(LOCAL_PATH, "r", encoding="utf-8") as f:
        return f.read()

Then in build_or_load_index(), swap:

text_path = download_text()
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

for:

text = load_local_text()

3) Run it

python tom_sawyer_qa.py "Why do Tom and Huck go to the graveyard?"

4) Alternative: Quick standalone script

If you don’t want caching/indexing logic, here’s a minimal offline script that just reads your file, splits it into chunks, and answers extractively:

import re, sys

LOCAL_PATH = "/home/userland/tom_sawyer.txt"

def load_text():
    with open(LOCAL_PATH, "r", encoding="utf-8") as f:
        return f.read()

def simple_search(question, text, window=300):
    # crude keyword search
    qwords = [w.lower() for w in re.findall(r"\w+", question)]
    best, idx = 0, 0
    for i in range(len(text) - window):
        chunk = text[i:i+window].lower()
        score = sum(chunk.count(w) for w in qwords)
        if score > best:
            best, idx = score, i
    return text[idx:idx+window]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python offline_tom_sawyer.py 'your question'")
        sys.exit(1)
    text = load_text()
    q = sys.argv[1]
    passage = simple_search(q, text)
    print("Question:", q)
    print("Answer (extractive guess):", passage)

👉 Do you want me to prepare a ready-to-run offline version of the TF-IDF QA script (so you don’t have to edit it yourself), or would the simpler keyword-search version above be good enough?
You said:
ready to run offline version
ChatGPT said:

Perfect — here’s a single-file, dependency-free, fully offline QA tool for Tom Sawyer.
It uses a custom TF-IDF retriever (no NumPy / scikit-learn), reads your local file, and caches its index for fast repeats.
How to run

# 1) Save the script below as: tom_sawyer_offline_qa.py
# 2) Point it at your local text file and ask a question:
python3 tom_sawyer_offline_qa.py --file "/home/userland/tom_sawyer.txt" "How does Tom avoid whitewashing the fence?"

# Optional flags:
#   --reindex        Force rebuild the index
#   --topk 5         Change number of supporting chunks (default 5)
#   --words 180      Chunk size in words (default 180)

The script (drop-in, no extra installs)

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

Notes

    No internet, no extra packages. Everything is Python standard library.

    Works on armv7/UserLAnd without NumPy/Meson headaches.

    Re-run with --reindex if you change the source text or want a different --words chunk size.

    If you later want a small web UI (still offline), I can add a minimal http.server or Textual TUI on top of this without adding heavy dependencies.


ChatGPT can make mistakes. Check important info.
