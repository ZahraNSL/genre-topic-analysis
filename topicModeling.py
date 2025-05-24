# ------------------------------------------------------------------
# 0. environment – install once, then restart the interpreter
# ------------------------------------------------------------------
# pip install datasets bertopic sentence-transformers nltk
# pip install gensim scikit-learn transformers duckdb tqdm

# ------------------------------------------------------------------
# 1. small helpers and configuration
# ------------------------------------------------------------------
import builtins, gzip, json, itertools, os, re
from pathlib import Path
from tqdm import tqdm

RAW = Path("blitt/episodeLevelData.jsonl.gz")          # adjust if file is plain .jsonl
GENRES = ["Sports", "Comedy", "True Crime", "News"]    # pick any four labels you like
MAX_DOCS_PER_GENRE = 12_000                            # keeps RAM modest (< 4 GB)
SUBSET_DIR = Path("genre_subsets")
SUBSET_DIR.mkdir(exist_ok=True)

def opener(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else builtins.open(path, "rt")

# very quick tokenizer, good enough for topic modelling
TOKEN_RE = re.compile(r"[a-z]{3,}")
def tokenize(text):
    return TOKEN_RE.findall(text.lower())

# ------------------------------------------------------------------
# 2. stream once through the big file → create one compressed
#    JSONL per chosen genre, capped at MAX_DOCS_PER_GENRE
# ------------------------------------------------------------------
counters = {g: 0 for g in GENRES}
writers  = {g: gzip.open(SUBSET_DIR / f"{g}.jsonl.gz", "wt") for g in GENRES}

with opener(RAW) as fh:
    for line in tqdm(fh, desc="scanning corpus"):
        o = json.loads(line)
        g = o.get("category1")                         # SPoRC primary genre label
        if g in GENRES and counters[g] < MAX_DOCS_PER_GENRE:
            writers[g].write(json.dumps({
                "transcript": o["transcript"],
                "episode":    o["epTitle"],
                "show":       o["podTitle"]
            }) + "\n")
            counters[g] += 1
        if all(c >= MAX_DOCS_PER_GENRE for c in counters.values()):
            break

for w in writers.values():
    w.close()

print("documents kept:", counters)

# ------------------------------------------------------------------
# 3. build one BERTopic model per genre
# ------------------------------------------------------------------
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, disable_caching
disable_caching()                                       # avoids huge cache copies

topic_models = {}
for g in GENRES:
    ds = load_dataset("json",
                      data_files=str(SUBSET_DIR / f"{g}.jsonl.gz"),
                      split="train")                    # small enough to materialise
    docs = ds["transcript"]
    model = BERTopic(embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
                     language="english",
                     calculate_probabilities=True,
                     verbose=True)
    topics, probs = model.fit_transform(docs)
    model.save(f"bertopic_{g}")
    topic_models[g] = (model, topics, probs)
    print(f"{g}: {model.get_topic_info().shape[0]} topics")

# ------------------------------------------------------------------
# 4. derive one averaged topic-probability vector per genre
#    then compute Jensen–Shannon divergence between genres
# ------------------------------------------------------------------
import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon

all_topic_words = set()
for m,_,_ in topic_models.values():
    for tid in m.get_topic_freq().Topic:
        if tid != -1:
            all_topic_words.add(" ".join(w for w,_ in m.get_topic(tid)[:10]))
all_topic_words = sorted(all_topic_words)

def genre_vector(model, topics, probs):
    # probability mass aggregated over documents
    vec = np.zeros(len(all_topic_words))
    for doc_topics in probs:
        for tid, p in enumerate(doc_topics):
            if tid == -1 or p == 0:
                continue
            word_sig = " ".join(w for w,_ in model.get_topic(tid)[:10])
            idx = all_topic_words.index(word_sig)
            vec[idx] += p
    vec /= vec.sum()
    return vec

vectors = [genre_vector(*topic_models[g]) for g in GENRES]
distmat = squareform(pdist(np.vstack(vectors), metric="jensenshannon"))
print("\nJensen–Shannon divergence matrix:")
print(distmat.round(3))

# ------------------------------------------------------------------
# 5. flag potentially unsafe topics with an off-the-shelf
#    moderation classifier, then tally by genre
# ------------------------------------------------------------------
from transformers import pipeline
mod = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", truncation=True)

def is_unsafe(label):
    return label["label"] in {"toxic", "hate", "sexual_explicit", "violent"}

unsafe_share = {}
for g in GENRES:
    model, topics, probs = topic_models[g]
    unsafe = 0
    total  = 0
    for tid in model.get_topic_freq().Topic:
        if tid == -1:
            continue
        text = " ".join(w for w,_ in model.get_topic(tid)[:20])
        label = mod(text, max_length=128, truncation=True)[0]
        if is_unsafe(label):
            unsafe += 1
        total  += 1
    unsafe_share[g] = unsafe / total

print("\nfraction of topics flagged unsafe:")
for g,v in unsafe_share.items():
    print(f"{g:10s}  {v:.2%}")

# ------------------------------------------------------------------
# 6. simple interpretation
# ------------------------------------------------------------------
print("\nInterpretation:")
least, most = min(unsafe_share, key=unsafe_share.get), max(unsafe_share, key=unsafe_share.get)
print(f"Among these four genres, {most} shows the highest proportion "
      f"of potentially unsafe topics, while {least} shows the lowest. "
      f"The divergence matrix suggests how firmly topic usage separates each genre; "
      f"pairs with lower JS values (closer to zero) share more thematic ground.")