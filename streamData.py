import builtins
import gzip
import json
import re

from tqdm import tqdm


def opener(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else builtins.open(path, "rt")

# very quick tokenizer, good enough for topic modelling
TOKEN_RE = re.compile(r"[a-z]{3,}")
def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def streamData(GENRES, SUBSET_DIR, MAX_DOCS_PER_GENRE,RAW ):

    counters = {g: 0 for g in GENRES}
    writers = {g: gzip.open(SUBSET_DIR / f"{g}.jsonl.gz", "wt") for g in GENRES}

    with opener(RAW) as fh:
        for line in tqdm(fh, desc="scanning corpus"):
            o = json.loads(line)
            g = o.get("category1")  # SPoRC primary genre label
            if g in GENRES and counters[g] < MAX_DOCS_PER_GENRE:
                writers[g].write(
                    json.dumps(
                        {
                            "transcript": o["transcript"],
                            "episode": o["epTitle"],
                            "show": o["podTitle"],
                        }
                    )
                    + "\n"
                )
                counters[g] += 1
            if all(c >= MAX_DOCS_PER_GENRE for c in counters.values()):
                break

    for w in writers.values():
        w.close()
