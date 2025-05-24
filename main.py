import builtins, gzip, json, itertools, os, re
from pathlib import Path
from tqdm import tqdm

from BERTopic import BERTopicModel
from averagedTopicProbabilities import averagedTopicProbabilities
from defineUnsafeGenres import unsafeGenres
from streamData import streamData

RAW = Path("blitt/episodeLevelData.jsonl.gz")          # adjust if file is plain .jsonl
GENRES = ["kids", "religion", "society", "news"]    # pick any four labels you like
MAX_DOCS_PER_GENRE = 12_000                            # keeps RAM modest (< 4 GB)
SUBSET_DIR = Path("genre_subsets")
SUBSET_DIR.mkdir(exist_ok=True)


#streamData(GENRES, SUBSET_DIR, MAX_DOCS_PER_GENRE,RAW );
topic_models = BERTopicModel(GENRES, SUBSET_DIR);
averagedTopicProbabilities(topic_models,GENRES);
unsafeGenres(GENRES, topic_models);