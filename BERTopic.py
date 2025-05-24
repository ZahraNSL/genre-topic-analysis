from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, disable_caching
disable_caching()                                       # avoids huge cache copies

def BERTopicModel(GENRES, SUBSET_DIR):
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


    return topic_models