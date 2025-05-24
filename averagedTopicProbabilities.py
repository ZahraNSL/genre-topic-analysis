import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon


def genre_vector(model, probs, all_topic_words):
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

def averagedTopicProbabilities(topic_models,GENRES):
    all_topic_words = set()
    for m,_,_ in topic_models.values():
        for tid in m.get_topic_freq().Topic:
            if tid != -1:
                all_topic_words.add(" ".join(w for w,_ in m.get_topic(tid)[:10]))
    all_topic_words = sorted(all_topic_words)
    #vectors = [genre_vector(*topic_models[g]) for g in GENRES]
    vectors = [
    genre_vector(topic_models[g][0], topic_models[g][2], all_topic_words)
    for g in GENRES
    ]
    distmat = squareform(pdist(np.vstack(vectors), metric="jensenshannon"))
    print("\nJensen–Shannon divergence matrix:")
    print(distmat.round(3))


