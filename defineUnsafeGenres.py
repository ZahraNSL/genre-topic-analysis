from transformers import pipeline
from visualise_topics import dump_topics, plot_unsafe_bar

def is_unsafe(label):
    return label["label"] in {"toxic", "hate", "sexual_explicit", "violent"}

def unsafeGenres(GENRES, topic_models):
    unsafe_share = {}
    mod = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", truncation=True)
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

    for g in GENRES:
        model, _, _ = topic_models[g]
        dump_topics(model, out_path=f"topics_{g}.txt")

    plot_unsafe_bar(unsafe_share, out_png="unsafe_share.png")