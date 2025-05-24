
import json, gzip, itertools, pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   # light wrapper for prettier grids
from matplotlib import ticker

def dump_topics(model, out_path="topics.txt", top_n_words=10):

    with open(out_path, "w", encoding="utf-8") as f:
        for tid in model.get_topic_freq().Topic:
            if tid == -1:
                continue
            words = " ".join(w for w, _ in model.get_topic(tid)[:top_n_words])
            f.write(f"{tid}\t{words}\n")
    print("Topics written to", out_path)

def plot_unsafe_bar(unsafe_dict, out_png="unsafe_share.png"):
   
    genres  = list(unsafe_dict.keys())
    shares  = [100 * unsafe_dict[g] for g in genres]  # convert to %
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(genres, shares)
    ax.set_ylabel("% of topics flagged unsafe")
    ax.set_ylim(0, max(shares) * 1.25 + 0.1)
    ax.set_title("Unsafe topic proportion per genre")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    for bar, val in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}%",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("Bar chart saved to", out_png)