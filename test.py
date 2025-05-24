import builtins, gzip, json, os

path   = "genre_dataset/Comedy.jsonl.gz"      # ← change if you decompressed
open_f = gzip.open if path.endswith(".gz") else builtins.open


pod_titles = set()
categories = set()

with open_f(path, "rt") as fh:
    for line in fh:                             # never loads more than one line
        obj = json.loads(line)

        # 2a.  podcast title
        title = obj.get("podTitle")
        if title:
            pod_titles.add(title)

        # 2b.  up to ten category columns
        for i in range(1):
            cat = obj.get(f"category{i}")
            if cat:
                categories.add(cat)

print(f"{len(pod_titles):,} unique podcast titles")
print(f"{len(categories):,} unique categories/genres")

# peek at a few examples
for t in list(pod_titles)[:10]:
    print("•", t)
print("\nSample categories:", list(categories))


# quick peek
for t in list(pod_titles)[:10]:
    print("•", t)
