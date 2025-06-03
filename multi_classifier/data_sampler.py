import gzip, json, pathlib, collections, random
from typing import List, Tuple

def stream_sporc(path: str, cap_per_label: int | None = None,
                 min_tokens:int=200) -> List[Tuple[str, List[str]]]:

    path = pathlib.Path(path)
    op   = gzip.open if path.suffix == ".gz" else open
    cap_counter = collections.Counter()
    data = []

    with op(path, "rt") as fh:
        for line in fh:
            o   = json.loads(line)
            txt = o.get("transcript", "")
            if len(txt.split()) < min_tokens:
                continue

            c1, c2 = o.get("category1"), o.get("category2")
            if not c1:                             # must have primary label
                continue
            if cap_per_label and cap_counter[c1] >= cap_per_label:
                continue

            labels = [c1]
            if c2 and c2 != c1:
                labels.append(c2)
            cap_counter[c1] += 1
            data.append((txt, labels))

    random.shuffle(data)
    return data