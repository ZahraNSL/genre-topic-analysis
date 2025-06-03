from data_sampler import stream_sporc
from features import tfidf_matrix, tfidf_plus_svd, tfidf_plus_nmf
from models   import fasttext_clf, linear_clf, rf_clf
from metrics  import multilabel_f1
from sklearn.model_selection import train_test_split
import argparse, json, scipy.sparse as sp
from sklearn.preprocessing import MultiLabelBinarizer

def run(args):
    data = stream_sporc(args.file, cap_per_label=args.cap)
    texts, labels = zip(*data)

    mlb = MultiLabelBinarizer()

    X_train_txt, X_test_txt, y_train_raw, y_test_raw = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    y_train = mlb.fit_transform(y_train_raw)
    y_test  = mlb.transform(y_test_raw)
    # ----- choose feature pipeline -----
    Xtr_tfidf, Xte_tfidf = tfidf_matrix(X_train_txt, X_test_txt)

    if args.feature == "tfidf":
        Xtr, Xte = Xtr_tfidf, Xte_tfidf
    elif args.feature == "svd":
        Xtr, Xte = tfidf_plus_svd(Xtr_tfidf, Xte_tfidf, n=500)
    elif args.feature == "nmf":
        Xtr, Xte = tfidf_plus_nmf(Xtr_tfidf, Xte_tfidf, n=50)
    else:
        raise SystemExit("feature must be tfidf|svd|nmf")

    # ----- choose model -----
    if args.model == "fasttext":
        y_pred = fasttext_clf(X_train_txt, y_train_raw, X_test_txt)
    else:
        if args.model == "sgd":
            clf = linear_clf("sgd")
        elif args.model == "logreg":
            clf = linear_clf("logreg")
        elif args.model == "rf":
            clf = rf_clf()
        else:
            raise SystemExit("model must be fasttext|sgd|logreg|rf")

        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)

    f1 = multilabel_f1(y_test, y_pred)
    print(json.dumps(f1, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="blitt/episodeLevelData.jsonl.gz")
    p.add_argument("--cap", type=int, default=None,
                   help="optional cap per primary genre")
    p.add_argument("--feature", choices=["tfidf", "svd", "nmf"], default="tfidf")
    p.add_argument("--model",   choices=["fasttext","sgd","logreg","rf"],
                   default="sgd")
    run(p.parse_args())