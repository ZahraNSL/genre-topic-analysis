from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import fasttext
import tempfile, os

def fasttext_clf(train_txt, train_labels, test_txt):
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
    for txt, labels in zip(train_txt, train_labels):
        tmp.write("__label__" + " __label__".join(labels) + " " + txt + "\n")
    tmp.close()
    model = fasttext.train_supervised(tmp.name, epoch=25, lr=1.0,
                                      wordNgrams=2, dim=100, loss="ova")
    os.unlink(tmp.name)
    preds = []
    for t in test_txt:
        labs, _ = model.predict(t, k=5)   # at most 5 labels
        preds.append([lab.replace("__label__", "") for lab in labs])
    return preds

def linear_clf(backend="sgd"):
    base = (SGDClassifier(loss="log_loss", max_iter=1000) if backend=="sgd"
            else LogisticRegression(max_iter=400, multi_class="ovr"))
    return MultiOutputClassifier(base)

def rf_clf():
    return MultiOutputClassifier(RandomForestClassifier(
        n_estimators=400, n_jobs=-1))