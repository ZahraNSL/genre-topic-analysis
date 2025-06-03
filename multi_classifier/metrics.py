from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

mlb = MultiLabelBinarizer()

def multilabel_f1(y_true, y_pred):

    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)
    return dict(
        weighted=f1_score(Y_true, Y_pred, average="weighted", zero_division=0),
        micro   =f1_score(Y_true, Y_pred, average="micro",    zero_division=0),
        macro   =f1_score(Y_true, Y_pred, average="macro",    zero_division=0)
    )