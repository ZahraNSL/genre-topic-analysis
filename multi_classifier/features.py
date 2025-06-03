from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
import scipy.sparse as sp

tfidf = TfidfVectorizer(max_features=40_000,
                        ngram_range=(1,2),
                        stop_words="english")

def tfidf_matrix(train, test):
    return tfidf.fit_transform(train), tfidf.transform(test)

def tfidf_plus_svd(Xtr, Xte, n=500):
    svd = TruncatedSVD(n_components=n, random_state=0)
    return svd.fit_transform(Xtr), svd.transform(Xte)

def tfidf_plus_nmf(Xtr, Xte, n=50):
    nmf = NMF(n_components=n, random_state=0)
    Ztr = nmf.fit_transform(Xtr)
    Zte = nmf.transform(Xte)
    return sp.hstack([Xtr, Ztr]), sp.hstack([Xte, Zte])