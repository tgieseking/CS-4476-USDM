import numpy as np
from sklearn.cluster import KMeans


def createVocab(sift_features, vocab_size):
    vocab = KMeans(vocab_size).fit(sift_features)
    return vocab

def createHist(sift_features, vocab, vocab_size):
    clusters = vocab.predict(sift_features)
    vocab_hist = np.bincount(clusters, minlength = vocab_size)
    if vocab_hist.size != vocab_size:
        raise HistLengthMismatch()
    return vocab_hist

class HistLengthMismatch(Exception):
    pass
