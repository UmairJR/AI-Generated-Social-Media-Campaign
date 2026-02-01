from sentence_transformers import SentenceTransformer
import hdbscan

_model = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embeddings(texts):
    model = load_model()
    return model.encode(texts)


def sbert_hdbscan(texts):
    X = get_embeddings(texts)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(X)

    clusters = {}
    for label, text in zip(labels, texts):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(text)

    return list(clusters.values()), labels


