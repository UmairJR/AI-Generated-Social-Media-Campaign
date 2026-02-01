from textclust import TextClust
from semantic_clustering import sbert_hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def tfidf_kmeans(texts, k=5):
    X = TfidfVectorizer(stop_words="english").fit_transform(texts)

    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)

    clusters = {}
    for label, text in zip(labels, texts):
        clusters.setdefault(label, []).append(text)

    return list(clusters.values()), labels


def run_clustering(texts, algorithm, threshold=0.05):
    if algorithm == "TextClust (Lexical)":
        tc = TextClust(threshold=threshold)
        tc.fit(texts)
        return tc.clusters, None

    if algorithm == "SBERT + HDBSCAN (Semantic)":
        return sbert_hdbscan(texts)


