from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextClust:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.clusters = []

    def fit(self, texts):
        texts = list(texts)  # Convert pandas Series to list
        if not texts:
            return

        X = self.vectorizer.fit_transform(texts)
        assigned = [False] * len(texts)

        for i, vec in enumerate(X):
            if assigned[i]:
                continue

            cluster = [texts[i]]
            assigned[i] = True

            for j in range(i + 1, len(texts)):
                if not assigned[j]:
                    sim = cosine_similarity(vec, X[j])[0][0]
                    if sim >= self.threshold:
                        cluster.append(texts[j])
                        assigned[j] = True

            self.clusters.append(cluster)

    def extract_keywords(self, cluster, top_k=5):
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(cluster)

        scores = np.asarray(X.mean(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()

        top_indices = scores.argsort()[::-1][:top_k]
        return [terms[i] for i in top_indices]

    def detect_pattern(self, tweet_count):
        return "burst" if tweet_count >= 5 else "slow"

