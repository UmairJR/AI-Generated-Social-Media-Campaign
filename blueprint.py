from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re


def time_to_minutes(t):
    if isinstance(t, str):
        h, m = t.split(":")
        return int(h) * 60 + int(m)
    else:  # assume Timestamp
        return t.hour * 60 + t.minute



def extract_keywords(cluster, top_k=5):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(cluster)
    scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(terms, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [term for term, _ in ranked[:top_k]]


def extract_anchor_phrases(cluster, min_freq=3):
    tokens = []
    for text in cluster:
        tokens.extend(re.findall(r"\b\w+\b", text.lower()))

    counts = Counter(tokens)
    return [w for w, c in counts.items() if c >= min_freq]


def detect_pattern(cluster_size, duration_minutes):
    if cluster_size >= 15 and duration_minutes <= 60:
        return "burst"
    elif cluster_size >= 7:
        return "coordinated"
    else:
        return "weak"


def extract_blueprint(cluster, timestamps=None):
    blueprint = {}

    blueprint["tweets"] = len(cluster)

    # ⏱ Duration calculation
    if timestamps and len(timestamps) >= 2:
        times = [time_to_minutes(t) for t in timestamps]
        duration = max(times) - min(times)
        blueprint["duration_minutes"] = duration
    else:
        blueprint["duration_minutes"] = None

    blueprint["keywords"] = extract_keywords(cluster)
    blueprint["anchor_phrases"] = extract_anchor_phrases(cluster)
    blueprint["topic"] = blueprint["keywords"][0]

    blueprint["pattern"] = detect_pattern(
        blueprint["tweets"],
        blueprint["duration_minutes"] or 0
    )

    return blueprint
