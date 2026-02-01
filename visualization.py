import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA



def plot_similarity_heatmap(texts, title):
    """
    texts: list[str]
    title: str
    """
    if len(texts) < 2:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        sim_matrix,
        cmap="YlOrRd",
        square=True,
        cbar=True,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Tweet index")
    ax.set_ylabel("Tweet index")

    return fig

def plot_cluster_weight_over_time1(texts, clusters, window=5):
    """
    texts: full tweet stream (ordered)
    clusters: output of TextClust
    window: sliding window size
    """
    tweet_to_cluster = {}
    for cid, cluster in enumerate(clusters):
        for t in cluster:
            tweet_to_cluster[t] = cid

    data = []

    for i in range(len(texts)):
        window_texts = texts[max(0, i - window): i + 1]
        counts = {}

        for t in window_texts:
            cid = tweet_to_cluster.get(t, -1)
            counts[cid] = counts.get(cid, 0) + 1

        for cid, weight in counts.items():
            data.append({
                "time": i,
                "cluster": cid,
                "weight": weight
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=df,
        x="time",
        y="weight",
        hue="cluster",
        marker="o",
        ax=ax
    )

    ax.set_title("Cluster Weight Over Time")
    ax.set_xlabel("Tweet index (time)")
    ax.set_ylabel("Cluster weight")

    return fig

def get_2d_embeddings(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    return X_2d

def get_cluster_labels(texts, clusters):
    labels = [-1] * len(texts)

    for cluster_id, cluster in enumerate(clusters):
        for tweet in cluster:
            idx = texts.index(tweet)
            labels[idx] = cluster_id

    return labels

def cluster_cohesion(cluster):
    if len(cluster) < 2:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(cluster)
    sim = cosine_similarity(X)

    return (sim.sum() - len(cluster)) / (len(cluster)**2 - len(cluster))

def plot_cluster_membership(texts, clusters):
    labels = [-1] * len(texts)

    for cid, cluster in enumerate(clusters):
        for t in cluster:
            if t in texts:
                labels[texts.index(t)] = cid

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(
        labels,
        range(len(texts)),
        c=labels,
        cmap="tab10",
        s=60
    )

    ax.set_xlabel("Tweet index (time)")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Tweet-to-Cluster Assignment Over Time")

    return fig

def plot_cluster_sizes(clusters):
    sizes = sorted(
        [(i, len(c)) for i, c in enumerate(clusters)],
        key=lambda x: x[1],
        reverse=True
    )

    ids, counts = zip(*sizes)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ids, counts, marker="o")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of tweets")
    ax.set_title("Cluster Size Distribution")

    return fig

def build_cluster_index(clusters):
    index = {}
    for cid, cluster in enumerate(clusters):
        for text in cluster:
            index[text] = cid
    return index

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_cluster_weight_over_time2(
    df,
    clusters,
    window=5,
    highlight_largest=True
):
    """
    df: dataframe with ['timestamp', 'text']
    clusters: list of clusters (list of lists)
    window: sliding window size (in tweets)
    """

    cluster_index = build_cluster_index(clusters)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Assign cluster id to each tweet in time order
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["cluster_id"] = df["text"].map(cluster_index)

    times = df["timestamp"]

    # Sliding window counts
    weights = {cid: [] for cid in range(len(clusters))}

    for i in range(len(df)):
        start = max(0, i - window + 1)
        window_slice = df.iloc[start:i + 1]

        for cid in weights:
            weights[cid].append(
                (window_slice["cluster_id"] == cid).sum()
            )

    fig, ax = plt.subplots(figsize=(9, 4))

    largest_cluster = max(weights, key=lambda k: max(weights[k]))

    for cid, values in weights.items():
        if highlight_largest and cid == largest_cluster:
            ax.plot(
                times,
                values,
                color="red",
                linewidth=2.5,
                label="Detected Campaign"
            )
        else:
            ax.plot(
                times,
                values,
                color="grey",
                alpha=0.4
            )

    ax.set_title("Cluster Weight Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster Weight")

    ax.set_xlim(df["timestamp"].min(), df["timestamp"].max())

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax.tick_params(axis='x', rotation=0)

    return fig

import matplotlib.dates as mdates

def plot_cluster_weight_over_time_fixed(
    df,
    clusters,
    window_minutes=60,
    highlight_largest=True
):
    """
    df: DataFrame with ['timestamp', 'text']
    clusters: list of clusters (list of lists)
    window_minutes: size of sliding time window in minutes
    """

    cluster_index = build_cluster_index(clusters)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cluster_id"] = df["text"].map(cluster_index)

    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()

    # Create time bins
    bins = pd.date_range(start=start_time, end=end_time + pd.Timedelta(minutes=window_minutes),
                         freq=f"{window_minutes}min")

    weights = {cid: [] for cid in range(len(clusters))}
    time_labels = []

    for i in range(len(bins) - 1):
        window_slice = df[(df["timestamp"] >= bins[i]) & (df["timestamp"] < bins[i + 1])]
        time_labels.append(bins[i])

        for cid in weights:
            weights[cid].append((window_slice["cluster_id"] == cid).sum())

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))

    largest_cluster = max(weights, key=lambda k: max(weights[k]))

    for cid, values in weights.items():
        if highlight_largest and cid == largest_cluster:
            ax.plot(time_labels, values, color="red", linewidth=2.5, label="Detected Campaign")
        else:
            ax.plot(time_labels, values, color="grey", alpha=0.4)

    ax.set_title("Cluster Weight Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster Weight")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', rotation=45)

    return fig

def plot_cluster_weight_over_time(
    df,
    clusters,
    window=50,
    highlight_largest=True
):
    cluster_index = build_cluster_index(clusters)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["cluster_id"] = df["text"].map(cluster_index)

    times = df["timestamp"]

    weights = {cid: [] for cid in range(len(clusters))}

    for i in range(len(df)):
        start = max(0, i - window + 1)
        window_slice = df.iloc[start:i + 1]

        for cid in weights:
            weights[cid].append(
                (window_slice["cluster_id"] == cid).sum()
            )

    fig, ax = plt.subplots(figsize=(9, 4))

    largest_cluster = max(weights, key=lambda k: max(weights[k]))

    for cid, values in weights.items():
        if highlight_largest and cid == largest_cluster:
            ax.plot(times, values, color="red", linewidth=2.5)
        else:
            ax.plot(times, values, color="grey", alpha=0.4)

    ax.set_title("Cluster Weight Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster Weight (window = 50)")
    ax.set_ylim(0, window)  # 🔑 KEY LINE

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    return fig

