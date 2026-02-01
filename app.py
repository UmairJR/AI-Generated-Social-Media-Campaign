import streamlit as st
import pandas as pd
from clustering_runner import run_clustering
from data import create_initial_stream, replace_campaign
from blueprint import extract_blueprint
from gpt_rewrite import rewrite_with_gpt
from visualization import plot_cluster_weight_over_time, plot_cluster_membership

st.set_page_config(
    page_title="AI Campaign Evasion Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.title("🔧 Experiment Settings")

threshold = st.sidebar.slider(
    "TextClust similarity threshold",
    min_value=0.01,
    max_value=0.2,
    value=0.05,
    step=0.01
)

window = st.sidebar.slider(
    "Temporal window size",
    min_value=3,
    max_value=50,
    value=5
)
st.sidebar.title("🧪 Clustering Algorithm")

algorithm = st.sidebar.selectbox(
    "Select detection algorithm",
    [
        "TextClust (Lexical)",
        "SBERT + HDBSCAN (Semantic)",
        "SBERT + KMeans (Semantic)",
        "TF-IDF + KMeans"
    ]
)

st.markdown("""
## 🧠 Research Objective

Simple Prototype replicating steps Done in Pohl et al. (2022) – Artificial Social Media Campaign Creation
""")
st.header("1️⃣ Original Social Media Stream")
df = create_initial_stream()
st.dataframe(df)

texts = df["text"].tolist()
st.markdown("## 1️⃣ Social Media Stream")

with st.expander("📄 View raw tweet stream"):
    st.dataframe(df, use_container_width=True)

st.markdown("## 2️⃣ Campaign Detection")

clusters, labels = run_clustering(
    texts=texts,
    algorithm=algorithm,
    threshold=threshold
)
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Detected Clusters")

    cluster_labels = [
        f"Cluster {i} ({len(c)} tweets)"
        for i, c in enumerate(clusters)
    ]

    selected_cluster_idx = st.selectbox(
        "Select a cluster to inspect",
        range(len(clusters)),
        format_func=lambda i: cluster_labels[i]
    )

    with st.expander("🧾 Tweets in selected cluster"):
        for t in clusters[selected_cluster_idx]:
            st.write("•", t)

with col2:
    st.markdown("### Cluster Membership Over Time")
    fig_cluster_map = plot_cluster_membership(texts, clusters)
    st.pyplot(fig_cluster_map)



# st.markdown("## 3️⃣ Temporal Campaign Dynamics")
#
# fig_weight = plot_cluster_weight_over_time(
#     texts,
#     clusters,
#     window=window
# )
#
# st.pyplot(fig_weight)

st.markdown("## 3️⃣ Temporal Campaign Dynamics (Real Dataset)")

fig_weight = plot_cluster_weight_over_time(
    df=df,
    clusters=clusters,
    # window=window
)

st.pyplot(fig_weight)

st.caption(
    "The red curve indicates a coordinated campaign burst, "
    "while grey curves represent background activity."
)

st.markdown("## 4️⃣ Extracted Campaign Blueprint")

largest_cluster = max(clusters, key=len)

cluster_times = (
    df[df["text"].isin(largest_cluster)]["timestamp"].tolist()
    if "timestamp" in df.columns else None
)

blueprint = extract_blueprint(
    largest_cluster,
    timestamps=cluster_times
)

st.json(blueprint)
st.caption(
    "The blueprint captures structural properties of the detected campaign, "
    "used later to generate AI-based variants."
)

st.markdown("## 5️⃣ AI-Generated Campaign")

if st.button("🚀 Generate AI Variant"):
    rewritten = rewrite_with_gpt(largest_cluster)
    st.session_state["rewritten"] = rewritten

if "rewritten" in st.session_state:
    with st.expander("🧠 View AI-generated tweets"):
        for t in st.session_state["rewritten"]:
            st.write("•", t)


st.markdown("## 6️⃣ Detection After AI Injection")

if "rewritten" in st.session_state:
    updated_texts = replace_campaign(
        texts,
        largest_cluster,
        st.session_state["rewritten"]
    )

    df_ai = df.copy()
    df_ai["text"] = updated_texts


    cluster_indices = df_ai[df_ai["text"].isin(largest_cluster)].index


    # tc2 = TextClust(threshold=threshold)
    # tc2.fit(updated_texts)
    tc2_clusters, _ = run_clustering(
        texts=df_ai["text"].tolist(),
        algorithm=algorithm,
        threshold=threshold
    )

    from visualization import cluster_cohesion

    st.markdown("### 📊 Campaign Detectability Comparison")

    orig_cohesion = cluster_cohesion(largest_cluster)

    ai_cluster = max(tc2_clusters, key=len)

    ai_cohesion = cluster_cohesion(ai_cluster)

    comparison_df = pd.DataFrame({
        "Stream": ["Original Campaign", "AI-Generated Campaign"],
        "Cohesion Score": [orig_cohesion, ai_cohesion]
    })

    st.markdown("### Inspect Cluster Content")

    cluster_labels = [
        f"Cluster {i} ({len(c)} tweets)"
        for i, c in enumerate(tc2_clusters)
    ]

    selected_cluster = st.selectbox(
        "Select cluster",
        range(len(tc2_clusters)),
        format_func=lambda i: cluster_labels[i]
    )

    with st.expander("🧾 Tweets in selected cluster"):
        for t in tc2_clusters[selected_cluster]:
            st.write("•", t)

    st.dataframe(df_ai)
    st.markdown("## 6️⃣ Temporal Campaign Dynamics (Artificial Dataset)")

    orig_cohesion = cluster_cohesion(largest_cluster)
    # plot_cluster_sizes(tc2_clusters)

    ai_cluster = max(tc2_clusters, key=len)
    ai_cohesion = cluster_cohesion(ai_cluster)

    fig_weight = plot_cluster_weight_over_time(
        df=df_ai,
        clusters=tc2_clusters,
        # window=window
    )

    st.pyplot(fig_weight)

    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.markdown("### On Real Campaign Dataset")

        fig_weight = plot_cluster_weight_over_time(
            df=df,
            clusters=clusters,
            # window=window
        )

        st.pyplot(fig_weight)

    with col2:
        st.markdown("### On Artificial Campaign Dataset")
        fig_weight = plot_cluster_weight_over_time(
            df=df_ai,
            clusters=tc2_clusters,
            # window=window
        )
        st.pyplot(fig_weight)

    st.success(
        "AI rewriting preserves campaign size but significantly "
        "reduces lexical similarity, causing detection failure."
    )


