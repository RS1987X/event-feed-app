# unsupervised_cluster_press_releases_multilingual.py
# Python 3.9+
# pip install scikit-learn pyarrow gcsfs pandas numpy fasttext-wheel

import os
import re
import tempfile
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import gcsfs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import fasttext  # installed via fasttext-wheel


# =========================
# CONFIG
# =========================
GCS_SILVER_ROOT = "event-feed-app-data/silver_normalized/table=press_releases"

# Where to save local artifacts for review
OUT_LOCAL_SUMMARY_CSV = "cluster_summaries.csv"
OUT_LOCAL_ASSIGNMENTS_CSV = "cluster_assignments_sample.csv"

# FastText language-ID model (either local path or GCS path)
# Download lid.176.ftz once and store e.g. at gs://.../models/lid.176.ftz
FT_MODEL_PATH = "gs://event-feed-app-data/models/lid.176.ftz"  # or "/models/lid.176.ftz"

# Allowed language codes for your universe
ALLOWED_LANG = {"en", "sv", "da", "fi"}

# Partitioning (Hive: .../release_date=YYYY-MM-DD/...)
PARTITIONING = ds.partitioning(
    schema=pa.schema([pa.field("release_date", pa.large_string())]),
    flavor="hive",
)

# Columns to read (adjust to your schema)
READ_COLS = ["release_date", "company_name", "title", "full_text", "category"]

# Candidate k for clustering, pick best by silhouette
CANDIDATE_K = list(range(10, 41, 5))  # 10,15,...,40

# Lead length for text (avoid boilerplate)
LEAD_CHARS = 1200

# Whether to also compute topic hints via word-level NMF (optional but helpful)
USE_NMF_HINTS = True
NMF_MAX_COMPONENTS = 30  # caps number of topics used for hints


# =========================
# FASTTEXT LID HELPERS
# =========================
_FT_MODEL = None

def _load_ft_model(path_or_gcs: str):
    """Load FastText LID model from local path or GCS into a cached global."""
    global _FT_MODEL
    if _FT_MODEL is not None:
        return _FT_MODEL

    if path_or_gcs.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(path_or_gcs, "rb") as fsrc, tempfile.NamedTemporaryFile(delete=False, suffix=".ftz") as fdst:
            fdst.write(fsrc.read())
            tmp_path = fdst.name
        _FT_MODEL = fasttext.load_model(tmp_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    else:
        _FT_MODEL = fasttext.load_model(path_or_gcs)

    return _FT_MODEL

def batch_detect_lang_ft(texts: Iterable[str], min_conf: float = 0.70) -> List[str]:
    """
    Detect language for a list of texts using FastText LID.
    Returns 'en'/'sv'/'da'/'fi' or 'unk' if below threshold or outside allowed.
    """
    model = _load_ft_model(FT_MODEL_PATH)
    cleaned = [" ".join((t or "").split())[:4000] for t in texts]
    labels_list, probs_list = model.predict(cleaned, k=1)
    out = []
    for lbl, pr in zip(labels_list, probs_list):
        code = lbl[0].replace("__label__", "")
        conf = float(pr[0])
        out.append(code if code in ALLOWED_LANG and conf >= min_conf else "unk")
    return out


# =========================
# UTILITIES
# =========================
def build_lead_text(title: str, full_text: str, lead_chars: int = LEAD_CHARS) -> str:
    t = (title or "").strip()
    f = (full_text or "")[:lead_chars]
    return re.sub(r"\s+", " ", f"{t} {f}").strip()

def representative_titles_for_cluster(X_reduced: np.ndarray,
                                     labels: np.ndarray,
                                     kmeans: MiniBatchKMeans,
                                     cluster_id: int,
                                     titles: pd.Series,
                                     top_n: int = 8) -> List[str]:
    idx = np.where(labels == cluster_id)[0]
    if idx.size == 0:
        return []
    centroid = kmeans.cluster_centers_[cluster_id]
    # X_reduced rows are L2-normalized (after Normalizer), so dot = cosine sim
    sims = (X_reduced[idx] @ centroid)
    top_idx_local = idx[np.argsort(sims)[::-1][:top_n]]
    return titles.iloc[top_idx_local].tolist()


# =========================
# MAIN
# =========================
def main():
    # --- Load dataset from GCS ---
    fs = gcsfs.GCSFileSystem()
    dataset = ds.dataset(
        GCS_SILVER_ROOT,
        filesystem=fs,
        format="parquet",
        partitioning=PARTITIONING,
    )

    available_cols = [f.name for f in dataset.schema]
    cols = [c for c in READ_COLS if c in available_cols]
    table = dataset.to_table(columns=cols)
    df = table.to_pandas()

    # Basic hygiene
    for c in ["title", "full_text"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    # Build the text used for lang detection and clustering (title + lead)
    X_text = pd.Series(
        [build_lead_text(ti, tx) for ti, tx in zip(df["title"], df["full_text"])],
        index=df.index
    )

    # --- Language detection ---
    try:
        df["lang"] = batch_detect_lang_ft(X_text.tolist(), min_conf=0.70)
    except Exception as e:
        raise RuntimeError(
            f"Failed to run FastText language detection. "
            f"Check FT_MODEL_PATH={FT_MODEL_PATH}. Underlying error: {e}"
        )

    print("Language distribution:\n", df["lang"].value_counts(dropna=False), "\n")

    # --- Vectorization (character n-grams; language-agnostic) + SVD ---
    tfidf_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.80,
        max_features=200_000
    )
    svd = TruncatedSVD(n_components=200, random_state=42)
    l2norm = Normalizer(copy=False)
    lsa = make_pipeline(tfidf_char, svd, l2norm)

    X_reduced = lsa.fit_transform(X_text)

    # --- Choose k by silhouette ---
    sil_scores: List[Tuple[int, float]] = []
    models = {}

    for k in CANDIDATE_K:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init="auto")
        labels = km.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, labels, metric="euclidean")
        sil_scores.append((k, score))
        models[k] = (km, labels)

    best_k, best_score = max(sil_scores, key=lambda x: x[1])
    kmeans, labels = models[best_k]

    print("Silhouette scores:", sil_scores)
    print(f"Selected k={best_k} (silhouette={best_score:.3f})\n")

    df["cluster_k"] = best_k
    df["cluster"] = labels

    # --- Topic hints (optional, word-level NMF for interpretability) ---
    topic_hints_by_cluster = {}
    if USE_NMF_HINTS:
        tfidf_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.70,
            max_features=80_000,
            strip_accents="unicode"
        )
        TF_word = tfidf_word.fit_transform(X_text)
        n_topics = min(best_k, NMF_MAX_COMPONENTS)
        nmf = NMF(n_components=n_topics, random_state=42, init="nndsvd", max_iter=300)
        W = nmf.fit_transform(TF_word)   # (docs x topics)
        H = nmf.components_              # (topics x terms)
        terms = np.array(tfidf_word.get_feature_names_out())

        def top_words(component, n=12):
            idx = np.argsort(component)[::-1][:n]
            return ", ".join(terms[idx])

        nmf_topics = [top_words(H[i]) for i in range(H.shape[0])]

        # Map each cluster to its majority NMF topic (by average W over its docs)
        for c in range(best_k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                topic_hints_by_cluster[c] = ""
                continue
            avg_topic_weights = W[idx].mean(axis=0)
            maj_topic = int(np.argmax(avg_topic_weights))
            topic_hints_by_cluster[c] = nmf_topics[maj_topic]

    # --- Build cluster summaries ---
    rows = []
    for c in range(best_k):
        idx = np.where(labels == c)[0]
        size = int(len(idx))
        rep_titles = representative_titles_for_cluster(
            X_reduced=X_reduced,
            labels=labels,
            kmeans=kmeans,
            cluster_id=c,
            titles=df["title"],
            top_n=8
        )
        topic_hint = topic_hints_by_cluster.get(c, "") if USE_NMF_HINTS else ""
        rows.append({
            "cluster": c,
            "size": size,
            "topic_hint": topic_hint,
            "sample_titles_representative": " | ".join(rep_titles),
        })

    summary = pd.DataFrame(rows).sort_values("size", ascending=False)
    summary.to_csv(OUT_LOCAL_SUMMARY_CSV, index=False)
    print(f"Wrote {OUT_LOCAL_SUMMARY_CSV}")

    # --- Dump a small sample of assignments for manual review ---
    show_cols = [c for c in ["release_date", "company_name", "title", "lang", "cluster", "cluster_k", "category"] if c in df.columns]
    assign_sample = df.sample(min(2000, len(df)), random_state=42)[show_cols]
    assign_sample.to_csv(OUT_LOCAL_ASSIGNMENTS_CSV, index=False)
    print(f"Wrote {OUT_LOCAL_ASSIGNMENTS_CSV}")

    print("\nNext steps:")
    print("- Open cluster_summaries.csv and map cluster IDs to your taxonomy (Earnings, CEO_Change, etc.).")
    print("- Apply that mapping to unknown rows to create initial labels.")
    print("- Train a simple supervised model (TF-IDF + Logistic Regression) on those labels for higher accuracy.")


if __name__ == "__main__":
    main()
