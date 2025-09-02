# unsupervised_cluster_press_releases_multilingual.py
# Python 3.9+
# pip install: scikit-learn pyarrow gcsfs pandas numpy fasttext-wheel

import os
import re
import json
import tempfile
import datetime
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import gcsfs

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import fasttext  # installed via fasttext-wheel


# =========================
# CONFIG (env-overridable)
# =========================
GCS_SILVER_ROOT   = os.getenv("GCS_SILVER_ROOT", "event-feed-app-data/silver_normalized/table=press_releases")
FT_MODEL_PATH     = os.getenv("FT_MODEL_PATH",     "gs://event-feed-app-data/models/lid.176.ftz")  # or "models/lid.176.ftz"
OUTPUT_GCS_PREFIX = os.getenv("OUTPUT_GCS_PREFIX", "gs://event-feed-app-data/unsup_clusters")
MAPPING_GCS_PATH  = os.getenv("MAPPING_GCS_PATH",  "")  # e.g. "gs://.../cluster_mapping.json" (optional)

RUN_TS = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

# Local artifact names
OUT_LOCAL_SUMMARY_CSV            = "cluster_summaries.csv"
OUT_LOCAL_ASSIGNMENTS_SAMPLE_CSV = "cluster_assignments_sample.csv"
OUT_LOCAL_ALL_ASSIGNMENTS_CSV    = "cluster_assignments_all.csv"
OUT_LOCAL_MAPPING_TEMPLATE_JSON  = "cluster_mapping_template.json"
OUT_LOCAL_ASSIGNMENTS_MAPPED_CSV = "cluster_assignments_with_mapping.csv"  # only if mapping provided

# Clustering knobs
ALLOWED_LANG = {"en", "sv", "da", "fi"}
PARTITIONING = ds.partitioning(
    schema=pa.schema([pa.field("release_date", pa.string())]),
    flavor="hive",
)


# Add a canonical schema (match your silver schema)
CANONICAL_SCHEMA = pa.schema([
    ("press_release_id", pa.string()),
    ("company_name",     pa.string()),
    ("category",         pa.string()),
    ("release_date",     pa.string()),
    ("ingested_at",      pa.string()),
    ("title",            pa.string()),
    ("full_text",        pa.string()),
    ("source",           pa.string()),
    ("source_url",       pa.string()),
    ("parser_version",   pa.int32()),
    ("schema_version",   pa.int32()),
])

READ_COLS = ["release_date", "company_name", "title", "full_text", "category"]
CANDIDATE_K = list(range(3, 101, 5))  # 10,15,...,40
LEAD_CHARS = 1200
USE_NMF_HINTS = True
NMF_MAX_COMPONENTS = 30

from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
# define stopwords


STOP_EN = set(stopwords.words("english"))
STOP_SV = set(stopwords.words("swedish"))
STOP_DA = set(stopwords.words("danish"))
STOP_FI = set(stopwords.words("finnish"))

# --- Build company/org stopwords from your data ---
def make_company_stopwords(df: pd.DataFrame) -> set[str]:
    org_suffixes = {
        "ab","abp","oyj","asa","a/s","plc","as","oy","hf","hf.","aps","publ",
        "group","company","corporation","corp","holding","koncern","aktiebolag","aktieselskab"
    }
    tokens = set()
    if "company_name" in df.columns:
        for name in df["company_name"].fillna("").astype(str):
            for t in TOKEN_RE.findall(name.lower()):
                if len(t) > 2:
                    tokens.add(t)
    # Keep only tokens that look like org identifiers or frequent brand tokens
    # (the simple version: include all tokens from company_name + suffixes)
    return tokens | org_suffixes


MONTHS = {
    # English
    "january","february","march","april","may","june","july","august",
    "september","october","november","december",
    # Swedish
    "januari","februari","mars","april","maj","juni","juli","augusti",
    "september","oktober","november","december",
    # Danish/Norwegian
    "januar","februar","marts","april","maj","juni","juli","august",
    "september","oktober","november","december",
    # Finnish
    "tammikuu","helmikuu","maaliskuu","huhtikuu","toukokuu","kesäkuu",
    "heinäkuu","elokuu","syyskuu","lokakuu","marraskuu","joulukuu",
}



# # 2) Stopword dictionaries (seed lists; extend as you see patterns)
# STOP_EN = {
#     # base English
#     "the","and","of","in","for","to","a","on","as","with","by","from","at","is","are","be","will",
# }
# STOP_SV = {
#     "och","att","av","på","med","som","är","en","ett","till","för","den","det","har","vi","i",
# }
# STOP_DA = {
#     "og","at","af","på","med","som","er","en","et","til","for","den","det","har","vi","i",
#     "as","a/s",
# }
# STOP_FI = {
#     "ja","että","on","se","ne","tämä","nämä","joka","joka","tässä","myös","kun","kanssa","että",
# }

LANG2STOP = {
    "en": STOP_EN,
    "sv": STOP_SV,
    "da": STOP_DA,
    "fi": STOP_FI,
}

# 3) Tokenizer / cleaner
TOKEN_RE = re.compile(r"[A-Za-zÅÄÖåäöÆØæøÐÞðþÁÉÍÓÚÝáéíóúýÄÖäöÜüÅåØøÆæ]+")  # letters in nordic langs

# Put near your other utils
PHRASES = [
    r"rights\s+issue", r"share\s+buy[-\s]*back", r"general\s+meeting",
    r"extra\s+bolagsstämma", r"kommuniké\s+från\s+extra\s+bolagsstämma",
    r"delårsrapport", r"halvårsrapport", r"puolivuosikatsaus",
    r"certified\s+adviser", r"managers[’']?\s+transactions",
    r"admitted\s+to\s+trading", r"bond\s+loan", r"mini\s+futures",
    r"turbo\s+warrants", r"half[-\s]*year\s+report", r"interim\s+report",
    r"notice\s+of\s+annual\s+general\s+meeting", r"annual\s+general\s+meeting",
]

PHRASE_PATTERNS = [re.compile(p, re.I) for p in PHRASES]

def protect_phrases(txt: str) -> str:
    if not isinstance(txt, str) or not txt:
        return ""
    s = txt
    for pat in PHRASE_PATTERNS:
        s = pat.sub(lambda m: m.group(0).replace(" ", "_").replace("-", "_"), s)
    return s


from bs4 import BeautifulSoup
import html as ihtml

URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
WS_RE    = re.compile(r"\s+")

# Titles/bodies that are clearly not press releases
TITLE_BLOCKLIST = [
    r"\b(confirm your subscription|verify your subscription|welcome( as)? (a )?subscriber)\b",
    r"\bsubscription (activation|request)\b",
    r"\b(email alerting service|validate account)\b",
    r"\bstock quote notification\b",
    r"\bverify your subscription\b",
    r"\bsubscription (activation|removal)\b",
]
TITLE_BLOCKLIST_RE = re.compile("(" + "|".join(TITLE_BLOCKLIST) + ")", re.I)

BODY_HINTS_BLOCKLIST = re.compile(
    r"\b(unsubscribe|manage (your )?subscription|click (the )?link|copy and paste|browser|"
    r"privacy policy|cookie policy|email alerting service|verify (your )?subscription|"
    r"confirm( your)? subscription)\b", re.I
)

def strip_html(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    # fast path: remove obvious tags first
    txt = ihtml.unescape(text)
    soup = BeautifulSoup(txt, "lxml")  # or "html.parser" if lxml unavailable
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    txt = soup.get_text(separator=" ")
    txt = URL_RE.sub(" ", txt)
    txt = EMAIL_RE.sub(" ", txt)
    txt = WS_RE.sub(" ", txt).strip()
    return txt


# =========================
# GCS helpers
# =========================
def write_local_to_gcs(fs: gcsfs.GCSFileSystem, local_path: str, gcs_dest_path: str):
    with fs.open(gcs_dest_path, "wb") as fout:
        with open(local_path, "rb") as fin:
            fout.write(fin.read())

def load_json_from_gcs(fs: gcsfs.GCSFileSystem, gcs_path: str) -> Dict:
    with fs.open(gcs_path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

def save_json_local(obj: Dict, local_path: str):
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    sims = (X_reduced[idx] @ centroid)  # dot == cosine sim after L2 norm
    top_idx_local = idx[np.argsort(sims)[::-1][:top_n]]
    return titles.iloc[top_idx_local].tolist()



def clean_and_tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    return TOKEN_RE.findall(text)

def preprocess_docs_word_ngrams(df, text_series, lang_series, company_stop:set):
    """
    For each doc, remove stopwords for its detected language + domain stopwords.
    Returns a list of cleaned strings to feed into word-level TF-IDF.
    """
    out = []
    for txt, lang in zip(text_series, lang_series):
        txt = protect_phrases(txt)
        toks = clean_and_tokenize(txt.lower())
        #stop = set(STOP_DOMAIN)
        stop = LANG2STOP.get(lang, set().union(*LANG2STOP.values()))  # unk -> union
        stop |= company_stop
        stop |= MONTHS
        kept = [t for t in toks if t not in stop and len(t) > 2]
        out.append(" ".join(kept))
    return out

def lead(text, n=1200): return (text or "")[:n]

# =========================
# MAIN
# =========================
def main():
    fs = gcsfs.GCSFileSystem()

    # --- Load dataset from GCS ---
    dataset = ds.dataset(
        GCS_SILVER_ROOT,
        filesystem=fs,
        format="parquet",
        partitioning=PARTITIONING,
        schema=CANONICAL_SCHEMA,
    )

    available_cols = [f.name for f in dataset.schema]
    cols = [c for c in READ_COLS if c in available_cols]
    table = dataset.to_table(columns=cols)

    df = table.to_pandas(types_mapper=pd.ArrowDtype)

    # Basic hygiene
    for c in ["title", "full_text"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    # Clean HTML/noise
    df["title_clean"] = df["title"].map(strip_html)
    df["full_text_clean"] = df["full_text"].map(strip_html)

    # Filter out non-PR system emails
    mask_system = (
        df["title_clean"].str.contains(TITLE_BLOCKLIST_RE, na=False)
        | df["full_text_clean"].str.contains(BODY_HINTS_BLOCKLIST, na=False)
    )
    if mask_system.any():
        df[mask_system].to_csv("filtered_out_system_emails.csv", index=False)
        df = df[~mask_system].copy()
        print(f"Filtered out {mask_system.sum()} system/notification items.")

  

    X_raw = (df["title"].fillna("").str.strip() + " " + df["full_text"].fillna("").map(lambda t: lead(t, 1200)))

    
    # --- Language detection ---
    try:
        df["lang"] = batch_detect_lang_ft(X_raw.tolist(), min_conf=0.70)
    except Exception as e:
        raise RuntimeError(
            f"Failed to run FastText language detection. "
            f"Check FT_MODEL_PATH={FT_MODEL_PATH}. Underlying error: {e}"
        )

    print("Language distribution:\n", df["lang"].value_counts(dropna=False), "\n")


    # after you have df loaded/cleaned:
    COMPANY_STOP = make_company_stopwords(df)

    # ---- Preprocess per language
    X_clean = preprocess_docs_word_ngrams(df, X_raw, df["lang"], COMPANY_STOP)

    # --- Vectorization (character n-grams; language-agnostic) + SVD ---
    # 4) Word-level TF-IDF (with bigrams)
    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.70,
        max_features=200_000,
        # We already removed stopwords per doc, so no global stop_words here.
        strip_accents="unicode",
    )
    
    # 5) Dimensionality reduction + normalize (same as before)
    svd = TruncatedSVD(n_components=200, random_state=42)
    l2norm = Normalizer(copy=False)
    lsa_word = make_pipeline(tfidf_word, svd, l2norm)

    X_reduced = lsa_word.fit_transform(X_clean)   # -> (n_docs x 200) dense matrix

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

    # --- Per-cluster silhouette diagnostics ---
    # Per-sample silhouettes (Euclidean is fine since you L2-normalize)
    sil_samples = silhouette_samples(X_reduced, labels, metric="euclidean")
    df["_silhouette"] = sil_samples

    # Aggregate per cluster
    diag_rows = []
    for c in range(best_k):
        mask = (labels == c)
        ss = sil_samples[mask]
        if ss.size == 0:
            continue
        diag_rows.append({
            "cluster": c,
            "sil_mean": float(np.mean(ss)),
            "sil_median": float(np.median(ss)),
            "sil_p10": float(np.percentile(ss, 10)),
            "sil_p25": float(np.percentile(ss, 25)),
            "sil_p75": float(np.percentile(ss, 75)),
            "sil_p90": float(np.percentile(ss, 90)),
            "size": int(mask.sum()),
        })

    diag = pd.DataFrame(diag_rows).sort_values("sil_mean", ascending=True)
    diag.to_csv("cluster_diagnostics.csv", index=False)
    print("Wrote cluster_diagnostics.csv (per-cluster silhouette stats)")




    df["cluster_k"] = best_k
    df["cluster"] = labels  # <-- attach unsupervised cluster id

    # --- Save ALL assignments early (local) ---
    df.to_csv(OUT_LOCAL_ALL_ASSIGNMENTS_CSV, index=False)
    print(f"Wrote {OUT_LOCAL_ALL_ASSIGNMENTS_CSV}")

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
        TF_word = tfidf_word.fit_transform(X_clean)
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
    summary = summary.merge(
        diag[["cluster","sil_mean","sil_p25","sil_p75"]],  # pick what you want to show
        on="cluster", how="left"
    )
    summary.to_csv(OUT_LOCAL_SUMMARY_CSV, index=False)
    print(f"Wrote {OUT_LOCAL_SUMMARY_CSV} (with per-cluster silhouette)")

    # --- Dump a small sample of assignments for manual review ---
    show_cols = [c for c in ["release_date", "company_name", "title", "lang", "cluster", "cluster_k", "category"] if c in df.columns]
    
    assign_sample = (df.sample(min(1000, len(df)), random_state=42)[show_cols]
        .sort_values(["cluster"])
        .reset_index(drop=True)
    )
    assign_sample.to_csv(OUT_LOCAL_ASSIGNMENTS_SAMPLE_CSV, index=False)
    print(f"Wrote {OUT_LOCAL_ASSIGNMENTS_SAMPLE_CSV}")

    # --- Emit a mapping template JSON (cluster -> suggested label) ---
    # You (or a reviewer) will edit 'suggested_label' values and save to GCS (or pass via MAPPING_GCS_PATH next run)
    mapping_template = {
        "run_ts": RUN_TS,
        "cluster_k": best_k,
        "clusters": [
            {
                "cluster": int(row["cluster"]),
                "size": int(row["size"]),
                "topic_hint": row["topic_hint"],
                "suggested_label": ""  # fill manually: e.g., "earnings_report", "share_buyback", "agm/egm", "managers_transactions"
            }
            for row in summary.to_dict(orient="records")
        ]
    }
    save_json_local(mapping_template, OUT_LOCAL_MAPPING_TEMPLATE_JSON)
    print(f"Wrote {OUT_LOCAL_MAPPING_TEMPLATE_JSON}")

    # --- If an existing mapping is provided, apply it and emit a pre-labeled CSV ---
    if MAPPING_GCS_PATH:
        try:
            user_map = load_json_from_gcs(fs, MAPPING_GCS_PATH)
            # Expect format: {"clusters": [{"cluster": 24, "label": "earnings_report"}, ...]}
            lut = {}
            for item in user_map.get("clusters", []):
                cid = int(item["cluster"])
                lbl = str(item.get("label", "") or item.get("suggested_label", "")).strip()
                if lbl:
                    lut[cid] = lbl

            mapped = df.copy()
            mapped["unsup_cluster"] = mapped["cluster"]
            mapped["unsup_label"] = mapped["unsup_cluster"].map(lut).fillna("")
            mapped["manual_label"] = ""       # to be filled by a reviewer/UI
            mapped["supervised_label"] = ""   # to be filled later by your trained classifier

            mapped_cols = [c for c in ["release_date","company_name","title","lang","unsup_cluster","unsup_label","manual_label","supervised_label","category","full_text"] if c in mapped.columns]
            mapped[mapped_cols].to_csv(OUT_LOCAL_ASSIGNMENTS_MAPPED_CSV, index=False)
            print(f"Wrote {OUT_LOCAL_ASSIGNMENTS_MAPPED_CSV} (applied mapping from {MAPPING_GCS_PATH})")
        except Exception as e:
            print(f"WARNING: Failed to load/apply mapping from {MAPPING_GCS_PATH}: {e}")

    # --- Upload artifacts to GCS (timestamped folder) ---
    dst_dir = f"{OUTPUT_GCS_PREFIX}/{RUN_TS}"
    write_local_to_gcs(fs, OUT_LOCAL_SUMMARY_CSV,            f"{dst_dir}/{OUT_LOCAL_SUMMARY_CSV}")
    write_local_to_gcs(fs, OUT_LOCAL_ASSIGNMENTS_SAMPLE_CSV, f"{dst_dir}/{OUT_LOCAL_ASSIGNMENTS_SAMPLE_CSV}")
    write_local_to_gcs(fs, OUT_LOCAL_ALL_ASSIGNMENTS_CSV,    f"{dst_dir}/{OUT_LOCAL_ALL_ASSIGNMENTS_CSV}")
    #write_local_to_gcs(fs, OUT_LOCAL_MAPPING_TEMPLATE_JSON,  f"{dst_dir}/{OUT_LOCAL_MAPPING_TEMPLATE_JSON}")
    #if os.path.exists(OUT_LOCAL_ASSIGNMENTS_MAPPED_CSV):
    #    write_local_to_gcs(fs, OUT_LOCAL_ASSIGNMENTS_MAPPED_CSV, f"{dst_dir}/{OUT_LOCAL_ASSIGNMENTS_MAPPED_CSV}")

    print(f"\nAll artifacts saved locally and to {dst_dir}")

    # print("\nNext steps:")
    # print(f"- Edit {dst_dir}/cluster_mapping_template.json (or create your own mapping JSON).")
    # print("- Re-run with MAPPING_GCS_PATH pointing to that mapping to produce a pre-labeled CSV for review.")
    # print("- Use that reviewed CSV to train a supervised classifier (stage 2).")


if __name__ == "__main__":
    main()