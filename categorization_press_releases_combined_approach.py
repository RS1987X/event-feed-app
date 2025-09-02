# Python 3.9+
# pip install: sentence-transformers pyarrow gcsfs pandas numpy bs4 lxml

import os, json, html as ihtml
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import gcsfs
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from housekeeping_gate import apply_housekeeping_filter
from dedup_utils import dedup_df
from near_dedup import near_dedup_embeddings
from company_name_remover import clean_df_from_company_names

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer


FUSE_ALPHA = float(os.getenv("FUSE_ALPHA", "0.6"))  # weight for embeddings in fusion
TEMP_EMB   = float(os.getenv("TEMP_EMB",   "1.0"))  # softmax temp for embeddings sims
TEMP_LSA   = float(os.getenv("TEMP_LSA",   "1.3"))  # softmax temp for LSA sims

# =========================
# METRICS CONFIG
# =========================
RUN_METRICS_CSV   = os.getenv("RUN_METRICS_CSV", "runs_metrics.csv")
TAXONOMY_VERSION  = os.getenv("TAXONOMY_VERSION", "v3")
KNN_K             = int(os.getenv("KNN_K", "10"))  # for neighbor-consistency on embeddings



# =========================
# CONFIG
# =========================
GCS_SILVER_ROOT   = os.getenv("GCS_SILVER_ROOT", "event-feed-app-data/silver_normalized/table=press_releases")
OUTPUT_LOCAL_CSV  = os.getenv("OUTPUT_LOCAL_CSV", "assignments.csv")
OUTPUT_GCS_URI    = os.getenv("OUTPUT_GCS_URI", "")  # e.g., gs://event-feed-app-data/unsup_clusters/assignments_<ts>.csv

EMB_MODEL_NAME    = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-base")
E5_PREFIX         = bool(int(os.getenv("E5_PREFIX", "1")))  # set 0 if using non-E5 like 'paraphrase-multilingual-MiniLM-L12-v2'
EMB_BATCH_SIZE    = int(os.getenv("EMB_BATCH_SIZE", "32"))
MAX_SEQ_TOKENS    = int(os.getenv("MAX_SEQ_TOKENS", "384"))  # truncate to keep speed/memory predictable

CAT_ASSIGN_MIN_SIM = float(os.getenv("CAT_ASSIGN_MIN_SIM", "0.62"))
CAT_LOW_MARGIN     = float(os.getenv("CAT_LOW_MARGIN", "0.02"))
SNIPPET_CHARS      = int(os.getenv("SNIPPET_CHARS", "600"))

RUN_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#Import categories and descriptions
from event_taxonomies import l1_v3_en
L1_CATEGORIES = l1_v3_en.L1_CATEGORIES

CID2NAME = {cid: name for cid, name, _ in L1_CATEGORIES}

# pip install: langid
import langid

# Optional: restrict to common langs you expect (improves accuracy/speed).
# Comma-separated ISO 639-1 codes, e.g. "en,sv,fi,da,no,de,fr"
LANG_WHITELIST = os.getenv("LANG_WHITELIST", "en,sv,fi,da,no,de,fr,is")
if LANG_WHITELIST.strip():
    langid.set_languages([s.strip() for s in LANG_WHITELIST.split(",") if s.strip()])


# def softmax_rows(S: np.ndarray, T: float = 1.0) -> np.ndarray:
#     S = S / max(T, 1e-6)
#     S = S - S.max(axis=1, keepdims=True)
#     np.exp(S, out=(S := np.exp(S)))                  # reuse buffer
#     S_sum = S.sum(axis=1, keepdims=True) + 1e-12
#     return S / S_sum

from scipy import sparse

def softmax_rows(S, T: float = 1.0) -> np.ndarray:
    # Make sure we work with a dense ndarray
    if sparse.issparse(S):
        S = S.toarray()
    else:
        S = np.asarray(S)
    S = S.astype(np.float64, copy=False)

    # Temperature + log-sum-exp for stability
    S = S / max(T, 1e-6)
    S -= np.max(S, axis=1, keepdims=True)
    S = np.exp(S)
    denom = np.sum(S, axis=1, keepdims=True) + 1e-12
    return S / denom

def entropy_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # P is already row-stochastic; safe entropy in nats
    P_clip = np.clip(P, eps, 1.0)
    return -np.sum(P_clip * np.log(P_clip), axis=1)

def topk_sum(P: np.ndarray, k: int = 3) -> np.ndarray:
    # Sum of top-k probs per row
    if P.shape[1] <= k:
        return np.sum(P, axis=1)
    part = np.partition(P, -k, axis=1)
    return np.sum(part[:, -k:], axis=1)

def append_row(path: str, row: dict):
    # Append one row to a CSV, write header if file doesn't exist
    import csv, os
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def detect_language_batch(titles: pd.Series, bodies: pd.Series, max_len: int = 1200):
    """
    Returns two lists: language codes and confidences (0..1).
    Uses title + first max_len chars of body. Falls back to 'und' if empty.
    """
    langs, confs = [], []
    for t, b in zip(titles.fillna(""), bodies.fillna("")):
        txt = (str(t).strip() + " " + str(b).strip()[:max_len]).strip()
        if not txt:
            langs.append("und"); confs.append(0.0); continue
        lang, score = langid.classify(txt)
        # langid score is a log-prob-ish margin; squash to 0..1 via a simple mapping
        # Empirically, scores ~5–20 are confident. Map with 1 - exp(-score/10).
        try:
            conf = 1.0 - float(np.exp(-float(score) / 10.0))
        except Exception:
            conf = 0.0
        langs.append(lang); confs.append(round(conf, 4))
    return langs, confs

# =========================
# TEXT CLEANING (minimal)
# =========================

def build_doc_for_embedding(title: str, body: str, e5_prefix: bool = True) -> str:
    title = (title or "").strip()
    body  = (body or "").strip()
    txt   = f"[TITLE] {title}\n\n{body}" if title else body
    return f"passage: {txt}" if e5_prefix else txt

def build_cat_query(name: str, desc: str, e5_prefix: bool = True) -> str:
    q = f"category: {name}. description: {desc}"
    return "query: " + q if e5_prefix else q

# =========================
# EMBEDDING HELPERS
# =========================
def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

# =========================
# MAIN
# =========================
def main():
    # --- Load dataset from GCS ---
    fs = gcsfs.GCSFileSystem()
    dataset = ds.dataset(GCS_SILVER_ROOT, filesystem=fs, format="parquet")
    df = dataset.to_table().to_pandas()

    # --- Housekeeping filter (modular) ---
    df, stats = apply_housekeeping_filter(df)
    
    print(f"Housekeeping filter skipped {stats['skipped']} / {stats['total']} ({(stats['skipped']/max(stats['total'],1))*100:.1f}%).")

    df = clean_df_from_company_names(df)

    # 2) Deduplicate (id + content)
    df, dd_stats = dedup_df(
        df,
        id_col="press_release_id",     # set to None if you don’t have it
        title_col="title_clean",
        body_col="full_text_clean",
        ts_col="release_date",         # optional; keeps newest when duplicates exist
    )
    print(dd_stats)

    #Build a diverse subset to speed up
    # --- 1) Diverse subset (no embeddings needed yet) ---
    texts = df["full_text_clean"].tolist()

    hv = HashingVectorizer(analyzer="char", ngram_range=(3,5), n_features=2**20,
                        alternate_sign=False, strip_accents="unicode")
    X_hash = hv.transform(texts)                       # sparse, fast
    svd_sketch = TruncatedSVD(n_components=96, random_state=42)
    X_sketch = svd_sketch.fit_transform(X_hash)       # dense (n x 96)
    X_sketch = Normalizer(copy=False).fit_transform(X_sketch)

    K = 100
    mbk = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096)
    labels = mbk.fit_predict(X_sketch)

    # pick 6 docs per cluster (or fewer if small)
    subset_idx = (pd.Series(range(len(texts)))
                    .groupby(labels, group_keys=False)
                    .apply(lambda s: s.sample(min(6, len(s)), random_state=42))
                    .tolist())
    df_sub = df.iloc[subset_idx].reset_index(drop=True)

    # LSA text (no E5 prefixes; just natural text)
    cat_texts_lsa = [f"{name}. {desc}" for _, name, desc in L1_CATEGORIES]


    USE_SUBSET = bool(int(os.getenv("USE_SUBSET", "1")))  # 1=use df_sub, 0=use full df
    doc_df = df_sub if USE_SUBSET else df

    
    # --- Embedding Model & Vectorization (Docs + Categories) ---
    model = SentenceTransformer(EMB_MODEL_NAME)
    # Control truncation length cleanly
    limit = getattr(model, "max_seq_length", 512) or 512
    model.max_seq_length = max(32, min(MAX_SEQ_TOKENS, int(limit)))


    # Build documents for embedding
    docs = [
        build_doc_for_embedding(t, b, E5_PREFIX)
        for t, b in zip(doc_df["title_clean"].tolist(), doc_df["full_text_clean"].tolist())
    ]

    # Encode documents
    X = encode_texts(model, docs, EMB_BATCH_SIZE)  # (N, d)

    # --- Categories (for both spaces) ---
    cat_names = [name for _, name, _ in L1_CATEGORIES]
    cat_descs = [desc for _, _, desc in L1_CATEGORIES]

    # Embedding queries (E5 requires query: prefix)
    cat_texts_emb = [build_cat_query(n, d, E5_PREFIX) for n, d in zip(cat_names, cat_descs)]
    Q = encode_texts(model, cat_texts_emb, EMB_BATCH_SIZE)   # (C, d)

    # --- Language detection (title + truncated body) ---
    langs, lang_confs = detect_language_batch(doc_df["title_clean"], doc_df["full_text_clean"])
    doc_df = doc_df.copy()
    doc_df["language"] = langs
    doc_df["lang_conf"] = lang_confs

    # mark-only dedup using language-confidence representative
    doc_df, X, near_dedupe_stats, rep_keep_idx = near_dedup_embeddings(
        doc_df, X, threshold=0.97, drop=False, RUN_TS=RUN_TS,
        rep_strategy="lang_conf", lang_conf_col="lang_conf", NEAR_DUP_AUDIT_CSV= "near_dedup_audit.csv"
    )
    print(near_dedupe_stats)

    # --- Vectorization (word n-grams; unigrams + bigrams + trigrams)
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
    #svd = TruncatedSVD(n_components=200, random_state=42)
    l2norm = Normalizer(copy=False)
    tfidf_pipe = make_pipeline(tfidf_word, l2norm)

    # 3) TF-IDF: fit on representatives ONLY, transform ALL docs
    tfidf_input = (doc_df["title_clean"].astype(str) + " " + doc_df["full_text_clean"].astype(str))
    # tfidf_pipe = make_pipeline(
    #     TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=3, max_df=0.70, max_features=200_000, strip_accents="unicode"),
    #     Normalizer(copy=False)
    # )
    V_docs_fit = tfidf_pipe.fit_transform(tfidf_input.iloc[rep_keep_idx])   # fit on reps
    V_docs = tfidf_pipe.transform(tfidf_input)                               # transform all docs
    V_cats = tfidf_pipe.transform(cat_texts_lsa)

    # tfidf_input = (doc_df["title_clean"].astype(str) + " " + doc_df["full_text_clean"].astype(str))
    # #tfidf_input = doc_df["title_clean"].astype(str)
    
    # V_docs = tfidf_pipe.fit_transform(tfidf_input)

    # #V_docs = lsa.fit_transform(doc_df["full_text_clean"])  # (N_docs, d_lsa), L2-normalized
    # V_cats = tfidf_pipe.transform(cat_texts_lsa)                  # (C, d_lsa)

    # --- Similarities in each space ---
    sims_emb = X @ Q.T              # cosine (X, Q are L2-normalized)
    sims_tfidf = V_docs @ V_cats.T    # cosine (LSA pipeline L2-normalizes)

    # --- Per-doc calibration and fusion ---
    P_emb = softmax_rows(sims_emb, T=TEMP_EMB)             # (N_docs, C)
    P_tfidf = softmax_rows(sims_tfidf, T=TEMP_LSA)             # (N_docs, C)
    P_fused = FUSE_ALPHA * P_emb + (1.0 - FUSE_ALPHA) * P_tfidf

    # --- Argmax + margin on fused probs ---
    best_idx = np.argmax(P_fused, axis=1)
    best_prob = P_fused[np.arange(len(P_fused)), best_idx]

    if P_fused.shape[1] >= 2:
        # get 2nd best via partial sort
        part = np.partition(P_fused, -2, axis=1)
        second_best = part[:, -2]
        margin = best_prob - second_best
    else:
        second_best = np.zeros_like(best_prob)
        margin = np.zeros_like(best_prob)

    assigned_cids  = [L1_CATEGORIES[i][0] for i in best_idx]
    assigned_names = [CID2NAME[cid] for cid in assigned_cids]

    def decide(prob, mar):
        # Thresholds now operate on fused probabilities (0..1).
        # Start with your old values; tune as needed.
        if prob >= CAT_ASSIGN_MIN_SIM and mar >= CAT_LOW_MARGIN:
            return "auto_assign"
        elif prob >= CAT_ASSIGN_MIN_SIM:
            return "needs_review_margin"
        else:
            return "needs_review_low_sim"

    decisions = [decide(float(p), float(m)) for p, m in zip(best_prob, margin)]

    # --- Per-doc diagnostics ---
    entropy_emb   = entropy_rows(P_emb)
    entropy_lsa   = entropy_rows(P_tfidf)
    entropy_fused = entropy_rows(P_fused)

    # top1 probs already computed as best_prob (fused); do for each view:
    p1_emb   = P_emb[np.arange(len(P_emb)), np.argmax(P_emb, axis=1)]
    p1_tfidf   = P_tfidf[np.arange(len(P_tfidf)), np.argmax(P_tfidf, axis=1)]

    # top-k concentration
    top3_fused = topk_sum(P_fused, k=3)

    # agreement between views (argmax)
    argmax_emb = np.argmax(P_emb, axis=1)
    argmax_tfidf = np.argmax(P_tfidf, axis=1)
    agreement_emb_tfidf = (argmax_emb == argmax_tfidf).astype(np.bool_)
    #agreement_emb_lsa = (argmax_emb == argmax_lsa)

    # capture top-3 categories (for reviewer UX)
    top3_idx = np.argsort(-P_fused, axis=1)[:, :3]
    top3_cids = [[L1_CATEGORIES[j][0] for j in row] for row in top3_idx]
    top3_probs = [P_fused[i, top3_idx[i]].round(4).tolist() for i in range(len(P_fused))]

    # --- Neighbor consistency (embedding space; cosine) ---
    # X is L2-normalized embeddings (docs). Cosine distance is fine.
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(KNN_K + 1, len(X)), metric="cosine", algorithm="auto")
    nbrs.fit(X)
    dist, idx = nbrs.kneighbors(X, return_distance=True)

    # Drop self neighbor at position 0
    idx_no_self = idx[:, 1:]
    # Compare fused argmax label with neighbors' fused argmax labels
    argmax_fused = best_idx  # already computed above
    neighbor_labels = argmax_fused[idx_no_self]
    neighbor_consistency = (neighbor_labels == argmax_fused.reshape(-1,1)).mean(axis=1)

    # CONSENSUS BY DUP GROUP:
    # We marked dup groups earlier (drop=False), so groups may have >1 members.
    # This aggregates probs per group and writes consensus back to each member.

    # Ensure dup_group_id exists; if not, make singleton groups
    if "dup_group_id" not in doc_df.columns:
        doc_df = doc_df.copy()
        doc_df["dup_group_id"] = (doc_df.get("press_release_id", pd.Series(range(len(doc_df)))).astype(str))

    N_docs, N_c = P_fused.shape

    # Build weights: language confidence + body length (tune if you like)
    body_len = doc_df["full_text_clean"].astype(str).str.len().fillna(0).to_numpy()
    len_norm = body_len / (body_len.mean() + 1e-9)
    lang_conf = pd.to_numeric(doc_df.get("lang_conf", pd.Series(0.5, index=doc_df.index)), errors="coerce").fillna(0.5).to_numpy()
    w = np.clip(0.7 * lang_conf + 0.3 * len_norm, 0.1, None)

    groups = doc_df["dup_group_id"].astype(str).to_numpy()

    # Weighted-mean consensus per group
    idx_df = pd.DataFrame({"g": groups, "i": np.arange(N_docs), "w": w})
    grouped = idx_df.groupby("g", sort=False)
    P_fused_g = np.empty_like(P_fused)

    for g, block in grouped:
        ii = block["i"].to_numpy()
        ww = block["w"].to_numpy()
        ww = ww / (ww.sum() + 1e-12)
        Pg = (P_fused[ii] * ww[:, None]).sum(axis=0)
        s = Pg.sum()
        if s > 0:
            Pg = Pg / s
        P_fused_g[ii, :] = Pg

    # Derive group-level decisions/diagnostics
    best_idx_g  = np.argmax(P_fused_g, axis=1)
    best_prob_g = P_fused_g[np.arange(N_docs), best_idx_g]
    second_g    = np.partition(P_fused_g, -2, axis=1)[:, -2] if N_c >= 2 else np.zeros_like(best_prob_g)
    margin_g    = best_prob_g - second_g

    # def decide(prob, mar):
    #     if prob >= CAT_ASSIGN_MIN_SIM and mar >= CAT_LOW_MARGIN:
    #         return "auto_assign"
    #     elif prob >= CAT_ASSIGN_MIN_SIM:
    #         return "needs_review_margin"
    #     else:
    #         return "needs_review_low_sim"

    decisions_g = [decide(float(p), float(m)) for p, m in zip(best_prob_g, margin_g)]
    entropy_fused_g = entropy_rows(P_fused_g)

    # Group size + representative marker (useful for dedup-aware counts later)
    doc_df = doc_df.copy()
    doc_df["__body_len_tmp"] = body_len  # <-- add temp column first

    # now create the groupby on the same DataFrame that has the column
    grp = doc_df.groupby("dup_group_id", dropna=False)

    # dup group sizes
    doc_df["dup_group_size"] = grp["dup_group_id"].transform("size").astype(int)

    def _pick_rep(g):
        cols = ["__body_len_tmp"]; asc = [False]
        if "release_date" in g.columns:
            cols.append("release_date"); asc.append(True)
        return g.sort_values(cols, ascending=asc, kind="mergesort").index[0]

    rep_idx = grp.apply(_pick_rep)
    rep_idx = rep_idx.values if hasattr(rep_idx, "values") else rep_idx
    is_rep = pd.Series(False, index=doc_df.index)
    is_rep.loc[rep_idx] = True

    # clean up temp col
    doc_df.drop(columns=["__body_len_tmp"], inplace=True)


    # Sample weight so each dup group totals to 1 when aggregating
    sample_weight = 1.0 / doc_df["dup_group_size"].clip(lower=1)

    # Stash everything to add to `out` later
    _consensus_payload = {
        "dup_group_id":        doc_df["dup_group_id"],
        "dup_group_size":      doc_df["dup_group_size"],
        "is_group_rep":        is_rep.values,
        "sample_weight_dedup": np.round(sample_weight, 6),
        "cid_group":           [L1_CATEGORIES[i][0] for i in best_idx_g],
        "cat_sim_group":       np.round(best_prob_g, 4),
        "cat_margin_group":    np.round(margin_g, 4),
        "decision_group":      decisions_g,
        "entropy_fused_group": np.round(entropy_fused_g, 4),
    }



    # --- Build output on doc_df (post-dedup) ---
    snippet = doc_df["full_text_clean"].astype(str).str.slice(0, SNIPPET_CHARS)
    id_cols = [c for c in ["press_release_id","release_date","company_name","category","source","source_url"]
            if c in doc_df.columns]
    out = doc_df[id_cols].copy() if id_cols else pd.DataFrame(index=doc_df.index)
    out["cid"]           = assigned_cids
    out["category_name"] = assigned_names
    out["cat_sim"]       = best_prob.round(4)   # now fused probability
    out["cat_margin"]    = margin.round(4)
    out["decision"]      = decisions
    out["title"]         = doc_df["title_clean"]
    out["snippet"]       = snippet

    out["run_id"]              = RUN_TS
    out["taxonomy_version"]    = TAXONOMY_VERSION
    out["p1_emb"]              = np.round(p1_emb, 4)
    out["p1_lsa"]              = np.round(p1_tfidf, 4)
    out["entropy_emb"]         = np.round(entropy_emb, 4)
    out["entropy_lsa"]         = np.round(entropy_lsa, 4)
    out["entropy_fused"]       = np.round(entropy_fused, 4)
    out["top3_fused_sum"]      = np.round(top3_fused, 4)
    out["agreement_emb_lsa"]   = agreement_emb_tfidf.astype(bool)
    out["neighbor_consistency_k{}".format(KNN_K)] = np.round(neighbor_consistency, 4)
    # Optional: store top-3 as JSON strings for readability
    out["top3_cids"]           = [json.dumps(v) for v in top3_cids]
    out["top3_probs"]          = [json.dumps(v) for v in top3_probs]
    
    for k, v in _consensus_payload.items():
        out[k] = v

    out.to_csv(OUTPUT_LOCAL_CSV, index=False)
    print(f"[ok] Wrote {len(out)} assignments to {os.path.abspath(OUTPUT_LOCAL_CSV)}")

    # --- Per-run summary metrics ---
    from collections import Counter

    N = len(out)
    dec_counts = Counter(out["decision"].tolist())
    cat_counts = Counter(out["cid"].tolist())

    summary = {
        "run_id": RUN_TS,
        "taxonomy_version": TAXONOMY_VERSION,
        "timestamp_utc": RUN_TS,
        "n_docs": N,
        "use_subset": int(bool(int(os.getenv("USE_SUBSET", "1")))),
        "emb_model": EMB_MODEL_NAME,
        "fuse_alpha": FUSE_ALPHA,
        "temp_emb": TEMP_EMB,
        "temp_lsa": TEMP_LSA,
        "cat_assign_min_sim": CAT_ASSIGN_MIN_SIM,
        "cat_low_margin": CAT_LOW_MARGIN,
        # Agreement & uncertainty
        "agreement_rate_emb_lsa": float(np.mean(agreement_emb_tfidf)) if N else 0.0,
        "avg_margin_fused": float(np.mean(out["cat_margin"])) if N else 0.0,
        "avg_entropy_fused": float(np.mean(entropy_fused)) if N else 0.0,
        "avg_neighbor_consistency_k{}".format(KNN_K): float(np.mean(neighbor_consistency)) if N else 0.0,
        "p10_neighbor_consistency_k{}".format(KNN_K): float(np.percentile(neighbor_consistency, 10)) if N else 0.0,
        # Decision bucket rates
        "rate_auto_assign": dec_counts.get("auto_assign", 0) / N if N else 0.0,
        "rate_needs_review_margin": dec_counts.get("needs_review_margin", 0) / N if N else 0.0,
        "rate_needs_review_low_sim": dec_counts.get("needs_review_low_sim", 0) / N if N else 0.0,
        # Distributions as JSON for easy parsing later
        "category_counts_json": json.dumps(cat_counts, ensure_ascii=False),
        "decision_counts_json": json.dumps(dec_counts, ensure_ascii=False),
    }

    append_row(RUN_METRICS_CSV, summary)
    print(f"[ok] Appended run summary to {os.path.abspath(RUN_METRICS_CSV)}")

if __name__ == "__main__":
    main()