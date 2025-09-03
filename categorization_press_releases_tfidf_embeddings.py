# Python 3.9+
# pip install: sentence-transformers pyarrow gcsfs pandas numpy bs4 lxml langid scikit-learn

import os, json, html as ihtml
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import gcsfs
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors

# External modules you already have
from src.event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter
from dedup_utils import dedup_df
from near_dedup import near_dedup_embeddings
from company_name_remover import clean_df_from_company_names
from event_taxonomies import l1_v3_en
from event_taxonomies.keyword_rules import keywords
from event_taxonomies.taxonomy_rules_classifier import classify_batch, classify_press_release

import logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
else:
    logging.getLogger().setLevel(logging.INFO)

# ---------- Config ----------

@dataclass
class Config:
    # Fusion / temps
    FUSE_ALPHA: float = float(os.getenv("FUSE_ALPHA", "0.6"))
    TEMP_EMB: float   = float(os.getenv("TEMP_EMB",   "1.0"))
    #TEMP_LSA: float   = float(os.getenv("TEMP_LSA",   "1.3"))
    TEMP_LSA: float   = float(os.getenv("TEMP_LSA",   "0.8"))
    

    # Thresholds
    CAT_ASSIGN_MIN_SIM: float = float(os.getenv("CAT_ASSIGN_MIN_SIM", "0.62"))
    CAT_LOW_MARGIN: float     = float(os.getenv("CAT_LOW_MARGIN", "0.02"))

    # KNN
    KNN_K: int = int(os.getenv("KNN_K", "10"))

    # IO
    GCS_SILVER_ROOT: str  = os.getenv("GCS_SILVER_ROOT", "event-feed-app-data/silver_normalized/table=press_releases")
    OUTPUT_LOCAL_CSV: str = os.getenv("OUTPUT_LOCAL_CSV", "assignments.csv")
    OUTPUT_GCS_URI: str   = os.getenv("OUTPUT_GCS_URI", "")

    # Embeddings
    EMB_MODEL_NAME: str = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-base")
    #EMB_MODEL_NAME: str = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-large")
    
    E5_PREFIX: bool     = bool(int(os.getenv("E5_PREFIX", "1")))
    EMB_BATCH_SIZE: int = int(os.getenv("EMB_BATCH_SIZE", "32"))
    MAX_SEQ_TOKENS: int = int(os.getenv("MAX_SEQ_TOKENS", "384"))

    # Subset
    USE_SUBSET: bool = bool(int(os.getenv("USE_SUBSET", "0")))
    SUBSET_CLUSTERS: int = 100
    SUBSET_PER_CLUSTER: int = 6

    # Misc
    SNIPPET_CHARS: int = int(os.getenv("SNIPPET_CHARS", "600"))
    TAXONOMY_VERSION: str = os.getenv("TAXONOMY_VERSION", "v3")
    RUN_METRICS_CSV: str  = os.getenv("RUN_METRICS_CSV", "runs_metrics_v2.csv")

    # LangID
    LANG_WHITELIST: str = os.getenv("LANG_WHITELIST", "en,sv,fi,da,no,de,fr,is")


# ---------- Utils: math & small helpers ----------

def softmax_rows(S, T: float = 1.0) -> np.ndarray:
    if sparse.issparse(S):
        S = S.toarray()
    else:
        S = np.asarray(S)
    S = S.astype(np.float64, copy=False)
    S = S / max(T, 1e-6)
    S -= np.max(S, axis=1, keepdims=True)
    S = np.exp(S)
    denom = np.sum(S, axis=1, keepdims=True) + 1e-12
    return S / denom

def entropy_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P_clip = np.clip(P, eps, 1.0)
    return -np.sum(P_clip * np.log(P_clip), axis=1)

def topk_sum(P: np.ndarray, k: int = 3) -> np.ndarray:
    if P.shape[1] <= k:
        return np.sum(P, axis=1)
    part = np.partition(P, -k, axis=1)
    return np.sum(part[:, -k:], axis=1)

def append_row(path: str, row: dict):
    import csv, os
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

# ---------- Taxonomy ----------

L1_CATEGORIES = l1_v3_en.L1_CATEGORIES  # list of (cid, name, desc)
CID2NAME = {cid: name for cid, name, _ in L1_CATEGORIES}

# ---------- Language detection ----------

def setup_langid(whitelist: str):
    import langid
    if whitelist.strip():
        langid.set_languages([s.strip() for s in whitelist.split(",") if s.strip()])
    return langid

def detect_language_batch(titles: pd.Series, bodies: pd.Series, max_len: int = 1200, langid=None):
    langs, confs = [], []
    for t, b in zip(titles.fillna(""), bodies.fillna("")):
        txt = (str(t).strip() + " " + str(b).strip()[:max_len]).strip()
        if not txt:
            langs.append("und"); confs.append(0.0); continue
        lang, score = langid.classify(txt)
        try:
            conf = 1.0 - float(np.exp(-float(score) / 10.0))
        except Exception:
            conf = 0.0
        langs.append(lang); confs.append(round(conf, 4))
    return langs, confs

# ---------- Text prep ----------

def build_doc_for_embedding(title: str, body: str, e5_prefix: bool) -> str:
    title = (title or "").strip()
    body  = (body or "").strip()
    txt   = f"[TITLE] {title}\n\n{body}" if title else body
    return f"passage: {txt}" if e5_prefix else txt

def build_cat_query(name: str, desc: str, e5_prefix: bool) -> str:
    q = f"category: {name}. description: {desc}"
    return "query: " + q if e5_prefix else q

# ---------- Data IO ----------

def load_from_gcs(uri: str) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    dataset = ds.dataset(uri, filesystem=fs, format="parquet")
    return dataset.to_table().to_pandas()

# ---------- Subset selection (diverse without embeddings) ----------

def pick_diverse_subset(df: pd.DataFrame, text_col: str, k_clusters: int, per_cluster: int, random_state: int = 42) -> pd.DataFrame:
    texts = df[text_col].fillna("").astype(str).tolist()
    hv = HashingVectorizer(analyzer="char", ngram_range=(3,5), n_features=2**20,
                           alternate_sign=False, strip_accents="unicode")
    X_hash = hv.transform(texts)
    svd_sketch = TruncatedSVD(n_components=96, random_state=random_state)
    X_sketch = svd_sketch.fit_transform(X_hash)
    X_sketch = Normalizer(copy=False).fit_transform(X_sketch)

    mbk = MiniBatchKMeans(n_clusters=k_clusters, random_state=random_state, batch_size=4096)
    labels = mbk.fit_predict(X_sketch)

    subset_idx = (pd.Series(range(len(texts)))
                  .groupby(labels, group_keys=False)
                  .apply(lambda s: s.sample(min(per_cluster, len(s)), random_state=random_state))
                  .tolist())
    return df.iloc[subset_idx].reset_index(drop=True)

# ---------- Embeddings ----------

def get_embedding_model(name: str, max_seq_tokens: int) -> SentenceTransformer:
    model = SentenceTransformer(name)
    limit = getattr(model, "max_seq_length", 512) or 512
    model.max_seq_length = max(32, min(max_seq_tokens, int(limit)))
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return model

def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

# ---------- Per-language TF-IDF ----------

def multilingual_tfidf_cosine(
    doc_df: pd.DataFrame,
    cat_texts_lsa_default: List[str],
    rep_keep_idx: Iterable[int],
    lang_to_translated_cats: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """
    Build one TF-IDF space per detected language and compute cosine(doc, cats) within that space.
    Returns sims_lsa (N_docs x N_cats).
    """
    from collections import defaultdict

    N_docs = len(doc_df)
    N_cats = len(cat_texts_lsa_default)
    sims_tfidf = np.zeros((N_docs, N_cats), dtype=np.float64)

    tfidf_input_all = (doc_df["title_clean"].astype(str) + " " + doc_df["full_text_clean"].astype(str))

    lang_to_idx = defaultdict(list)
    for i, lang in enumerate(doc_df["language"].astype(str).fillna("und")):
        lang_to_idx[lang].append(i)

    rep_keep_idx_set = set(int(x) for x in np.asarray(list(rep_keep_idx)).tolist())
    lang_to_rep_idx = {lang: [i for i in idxs if i in rep_keep_idx_set] for lang, idxs in lang_to_idx.items()}

    def make_tfidf(min_df_val=3):
        return make_pipeline(
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=min_df_val,
                max_df=0.70,
                max_features=200_000,
                strip_accents="unicode",
            ),
            Normalizer(copy=False),
        )

    for lang, idxs in lang_to_idx.items():
        if not idxs:
            continue
        n_lang_docs = len(idxs)
        min_df_lang = 1 if n_lang_docs < 50 else 2 if n_lang_docs < 200 else 3
        tfidf_pipe_lang = make_tfidf(min_df_val=min_df_lang)

        rep_idx_lang = lang_to_rep_idx.get(lang, []) or idxs
        _ = tfidf_pipe_lang.fit_transform(tfidf_input_all.iloc[rep_idx_lang])

        V_docs_lang = tfidf_pipe_lang.transform(tfidf_input_all.iloc[idxs])
        cat_texts_lang = (lang_to_translated_cats or {}).get(lang, cat_texts_lsa_default)
        V_cats_lang = tfidf_pipe_lang.transform(cat_texts_lang)

        sims_lang = V_docs_lang @ V_cats_lang.T
        sims_tfidf[np.array(idxs), :] = sims_lang.toarray() if sparse.issparse(sims_lang) else sims_lang

    return sims_tfidf

from importlib import import_module

def load_taxonomy(version: str, lang: str):
    """
    Attempts to import event_taxonomies.l1_<version>_<lang>.
    Falls back to English if that language module doesn't exist.
    """
    mods_to_try = [f"event_taxonomies.l1_{version}_{lang}",
                   f"event_taxonomies.l1_{version}_en"]
    last_err = None
    for mod in mods_to_try:
        try:
            return import_module(mod).L1_CATEGORIES
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load taxonomy for {version}/{lang}: {last_err}")

def make_cat_texts_in_base_order(base_L1, localized_L1):
    """
    Return ['<name>. <desc>', ...] in the exact CID order of base_L1.
    """
    # maps in localized taxonomy: cid -> (name, desc)
    loc_by_cid = {cid: (name, desc) for cid, name, desc in localized_L1}
    cat_texts = []
    for cid, _, _ in base_L1:
        name, desc = loc_by_cid.get(cid, ("", ""))  # fall back to empty strings if missing
        cat_texts.append(f"{name}. {desc}".strip(". "))  # keep tidy even if one part is empty
    return cat_texts

# normalize language codes to the ones you actually have files for
def normalize_lang(code: str) -> str:
    code = (code or "und").lower()
    if code.startswith("sv"): return "sv"
    if code.startswith("fi"): return "fi"
    if code.startswith("da"): return "da"
    # keep 'en' for English; everything else returns as-is and will fall back to English
    if code.startswith("en"): return "en"
    return code

# ---------- Fusion & decisions ----------

def decide(prob: float, margin: float, min_sim: float, low_margin: float) -> str:
    if prob >= min_sim and margin >= low_margin:
        return "auto_assign"
    elif prob >= min_sim:
        return "needs_review_margin"
    else:
        return "needs_review_low_sim"

# ---------- Neighbor consistency ----------

def neighbor_consistency_scores(X_emb: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X_emb)), metric="cosine", algorithm="auto")
    nbrs.fit(X_emb)
    dist, idx = nbrs.kneighbors(X_emb, return_distance=True)
    idx_no_self = idx[:, 1:]
    neighbor_labels = labels[idx_no_self]
    return (neighbor_labels == labels.reshape(-1,1)).mean(axis=1)

# ---------- Group consensus ----------

def consensus_by_dup_group(doc_df: pd.DataFrame, P_fused: np.ndarray) -> Dict[str, np.ndarray]:
    N_docs, N_c = P_fused.shape

    if "dup_group_id" not in doc_df.columns:
        doc_df = doc_df.copy()
        doc_df["dup_group_id"] = (doc_df.get("press_release_id", pd.Series(range(N_docs))).astype(str))

    body_len = doc_df["full_text_clean"].astype(str).str.len().fillna(0).to_numpy()
    len_norm = body_len / (body_len.mean() + 1e-9)
    lang_conf = pd.to_numeric(doc_df.get("lang_conf", pd.Series(0.5, index=doc_df.index)), errors="coerce").fillna(0.5).to_numpy()
    w = np.clip(0.7 * lang_conf + 0.3 * len_norm, 0.1, None)

    groups = doc_df["dup_group_id"].astype(str).to_numpy()
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

    best_idx_g  = np.argmax(P_fused_g, axis=1)
    best_prob_g = P_fused_g[np.arange(N_docs), best_idx_g]
    second_g    = np.partition(P_fused_g, -2, axis=1)[:, -2] if N_c >= 2 else np.zeros_like(best_prob_g)
    margin_g    = best_prob_g - second_g
    entropy_fused_g = entropy_rows(P_fused_g)

    # group size + representative marker
    doc_df = doc_df.copy()
    doc_df["__body_len_tmp"] = body_len
    grp = doc_df.groupby("dup_group_id", dropna=False)
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
    doc_df.drop(columns=["__body_len_tmp"], inplace=True)

    sample_weight = 1.0 / doc_df["dup_group_size"].clip(lower=1)

    return dict(
        P_fused_g=P_fused_g,
        best_idx_g=best_idx_g,
        best_prob_g=best_prob_g,
        margin_g=margin_g,
        entropy_fused_g=entropy_fused_g,
        dup_group_id=doc_df["dup_group_id"].to_numpy(),
        dup_group_size=doc_df["dup_group_size"].to_numpy(),
        is_group_rep=is_rep.values,
        sample_weight_dedup=np.round(sample_weight, 6),
    )
def report_rule_metrics(
    df: pd.DataFrame,
    assigned_col: str = "final_source",
    rule_id_col: str = "rule_id",
    rules_cat_col: str = "rule_pred_cid",
    model_cat_col: str = "ml_pred_cid",
    final_col: str = "final_cid",
):
    import logging, numpy as np, pandas as pd
    logger = logging.getLogger("cat.metrics")

    total = int(len(df))
    if total == 0:
        logger.info("report_rule_metrics: empty dataframe; skipping.")
        return

    # --- derive/sanitize assigned_by ---
    if assigned_col in df.columns:
        assigned = df[assigned_col]
    elif "final_source" in df.columns:
        assigned = df["final_source"].map({"rules": "rules", "ml": "model"}).fillna("none")
    else:
        rules_has = df.get(rules_cat_col)
        model_has = df.get(model_cat_col)
        assigned = np.where(
            (rules_has.notna() if rules_has is not None else False), "rules",
            np.where((model_has.notna() if model_has is not None else False), "model", "none")
        )

    def _scalarize(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray):
            if x.ndim == 0: return x.item()
            if x.size == 1: return x.reshape(-1)[0]
            return ",".join(map(str, x.tolist()))
        if isinstance(x, (list, tuple)):
            return x[0] if len(x) == 1 else ",".join(map(str, x))
        return x

    assigned_s = pd.Series(assigned).map(_scalarize).astype("string")

    # --- assignments by source ---
    by_source = (
        assigned_s.value_counts(dropna=False)
        .rename_axis("source").to_frame("n")
        .assign(pct=lambda s: (s["n"] / max(total, 1)).round(4))
        .reset_index()
    )
    logger.info("Assignments by source:\n%s", by_source.to_string(index=False))

    # --- rules coverage + top rules ---
    rules_mask = assigned_s.eq("rules")
    n_rules = int(rules_mask.sum())
    logger.info("Rules-assigned: %d/%d (%.2f%%)", n_rules, total, 100.0 * n_rules / max(total, 1))

    if rule_id_col in df.columns and n_rules:
        by_rule = (
            df.loc[rules_mask, rule_id_col]
              .astype("string")
              .value_counts()
              .rename_axis("rule_id")
              .to_frame("n")
        )
        by_rule["pct_of_rules"] = (by_rule["n"] / max(n_rules, 1)).round(4)
        logger.info("Top rules by coverage (up to 20):\n%s",
                    by_rule.head(20).to_string())

    # --- rules vs model conflicts ---
    if rules_cat_col in df.columns and model_cat_col in df.columns:
        rc = df[rules_cat_col].astype("string")
        mc = df[model_cat_col].astype("string")
        conflicts = rc.notna() & mc.notna() & rc.ne(mc)
        n_conflicts = int(conflicts.sum())
        logger.info("Rules vs Model conflicts: %d (%.2f%%)",
                    n_conflicts, 100.0 * n_conflicts / max(total, 1))

    # --- unassigned after final resolution ---
    if final_col in df.columns:
        unassigned = int(df[final_col].isna().sum())
        logger.info("Unassigned after resolution: %d (%.2f%%)",
                    unassigned, 100.0 * unassigned / max(total, 1))



# ---------- Output builders ----------
def build_output_df(
    base_df: pd.DataFrame,
    best_prob: np.ndarray,
    margin: np.ndarray,
    P_emb: np.ndarray,
    P_lsa: np.ndarray,
    P_fused: np.ndarray,
    top3_idx: np.ndarray,
    cfg: Config,
    consensus_payload: Dict[str, np.ndarray],
) -> pd.DataFrame:

    id_cols = [c for c in ["press_release_id","release_date","company_name","source","source_url"]
               if c in base_df.columns]
    out = base_df[id_cols].copy() if id_cols else pd.DataFrame(index=base_df.index)

    # Final assignment
    out["final_cid"]          = base_df["final_cid"]
    out["final_category_name"]= base_df["final_category_name"]
    out["final_source"]       = base_df["final_source"]
    out["final_decision"] = base_df["final_decision"]
    out["ml_decision"]    = base_df["ml_decision"]

    # Rules provenance
    for c in ["rule_id","rule_conf","rule_lock","rule_needs_review","rule_pred_cid"]:
        if c in base_df.columns:
            out[c] = base_df[c]

    # ML diagnostics (fused)
    out["ml_pred_cid"]        = base_df["ml_pred_cid"]
    out["ml_conf_fused"]      = np.round(best_prob, 4)
    out["ml_margin_fused"]    = np.round(margin, 4)
    out["ml_entropy_emb"]     = base_df["ml_entropy_emb"]
    out["ml_entropy_lsa"]     = base_df["ml_entropy_lsa"]
    out["ml_entropy_fused"]   = base_df["ml_entropy_fused"]
    out["ml_p1_emb"]          = base_df["ml_p1_emb"]
    out["ml_p1_lsa"]          = base_df["ml_p1_lsa"]
    out["ml_agreement_emb_lsa"]= base_df["ml_agreement_emb_lsa"]
    out["ml_top3_cids"]       = base_df["ml_top3_cids"]
    out["ml_top3_probs"]      = base_df["ml_top3_probs"]
    out["ml_version"]         = base_df["ml_version"]

    # Text
    out["title"]   = base_df["title_clean"]
    out["snippet"] = base_df["full_text_clean"].astype(str).str.slice(0, cfg.SNIPPET_CHARS)

    # Group consensus exports
    for k, v in consensus_payload.items():
        if k == "P_fused_g":
            continue
        if k == "best_idx_g":
            out["final_cid_group"] = [L1_CATEGORIES[i][0] for i in v]
        elif k == "best_prob_g":
            out["ml_conf_fused_group"] = np.round(v, 4)
        elif k == "margin_g":
            out["ml_margin_fused_group"] = np.round(v, 4)
        elif k == "entropy_fused_g":
            out["ml_entropy_fused_group"] = np.round(v, 4)
        else:
            out[k] = v

    # Run metadata
    out["run_id"]           = RUN_TS
    out["taxonomy_version"] = cfg.TAXONOMY_VERSION
    return out

def summarize_run(out: pd.DataFrame, neighbor_consistency: np.ndarray, cfg: Config) -> dict:
    from collections import Counter
    N = len(out)

    ml_dec = out.get("ml_decision")
    if ml_dec is None:
        ml_dec = pd.Series([], dtype=str)
    dec_counts = Counter(ml_dec.tolist())

    cat_counts = Counter(out["final_cid"].tolist()) if "final_cid" in out else Counter()

    share_final_by_rules = float((out["final_source"] == "rules").mean()) if "final_source" in out and N else 0.0

    return {
        "run_id": RUN_TS,
        "taxonomy_version": cfg.TAXONOMY_VERSION,
        "timestamp_utc": RUN_TS,
        "n_docs": N,
        "use_subset": int(cfg.USE_SUBSET),
        "emb_model": cfg.EMB_MODEL_NAME,
        "fuse_alpha": cfg.FUSE_ALPHA,
        "temp_emb": cfg.TEMP_EMB,
        "temp_lsa": cfg.TEMP_LSA,

        "agreement_rate_emb_lsa": float(np.mean(out["ml_agreement_emb_lsa"])) if N and "ml_agreement_emb_lsa" in out else 0.0,
        "avg_margin_fused": float(np.mean(out["ml_margin_fused"])) if N and "ml_margin_fused" in out else 0.0,
        "avg_entropy_fused": float(np.mean(out["ml_entropy_fused"])) if N and "ml_entropy_fused" in out else 0.0,

        f"avg_neighbor_consistency_k{cfg.KNN_K}": float(np.mean(neighbor_consistency)) if len(neighbor_consistency) else 0.0,
        f"p10_neighbor_consistency_k{cfg.KNN_K}": float(np.percentile(neighbor_consistency, 10)) if len(neighbor_consistency) else 0.0,

        "rate_auto_assign": dec_counts.get("auto_assign", 0) / N if N else 0.0,
        "rate_needs_review_margin": dec_counts.get("needs_review_margin", 0) / N if N else 0.0,
        "rate_needs_review_low_sim": dec_counts.get("needs_review_low_sim", 0) / N if N else 0.0,

        "share_final_by_rules": share_final_by_rules,

        "category_counts_json": json.dumps(dict(cat_counts), ensure_ascii=False),
        "decision_counts_json": json.dumps(dict(Counter(out.get("final_decision", pd.Series([], dtype=str)).tolist())), ensure_ascii=False),
    }



# ---------- Main pipeline ----------

RUN_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

def run(cfg: Config):
    # 1) Load & housekeeping
    df = load_from_gcs(cfg.GCS_SILVER_ROOT)
    df, stats = apply_housekeeping_filter(df)
    print(f"Housekeeping filter skipped {stats['skipped']} / {stats['total']} ({(stats['skipped']/max(stats['total'],1))*100:.1f}%).")

    df = clean_df_from_company_names(df)

    # 2) Hard dedup (id+content)
    df, dd_stats = dedup_df(
        df,
        id_col="press_release_id",
        title_col="title_clean",
        body_col="full_text_clean",
        ts_col="release_date",
    )
    print(dd_stats)

    # 3) Optional diverse subset
    base_df = pick_diverse_subset(df, "full_text_clean", cfg.SUBSET_CLUSTERS, cfg.SUBSET_PER_CLUSTER) if cfg.USE_SUBSET else df

    base_df = base_df.reset_index(drop=True)

    # 3b) Rules-first classification (keywords + smart rules)
    # FIXED: removed stray parenthesis and passed keywords correctly
    df_pred = classify_batch(
        base_df,
        text_cols=("title_clean", "full_text_clean"),
        keywords=keywords
    )
    # Keep the rule-based predictions in the working dataframe
    base_df = df_pred  # carri
        
    # --- provenance from rules ---
    base_df["rule_id"]           = base_df["rule_pred_details"].map(lambda d: (d or {}).get("rule"))
    base_df["category_rules"]    = base_df["rule_pred_cid"]  # back-compat alias if other code expects it
    base_df["rule_conf"]         = base_df["rule_pred_details"].map(lambda d: float((d or {}).get("rule_conf", 0.0)))
    base_df["rule_lock"]         = base_df["rule_pred_details"].map(lambda d: bool((d or {}).get("lock", False)))
    base_df["rule_needs_review"] = base_df["rule_pred_details"].map(lambda d: bool((d or {}).get("needs_review_low_sim", False)))


    # Mark confident rule hits (lock=True or rule_conf >= 0.75) and not "other"
    RULE_CONF_THR = 0.75

    def is_rules_final(row) -> bool:
        det = row.get("rule_pred_details") or {}
        if det.get("lock"):
            return row.get("rule_pred_cid") != "other_corporate_update"
        if det.get("needs_review_low_sim", False):
            return False
        conf = float(det.get("rule_conf", 0.0))
        return (row.get("rule_pred_cid") not in (None, "", "other_corporate_update")) and (conf >= RULE_CONF_THR)

    mask_rules_final = base_df.apply(is_rules_final, axis=1)


    # 4) Taxonomy text
    cat_names = [name for _, name, _ in L1_CATEGORIES]
    cat_descs = [desc for _, _, desc in L1_CATEGORIES]
    cat_texts_lsa_default = [f"{n}. {d}" for n, d in zip(cat_names, cat_descs)]
    cat_texts_emb = [build_cat_query(n, d, cfg.E5_PREFIX) for n, d in zip(cat_names, cat_descs)]

    # 5) Embedding model + docs/cats embeddings
    model = get_embedding_model(cfg.EMB_MODEL_NAME, cfg.MAX_SEQ_TOKENS)
    docs_for_emb = [
        build_doc_for_embedding(t, b, cfg.E5_PREFIX)
        for t, b in zip(base_df["title_clean"].tolist(), base_df["full_text_clean"].tolist())
    ]
    X = encode_texts(model, docs_for_emb, cfg.EMB_BATCH_SIZE)                 # (N, d)
    Q = encode_texts(model, cat_texts_emb, cfg.EMB_BATCH_SIZE)                # (C, d)

    # 6) Language detection
    langid = setup_langid(cfg.LANG_WHITELIST)
    langs, lang_confs = detect_language_batch(base_df["title_clean"], base_df["full_text_clean"], langid=langid)
    base_df = base_df.copy()
    base_df["language"] = langs
    base_df["lang_conf"] = lang_confs

    # 7) Near-dedup (embedding space)
    base_df, X, near_dedupe_stats, rep_keep_idx = near_dedup_embeddings(
        base_df, X, threshold=0.97, drop=False, RUN_TS=RUN_TS,
        rep_strategy="lang_conf", lang_conf_col="lang_conf",
        NEAR_DUP_AUDIT_CSV="near_dedup_audit.csv"
    )
    print(near_dedupe_stats)


    

    # 8) Per-language TF-IDF similarities with localized category texts

    # L1_CATEGORIES
    # CID2NAME
    base_L1 = L1_CATEGORIES
    base_cids = [cid for cid, _, _ in base_L1]

    langs_present = set(normalize_lang(x) for x in base_df["language"].astype(str).fillna("und"))

    lang_to_translated_cats = {}
    for lang in langs_present:
        try:
            # load localized taxonomy (or English fallback)
            localized_L1 = load_taxonomy(cfg.TAXONOMY_VERSION, lang)
            # sanity: same set of CIDs as base
            loc_cids = {cid for cid, _, _ in localized_L1}
            if set(base_cids) != loc_cids:
                # if mismatch, skip localization for this lang to avoid misalignment
                # (you could log a warning here instead)
                continue
            lang_to_translated_cats[lang] = make_cat_texts_in_base_order(base_L1, localized_L1)
        except Exception:
            # if module not found or any error, just let TF-IDF fall back to English
            pass


    # Always keep your English default list (already built earlier in your code)
    # cat_texts_lsa_default = [f"{n}. {d}" for _, n, d in base_L1]

    # --- Per-language TF-IDF similarities with localized category texts ---
    sims_lsa = multilingual_tfidf_cosine(
        doc_df=base_df,
        cat_texts_lsa_default=cat_texts_lsa_default,   # English fallback
        rep_keep_idx=rep_keep_idx,
        lang_to_translated_cats=lang_to_translated_cats,  # <- pass your localized cats
    )

    # 9) Similarities (embeddings) + softmaxes + fusion
    sims_emb = X @ Q.T
    P_emb = softmax_rows(sims_emb, T=cfg.TEMP_EMB)
    P_lsa = softmax_rows(sims_lsa, T=cfg.TEMP_LSA)
    P_fused = cfg.FUSE_ALPHA * P_emb + (1.0 - cfg.FUSE_ALPHA) * P_lsa

    # 10) Decisions
    best_idx = np.argmax(P_fused, axis=1)
    best_prob = P_fused[np.arange(len(P_fused)), best_idx]
    if P_fused.shape[1] >= 2:
        part = np.partition(P_fused, -2, axis=1)
        second_best = part[:, -2]
        margin = best_prob - second_best
    else:
        second_best = np.zeros_like(best_prob)
        margin = np.zeros_like(best_prob)
    
    
    # --- ML (fused) provenance + diagnostics (Patch 2) ---

    # Assigned by fusion
    assigned_cids = [L1_CATEGORIES[i][0] for i in best_idx]

    # Entropies and top-1 per source
    entropy_emb   = entropy_rows(P_emb)
    entropy_lsa   = entropy_rows(P_lsa)
    entropy_fused = entropy_rows(P_fused)

    p1_emb = P_emb[np.arange(len(P_emb)), np.argmax(P_emb, axis=1)]
    p1_lsa = P_lsa[np.arange(len(P_lsa)), np.argmax(P_lsa, axis=1)]

    # Top-3
    top3_idx   = np.argsort(-P_fused, axis=1)[:, :3]
    top3_cids  = [[L1_CATEGORIES[j][0] for j in row] for row in top3_idx]
    top3_probs = [P_fused[i, top3_idx[i]].round(4).tolist() for i in range(len(P_fused))]

    # Normalized ML columns
    base_df["ml_pred_cid"]          = assigned_cids
    base_df["ml_conf_fused"]        = best_prob
    base_df["ml_margin_fused"]      = margin
    base_df["ml_entropy_emb"]       = np.round(entropy_emb, 4)
    base_df["ml_entropy_lsa"]       = np.round(entropy_lsa, 4)
    base_df["ml_entropy_fused"]     = np.round(entropy_fused, 4)
    base_df["ml_p1_emb"]            = np.round(p1_emb, 4)
    base_df["ml_p1_lsa"]            = np.round(p1_lsa, 4)
    base_df["ml_agreement_emb_lsa"] = (np.argmax(P_emb, axis=1) == np.argmax(P_lsa, axis=1)).astype(bool)
    base_df["ml_top3_cids"]         = [json.dumps(v) for v in top3_cids]
    base_df["ml_top3_probs"]        = [json.dumps(v) for v in top3_probs]
    base_df["ml_version"]           = f"e5+tfidf:{cfg.TAXONOMY_VERSION}|alpha={cfg.FUSE_ALPHA}|T={cfg.TEMP_EMB}/{cfg.TEMP_LSA}"

    # Back-compat aliases so existing scripts keep working
    base_df["category_model"] = base_df["ml_pred_cid"]
    base_df["model_conf"]     = base_df["ml_conf_fused"]
    base_df["model_version"]  = base_df["ml_version"]


    # Rules-first fusion: rules decide where confident; ML handles the rest
    final_cids   = []
    final_source = []
    for i in range(len(base_df)):
        if mask_rules_final.iloc[i]:
            final_cids.append(base_df.loc[i, "rule_pred_cid"])
            final_source.append("rules")
        else:
            final_cids.append(base_df.loc[i, "ml_pred_cid"])
            final_source.append("ml")

    base_df["final_cid"]     = final_cids
    base_df["final_source"]  = final_source

    # Decision string for the fused ML probability/margin (still useful to log)
    # decisions = [decide(float(p), float(m), cfg.CAT_ASSIGN_MIN_SIM, cfg.CAT_LOW_MARGIN)
    #             for p, m in zip(best_prob, margin)]
    
    # ML decision string for diagnostics
    base_df["ml_decision"] = [decide(float(p), float(m), cfg.CAT_ASSIGN_MIN_SIM, cfg.CAT_LOW_MARGIN)
                          for p, m in zip(best_prob, margin)]

    # Final decision: if rules picked the label, mark that
    base_df["final_decision"] = np.where(
        base_df["final_source"].eq("rules"),
        "by_rules",
        base_df["ml_decision"]
    )

    # Display name
    base_df["final_category_name"] = [CID2NAME[cid] for cid in base_df["final_cid"]]

    # Back-compat aliases for metrics/eval that expect these
    base_df["assigned_by"] = base_df["final_source"].map({"rules":"rules", "ml":"model"}).astype("string")
    base_df["category"]    = base_df["final_cid"]


    report_rule_metrics(base_df)

    # 11) Diagnostics
    #top3_idx = np.argsort(-P_fused, axis=1)[:, :3]
    neighbor_consistency = neighbor_consistency_scores(X, best_idx, cfg.KNN_K)

    # 12) Group consensus
    consensus_payload = consensus_by_dup_group(base_df, P_fused)
    decisions_g = [
        decide(float(p), float(m), cfg.CAT_ASSIGN_MIN_SIM, cfg.CAT_LOW_MARGIN)
        for p, m in zip(consensus_payload["best_prob_g"], consensus_payload["margin_g"])
    ]
    # expose group-level decisions
    consensus_payload["decision_group"] = decisions_g

    # 13) Output dataframe
    out = build_output_df(
        base_df=base_df,
        best_prob=best_prob,
        margin=margin,
        P_emb=P_emb,
        P_lsa=P_lsa,
        P_fused=P_fused,
        top3_idx=top3_idx,
        cfg=cfg,
        consensus_payload=consensus_payload,
    )
    
    
    # out = build_output_df(
    #     base_df=base_df,
    #     assigned_cids=base_df["final_cid"].tolist(),
    #     best_prob=best_prob,
    #     margin=margin,
    #     decisions=decisions,
    #     P_emb=P_emb,
    #     P_lsa=P_lsa,
    #     P_fused=P_fused,
    #     top3_idx=top3_idx,
    #     cfg=cfg,
    #     consensus_payload=consensus_payload,
        
    # )

    # 14) Write outputs
    output_csv = cfg.OUTPUT_LOCAL_CSV
    root, ext = os.path.splitext(output_csv)
    output_csv_ts = f"{root}_{RUN_TS}{ext}"
    out.to_csv(output_csv_ts, index=False)
    print(f"[ok] Wrote {len(out)} assignments to {os.path.abspath(output_csv_ts)}")

    # 15) Run summary
    summary = summarize_run(out, neighbor_consistency, cfg)
    append_row(cfg.RUN_METRICS_CSV, summary)
    print(f"[ok] Appended run summary to {os.path.abspath(cfg.RUN_METRICS_CSV)}")

def main():
    cfg = Config()
    run(cfg)

if __name__ == "__main__":
    main()
