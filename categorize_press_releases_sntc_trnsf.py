# Python 3.9+
# pip install: sentence-transformers pyarrow gcsfs pandas numpy bs4 lxml

import os, json, html as ihtml
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import gcsfs
from bs4 import BeautifulSoup
import re

from sentence_transformers import SentenceTransformer
from src.event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter
from src.event_feed_app.preprocessing.dedup_utils import dedup_df
from src.event_feed_app.preprocessing.near_dedup import near_dedup_embeddings


FUSE_ALPHA = float(os.getenv("FUSE_ALPHA", "0.6"))  # weight for embeddings in fusion
TEMP_EMB   = float(os.getenv("TEMP_EMB",   "1.0"))  # softmax temp for embeddings sims
TEMP_LSA   = float(os.getenv("TEMP_LSA",   "1.3"))  # softmax temp for LSA sims

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

# =========================
# LEVEL-1 TAXONOMY
# =========================
L1_CATEGORIES: List[Tuple[str, str, str]] = [
    ("earnings_report", "Earnings report",
     "Periodic financial results: interim/quarter/half-year/year-end reports; may include KPIs, guidance commentary, and statutory statements."),
    ("share_buyback", "Share buyback",
     "Company repurchases of its own shares: program launches, extensions, updates, completions."),
    ("dividend", "Dividend",
     "Dividend declarations, proposals, changes, schedules (ex-date, record date, payment date)."),
    ("agm_egm_governance", "AGM/EGM & governance",
     "Annual/extraordinary general meeting notices and outcomes; board/nominations; bylaws changes; auditor matters."),
    ("capital_raise_rights_issue", "Capital raise / Rights issue",
     "Equity financing events: rights issues, directed issues, private placements, units/warrants; subscription details and use of proceeds."),
    ("pdmr_managers_transactions", "Managers' transactions (PDMR)",
     "Insider dealings by persons discharging managerial responsibilities (PDMR): purchases/sales/options exercises."),
    ("admission_listing", "Admission / Listing",
     "Listings and admissions to trading: first day of trading, uplistings, venue changes."),
    ("debt_bond_issue", "Debt / Bond issue",
     "Debt financing: bonds/notes issuance, convertibles, syndicated loans, green/sustainability-linked bonds."),
    ("guidance_outlook", "Guidance / Outlook",
     "Forward-looking statements: guidance initiations/updates/withdrawals; trading updates; profit warnings."),
    ("mna", "M&A",
     "Mergers, acquisitions, divestments, spin-offs, asset sales; includes deal terms, valuations, conditions, and timelines."),
    ("orders_contracts", "Orders / Contracts",
     "Order wins, contract awards, framework agreements, renewals, significant tender results."),
    ("product_launch_partnership", "Product / Launch / Partnership",
     "Product/service launches and upgrades; partnerships, collaborations, joint ventures, pilot programs."),
    ("personnel_management_change", "Personnel / Management change",
     "Executive and board changes: appointments, resignations, interim roles, succession plans."),
    ("sustainability_esg", "Sustainability / ESG",
     "Sustainability reports and initiatives; ESG targets/metrics; certifications; climate goals."),
    ("legal_regulatory_compliance", "Legal / Regulatory / Compliance",
     "Litigation, investigations, regulatory approvals/denials, fines/penalties, compliance disclosures."),
    ("credit_ratings", "Credit & Ratings",
     "Issuer/instrument rating actions and outlooks by rating agencies; watch placements."),
    ("equity_actions_non_buyback", "Equity actions (non-buyback)",
     "Share splits/reverse splits, ticker/name changes, conversions, warrant exercises, share class changes."),
    ("labor_workforce", "Labor / Workforce",
     "Workforce actions and relations: layoffs, plant closures, strikes/union matters, large hiring plans."),
    ("incidents_controversies", "Incidents / Controversies",
     "Operational incidents and controversies: safety/accidents, cybersecurity breaches, data/privacy incidents, sanctions."),
    ("admission_delisting", "Delisting / Trading halt",
     "Delistings, trading suspensions/halts, market notices affecting trading status."),
    ("other_corporate_update", "Other / Corporate update",
     "General corporate updates that do not fit the above categories; housekeeping announcements.")
]

CID2NAME = {cid: name for cid, name, _ in L1_CATEGORIES}


# def softmax_rows(S: np.ndarray, T: float = 1.0) -> np.ndarray:
#     S = S / max(T, 1e-6)
#     S = S - S.max(axis=1, keepdims=True)
#     np.exp(S, out=(S := np.exp(S)))                  # reuse buffer
#     S_sum = S.sum(axis=1, keepdims=True) + 1e-12
#     return S / S_sum

from scipy import sparse
import numpy as np

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

# =========================
# TEXT CLEANING (minimal)
# =========================
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
WS_RE    = re.compile(r"\s+")

def strip_html(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    txt = ihtml.unescape(text)
    soup = BeautifulSoup(txt, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    txt = soup.get_text(separator=" ")
    txt = URL_RE.sub(" ", txt)
    txt = EMAIL_RE.sub(" ", txt)
    return WS_RE.sub(" ", txt).strip()

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

    # Ensure text columns exist
    for c in ["title", "full_text"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    # Clean text
    df["title_clean"] = df["title"].map(strip_html)
    df["full_text_clean"] = df["full_text"].map(strip_html)

    # --- Housekeeping filter (modular) ---
    df, stats = apply_housekeeping_filter(
        df,
        title_col="title",
        body_col="full_text",
        use_clean_columns=True,        # use *_clean we just created
        strip_html_fn=None,            # not needed since we use clean cols
        reason_col=None,               # or "_skip_reason" if you want to inspect later
    )
    print(f"Housekeeping filter skipped {stats['skipped']} / {stats['total']} ({(stats['skipped']/max(stats['total'],1))*100:.1f}%).")

    # 2) Deduplicate (id + content)
    df, dd_stats = dedup_df(
        df,
        id_col="press_release_id",     # set to None if you don’t have it
        title_col="title_clean",
        body_col="full_text_clean",
        ts_col="release_date",         # optional; keeps newest when duplicates exist
    )
    print(dd_stats)

    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import Normalizer

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




    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    #from sklearn.decomposition import TruncatedSVD, NMF
    from sklearn.cluster import MiniBatchKMeans
    
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
    # lsa = make_pipeline(tfidf_word, l2norm)

    USE_SUBSET = bool(int(os.getenv("USE_SUBSET", "1")))  # 1=use df_sub, 0=use full df
    doc_df = df_sub if USE_SUBSET else df

    lsa_input = (doc_df["title_clean"].astype(str) + " " + doc_df["full_text_clean"].astype(str))
    # V_docs = lsa.fit_transform(lsa_input)

    # #V_docs = lsa.fit_transform(doc_df["full_text_clean"])  # (N_docs, d_lsa), L2-normalized
    # V_cats = lsa.transform(cat_texts_lsa)                  # (C, d_lsa)

    # --- Model ---
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



    # --- Similarities in each space ---
    sims_emb = X @ Q.T              # cosine (X, Q are L2-normalized)
    #sims_lsa = V_docs @ V_cats.T    # cosine (LSA pipeline L2-normalizes)

    # --- Per-doc calibration and fusion ---
    P_emb = softmax_rows(sims_emb, T=TEMP_EMB)             # (N_docs, C)
    #P_lsa = softmax_rows(sims_lsa, T=TEMP_LSA)             # (N_docs, C)
    #P_fused = FUSE_ALPHA * P_emb + (1.0 - FUSE_ALPHA) * P_lsa

    # --- Argmax + margin on fused probs ---
    best_idx = np.argmax(P_emb, axis=1)
    best_prob = P_emb[np.arange(len(P_emb)), best_idx]

    if P_emb.shape[1] >= 2:
        # get 2nd best via partial sort
        part = np.partition(P_emb, -2, axis=1)
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

    out.to_csv(OUTPUT_LOCAL_CSV, index=False)
    print(f"[ok] Wrote {len(out)} assignments to {os.path.abspath(OUTPUT_LOCAL_CSV)}")

if __name__ == "__main__":
    main()