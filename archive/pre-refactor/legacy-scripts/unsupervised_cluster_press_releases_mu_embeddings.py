# unsup_cluster_and_categorize_st.py
# Python 3.9+
# pip install: scikit-learn pyarrow gcsfs pandas numpy sentence-transformers transformers bs4 lxml regex

import os, re, json, datetime, html as ihtml
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import gcsfs

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
import regex as re2

# =========================
# CONFIG
# =========================
GCS_SILVER_ROOT   = os.getenv("GCS_SILVER_ROOT", "event-feed-app-data/silver_normalized/table=press_releases")
OUTPUT_GCS_PREFIX = os.getenv("OUTPUT_GCS_PREFIX", "gs://event-feed-app-data/unsup_clusters")
#RUN_TS = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

from datetime import datetime, UTC
RUN_TS = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")

# Embedding config
#EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-base")
E5_PREFIX      = bool(int(os.getenv("E5_PREFIX", "1")))   # if using an E5 model, set to 1
CHUNK_TOKENS   = int(os.getenv("CHUNK_TOKENS", "384"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "32"))
EMB_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "8"))
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# K search
CANDIDATE_K = list(range(10, 41, 5))  # 10..40 step 5

# Decision thresholds for category assignment (doc -> category)
CAT_ASSIGN_MIN_SIM = float(os.getenv("CAT_ASSIGN_MIN_SIM", "0.62"))
CAT_LOW_MARGIN     = float(os.getenv("CAT_LOW_MARGIN", "0.04"))

# Review pack / snippets
ENABLE_REVIEW_PACK = bool(int(os.getenv("ENABLE_REVIEW_PACK", "1")))
REVIEW_PER_CLUSTER = int(os.getenv("REVIEW_PER_CLUSTER", "30"))
SNIPPET_CHARS      = int(os.getenv("SNIPPET_CHARS", "600"))

# Outputs
OUT_MODELSEL_CSV         = "model_selection.csv"
OUT_DIAG_CSV             = "cluster_diagnostics.csv"
OUT_CENTSIM_CSV          = "centroid_cosine_similarity.csv"
OUT_SUMMARY_CSV          = "cluster_summaries.csv"
OUT_ASSIGNMENTS_CSV      = "assignments_with_categories.csv"   # ID-only + category + confidence
OUT_REVIEW_PACK_CSV      = "review_samples.csv"
OUT_CATEGORIES_JSON      = "categories.json"                   # category defs + learned centroids
OUT_CATEGORY_SCORES_CSV  = "category_scores.csv"               # per-cluster category match metrics

# Review sets
OUT_REVIEW_UNCERTAIN_CSV = "review_uncertain.csv"
OUT_REVIEW_SHOWCASE_CSV  = "review_showcase.csv"

REVIEW_UNCERTAIN_TOTAL   = int(os.getenv("REVIEW_UNCERTAIN_TOTAL", "1500"))  # cap for uncertain queue
SHOWCASE_PER_CATEGORY    = int(os.getenv("SHOWCASE_PER_CATEGORY", "20"))     # high-confidence examples per category
SHOWCASE_MIN_MARGIN      = float(os.getenv("SHOWCASE_MIN_MARGIN", "0.05"))   # ensure clear separation


# =========================
# LEVEL-1 TAXONOMY (NAME + DESCRIPTION)
# =========================
# (from your approved CSV)
L1_CATEGORIES = [
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
CID2DESC = {cid: desc for cid, _, desc in L1_CATEGORIES}

# =========================
# TEXT CLEANING
# =========================
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
WS_RE    = re.compile(r"\s+")
WS_MULTI_RE = re.compile(r"\s+")

BOILERPLATE_RE = re2.compile(
    r"(?is)\b(This information is information .*? Market Abuse Regulation.*?$)|"
    r"(?is)\b(Forward-looking statements.*?$)|"
    r"(?is)\b(For further information, please contact:.*?$)|"
    r"(?is)\b(Privacy Policy|Cookie Policy|Unsubscribe|Manage your subscription).*$"
)
LONG_NUM_RE = re.compile(r"(?<!\d)(\d{5,})(?!\d)")

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
    txt = WS_RE.sub(" ", txt).strip()
    return txt

def remove_boilerplate(text: str) -> str:
    return BOILERPLATE_RE.sub("", text or "").strip()

def normalize_numbers(text: str) -> str:
    return LONG_NUM_RE.sub("<NUM>", text or "")

def build_doc_for_embedding(title: str, body: str, e5_prefix: bool = False) -> str:
    title = (title or "").strip()
    body  = normalize_numbers(remove_boilerplate((body or "").strip()))
    title = WS_MULTI_RE.sub(" ", title)
    body  = WS_MULTI_RE.sub(" ", body)
    txt   = f"[TITLE] {title}\n\n{body}" if title else body
    return f"passage: {txt}" if e5_prefix else txt

# --- System/notification filters (kill alert emails, subscription churn) ---
TITLE_BLOCKLIST = [
    r"\b(confirm your subscription|verify your subscription|welcome( as)? (a )?subscriber)\b",
    r"\bsubscription (activation|request|removal)\b",
    r"\b(email alerting service|stock quote notification|validate account)\b",
]
TITLE_BLOCKLIST_RE = re.compile("(" + "|".join(TITLE_BLOCKLIST) + ")", re.I)

BODY_HINTS_BLOCKLIST = re.compile(
    r"\b(unsubscribe|manage (your )?subscription|email alerting service|"
    r"confirm( your)? subscription|verify (your )?subscription|cookie policy|privacy policy)\b",
    re.I
)

# --- Signature-based dedupe (near-exact) ---
from unicodedata import normalize as ucnorm

def _norm(s: str) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = ucnorm("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_signature(df: pd.DataFrame, body_head_chars: int = 800) -> pd.Series:
    t = df["title_clean"].map(_norm)
    b = df["full_text_clean"].str[:body_head_chars].map(_norm)
    return t + " || " + b

# =========================
# EMBEDDING + CHUNKING
# =========================
# def chunk_by_tokens(text: str, tokenizer, max_tokens: int = 240, overlap: int = 24) -> List[str]:
#     tokens = tokenizer.encode(text, add_special_tokens=False)
#     if not tokens: return []
#     chunks, start = [], 0
#     while start < len(tokens):
#         end = min(start + max_tokens, len(tokens))
#         piece = tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()
#         if piece: chunks.append(piece)
#         if end == len(tokens): break
#         start = max(0, end - overlap)
#     return chunks

def chunk_by_tokens(text: str, tokenizer, max_tokens: int = 240, overlap: int = 24) -> List[str]:
    # Encode with the SAME tokenizer the model will use
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    max_tokens = max(8, int(max_tokens))
    overlap    = max(0, min(int(overlap), max_tokens - 1))
    stride     = max_tokens - overlap

    chunks = []
    for start in range(0, len(ids), stride):
        piece_ids = ids[start:start + max_tokens]
        # IMPORTANT: avoid tokenization changes on decode
        piece = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()
        if piece:
            chunks.append(piece)
    return chunks

def embed_documents(titles, bodies, model_name, e5_prefix=False, batch_size=64,
                    chunk_tokens=240, chunk_overlap=24) -> np.ndarray:
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer  # <-- use the SAME tokenizer

    # Determine the true model limit and reserve room for special tokens
    model_limit = getattr(tokenizer, "model_max_length", 512)
    # Some tokenizers report a huge sentinel for "no limit"; clamp sensibly
    if model_limit is None or model_limit > 50000:
        model_limit = 512
    hard_cap = int(model_limit) - 2  # <cls>/<sep> etc.

    # Effective chunk params (never exceed the model limit)
    eff_chunk   = min(int(chunk_tokens), hard_cap)
    eff_overlap = min(int(chunk_overlap), max(0, eff_chunk // 8))

    # Tell SentenceTransformers to truncate internally as a second safety net
    model.max_seq_length = eff_chunk

    docs = [build_doc_for_embedding(t, b, e5_prefix) for t, b in zip(titles, bodies)]

    chunk_texts, owners = [], []
    for i, d in enumerate(docs):
        chs = chunk_by_tokens(d, tokenizer, eff_chunk, eff_overlap) or [d]
        # Final safety check: if anything is still too long, hard-trim it
        safe = []
        for c in chs:
            if len(tokenizer.encode(c, add_special_tokens=True)) > model_limit:
                ids = tokenizer.encode(c, add_special_tokens=False)[:hard_cap]
                c = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            safe.append(c)
        for c in safe:
            chunk_texts.append(c); owners.append(i)

    chunk_embs = model.encode(
        chunk_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Length-weighted mean pooling across chunks (using THIS tokenizer)
    lens = np.array([len(tokenizer.encode(c, add_special_tokens=False)) for c in chunk_texts], dtype=np.float32)
    owners = np.array(owners, dtype=np.int32)
    D, dim = len(docs), chunk_embs.shape[1]
    out = np.zeros((D, dim), dtype=np.float32)
    weights = np.zeros(D, dtype=np.float32)
    for emb, w, oid in zip(chunk_embs, lens, owners):
        out[oid] += emb * max(w, 1.0)
        weights[oid] += max(w, 1.0)
    out /= weights[:, None]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out

# =========================
# CLUSTERING DIAGNOSTICS
# =========================
def compute_nn_same_cluster_rate(X: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    if len(X) == 0: return np.nan
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(X)), metric="cosine", n_jobs=-1)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)
    same = sum(any(labels[j]==labels[i] for j in neigh[1:]) for i, neigh in enumerate(indices))
    return same / len(X)

def representative_titles_for_cluster(X_emb, labels, kmeans, cluster_id, titles, top_n=5):
    idx = np.where(labels == cluster_id)[0]
    if idx.size == 0: return []
    c = kmeans.cluster_centers_[cluster_id]
    c = c / max(np.linalg.norm(c), 1e-8)
    sims = X_emb[idx] @ c
    top_idx = idx[np.argsort(sims)[::-1][:top_n]]
    return titles.iloc[top_idx].tolist()

def _unit_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return M / n

# =========================
# CATEGORY (QUERY) ENCODING
# =========================
def _encode_category_queries(model: SentenceTransformer, e5_prefix: bool) -> Tuple[np.ndarray, list, list, list]:
    """
    Build a query text for each category using name + description.
    For E5 models we prefix with 'query:'; for others we just pass the text.
    Returns:
      Q  (C x d) normalized,
      cids, names, raw_texts
    """
    texts, cids, names = [], [], []
    for cid, name, desc in L1_CATEGORIES:
        q = f"category: {name}. description: {desc}"
        if e5_prefix:
            q = "query: " + q
        texts.append(q); cids.append(cid); names.append(name)
    Q = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return Q, cids, names, texts

# =========================
# GCS HELPER
# =========================
def write_local_to_gcs(fs: gcsfs.GCSFileSystem, local_path: str, gcs_dest_path: str):
    with fs.open(gcs_dest_path, "wb") as fout, open(local_path, "rb") as fin:
        fout.write(fin.read())

# =========================
# MAIN
# =========================
def main():
    fs = gcsfs.GCSFileSystem()
    dataset = ds.dataset(GCS_SILVER_ROOT, filesystem=fs, format="parquet")
    df = dataset.to_table().to_pandas()

    # Ensure text columns exist for embedding
    for c in ["title","full_text"]: df[c] = df.get(c,"").fillna("")
    df["title_clean"] = df["title"].map(strip_html)
    df["full_text_clean"] = df["full_text"].map(strip_html)
    

        
    # # --- Drop exact duplicates by source_url if available ---
    # if "source_url" in df.columns:
    #     before = len(df)
    #     df = df.drop_duplicates(subset=["source_url"], keep="first")
    #     if len(df) != before:
    #         print(f"[dedupe] Dropped {before - len(df)} duplicate rows by source_url")

    # --- Signature-based near-exact dedupe (title + head of body) ---
    sig = make_signature(df, body_head_chars=800)
    dupe_mask = sig.duplicated(keep="first")
    if dupe_mask.any():
        df[dupe_mask].to_csv("dupes_exact.csv", index=False)
        df = df[~dupe_mask].copy()
        print(f"[dedupe] Removed {dupe_mask.sum()} near-exact duplicates by signature")

    # --- Same-day same-title same-company (optional safety net) ---
    if {"company_name","title_clean","release_date"}.issubset(df.columns):
        dts = pd.to_datetime(df["release_date"], errors="coerce", utc=True)
        day = dts.dt.date.astype("string")  # stable calendar-day key

        before = len(df)
        tmp = df.assign(_day=day)
        df  = (
            tmp.drop_duplicates(subset=["company_name","title_clean","_day"], keep="first")
            .drop(columns="_day")                       # safe now, _day exists in tmp
            .copy()
        )
        if len(df) != before:
            print(f"[dedupe] Removed {before - len(df)} duplicates by (company, title, day)")
    
    df = df.reset_index(drop=True) 

    
    # Embeddings (documents)
    X_emb = embed_documents(
        df["title_clean"].tolist(),
        df["full_text_clean"].tolist(),
        model_name=EMB_MODEL_NAME,
        e5_prefix=E5_PREFIX,
        batch_size=EMB_BATCH_SIZE,
        chunk_tokens=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Model selection (clustering) — diagnostics only
    scores, models = [], {}
    for k in CANDIDATE_K:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init="auto")
        labels = km.fit_predict(X_emb)
        sil = silhouette_score(X_emb, labels, metric="cosine")
        dbi = davies_bouldin_score(X_emb, labels)
        chi = calinski_harabasz_score(X_emb, labels)
        nnr = compute_nn_same_cluster_rate(X_emb, labels, k=5)
        scores.append((k, sil, dbi, chi, nnr))
        models[k] = (km, labels)
    sel = pd.DataFrame(scores, columns=["k","sil_cos","dbi","chi","nnr"])
    sel.to_csv(OUT_MODELSEL_CSV, index=False)

    best_k = int(sel.sort_values("sil_cos", ascending=False).iloc[0]["k"])
    kmeans, labels = models[best_k]
    df["cluster"] = labels

    # Cluster diagnostics
    sil_samples = silhouette_samples(X_emb, labels, metric="cosine")
    df["_silhouette"] = sil_samples

    centroids = _unit_rows(kmeans.cluster_centers_)
    diag_rows = []
    for c in range(best_k):
        mask = labels == c
        sims = X_emb[mask] @ centroids[c]
        diag_rows.append({
            "cluster": c,
            "size": int(mask.sum()),
            "sil_mean": float(np.mean(sil_samples[mask])) if mask.any() else float("nan"),
            "centroid_mean_cos": float(np.mean(sims)) if sims.size else float("nan")
        })
    pd.DataFrame(diag_rows).to_csv(OUT_DIAG_CSV, index=False)
    if best_k > 1:
        C = centroids @ centroids.T
        pd.DataFrame(C).to_csv(OUT_CENTSIM_CSV, index=False)

    # Cluster summaries (titles only)
    rows = []
    for c in range(best_k):
        rep = representative_titles_for_cluster(X_emb, labels, kmeans, c, df["title"], top_n=5)
        rows.append({"cluster": c, "size": int((labels==c).sum()), "sample_titles": " | ".join(rep)})
    pd.DataFrame(rows).to_csv(OUT_SUMMARY_CSV, index=False)

    # =========================
    # CATEGORY CONTEXT ENCODINGS
    # =========================
    cat_model = SentenceTransformer(EMB_MODEL_NAME)
    cat_limit = getattr(cat_model.tokenizer, "model_max_length", 512) or 512
    if cat_limit > 50000:  # guard against "very large" sentinel
        cat_limit = 512
    cat_model.max_seq_length = int(min(512, cat_limit))
    Q, cat_cids, cat_names, cat_texts = _encode_category_queries(cat_model, E5_PREFIX)  # (C x d)

    
    # Your helper should return category embeddings Q and meta; ensure normalization:
    #Q, cat_cids, cat_names, cat_texts = _encode_category_queries(cat_model, E5_PREFIX)  # Q: (C, d)
    # If _encode_category_queries doesn't normalize, do it here:
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

    # Ensure centroids are L2-normalized for cosine sims
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)  # (K, d)


    # -------- CLUSTER -> CATEGORY (context) --------
    # match each cluster centroid to the best category query
    # S has shape (C, K): rows = categories, cols = clusters
    S = Q @ centroids.T

    top_idx = np.argmax(S, axis=0)                    # (K,)
    s1 = S[top_idx, np.arange(S.shape[1])]            # (K,)

    S2 = S.copy()
    S2[top_idx, np.arange(S.shape[1])] = -np.inf
    second_idx = np.argmax(S2, axis=0)
    s2 = S2[second_idx, np.arange(S.shape[1])]

    cluster_best = []
    for j in range(best_k):
        top = int(top_idx[j])
        runner = int(second_idx[j])
        cluster_best.append({
            "cluster": int(j),
            "best_cat_idx": top,                     # numeric index into cat lists
            "best_cat_cid": cat_cids[top],          # string ID like "legal_regulatory_compliance"
            "best_cat_name": cat_names[top],
            "best_sim": float(s1[j]),
            "margin": float(s1[j] - s2[j]),
            "runner_up_idx": runner,
            "runner_up_cid": cat_cids[runner],
            "runner_up_name": cat_names[runner],
        })

    pd.DataFrame(cluster_best).to_csv(OUT_CATEGORY_SCORES_CSV, index=False)

    # -------- DOC -> CATEGORY (context) --------
    # final per-doc category via direct similarity to category queries
    sims_doc_cat = X_emb @ Q.T   # (N x C)
    best_idx = np.argmax(sims_doc_cat, axis=1)
    second = np.partition(sims_doc_cat, -2, axis=1)[:, -2] if sims_doc_cat.shape[1] >= 2 else np.zeros(len(df))
    best_scores = sims_doc_cat[np.arange(len(df)), best_idx]
    best_margin = best_scores - second

    chosen_cid   = [cat_cids[i]  for i in best_idx]
    chosen_name  = [cat_names[i] for i in best_idx]
    chosen_conf  = best_scores
    chosen_margin= best_margin

    df["_cat_cid"]   = chosen_cid
    df["_cat_name"]  = chosen_name
    df["_cat_sim"]   = chosen_conf
    df["_cat_margin"]= chosen_margin

    # decision flags
    def decide(sim, margin):
        if sim >= CAT_ASSIGN_MIN_SIM and margin >= CAT_LOW_MARGIN:
            return "auto_assign"
        elif sim >= CAT_ASSIGN_MIN_SIM:
            return "needs_review_margin"
        else:
            return "needs_review_low_sim"
    df["_decision"] = [decide(s, m) for s, m in zip(chosen_conf, chosen_margin)]

    # Learned category centroids from assigned docs (EMA-style mean)
    cat2vecs = defaultdict(list)
    for emb, cid in zip(X_emb, df["_cat_cid"]):
        cat2vecs[cid].append(emb)
    learned_centroids = {}
    for cid in cat_cids:
        V = np.stack(cat2vecs[cid], axis=0) if cid in cat2vecs and len(cat2vecs[cid]) else None
        if V is None:
            learned_centroids[cid] = None
        else:
            m = V.mean(axis=0)
            learned_centroids[cid] = (m / (np.linalg.norm(m) or 1.0)).tolist()

    # ======= Write ID-only assignments + category =======
    id_cols = [c for c in ["press_release_id","release_date","company_name","category","source","source_url"] if c in df.columns]
    assign = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    assign["cluster"]      = labels
    assign["_silhouette"]  = df["_silhouette"]
    assign["cid"]          = df["_cat_cid"]
    assign["category_name"]= df["_cat_name"]
    assign["cat_sim"]      = df["_cat_sim"].round(4)
    assign["cat_margin"]   = df["_cat_margin"].round(4)
    assign["decision"]     = df["_decision"]
    assign.to_csv(OUT_ASSIGNMENTS_CSV, index=False)
        
        # ======= Build review sets: Uncertain & Showcase =======
    # We’ll use cleaned text already in memory to create a short snippet
    txt_title = df.get("title_clean", pd.Series([""]*len(df))).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    txt_body  = df.get("full_text_clean", pd.Series([""]*len(df))).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    snippet   = txt_body.str.slice(0, SNIPPET_CHARS)

    # Join minimal view with IDs + category decision columns
    id_cols = [c for c in ["press_release_id","release_date","company_name","category","source","source_url"] if c in df.columns]
    view = assign.merge(
        pd.DataFrame({
            "press_release_id": df["press_release_id"] if "press_release_id" in df.columns else pd.RangeIndex(len(df)),
            "title":  txt_title,
            "snippet": snippet
        }),
        on="press_release_id",
        how="left"
    )

    # Uncertainty flag
    view["_is_uncertain"] = (view["cat_sim"].astype(float) < CAT_ASSIGN_MIN_SIM) | (view["cat_margin"].astype(float) < CAT_LOW_MARGIN)

    # ---- A) Uncertain queue (worst first) ----
    uncertain = view.sort_values(
        by=["_is_uncertain","cat_sim","cat_margin","_silhouette"],
        ascending=[False, True, True, True]
    )

    # Keep all truly-uncertain first, then fill remainder (if quota not met) with next-worst confident rows
    uncertain_top = uncertain[uncertain["_is_uncertain"]]
    if len(uncertain_top) < REVIEW_UNCERTAIN_TOTAL:
        need = REVIEW_UNCERTAIN_TOTAL - len(uncertain_top)
        filler = uncertain[~uncertain["_is_uncertain"]].head(need)
        uncertain_final = pd.concat([uncertain_top, filler], ignore_index=True)
    else:
        uncertain_final = uncertain_top.head(REVIEW_UNCERTAIN_TOTAL)

    uncertain_cols = id_cols + ["cid","category_name","cat_sim","cat_margin","decision","_silhouette","cluster","title","snippet"]
    uncertain_final[uncertain_cols].to_csv(OUT_REVIEW_UNCERTAIN_CSV, index=False, float_format="%.4f")

    # ---- B) Showcase (balanced high-confidence per category) ----
    show = view[
        (view["cat_sim"].astype(float) >= CAT_ASSIGN_MIN_SIM) &
        (view["cat_margin"].astype(float) >= SHOWCASE_MIN_MARGIN)
    ].copy()

    # Highest confidence first inside each category
    show = show.sort_values(by=["cid","cat_sim","cat_margin"], ascending=[True, False, False])

    showcase = (
        show.groupby("cid", group_keys=False)
            .head(SHOWCASE_PER_CATEGORY)
            .sort_values(by=["cid","cat_sim"], ascending=[True, False])
    )

    showcase_cols = id_cols + ["cid","category_name","cat_sim","cat_margin","decision","_silhouette","cluster","title","snippet"]
    showcase[showcase_cols].to_csv(OUT_REVIEW_SHOWCASE_CSV, index=False, float_format="%.4f")




    # ======= Review pack (balanced per cluster + global most-uncertain) =======
    if ENABLE_REVIEW_PACK:
        txt_title = df["title_clean"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        txt_body  = df["full_text_clean"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        snippet   = txt_body.str.slice(0, SNIPPET_CHARS)

        sim    = df["_cat_sim"].astype(float).clip(0, 1)
        sil    = df["_silhouette"].astype(float)
        margin = df["_cat_margin"].astype(float).clip(lower=0)

        # Uncertainty: how far below thresholds + low/negative silhouette
        over_sim = np.maximum(0.0, CAT_ASSIGN_MIN_SIM - sim)
        over_mar = np.maximum(0.0, CAT_LOW_MARGIN     - margin)
        under_sil = np.maximum(0.0, 0.5 - sil)  # penalize very low silhouette
        uncertainty = over_sim + 0.5*over_mar + 0.25*under_sil
        order = np.argsort(-uncertainty.values)  # highest uncertainty first

        # Per-cluster reps: top to centroid + low-silhouette edge cases
        sim_to_cent = (X_emb * centroids[labels]).sum(axis=1)

        packs = []
        seen = set()
        def add_idx(i: int):
            if i in seen: return
            seen.add(i)
            packs.append({
                "cluster": int(labels[i]),
                "cid": df["_cat_cid"].iat[i],
                "category_name": df["_cat_name"].iat[i],
                "press_release_id": df.at[i, "press_release_id"] if "press_release_id" in df.columns else "",
                "release_date":     df.at[i, "release_date"]     if "release_date" in df.columns else "",
                "company_name":     df.at[i, "company_name"]     if "company_name" in df.columns else "",
                "title":   txt_title.iat[i],
                "snippet": snippet.iat[i],
                "_silhouette": float(sil.iat[i]),
                "cat_sim":   float(sim.iat[i]),
                "cat_margin":float(margin.iat[i]),
                "decision":  df["_decision"].iat[i] if "_decision" in df.columns else ""
            })

        for c in range(best_k):
            mask_idx = np.where(labels == c)[0]
            if mask_idx.size == 0: continue
            top_n  = max(1, REVIEW_PER_CLUSTER // 2)
            edge_n = max(1, REVIEW_PER_CLUSTER - top_n)
            top_idx = mask_idx[np.argsort(sim_to_cent[mask_idx])[::-1][:top_n]]
            low_idx = mask_idx[np.argsort(sil.values[mask_idx])[:edge_n]]
            for i in np.concatenate([top_idx, low_idx]):
                add_idx(int(i))

        # Also include the most uncertain across all docs (cap via env; default 200)
        REVIEW_UNCERTAIN_EXTRA = int(os.getenv("REVIEW_UNCERTAIN_EXTRA", "200"))
        for i in order[:REVIEW_UNCERTAIN_EXTRA]:
            add_idx(int(i))

        pd.DataFrame(packs).to_csv(OUT_REVIEW_PACK_CSV, index=False, float_format="%.4f")

    # ======= categories.json (context + learned centroids) =======
    categories = []
    for cid, name, desc in L1_CATEGORIES:
        entry = {
            "cid": cid,
            "label": name,
            "description": desc,
            # prototype = query embedding (context), centroid = mean of assigned docs (if any)
            "prototype": Q[cat_cids.index(cid)].tolist(),
            "centroid":  learned_centroids.get(cid),
            "active": True,
            "version": 1
        }
        categories.append(entry)
    categories_json = {
        "run_ts": RUN_TS,
        "model": {
            "name": EMB_MODEL_NAME,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
            "e5_prefix": E5_PREFIX
        },
        "thresholds": {
            "assign_min_sim": CAT_ASSIGN_MIN_SIM,
            "low_margin": CAT_LOW_MARGIN
        },
        "categories": categories
    }
    with open(OUT_CATEGORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(categories_json, f, ensure_ascii=False, indent=2)

    # --- Upload artifacts to GCS (timestamped folder) ---
    dst_dir = f"{OUTPUT_GCS_PREFIX}/{RUN_TS}"
    for local_name in [
        OUT_MODELSEL_CSV, OUT_DIAG_CSV, OUT_CENTSIM_CSV, OUT_SUMMARY_CSV,
        OUT_ASSIGNMENTS_CSV, OUT_REVIEW_PACK_CSV, OUT_CATEGORIES_JSON, OUT_CATEGORY_SCORES_CSV,
        OUT_REVIEW_UNCERTAIN_CSV, OUT_REVIEW_SHOWCASE_CSV
    ]:
        if os.path.exists(local_name := local_name):
            write_local_to_gcs(fs, local_name, f"{dst_dir}/{local_name}")

    print(f"\nAll artifacts saved locally and to {dst_dir}")

if __name__ == "__main__":
    main()
