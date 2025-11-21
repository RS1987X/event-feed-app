# build_labeling_pool.py
import pandas as pd
import numpy as np
from collections import defaultdict

ASSIGNMENTS_CSV = "assignments.csv"     # input: full predictions
LABEL_CSV       = "labeling_pool"   # output: for manual labeling

# --- CONFIG ---
TARGET_PER_CLASS = 10            # aim per predicted category
MIN_PER_LANG_PER_CLASS = 2       # try to include at least this many per language per class
# margin bands (tune after first run)
HARD_MAR = 0.015
MID_MAR  = 0.035
EASY_MAR = 0.07

# --- Load predictions ---
df = pd.read_csv(ASSIGNMENTS_CSV)

# Prefer group representatives to avoid dup-like items (if available)
if "is_group_rep" in df.columns:
    df = df[df["is_group_rep"] == True].copy()

# Required/expected columns (create if missing so code is robust)
for c in ["cid","category_name","cat_sim","cat_margin","language","title","snippet"]:
    if c not in df.columns:
        df[c] = ""

# Normalize language codes
def norm_lang(s: str) -> str:
    s = str(s).lower()
    for p in ["en","sv","fi","da","no","de","fr","es","it","nl","pt", "is"]:
        if s.startswith(p): return p
    return "other"

df["lang_norm"] = df["language"].map(norm_lang)

# Margin bands for diversity
def band(m: float) -> str:
    try:
        m = float(m)
    except Exception:
        m = 0.0
    if m < HARD_MAR: return "hard"
    if m < MID_MAR:  return "mid"
    if m < EASY_MAR: return "easy"
    return "very_easy"

df["margin_band"] = df["cat_margin"].apply(band)

# Helper: balanced sampling for one predicted category
def sample_category(block: pd.DataFrame, k: int) -> pd.DataFrame:
    # emphasize hard/mid
    weights = {"hard": 0.40, "mid": 0.35, "easy": 0.20, "very_easy": 0.05}
    pieces = []
    remaining = k

    for b in ["hard","mid","easy","very_easy"]:
        want = max(1, int(round(weights[b] * k)))
        take = block[block["margin_band"] == b]
        if len(take) > 0:
            grab = min(want, len(take))
            pieces.append(take.sample(grab, random_state=42))
            remaining -= grab

    # top-up from what’s left (prioritize lower margins)
    if remaining > 0:
        # order: hard -> mid -> easy -> very_easy by mapping
        mb_map = {"hard":0,"mid":1,"easy":2,"very_easy":3}
        used_idx = pd.Index([]) if not pieces else pd.concat(pieces).index
        rest = (block
                .drop(used_idx, errors="ignore")
                .assign(_mb_order=lambda x: x["margin_band"].map(mb_map))
                .sort_values(["_mb_order","cat_margin"], ascending=[True, True]))
        if len(rest) > 0:
            pieces.append(rest.head(remaining))

    samp = pd.concat(pieces) if pieces else block.sample(min(k, len(block)), random_state=42)

    # light language balancing
    want_per_lang = defaultdict(lambda: 0)
    for lang in samp["lang_norm"].unique():
        want_per_lang[lang] = MIN_PER_LANG_PER_CLASS

    for lang, want in want_per_lang.items():
        have = int((samp["lang_norm"] == lang).sum())
        need = max(0, want - have)
        if need > 0:
            candidates = block[(block["lang_norm"] == lang) & (~block.index.isin(samp.index))]
            if len(candidates) > 0:
                add = candidates.sample(min(need, len(candidates)), random_state=42)
                # drop highest-margin (easiest) rows to keep sample size ≈ k
                drop = samp.sort_values("cat_margin", ascending=False).head(len(add))
                samp = pd.concat([samp.drop(drop.index), add])

    return samp.head(k)

# Build per-class samples
samples = []
for pred_cid, grp in df.groupby("cid", sort=False):
    k = min(TARGET_PER_CLASS, len(grp))
    if k <= 0: 
        continue
    samples.append(sample_category(grp, k))

label_pool = pd.concat(samples).copy()

# Rename predictions for clarity
label_pool = label_pool.rename(columns={
    "cid":"pred_cid",
    "category_name":"pred_category",
    "cat_sim":"pred_prob",
    "cat_margin":"pred_margin",
})

# Add blank labeling columns
label_pool["true_cid"]   = ""    # <- fill this in manually
label_pool["true_notes"] = ""


# Order columns so you see title + snippet + labeling fields right away
keep_cols_order = [
    # identifiers / metadata (if present)
    "press_release_id","release_date","company_name","source","source_url",
    "language","lang_norm",
    # labeling fields (to fill out)
    "true_cid","true_notes","pred_cid",
    # text to read
    "title","snippet",
    # model predictions / diagnostics
    "pred_category","pred_prob","pred_margin","margin_band","decision",
    "top3_cids","top3_probs",
]

# Reorder columns
keep_cols = [c for c in keep_cols_order if c in label_pool.columns]
label_pool = label_pool[keep_cols].sort_values(["pred_cid","margin_band","pred_margin"])
import datetime
ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
path = f"data/labeled/{LABEL_CSV}_{ts}.csv"
# Write out
label_pool.to_csv(path, index=False, encoding="utf-8")
print(f"[ok] wrote {len(label_pool)} rows to {LABEL_CSV}")

