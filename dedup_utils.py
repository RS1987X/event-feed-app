# dedup_utils.py
from __future__ import annotations
import hashlib, re, unicodedata
from typing import Dict, Optional, Tuple
import pandas as pd

# ---------- helpers ----------

def _normalize_text(s: pd.Series) -> pd.Series:
    """Lower, strip accents, collapse whitespace, keep word chars & nordic letters."""
    def _norm_one(x: str) -> str:
        x = str(x or "")
        # strip accents but keep åäöæøß etc. by normalizing, then recompose
        x = unicodedata.normalize("NFKD", x)
        x = "".join(ch for ch in x if not unicodedata.combining(ch))
        x = unicodedata.normalize("NFKC", x).lower()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^\w åäöæøéèêëáàäüúùöß\-]", "", x)
        return x.strip()
    return s.astype(str).map(_norm_one)

def make_content_hash(
    df: pd.DataFrame,
    title_col: str = "title_clean",
    body_col: str  = "full_text_clean",
) -> pd.Series:
    txt = df[title_col].fillna("") + " " + df[body_col].fillna("")
    norm = _normalize_text(txt)
    return norm.apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())

def make_signature(
    df: pd.DataFrame,
    title_col: str = "title_clean",
    body_col: str  = "full_text_clean",
    body_head_chars: int = 800,
) -> pd.Series:
    """Near-exact signature: normalized title + head(body). Good for EN/SV duplicates from same source."""
    title = df[title_col].fillna("").astype(str)
    body  = df[body_col].fillna("").astype(str).str.slice(0, body_head_chars)
    sig_src = title.str.cat(body, sep=" || ")
    norm = _normalize_text(sig_src)
    return norm.apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())

# ---------- main API ----------

def dedup_df(
    df: pd.DataFrame,
    *,
    id_col: Optional[str]  = "press_release_id",
    title_col: str         = "title_clean",
    body_col: str          = "full_text_clean",
    company_col: str       = "company_name",
    ts_col: Optional[str]  = "release_date",
    # toggles:
    use_signature: bool    = True,
    body_head_chars: int   = 800,
    use_same_day_title_company: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Multi-step dedup:
      0) sort by ts (so keep='last' favors newest)
      1) drop duplicate IDs
      2) exact content hash (title+body)
      3) near-exact signature (title + head(body))
      4) same-day same-company same-title (safety net)

    Returns: (deduped_df, stats)
    """
    stats: Dict[str, int] = {
        "total": len(df),
        "drop_id_dupes": 0,
        "drop_content_dupes": 0,
        "drop_signature_dupes": 0,
        "drop_same_day_title_company": 0,
        "kept": 0,
    }

    out = df

    # 0) Sort by timestamp to make 'keep="last"' prefer the newest item
    if ts_col and ts_col in out.columns:
        try:
            out = out.assign(_ts=pd.to_datetime(out[ts_col], errors="coerce", utc=True))
            out = out.sort_values("_ts")
        except Exception:
            out = out.copy()
            out["_ts"] = None
    else:
        out = out.copy()
        out["_ts"] = None

    # 1) ID-level dupes
    if id_col and id_col in out.columns:
        before = len(out)
        out = out.drop_duplicates(subset=[id_col], keep="last")
        stats["drop_id_dupes"] = before - len(out)

    # 2) Exact content hash
    before = len(out)
    out["_content_sha1"] = make_content_hash(out, title_col=title_col, body_col=body_col)
    out = out.drop_duplicates(subset=["_content_sha1"], keep="last")
    stats["drop_content_dupes"] = before - len(out)

    # 3) Signature near-exact (title + head(body))
    if use_signature and {title_col, body_col}.issubset(out.columns):
        before = len(out)
        out["_sig_sha1"] = make_signature(out, title_col=title_col, body_col=body_col, body_head_chars=body_head_chars)
        out = out.drop_duplicates(subset=["_sig_sha1"], keep="last")
        stats["drop_signature_dupes"] = before - len(out)

    # 4) Same-day / same-company / same-title safety net
    if use_same_day_title_company and {"company_name", title_col}.issubset(out.columns):
        before = len(out)
        if ts_col and ts_col in out.columns:
            dts = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
            day = dts.dt.date.astype("string")
        else:
            # If no ts, use a single bucket to avoid accidental drops
            day = pd.Series(["__no_ts__"] * len(out), index=out.index, dtype="string")

        tmp = out.assign(_day=day)
        out = (
            tmp.drop_duplicates(subset=[company_col, title_col, "_day"], keep="first")
               .drop(columns=["_day"])
               .copy()
        )
        stats["drop_same_day_title_company"] = before - len(out)

    # cleanup temp cols
    for c in ["_content_sha1", "_sig_sha1", "_ts"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    stats["kept"] = len(out)
    return out, stats
