# housekeeping_gate.py
import re
import pandas as pd
from typing import Optional, Tuple, Dict
import re

# Title-level triggers (don’t rely on body/full_text)
SUB_TITLE_RE = re.compile(
    r"(?i)\b("
    r"opt[-\s]?in|"                          # Opt-In / Opt In
    r"(?:thank you|thanks).{0,30}subscrib|"  # “Thank you for subscribing…”
    r"you(?:'re| are) subscribed|"           # “You’re subscribed”
    r"registered your subscription|"         # “We’ve registered your subscription”
    r"subscription (?:confirmed|confirmation)|"
    r"activate (?:your )?(?:release )?subscription|"  # “Activate your release subscription”
    r"confirm (?:your )?subscription|"
    r"verify (?:your )?subscription"
    r")\b"
)

# Full-text triggers (body+title combined) you already had
SUB_RE = re.compile(
    r"(?i)(?:confirm(?: your| the)? (?:subscription|registration|email)|"
    r"verify (?:your )?(?:subscription|request)|"
    r"welcome(?: as)?(?: a)? subscriber|"
    r"subscription (?:confirmed|confirmation)|"
    r"activate(?: your)? (?:subscription|registration|email)|"
    r"manage(?: your)? subscriptions|"
    r"bekräfta din prenumeration|verifiera din prenumeration|välkommen som prenumerant|"
    r"bekräfta e-postadressen|aktivera.*prenumeration|hantera dina prenumerationer|avsluta prenumeration|"
    r"aktiver.*(?:abonnement|release)|bekræft registrering|nyhedsservice)"
)

# Broader account/security hygiene
ACCOUNT_HYGIENE_RE = re.compile(
    r"(?i)(?:security alert|privacy checkup|privacy settings|google account|welcome to google|your new google account)"
)

QUOTE_PREFIX_RE = re.compile(r"(?i)^stock quote notification")

# Allow-list for real news in titles (unchanged idea)
ALLOW_TITLE_RE = re.compile(
    r"(?i)(?:cancels?|announc(?:es|ed|ing)?|enters|agrees|acquire|acquisition|appoints?|"
    r"earnings|Q[1-4]\b|interim report|dividend|guidance|rights issue|capital raise|"
    r"orders?|contract|partnership|incident|strike|industrial action)"
)

# def apply_housekeeping_filter(
#     df: pd.DataFrame,
#     title_col: str = "title",
#     body_col: str = "full_text",
#     *,
#     use_clean_columns: bool = True,
#     strip_html_fn=None,
#     reason_col: Optional[str] = None,
# ) -> Tuple[pd.DataFrame, Dict[str, int]]:
#     """
#     Vectorized skip-filter for subscription/welcome/activation/account/quote emails.

#     - If use_clean_columns is True and `<col>_clean` exists, those are used.
#     - Otherwise, falls back to raw columns (optionally applying strip_html_fn if provided).
#     - If reason_col is set, writes a short reason string per skipped row (else no side-effects).

#     Returns: (filtered_df, stats_dict)
#     """
#     for c in (title_col, body_col):
#         if c not in df.columns:
#             df[c] = ""
#         df[c] = df[c].fillna("")




#     # choose series for matching
#     if use_clean_columns and all(
#         c in df.columns for c in (f"{title_col}_clean", f"{body_col}_clean")
#     ):
#         title_s = df[f"{title_col}_clean"].astype(str)
#         body_s = df[f"{body_col}_clean"].astype(str)
#     else:
#         if strip_html_fn is None:
#             strip_html_fn = lambda x: x
#         title_s = df[title_col].map(strip_html_fn).astype(str)
#         body_s = df[body_col].map(strip_html_fn).astype(str)

#     text_s  = title_s.str.cat(body_s, sep=" ", na_rep="")

#     # NEW: title-only triggers
#     title_hit = title_s.str.contains(SUB_TITLE_RE, regex=True, na=False)

#     housekeeping_match = (
#         title_hit |
#         title_s.str.contains(QUOTE_PREFIX_RE, regex=True, na=False) |
#         title_s.str.contains(ACCOUNT_HYGIENE_RE, regex=True, na=False) |
#         text_s.str.contains(SUB_RE, regex=True, na=False)
#     )

#     allow_hit = title_s.str.contains(ALLOW_TITLE_RE, regex=True, na=False)
#     skip_mask = housekeeping_match & ~allow_hit


#     if reason_col:
#         # annotate reasons only for skipped rows
#         reason = pd.Series(index=df.index, dtype=object)
#         reason.loc[title_s.str.contains(QUOTE_PREFIX_RE, regex=True, na=False)] = "market_quote_alert"
#         reason.loc[title_s.str.contains(ACCOUNT_HYGIENE_RE, regex=True, na=False)] = "account_security"
#         reason.loc[(text_s.str.contains(SUB_RE, regex=True, na=False)) & ~allow_hit & reason.isna()] = "subscription_or_welcome"
#         df[reason_col] = reason

#     filtered = df.loc[~skip_mask].copy()
#     stats = {"skipped": int(skip_mask.sum()), "kept": int((~skip_mask).sum()), "total": int(len(df))}
#     return filtered, stats

URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
WS_RE    = re.compile(r"\s+")
from bs4 import BeautifulSoup
import os, json, html as ihtml

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

# SEPARATOR_PATTERNS = [
#     r"[-]{5,}",    # 5 or more dashes
#     r"[=]{5,}",    # 5 or more equals
#     r"[_]{5,}",    # 5 or more underscores
#     r"[\*]{5,}",   # 5 or more asterisks
# ]

# def remove_separators(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     out = text
#     for pat in SEPARATOR_PATTERNS:
#         out = re.sub(pat, " ", out)
#     return re.sub(r"\s+", " ", out).strip()

def apply_housekeeping_filter(
    df: pd.DataFrame,
    title_col: str = "title",
    body_col: str = "full_text",
    *,
    reason_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Skip-filter for subscription/welcome/activation/account/quote press releases.

    - Always ensures *_clean columns exist and strips HTML.
    - Works on *_clean columns internally.
    - Optionally annotates skipped rows with reason_col.
    """
    # Ensure base cols exist
    for c in (title_col, body_col):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    # Always (re)compute clean cols
    df[f"{title_col}_clean"] = df[title_col].map(strip_html).astype(str)
    df[f"{body_col}_clean"]  = df[body_col].map(strip_html).astype(str)


    #remove separator lines (e.g. "-----") that can confuse regexes
    sep_re = r"[-_*=]{5,}"
    df[f"{title_col}_clean"] = (
        df[f"{title_col}_clean"].str.replace(sep_re, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df[f"{body_col}_clean"] = (
        df[f"{body_col}_clean"].str.replace(sep_re, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    
    # Use cleaned text for filtering
    title_s = df[f"{title_col}_clean"]
    body_s  = df[f"{body_col}_clean"]
    text_s  = title_s.str.cat(body_s, sep=" ", na_rep="")

    # --- housekeeping rules (fill in your regexes) ---
    title_hit = title_s.str.contains(SUB_TITLE_RE, regex=True, na=False)
    housekeeping_match = (
        title_hit |
        title_s.str.contains(QUOTE_PREFIX_RE, regex=True, na=False) |
        title_s.str.contains(ACCOUNT_HYGIENE_RE, regex=True, na=False) |
        text_s.str.contains(SUB_RE, regex=True, na=False)
    )
    allow_hit = title_s.str.contains(ALLOW_TITLE_RE, regex=True, na=False)
    skip_mask = housekeeping_match & ~allow_hit

    if reason_col:
        reason = pd.Series(index=df.index, dtype=object)
        reason.loc[title_s.str.contains(QUOTE_PREFIX_RE, regex=True, na=False)] = "market_quote_alert"
        reason.loc[title_s.str.contains(ACCOUNT_HYGIENE_RE, regex=True, na=False)] = "account_security"
        reason.loc[(text_s.str.contains(SUB_RE, regex=True, na=False)) & ~allow_hit & reason.isna()] = "subscription_or_welcome"
        df[reason_col] = reason

    filtered = df.loc[~skip_mask].copy()
    stats = {"skipped": int(skip_mask.sum()), "kept": int((~skip_mask).sum()), "total": int(len(df))}
    return filtered, stats