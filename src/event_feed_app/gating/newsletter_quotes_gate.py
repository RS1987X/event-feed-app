# newsletter_gate.py
from __future__ import annotations
import re
from typing import Tuple
import pandas as pd

# --- Title/body cues that characterize newsletters / quote dumps ---

NEWSLETTER_T = re.compile(
    r"(?i)\b("
    r"newsletter|news\s*letter|"
    r"weekly\s+(?:update|summary|wrap|overview|report)|"
    r"weekly\s+summary\s+alert|week(?:ly)?\s+in\s+review|"
    r"monthly\s+(?:update|summary|newsletter|report)|"
    r"market\s+(?:update|summary|wrap|review)|"
    r"investor\s+newsletter"
    r")\b"
)

# Title looks like a quote/price email
QUOTE_T = re.compile(
    r"(?i)\b("
    r"stock\s+quote|price\s+update|daily\s+prices?|closing\s+prices?|"
    r"end\s+of\s+day\s+(?:stock|price)|eod\s+prices?|quote\s+summary"
    r")\b"
)

# Body looks like tabular quote dump (rows of weekday/price/percent/volume)
QUOTE_BODY = re.compile(
    r"(?is)(monday|tuesday|wednesday|thursday|friday).{0,160}?"
    r"(opening\s+price|last\s+trade|percent\s+change|day\s+high|day\s+low|volume)"
)

# Strong “do NOT gate” (earnings/reporting) signals — apply to TITLE ONLY
STRONG_ABSTAIN_RE = re.compile(
    r"(?i)\b("
    r"interim\s+report|half[-\s]?year(?:\s+report)?|year[-\s]?end(?:\s+report)?|"
    r"full[-\s]?year(?:\s+report)?|annual\s+report|"
    r"q[1-4]\b|first|second|third|fourth\s+quarter|quarterly\s+(?:report|results?)|"
    r"results\s+(?:for|q[1-4])"
    r")\b"
)

def _title_looks_like_newsletter(title: str) -> bool:
    t = title or ""
    return bool(NEWSLETTER_T.search(t) or QUOTE_T.search(t))

def _has_strong_signal_in_title(title: str) -> bool:
    return bool(STRONG_ABSTAIN_RE.search(title or ""))

def apply_newsletter_gate(
    df: pd.DataFrame,
    title_col: str = "title_clean",
    body_col: str = "full_text_clean",
    drop: bool = True,
    *,
    title_only: bool = True,             # << new knob: detect newsletter via title only
    keep_body_quote_gate: bool = True,   # << keep hard body quote-dump gate
) -> Tuple[pd.DataFrame, dict]:
    """
    Flags rows that look like newsletters/quote dumps, unless strong competing
    signals are present (earnings/reporting) in the TITLE.

    If drop=True: returns df with those rows removed.
    If drop=False: keeps rows and adds a boolean column 'gate_newsletter'.
    """

    titles = df[title_col].fillna("").astype(str)
    bodies = df[body_col].fillna("").astype(str)

    gate_flag = []
    for t, b in zip(titles, bodies):
        # 1) Hard body quote-dump gate (always on by default)
        if keep_body_quote_gate and QUOTE_BODY.search(b):
            gate_flag.append(True)
            continue

        # 2) Title-only newsletter/quote detection
        looks_like = _title_looks_like_newsletter(t) if title_only else (_title_looks_like_newsletter(t) or bool(QUOTE_BODY.search(b)))

        # 3) Earnings/reporting allow-list — TITLE only
        abstain = _has_strong_signal_in_title(t)

        gate_flag.append(looks_like and not abstain)

    n_gated = int(sum(gate_flag))
    stats = {"total": int(len(df)), "gated": n_gated, "kept": int(len(df) - n_gated)}

    if drop:
        df_out = df.loc[~pd.Series(gate_flag, index=df.index)].copy()
        return df_out, stats

    df_out = df.copy()
    df_out["gate_newsletter"] = gate_flag
    return df_out, stats

__all__ = [
    "apply_newsletter_gate",
    "NEWSLETTER_T",
    "QUOTE_T",
    "QUOTE_BODY",
    "STRONG_ABSTAIN_RE",
]
