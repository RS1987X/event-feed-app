# event_taxonomies/rules/context.py
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ---------- Normalization & tokenization ----------

def _normalize_quotes(s: str) -> str:
    return (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("–", "-")
         .replace("—", "-")
    )

def norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    s = _normalize_quotes(s)
    # collapse whitespace and lowercase with casefold (better for intl)
    return " ".join(s.split()).casefold()

TOKEN_RE = re.compile(r"[a-z0-9åäöæøéèüïóáç\-]+", re.IGNORECASE)

def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(norm(s))

# ---------- Regex compilers ----------

def rx(words: List[str]) -> re.Pattern:
    """Compile whole-word OR regex for a list of phrases (case-insensitive)."""
    ws = [w for w in words if w]
    if not ws:
        return re.compile(r"(?!x)x")  # match nothing
    return re.compile(r"\b(" + "|".join(map(re.escape, ws)) + r")\b", re.IGNORECASE)

@lru_cache(maxsize=4096)
def rx_any(words_tuple: tuple) -> re.Pattern:
    """
    Back-compat helper used by older rules: same as rx(list(...)), but cached on tuple.
    Keep it so existing imports don't break.
    """
    words = [w for w in words_tuple if w]
    if not words:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)

# ---------- Presence helpers ----------

def any_present(text: str, terms: List[str]) -> List[str]:
    """Return list of terms present as whole-phrase matches (order-preserving, case-insensitive)."""
    t = norm(text)
    hits: List[str] = []
    seen = set()
    for w in terms:
        if not w:
            continue
        pat = r"\b" + re.escape(norm(w)) + r"\b"
        if re.search(pat, t, flags=re.IGNORECASE):
            lw = norm(w)
            if lw not in seen:
                seen.add(lw)
                hits.append(lw)
    return hits

@lru_cache(maxsize=4096)
def _build_morph_phrase_pattern(phrase: str, min_stem: int = 3) -> str:
    """
    Build a regex that matches the phrase allowing inflectional endings per token.
    - Adds \w* to tokens with length >= min_stem.
    - Preserves token order.
    - Allows spaces / hyphens / underscores between tokens.
    """
    toks = tokenize(phrase)
    if not toks:
        esc = re.escape(_normalize_quotes(unicodedata.normalize("NFKC", phrase)))
        return rf"\b{esc}\b"
    parts = []
    for tkn in toks:
        esc = re.escape(tkn)
        parts.append(esc + r"\w*" if len(tkn) >= min_stem else esc)
    inner = r"(?:[\s\-_]+)".join(parts)
    return rf"\b{inner}\b"

def any_present_morph(text: str, terms: List[str], min_stem: int = 3) -> List[str]:
    """
    Morph-friendly presence: allows simple inflection on each token (via \w*).
    Returns canonical terms (as provided), lowercased, order-preserving.
    """
    t = norm(text)
    hits: List[str] = []
    seen = set()
    for term in terms:
        if not term:
            continue
        pat = _build_morph_phrase_pattern(term, min_stem=min_stem)
        if re.search(pat, t, flags=re.IGNORECASE):
            lw = norm(term)
            if lw not in seen:
                seen.add(lw)
                hits.append(lw)
    return hits

def any_present_inflected(text: str, stems: List[str], suffix_group: str) -> List[str]:
    """
    Stem + restricted endings (e.g., EN: (?:|s|ed|ing), SV: (?:|ar|ade|at|er|te|t)).
    Returns stems (lowercased) that matched, order-preserving.
    """
    t = norm(text)
    hits: List[str] = []
    seen = set()
    for stem in stems:
        if not stem:
            continue
        pat = rf"\b{re.escape(stem)}{suffix_group}\b"
        if re.search(pat, t, flags=re.IGNORECASE):
            lw = stem.lower()
            if lw not in seen:
                seen.add(lw)
                hits.append(lw)
    return hits

# ---------- Proximity helpers ----------

def proximity_cooccurs(text: str, set_a: List[str], set_b: List[str], window: int = 12
                      ) -> Tuple[bool, Tuple[str, str]]:
    """
    True if first token of any term in set_a occurs within `window` tokens of first token of any term in set_b.
    """
    toks = tokenize(text)
    pos_a = [i for i, tok in enumerate(toks) if tok in {tokenize(x)[0] for x in set_a if tokenize(x)}]
    pos_b = [i for i, tok in enumerate(toks) if tok in {tokenize(x)[0] for x in set_b if tokenize(x)}]
    for ia in pos_a:
        for ib in pos_b:
            if abs(ia - ib) <= window:
                return True, (toks[ia], toks[ib])
    return False, ("","")

def proximity_cooccurs_morph(text: str, set_a: List[str], set_b: List[str], window: int = 12, min_stem: int = 3
                            ) -> Tuple[bool, Tuple[str, str]]:
    """
    Morph-aware proximity: a phrase matches if each token equals the text token OR
    the text token startswith the phrase token (for tokens >= min_stem).
    """
    toks = tokenize(text)

    def positions_morph(terms: List[str]) -> Dict[str, List[int]]:
        pos: Dict[str, List[int]] = {}
        for term in terms:
            ptoks = tokenize(term)
            if not ptoks:
                continue
            m = len(ptoks)
            hits: List[int] = []
            for i in range(0, len(toks) - m + 1):
                ok = True
                for j, pt in enumerate(ptoks):
                    tt = toks[i + j]
                    if len(pt) >= min_stem:
                        if not tt.startswith(pt):
                            ok = False; break
                    else:
                        if tt != pt:
                            ok = False; break
                if ok:
                    hits.append(i)
            if hits:
                pos[term] = hits
        return pos

    pos_a = positions_morph(set_a)
    pos_b = positions_morph(set_b)

    for ta, ia_list in pos_a.items():
        for tb, ib_list in pos_b.items():
            for ia in ia_list:
                for ib in ib_list:
                    if abs(ia - ib) <= window:
                        return True, (ta, tb)
    return False, ("", "")

# ---------- Simple heuristic person detector (fallback) ----------

def detect_person_name(original_text: str) -> Optional[str]:
    pat = r"\b[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+(?:\s+[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+)+\b"
    m = re.search(pat, original_text)
    return m.group(0) if m else None


# Title-only proximity (stricter window)
def title_pair_close(ctx: Ctx, set_a, set_b, window: int = 8, min_stem: int = 4) -> bool:
    return proximity_cooccurs_morph(ctx.title or "", set_a, set_b, window=window, min_stem=min_stem)[0]

# Body corroboration (looser; any of: pair close OR both sets present)
def body_corroborates(ctx: Ctx, set_a, set_b, window: int = 16, min_stem: int = 4) -> bool:
    if proximity_cooccurs_morph(ctx.body or "", set_a, set_b, window=window, min_stem=min_stem)[0]:
        return True
    has_a = bool(any_present_morph(ctx.body or "", list(set_a), min_stem=min_stem))
    has_b = bool(any_present_morph(ctx.body or "", list(set_b), min_stem=min_stem))
    return has_a and has_b


# ---------- Context container ----------

@dataclass
class Ctx:
    title: str
    body: str
    text_norm: str
    text_original: str
