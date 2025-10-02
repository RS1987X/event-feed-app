# event_feed_app/taxonomy/rules/textutils.py
from functools import lru_cache
import re
from typing import List, Dict, Tuple
from event_feed_app.utils.text import norm, tokenize

@lru_cache(maxsize=4096)
def _build_morph_phrase_pattern(phrase: str, min_stem: int = 3) -> str:
    toks = tokenize(phrase)
    if not toks: return rf"\b{re.escape(phrase)}\b"
    parts=[(re.escape(t) + r"\w*") if len(t)>=min_stem else re.escape(t) for t in toks]
    inner = r"(?:[\s\-_]+)".join(parts)
    return rf"\b{inner}\b"

def any_present_morph(text: str, terms: List[str], min_stem: int = 3) -> List[str]:
    t = norm(text); hits=[]; seen=set()
    for term in terms or []:
        if not term: continue
        pat = _build_morph_phrase_pattern(term, min_stem=min_stem)
        if re.search(pat, t) and (val:=term.casefold()) not in seen:
            seen.add(val); hits.append(val)
    return hits

def any_present_inflected(text: str, stems: List[str], suffix_group: str) -> List[str]:
    t = norm(text); hits=[]; seen=set()
    for stem in stems or []:
        if not stem: continue
        if re.search(rf"\b{re.escape(stem)}{suffix_group}\b", t, re.IGNORECASE):
            val = stem.casefold()
            if val not in seen: seen.add(val); hits.append(val)
    return hits

def proximity_cooccurs(text: str, set_a: List[str], set_b: List[str], window: int = 12) -> Tuple[bool, Tuple[str, str]]:
    toks = tokenize(text)
    idx = {i: tok for i, tok in enumerate(toks)}
    def first_positions(terms: List[str]) -> Dict[str,List[int]]:
        out={}
        for term in terms or []:
            tt = tokenize(term)
            if tt: out[term] = [i for i, tok in idx.items() if tok == tt[0]]
        return out
    pa, pb = first_positions(set_a), first_positions(set_b)
    for ta, ia_list in pa.items():
        for tb, ib_list in pb.items():
            for ia in ia_list:
                for ib in ib_list:
                    if abs(ia-ib) <= window:
                        return True, (ta, tb)
    return False, ("","")

def proximity_cooccurs_morph(text: str, set_a: List[str], set_b: List[str], window: int = 12, min_stem: int = 3) -> Tuple[bool, Tuple[str,str]]:
    """
    Return whether any phrase from `set_a` appears within `window` tokens of any phrase
    from `set_b` in `text`, using simple morphological (prefix) matching.

    Token/phrase matching:
      - The input `text` is tokenized via `tokenize(text)`.
      - Each term/phrase in `set_a` and `set_b` is tokenized via `tokenize(term)`.
      - A phrase matches a span of text tokens if all its tokens match in order:
          * If len(pattern_token) >= `min_stem`, the text token must start with the
            pattern token (prefix match), e.g. "regulat" matches "regulation".
          * Otherwise (short tokens), the text token must be exactly equal.
      - All matching is performed on tokens; behavior depends on `tokenize`.

    Proximity:
      - For every match of a `set_a` phrase and every match of a `set_b` phrase,
        measure distance as the absolute difference between their start token indices.
      - If `abs(start_a - start_b) <= window`, the function returns True.

    Args:
        text: The input text to search.
        set_a: Phrases/terms for the first side of the co-occurrence test.
        set_b: Phrases/terms for the second side of the co-occurrence test.
        window: Maximum allowed token distance between the starts of the two matches.
        min_stem: Minimum token length to enable prefix (morphological) matching.
                  Shorter tokens require exact equality.

    Returns:
        A tuple (found, (a_term, b_term)):
          - found: True if any qualifying co-occurrence is found, else False.
          - (a_term, b_term): The first pair of terms (by input order) whose matches
            satisfied the proximity constraint; ("", "") if not found.

    Notes:
        - Distance is computed between **start indices** of the phrase matches.
        - The first qualifying pair is returned; order of `set_a`/`set_b` may affect
          which pair you get when multiple pairs qualify.

    """
    toks = tokenize(text)
    def positions_morph(terms: List[str]) -> Dict[str,List[int]]:
        res={}
        for term in terms or []:
            ptoks = tokenize(term)
            if not ptoks: continue
            m=len(ptoks); hits=[]
            for i in range(0, len(toks)-m+1):
                ok=True
                for j, pt in enumerate(ptoks):
                    tt=toks[i+j]
                    if len(pt)>=min_stem:
                        if not tt.startswith(pt): ok=False; break
                    else:
                        if tt != pt: ok=False; break
                if ok: hits.append(i)
            if hits: res[term]=hits
        return res
    pa, pb = positions_morph(set_a), positions_morph(set_b)
    for ta, ia_list in pa.items():
        for tb, ib_list in pb.items():
            for ia in ia_list:
                for ib in ib_list:
                    if abs(ia-ib) <= window:
                        return True, (ta, tb)
    return False, ("","")

_TOKEN_RX = re.compile(r"[a-z0-9åäöæøéèüïóáç\-]+", re.IGNORECASE)
def tokenize(s: str) -> List[str]:
    return _TOKEN_RX.findall(norm(s))

def proximity_cooccurs_morph_spans(
    text,
    set_a,
    set_b,
    window=12,
    min_stem=3,
    same_sentence=True,
    negations=None,
    **_   # ignore any future kwargs safely
):
    """
    Find nearest co-occurrence of any term in set_a and set_b within a token window.
    Adds optional sentence constraint and simple negation guard.
    Returns (bool, meta) where meta includes spans and token distance.
    """
    # Tokenize (reuse your existing helpers)
    toks_spans = [(m.group(0), m.span()) for m in _TOKEN_RX.finditer(norm(text))]
    toks = [t for t, _ in toks_spans]
    n = len(toks)

    # Build sentence spans once (very lightweight char-based splitter)
    sent_bounds = []
    last = 0
    for m in re.finditer(r"[.!?;:]+", text):
        end = m.end()
        if end > last:
            sent_bounds.append((last, end))
            last = end
    if last < len(text):
        sent_bounds.append((last, len(text)))

    # Map a char offset to sentence index
    def sent_index_at(char_pos):
        for i, (s, e) in enumerate(sent_bounds):
            if s <= char_pos < e:
                return i
        return len(sent_bounds) - 1 if sent_bounds else 0

    # Optional: tokens that must match exactly (avoid 'will'→'willing', etc.)
    EXACT_ONLY = {norm(x) for x in ["will", "should", "we", "vi", "see", "ser", "att", "bör"]}

    def positions(terms):
        out = {}
        for term in terms or []:
            ptoks = tokenize(term)
            if not ptoks:
                continue
            m = len(ptoks)
            hits = []
            for i in range(0, n - m + 1):
                ok = True
                for j, pt in enumerate(ptoks):
                    tt = toks[i + j]
                    exact = (pt in EXACT_ONLY) or (len(pt) < min_stem)
                    if exact:
                        if tt != pt:
                            ok = False
                            break
                    else:
                        if not tt.startswith(pt):
                            ok = False
                            break
                if ok:
                    cs = toks_spans[i][1][0]
                    ce = toks_spans[i + m - 1][1][1]
                    hits.append((i, i + m, cs, ce))
            if hits:
                out[term] = hits
        return out

    pa, pb = positions(set_a), positions(set_b)
    best = None
    best_d = 10**9
    negations = [norm(x) for x in (negations or [])]

    for ta, a_hits in pa.items():
        for tb, b_hits in pb.items():
            for ia0, ia1, acs, ace in a_hits:
                for ib0, ib1, bcs, bce in b_hits:
                    # Sentence constraint (check by char positions)
                    if same_sentence:
                        if sent_index_at(acs) != sent_index_at(bcs):
                            continue

                    d = abs(ia0 - ib0)
                    if d > window:
                        continue

                    # Negation guard: skip if an unchanged/negation token appears between hits
                    if negations:
                        lo, hi = (min(ia1, ib0), max(ia0, ib1))
                        between = toks[lo:hi]
                        if any(any(tt.startswith(ng) for tt in between) for ng in negations):
                            continue

                    if d < best_d:
                        best_d = d
                        best = {
                            "cue_term": ta,
                            "cue_span": (acs, ace),
                            "metric_term": tb,
                            "metric_span": (bcs, bce),
                            "token_dist": d,
                        }

    return (best is not None, best)