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
