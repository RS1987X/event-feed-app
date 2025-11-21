# event_taxonomies/rules/context.py
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

def norm(s: str) -> str:
    s = (s or "")
    s = s.replace("’","'").replace("–","-").replace("—","-")
    return " ".join(s.split()).lower()

TOKEN_RE = re.compile(r"[a-z0-9åäöæøéèüïóáç\-]+", re.IGNORECASE)

def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(norm(s))

@lru_cache(maxsize=4096)
def rx_any(words_tuple: tuple) -> re.Pattern:
    words = [w for w in words_tuple if w]
    if not words:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)

def any_present(text: str, terms: List[str]) -> List[str]:
    t = norm(text)
    hits = []
    for w in terms:
        if not w: continue
        pat = r"\b" + re.escape(w.lower()) + r"\b"
        if re.search(pat, t, flags=re.IGNORECASE):
            hits.append(w.lower())
    return hits

def proximity_cooccurs(text: str, set_a: List[str], set_b: List[str], window: int = 12
                      ) -> Tuple[bool, Tuple[str, str]]:
    toks = tokenize(text)
    pos_a = [i for i, tok in enumerate(toks) if tok in {tokenize(x)[0] for x in set_a if tokenize(x)}]
    pos_b = [i for i, tok in enumerate(toks) if tok in {tokenize(x)[0] for x in set_b if tokenize(x)}]
    for ia in pos_a:
        for ib in pos_b:
            if abs(ia - ib) <= window:
                return True, (toks[ia], toks[ib])
    return False, ("","")

def detect_person_name(original_text: str) -> Optional[str]:
    pat = r"\b[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+(?:\s+[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+)+\b"
    m = re.search(pat, original_text)
    return m.group(0) if m else None

@dataclass
class Ctx:
    title: str
    body: str
    text_norm: str
    text_original: str
