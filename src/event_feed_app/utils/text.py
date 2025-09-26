
def build_doc_for_embedding(title: str, body: str, e5_prefix: bool) -> str:
    title = (title or "").strip()
    body  = (body or "").strip()
    txt   = f"[TITLE] {title}\n\n{body}" if title else body
    return f"passage: {txt}" if e5_prefix else txt

def build_cat_query(name: str, desc: str, e5_prefix: bool) -> str:
    q = f"category: {name}. description: {desc}"
    return "query: " + q if e5_prefix else q


# event_feed_app/utils/text.py
import re, unicodedata
from functools import lru_cache
from typing import List

def normalize_quotes(s: str) -> str:
    return (s or "").replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"').replace("–","-").replace("—","-")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return normalize_quotes(s).casefold()

_TOKEN_RX = re.compile(r"[a-z0-9åäöæøéèüïóáç\-]+", re.IGNORECASE)
def tokenize(s: str) -> List[str]:
    return _TOKEN_RX.findall(norm(s))

def rx(words: List[str]) -> re.Pattern:
    words = [w for w in words if w]
    if not words: return re.compile(r"(?!x)x")
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)

def any_present(text: str, terms: List[str]) -> List[str]:
    t = norm(text); hits=[]
    seen=set()
    for w in terms or []:
        if not w: continue
        tw = norm(w)
        if re.search(r"\b" + re.escape(tw) + r"\b", t) and tw not in seen:
            seen.add(tw); hits.append(tw)
    return hits
