
from typing import List, Dict, Any, Optional
import re

# ---- Optional dependency guard ----
try:
    from flair.models import SequenceTagger
    from flair.splitter import SegtokSentenceSplitter
    _FLAIR_AVAILABLE = True
except Exception:
    _FLAIR_AVAILABLE = False

_FLAIR_TAGGER = None
_FLAIR_SPLITTER = None

def _get_flair_tagger():
    """Lazy-load the multilingual BIOES NER tagger."""
    global _FLAIR_TAGGER
    if _FLAIR_TAGGER is None:
        if not _FLAIR_AVAILABLE:
            raise RuntimeError("Flair is not installed. `pip install flair`")
        _FLAIR_TAGGER = SequenceTagger.load("flair/ner-multi")
    return _FLAIR_TAGGER

def _get_flair_splitter():
    """Lazy-load the sentence splitter."""
    global _FLAIR_SPLITTER
    if _FLAIR_SPLITTER is None:
        if not _FLAIR_AVAILABLE:
            raise RuntimeError("Flair is not installed. `pip install flair`")
        _FLAIR_SPLITTER = SegtokSentenceSplitter()
    return _FLAIR_SPLITTER

# ---- Lightweight fallback sentence splitter (no deps) ----
_SENT_SPLIT_RX = re.compile(r'(?<=[\.\!\?])\s+(?=[A-ZÅÄÖÆØ])')
def _basic_split(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(_SENT_SPLIT_RX, t)
    return [p.strip() for p in parts if p.strip()]

# ---- Minimal regex fallback for person names (kept local to avoid cycles) ----
_NAME_WORD = r"(?:[A-ZÅÄÖÆØ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+|[A-Z]\.)"
_LOWER_PARTICLE = r"(?:de|del|da|di|la|le|van|von|af|av|bin|al|der|den|du|van der|van de|von der)"
_NAME_PATTERN = rf"\b{_NAME_WORD}(?:\s+(?:{_LOWER_PARTICLE}|{_NAME_WORD})){1,3}\b"
_NAME_RX = re.compile(_NAME_PATTERN)
_NON_NAME_TOKENS = {
    "Annual","General","Meeting","Extraordinary","Board","Company","Group",
    "AB","ASA","AS","Oyj","PLC","AB (publ)","Publ","Limited","Ltd"
}
def _regex_person_name(text: str) -> Optional[str]:
    m = _NAME_RX.search(text or "")
    while m:
        cand = m.group(0)
        toks = re.findall(r"[A-Za-zÅÄÖÆØÀ-ÖØ-öø-ÿ’'\-]+", cand)
        if not any(tok in _NON_NAME_TOKENS for tok in toks) and len(toks) >= 2:
            return cand
        m = _NAME_RX.search(text, m.end())
    return None

# ---- Public API ----
def ner_tag_sentences(text: str) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts, one per sentence:
      {
        "text": "...",
        "entities": [{"text": "...", "type": "PER|ORG|LOC|MISC", "score": 0.99}],
        "tokens":   [{"text": "...", "tag": "B-PER|O|S-ORG|...", "score": 0.99}]
      }
    If Flair isn't available, returns sentence splits with empty entities/tokens.
    """
    if not text:
        return []

    if not _FLAIR_AVAILABLE:
        return [{"text": s, "entities": [], "tokens": []} for s in _basic_split(text)]

    splitter = _get_flair_splitter()
    tagger = _get_flair_tagger()

    sents = splitter.split(text)
    if not sents:
        return []

    tagger.predict(sents, verbose=False)
    out: List[Dict[str, Any]] = []
    for s in sents:
        ents = []
        for span in s.get_spans("ner"):
            lab = span.get_label("ner")
            ents.append({"text": span.text, "type": lab.value, "score": lab.score})
        out.append({"text": s.to_plain_string(), "entities": ents, "tokens": []})
    return out

def detect_person_name_flair(text: str) -> Optional[str]:
    """
    Return the first PERSON entity text using Flair; fall back to regex if Flair isn't available
    or no PER entities were found.
    """
    if text:
        if _FLAIR_AVAILABLE:
            try:
                for sent in ner_tag_sentences(text):
                    for ent in sent["entities"]:
                        if ent["type"] in ("PER", "PERSON"):
                            return ent["text"]
            except Exception:
                pass
    # Fallback
    return _regex_person_name(text or "")

def flair_available() -> bool:
    return _FLAIR_AVAILABLE