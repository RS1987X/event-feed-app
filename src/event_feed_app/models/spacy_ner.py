# -*- coding: utf-8 -*-
"""
Fast NER adapter using spaCy's small multilingual model + light ORG regex boost.
API matches flair_ner: ner_tag_sentences(), detect_person_name_spacy()
"""
from typing import List, Dict, Any, Optional
import re
from functools import lru_cache

# ---- Optional dep guard ----
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

# Capitalized sequence + common company suffixes (Nordic + EN)
_ORG_SUFFIXES = r"(?:AB(?:\s*\(publ\))?|ASA|AS|A/S|Oyj|Oy|ASO|ApS|PLC|plc|Ltd|Limited|GmbH|AG|NV|SA|S\.p\.A|S\.A\.S|BV|B\.V\.)"
_ORG_CORE = r"(?:[A-ZÅÄÖÆØ][\w’'\-]+(?:\s+[A-ZÅÄÖÆØ][\w’'\-]+){0,5})"
_ORG_RX = re.compile(rf"\b{_ORG_CORE}\s+{_ORG_SUFFIXES}\b")

# Lightweight sentence splitter fallback
_SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+(?=[A-ZÅÄÖÆØ])')
def _basic_split(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in re.split(_SENT_SPLIT_RX, re.sub(r"\s+", " ", text.strip())) if s.strip()]

# Regex fallback for PER
_NAME_WORD = r"(?:[A-ZÅÄÖÆØ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+|[A-Z]\.)"
_LOWER_PARTICLE = r"(?:de|del|da|di|la|le|van|von|af|av|bin|al|der|den|du|van der|van de|von der)"
_NAME_RX = re.compile(rf"\b{_NAME_WORD}(?:\s+(?:{_LOWER_PARTICLE}|{_NAME_WORD})){1,3}\b")

@lru_cache(maxsize=1)
def _get_nlp():
    if not _SPACY_AVAILABLE:
        raise RuntimeError("spaCy is not installed. `pip install spacy`")
    try:
        # multilingual small NER
        return spacy.load("xx_ent_wiki_sm", exclude=["lemmatizer", "attribute_ruler"])
    except OSError:
        # if model missing, show clear error; up to caller to install
        raise RuntimeError("Model 'xx_ent_wiki_sm' missing. Run: python -m spacy download xx_ent_wiki_sm")

def ner_tag_sentences(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    if not _SPACY_AVAILABLE:
        # spaCy not available → sentence-only
        return [{"text": s, "entities": [], "tokens": []} for s in _basic_split(text)]

    try:
        nlp = _get_nlp()
        # ensure we have sentence boundaries quickly
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")  # no-op if already there

        # cap extreme input length
        doc = nlp(text[:60000])
        out: List[Dict[str, Any]] = []

        # Precollect model ents by sentence
        by_char = {}
        for ent in doc.ents:
            if ent.label_ in ("PER", "PERSON", "ORG"):
                by_char.setdefault((ent.start_char, ent.end_char), {"text": ent.text, "type": "PER" if ent.label_.startswith("PER") else "ORG", "score": getattr(ent, "kb_id_", None) or 1.0})

        # Regex-boost ORGs with suffixes (adds misses)
        for m in _ORG_RX.finditer(doc.text):
            k = (m.start(), m.end())
            if k not in by_char:
                by_char[k] = {"text": m.group(0), "type": "ORG", "score": 0.95}

        # Now assign ents to sentences
        for sent in doc.sents:
            s_start, s_end = sent.start_char, sent.end_char
            ents = []
            for (a, b), info in by_char.items():
                if a >= s_start and b <= s_end:
                    ents.append({"text": info["text"], "type": info["type"], "score": info.get("score", 1.0)})
            out.append({"text": sent.text, "entities": ents, "tokens": []})
        return out
    except Exception:
        # graceful fallback
        return [{"text": s, "entities": [], "tokens": []} for s in _basic_split(text)]

def detect_person_name_spacy(text: str, min_score: float = 0.0, max_chars: int = 6000) -> Optional[str]:
    if not text:
        return None
    t = text[:max_chars]
    if _SPACY_AVAILABLE:
        try:
            nlp = _get_nlp()
            doc = nlp(t)
            best = None
            for ent in doc.ents:
                if ent.label_ in ("PER", "PERSON"):
                    score = 1.0  # small model doesn't expose prob; treat as 1.0
                    if score >= min_score and (best is None or len(ent.text) > len(best)):
                        best = ent.text
            if best:
                return best
        except Exception:
            pass
    # regex fallback
    m = _NAME_RX.search(t)
    return m.group(0) if m else None
