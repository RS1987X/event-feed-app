import re, logging, spacy
from rapidfuzz import fuzz
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOPWORDS
from spacy.lang.sv.stop_words import STOP_WORDS as SV_STOPWORDS

logger = logging.getLogger(__name__)

# 1) Multilingual NER
nlp = spacy.load("xx_ent_wiki_sm")

# 2) Combined stop-words for English + Swedish
STOPWORDS = EN_STOPWORDS | SV_STOPWORDS

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\bab\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> set[str]:
    return {
        tok.text.lower()
        for tok in nlp(text)
        if tok.is_alpha and tok.text.lower() not in STOPWORDS
    }

def detect_mentioned_company(
    text: str,
    companies: list[dict],
    threshold: int = 90
) -> tuple[str, str] | None:
    """
    Returns (company_name, matched_token) or None.
    """
    try:
        text_tokens = tokenize(text)

        for comp in companies:
            name_tokens   = tokenize(comp["name"])
            symbol_tokens = tokenize(comp["symbol"])

            # 1) Exact token intersection
            intersection = name_tokens & text_tokens
            if not intersection:
                intersection = symbol_tokens & text_tokens
            if intersection:
                return comp["name"], intersection.pop()

            # 2) Fuzzy match on longer tokens
            for nt in name_tokens:
                if len(nt) < 4:
                    continue
                for tt in text_tokens:
                    if len(tt) < 4:
                        continue
                    if fuzz.ratio(nt, tt) >= threshold:
                        return comp["name"], tt

        return None

    except Exception as e:
        logger.warning(f"Company match error: {e} — text snippet: {text[:60]!r}")
        return None
    

    
def detect_mentioned_company_NER(
    text: str,
    companies: list[dict],
    threshold: int = 90
) -> tuple[str, str] | None:
    """
    1) NER‐based fuzzy match
    2) FALLBACK: exact token intersection on name or symbol
    """
    try:
        # ——— 1) NER fuzzy matching ———
        doc = nlp(text)
        candidates = {ent.text for ent in doc.ents if ent.label_ == "ORG"}
        best = None
        best_score = threshold

        for span in candidates:
            span_norm = normalize(span)
            if len(span_norm) < 2:
                continue

            for comp in companies:
                name_norm = normalize(comp["name"])
                # pick metric
                if len(span_norm) < 5:
                    score = fuzz.ratio(name_norm, span_norm)
                else:
                    score = fuzz.token_set_ratio(name_norm, span_norm)

                if score > best_score:
                    best, best_score = (comp["name"], span_norm), score

        if best:
            return best

        # ——— 2) FALLBACK exact token intersection ———
        text_tokens = tokenize(text)
        for comp in companies:
            name_tokens   = tokenize(comp["name"])
            symbol_tokens = tokenize(comp["symbol"])

            intersection = name_tokens & text_tokens
            if not intersection:
                intersection = symbol_tokens & text_tokens
            if intersection:
                return comp["name"], intersection.pop()

        return None

    except Exception as e:
        logger.warning(f"Company match error: {e} — text snippet: {text[:60]!r}")
        return None