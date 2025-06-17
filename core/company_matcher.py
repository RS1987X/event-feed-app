import re, logging, spacy
from rapidfuzz import fuzz
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOPWORDS
from spacy.lang.sv.stop_words import STOP_WORDS as SV_STOPWORDS
import json
import os

logger = logging.getLogger(__name__)

# 1) Multilingual NER
nlp = spacy.load("xx_ent_wiki_sm")

# 2) Combined stop-words for English + Swedish
STOPWORDS = EN_STOPWORDS | SV_STOPWORDS

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "generic_tokens.json")

with open(DATA_PATH, encoding="utf-8") as f:
    LOADED_GENERIC = set(json.load(f))

GENERIC_NOUNS = {"group", "holdings", "industries", "services", "systems", "solutions", 
"technologies","technology", "enterprises", "partners", "ventures", "management", 
"capital", "financial", "resources", "development", "logistics", 
"engineering", "consulting", "marketing", "communications", "investments"}

GENERIC_ADJ = {"global", "international", "national", "regional", "local", 
"corporate", "strategic", "digital", "integrated", "advanced", 
"dynamic", "premier", "innovative", "professional", "prime"}

GENERIC_NOUNS_SWE = {"grupp", "holding", "industri", "industrikoncern", "tjänster", "system", "lösningar", 
"teknik", "energi", "utveckling", "konsult", "kommunikation", "investeringar", 
"fastigheter", "förvaltning", "logistik", "transport", "media", "produktion", 
"service", "design", "innovation", "management", "kapital", "finans", "resurser", "utbildning", "momentum"}

GENERIC_TOKENS = LOADED_GENERIC | GENERIC_NOUNS | GENERIC_ADJ | GENERIC_NOUNS_SWE



from cleanco import basename

def strip_suffix(name: str) -> str:
    base = basename(name)
    return base if base else name

def normalize(text: str) -> str:
    text = strip_suffix(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens).strip()

def tokenize(text: str) -> set[str]:
    return {
        tok.text.lower()
        for tok in nlp(text)
        if tok.is_alpha and tok.text.lower() not in STOPWORDS
    }

def sliding_ngrams(tokens, max_n=4):
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            yield tokens[i:i+n]

def is_valid_fuzzy_match(candidate: str, score: int) -> bool:
    # if candidate.lower() in GENERIC_TOKENS:
    #     return False
    if score < 90:
        return False
    if len(candidate.split()) < 2 and score < 95:
        return False  # be stricter on short matches
    return True

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
        result_map = {}
        doc = nlp(text)
        candidates = {ent.text for ent in doc.ents if ent.label_ == "ORG"}
        best = None
        #best_score = threshold

        for span in candidates:
            span_norm = normalize(span)
            if len(span_norm) < 2:
                continue

            for comp in companies:
                if comp["name"] in result_map:
                    continue
                name_norm = normalize(comp["name"])
                # pick metric
                if len(span_norm) < 5:
                    score = fuzz.ratio(name_norm, span_norm)
                else:
                    score = fuzz.token_set_ratio(name_norm, span_norm)

                if score > threshold:
                    result_map[comp["name"]] = span
                    break
                    #best, best_score = (comp["name"], span_norm), score
        if len(result_map) < len(companies):
            tokens = text.split()
            for phrase_tokens in sliding_ngrams(tokens, max_n=4):
                phrase = " ".join(phrase_tokens)
                phrase_norm = normalize(phrase)
                phrase_toks = set(phrase_norm.split())

                for comp in companies:

                    if comp["name"] in result_map:
                        continue     
                    name_norm = normalize(comp["name"])
                    name_toks = set(name_norm.split())

                    if not name_toks:
                        continue

                    # skip if the entire phrase is “generic” (e.g. "global", "technology")
                    if phrase_toks <= GENERIC_TOKENS:
                        continue
                    
                    # 1) Enforce multi-token overlap ratio
                    overlap = name_toks & phrase_toks
                    
                    # **new**: if all overlapping tokens are generic, skip
                    if all(tok in GENERIC_TOKENS for tok in overlap):
                        continue
                    
                    
                    overlap_ratio = len(overlap) / len(name_toks)
                    if overlap_ratio < 0.5:
                        continue

                    score = fuzz.token_set_ratio(phrase_norm, name_norm)
                    if score > threshold and is_valid_fuzzy_match(phrase_norm, score):
                        result_map[comp["name"]] = phrase
                        break
                        
                        # best_match = (comp["name"], phrase)
                        # best_score = score


        # if best:
        #     return best

        # match = fuzzy_fallback_match(text, companies)
        
        return list(result_map.items())

    except Exception as e:
        logger.warning(f"Company match error: {e} — text snippet: {text[:60]!r}")
        return None
    
    
def fuzzy_fallback_match(text: str, companies: list[dict], threshold: int = 90) -> tuple[str, str] | None:
    """
    Try to fuzzy-match n-gram phrases from the text to known companies.
    Returns (company_name, matched_phrase) or None.
    """
    tokens = text.split()  # use cased tokens
    best_match = None
    best_score = threshold

    for phrase_tokens in sliding_ngrams(tokens, max_n=4):
        phrase = " ".join(phrase_tokens)
        phrase_norm = normalize(phrase)
        phrase_toks = set(phrase_norm.split())

        for comp in companies:

                
            name_norm = normalize(comp["name"])
            name_toks = set(name_norm.split())

            if not name_toks:
                continue

            # skip if the entire phrase is “generic” (e.g. "global", "technology")
            if phrase_toks <= GENERIC_TOKENS:
                continue
            
            # 1) Enforce multi-token overlap ratio
            overlap = name_toks & phrase_toks
            
            # **new**: if all overlapping tokens are generic, skip
            if all(tok in GENERIC_TOKENS for tok in overlap):
                continue
            
            
            overlap_ratio = len(overlap) / len(name_toks)
            if overlap_ratio < 0.5:
                continue

            score = fuzz.token_set_ratio(phrase_norm, name_norm)
            if score > best_score and is_valid_fuzzy_match(phrase_norm, score):
                best_match = (comp["name"], phrase)
                best_score = score

    return best_match
    

# def detect_mentioned_company(
#     text: str,
#     companies: list[dict],
#     threshold: int = 90
# ) -> tuple[str, str] | None:
#     """
#     Returns (company_name, matched_token) or None.
#     """
#     try:
#         text_tokens = tokenize(text)

#         for comp in companies:
#             name_tokens   = tokenize(comp["name"])
#             symbol_tokens = tokenize(comp["symbol"])

#             # 1) Exact token intersection
#             intersection = name_tokens & text_tokens
#             if not intersection:
#                 intersection = symbol_tokens & text_tokens
#             if intersection:
#                 return comp["name"], intersection.pop()

#             # 2) Fuzzy match on longer tokens
#             for nt in name_tokens:
#                 if len(nt) < 4:
#                     continue
#                 for tt in text_tokens:
#                     if len(tt) < 4:
#                         continue
#                     if fuzz.ratio(nt, tt) >= threshold:
#                         return comp["name"], tt

#         return None

#     except Exception as e:
#         logger.warning(f"Company match error: {e} — text snippet: {text[:60]!r}")
#         return None

if __name__ == "__main__":
    
    
    sample_text_1 = "The company ABB is in a leading position."

    sample_text_2 = "Apple Inc. is expanding its AI division in Sweden.Technology companies like are also making strides in AI."
    sample_text_3 = "Absolent Air care vinstvarnar. ABB stiger på rapport. Addvise rapporterar starka resultat."
    sample_text_4 = "Addvise rapporterar"
    sample_text_5 = "a global content and technology company"
    companies = [
        #{"name": "Apple Inc.", "symbol": "AAPL"},
        {"name": "Swedish Match", "symbol": "SWMA"},
        {"name": "Momentum Group", "symbol": "MMGR"},
        {"name": "ABB Ltd", "symbol": "ABB"},
        {"name" :"Adventure Box Technology AB" , "symbol" : "ADVBOX"},
        {"name" : "Absolent Air Care Group AB", "symbol" : "ABSO"},
        {"name" : "ADDvise Group AB", "symbol" : "ADDV A"},
    ]

    result = detect_mentioned_company_NER(sample_text_5, companies)
    print("NER result:", result)