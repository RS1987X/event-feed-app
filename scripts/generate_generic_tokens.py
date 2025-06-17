import json
from collections import Counter
import re

import nltk
from nltk.corpus import brown
from nltk.corpus import brown, stopwords
from nltk.corpus import reuters
from nltk import download
import csv
import os
# 1. Ensure data is available
download("brown")
download("universal_tagset")
download("reuters")

# def load_loughran_mcdonald(path="LoughranMcDonald_MasterDictionary_2021.csv"):
#     generic = set()
#     with open(path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             word = row["Word"].lower()
#             # if a word appears in many categories, it’s generic
#             if int(row["Uncertainty"]) > 0 or int(row["Litigious"]) > 0:
#                 generic.add(word)
#     return generic

import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")   # or your favorite English model

def build_business_vocab_from_reuters(top_n=200):
    freq = Counter()
    for fid in reuters.fileids():
        doc = nlp(reuters.raw(fid))
        freq.update([tok.text.lower() for tok in doc if tok.pos_ == "NOUN"])
    return {w for w, _ in freq.most_common(top_n)}

def build_business_vocab_from_reuters_fast(top_n=200, batch_size=500):
    freq = Counter()
    # generator of raw texts
    texts = (reuters.raw(fid) for fid in reuters.fileids())
    # disable unneeded components to speed up POS-tagging
    disable = ["parser", "ner", "lemmatizer"]  # keep only tagger
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=disable):
        freq.update([tok.text.lower() for tok in doc if tok.pos_ == "NOUN"])
    return {w for w, _ in freq.most_common(top_n)}


def build_generic_from_reuters(
    top_nouns: int = 200,
    top_adjs: int = 100,
    batch_size: int = 500,
) -> tuple[set[str], set[str]]:
    """
    Returns two sets:
      - most_common_nouns: top_nouns frequent NOUNs
      - most_common_adjs: top_adjs frequent ADJs
    """
    noun_freq = Counter()
    adj_freq  = Counter()

    # Only keep the tagger to speed things up
    disable = ["parser", "ner", "lemmatizer"]

    # Stream all Reuters docs through spaCy in batches
    texts = (reuters.raw(fid) for fid in reuters.fileids())
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=disable):
        for tok in doc:
            if tok.pos_ == "NOUN":
                noun_freq[tok.text.lower()] += 1
            elif tok.pos_ == "ADJ":
                adj_freq[tok.text.lower()] += 1

    # Grab the top‐frequencies
    most_common_nouns = {w for w, _ in noun_freq.most_common(top_nouns)}
    most_common_adjs  = {w for w, _ in adj_freq.most_common(top_adjs)}
    return most_common_nouns, most_common_adjs


def build_business_vocab_from_brown(top_n=200, category="news"):
    tagged = brown.tagged_words(categories=category, tagset="universal")
    nouns  = [w.lower() for w, t in tagged if t == "NOUN"]
    freq   = Counter(nouns)
    return {w for w, _ in freq.most_common(top_n)}

#business_vocab = build_business_vocab_from_brown(top_n=1000)

#business_vocab = build_business_vocab_from_reuters()

# def build_reuters_business_vocab(min_freq=50) -> set[str]:
#     # pick Reuters categories that are business-related
#     biz_cats = {"money-fx", "trade", "interest", "crude", "ship", "grain", "money-fx"}
#     # gather all words in those categories, lowercased
#     words = [w.lower() for w, t in reuters.tagged_words(tagset="universal")
#              if t == "NOUN" and reuters.categories(w)[0] in biz_cats]
#     freq = Counter(words)
#     # keep only sufficiently frequent nouns
#     return {w for w, c in freq.items() if c >= min_freq}


# 2. Build a frequency distribution over nouns in the Brown corpus
def load_generic_nouns(top_n=200) -> set[str]:
    tagged = brown.tagged_words(tagset="universal")
    # pick only NOUNs, lowercase
    nouns = [w.lower() for w, t in tagged if t == "NOUN"]
    freq = Counter(nouns)
    return {word for word, _ in freq.most_common(top_n)}

# # 3. Hand-curated corporate/generic tokens
# CORP_SUFFIXES = {
#     "ab", "asa", "as", "plc", "llc", "ltd", "inc", "corp", "co", "nv",
#     "oy", "gmbh", "ag", "sa"
# }

# MANUAL_GENERIC = {
#     "group", "technology", "capital", "momentum", "ventures",
#     "solutions", "partners", "systems", "time", "energy",
#     "global", "international"
# }
#business_vocab = build_business_vocab_from_reuters_fast(top_n=1000)
# 4. Combine into one set
#generic_nouns = load_generic_nouns(top_n=1000)

# lm_vocab = load_loughran_mcdonald()
nlp = spacy.load("en_core_web_sm")
def compute_adj_rank(word: str, batch_size=500):
    freq = Counter()
    texts = (reuters.raw(fid) for fid in reuters.fileids())
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["parser","ner","lemmatizer"]):
        freq.update(tok.text.lower() for tok in doc if tok.pos_ == "ADJ")
    ranked = [w for w,_ in freq.most_common()]
    try:
        return ranked.index(word.lower()) + 1
    except ValueError:
        return None

# 2) Build a global rank map at startup
def build_noun_rank_map(batch_size=500, n_process=2):
    """
    Returns a dict: { word -> rank }, where rank=1 is most frequent.
    """
    freq = Counter()
    texts = (reuters.raw(fid) for fid in reuters.fileids())

    # Disable everything except the tagger; use multi‐process batching
    disable = ["parser", "ner", "lemmatizer"]
    for doc in nlp.pipe(texts,
                        batch_size=batch_size,
                        disable=disable,
                        n_process=n_process):
        freq.update(tok.text.lower() for tok in doc if tok.pos_ == "NOUN")

    # Now create a rank map from most_common
    rank_map = {}
    for rank, (word, _) in enumerate(freq.most_common(), start=1):
        rank_map[word] = rank
    return rank_map

# Build once (takes ~20–30 s with batching & 2 cores)
NOUN_RANK = build_noun_rank_map(batch_size=500, n_process=2)

# 3) Instant rank lookup
def compute_noun_rank(word: str) -> int | None:
    return NOUN_RANK.get(word.lower())

# # Example usage:
# for w in ["systems", "holdings", "technologies", "partners", "ventures","financial"]:
#     print(f"{w!r} is rank #{compute_noun_rank(w)}")

# print("Systems is #", compute_noun_rank("systems"))
# print("Holdings is #", compute_noun_rank("holdings"))
# print("Technologies is #", compute_noun_rank("technologies"))
# print("Technologies is #", compute_noun_rank("partners"))
# print("Ventures is #", compute_noun_rank("ventures"))
# print("Fnancial is #", compute_noun_rank("financial"))


nouns, adjs = build_generic_from_reuters(top_nouns=500, top_adjs=250)

GENERIC_TOKENS = nouns | adjs #| lm_vocab


# 4. Ensure the data directory exists
os.makedirs("data", exist_ok=True)
# 5. Save to disk for later reuse
out_path = os.path.join("data", "generic_tokens.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sorted(GENERIC_TOKENS), f, ensure_ascii=False, indent=2)

print(f"Saved {len(GENERIC_TOKENS)} generic tokens to generic_tokens.json")
