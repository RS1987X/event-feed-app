import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Iterable, Optional
from scipy import sparse
import math
from collections import defaultdict

def multilingual_tfidf_cosine(
    doc_df: pd.DataFrame,
    cat_texts_lsa_default: List[str],
    rep_keep_idx: Iterable[int],
    lang_to_translated_cats: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """
    Build one TF-IDF space per detected language and compute cosine(doc, cats) within that space.
    Returns dense sims (N_docs x N_cats).
    """
    N_docs = len(doc_df)
    N_cats = len(cat_texts_lsa_default)
    sims_tfidf = np.zeros((N_docs, N_cats), dtype=np.float64)

    tfidf_input_all = (doc_df["title_clean"].astype(str) + " " +
                       doc_df["full_text_clean"].astype(str))

    # Map language -> all doc indices
    lang_to_idx = defaultdict(list)
    for i, lang in enumerate(doc_df["language"].astype(str).fillna("und")):
        lang_to_idx[lang].append(i)

    # Which indices are allowed for fitting (representative subset)
    rep_keep_idx_set = set(map(int, rep_keep_idx))
    lang_to_fit_idx = {
        lang: [i for i in idxs if i in rep_keep_idx_set] for lang, idxs in lang_to_idx.items()
    }

    def make_tfidf_for_subset(n_used: int):
        # min_df tiers based on what we actually fit on
        min_df_lang = 1 if n_used < 50 else 2 if n_used < 200 else 3

        # Safe max_df as integer doc-count
        if n_used <= 3:
            # Don’t strip high-DF terms in tiny corpora
            max_df_count = n_used
        else:
            max_df_count = max(min_df_lang, int(math.floor(0.70 * n_used)))

        return make_pipeline(
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=min_df_lang,          # ints (doc counts)
                max_df=max_df_count,         # ints (doc counts)
                max_features=200_000,
                strip_accents="unicode",
                dtype=np.float32,            # memory + speed
            ),
            Normalizer(copy=False),          # so dot = cosine
        )

    for lang, all_idx_lang in lang_to_idx.items():
        if not all_idx_lang:
            continue

        # If the representative sample is empty, fall back to all docs
        fit_idx_lang = lang_to_fit_idx.get(lang) or all_idx_lang
        n_used = len(fit_idx_lang)

        # Guardrail: if literally no docs, skip
        if n_used == 0:
            continue

        tfidf_pipe_lang = make_tfidf_for_subset(n_used)
        # Fit on the subset vocabulary
        _ = tfidf_pipe_lang.fit_transform(tfidf_input_all.iloc[fit_idx_lang])

        # Transform all language docs + the language-specific category texts
        V_docs_lang = tfidf_pipe_lang.transform(tfidf_input_all.iloc[all_idx_lang])

        cat_texts_lang = (lang_to_translated_cats or {}).get(lang, cat_texts_lsa_default)
        V_cats_lang = tfidf_pipe_lang.transform(cat_texts_lang)

        sims_lang = V_docs_lang @ V_cats_lang.T  # cosine similarities
        sims_tfidf[np.array(all_idx_lang), :] = (
            sims_lang.toarray() if sparse.issparse(sims_lang) else sims_lang
        )

    return sims_tfidf
