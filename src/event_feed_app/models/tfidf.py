import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Iterable, Optional
from scipy import sparse

def multilingual_tfidf_cosine(
    doc_df: pd.DataFrame,
    cat_texts_lsa_default: List[str],
    rep_keep_idx: Iterable[int],
    lang_to_translated_cats: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """
    Build one TF-IDF space per detected language and compute cosine(doc, cats) within that space.
    Returns sims_lsa (N_docs x N_cats).
    """
    from collections import defaultdict

    N_docs = len(doc_df)
    N_cats = len(cat_texts_lsa_default)
    sims_tfidf = np.zeros((N_docs, N_cats), dtype=np.float64)

    tfidf_input_all = (doc_df["title_clean"].astype(str) + " " + doc_df["full_text_clean"].astype(str))

    lang_to_idx = defaultdict(list)
    for i, lang in enumerate(doc_df["language"].astype(str).fillna("und")):
        lang_to_idx[lang].append(i)

    rep_keep_idx_set = set(int(x) for x in np.asarray(list(rep_keep_idx)).tolist())
    lang_to_rep_idx = {lang: [i for i in idxs if i in rep_keep_idx_set] for lang, idxs in lang_to_idx.items()}

    def make_tfidf(min_df_val=3):
        return make_pipeline(
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=min_df_val,
                max_df=0.70,
                max_features=200_000,
                strip_accents="unicode",
            ),
            Normalizer(copy=False),
        )

    for lang, idxs in lang_to_idx.items():
        if not idxs:
            continue
        n_lang_docs = len(idxs)
        min_df_lang = 1 if n_lang_docs < 50 else 2 if n_lang_docs < 200 else 3
        tfidf_pipe_lang = make_tfidf(min_df_val=min_df_lang)

        rep_idx_lang = lang_to_rep_idx.get(lang, []) or idxs
        _ = tfidf_pipe_lang.fit_transform(tfidf_input_all.iloc[rep_idx_lang])

        V_docs_lang = tfidf_pipe_lang.transform(tfidf_input_all.iloc[idxs])
        cat_texts_lang = (lang_to_translated_cats or {}).get(lang, cat_texts_lsa_default)
        V_cats_lang = tfidf_pipe_lang.transform(cat_texts_lang)

        sims_lang = V_docs_lang @ V_cats_lang.T
        sims_tfidf[np.array(idxs), :] = sims_lang.toarray() if sparse.issparse(sims_lang) else sims_lang

    return sims_tfidf