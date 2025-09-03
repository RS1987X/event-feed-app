import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Dict
from event_feed_app.utils.math import entropy_rows
#from src.event_feed_app.utils.math import softmax_rows
#from src.event_feed_app.utils.math import topk_sum

def decide(prob: float, margin: float, min_sim: float, low_margin: float) -> str:
    if prob >= min_sim and margin >= low_margin:
        return "auto_assign"
    elif prob >= min_sim:
        return "needs_review_margin"
    else:
        return "needs_review_low_sim"


def neighbor_consistency_scores(X_emb: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X_emb)), metric="cosine", algorithm="auto")
    nbrs.fit(X_emb)
    dist, idx = nbrs.kneighbors(X_emb, return_distance=True)
    idx_no_self = idx[:, 1:]
    neighbor_labels = labels[idx_no_self]
    return (neighbor_labels == labels.reshape(-1,1)).mean(axis=1)


def consensus_by_dup_group(doc_df: pd.DataFrame, P_fused: np.ndarray) -> Dict[str, np.ndarray]:
    N_docs, N_c = P_fused.shape

    if "dup_group_id" not in doc_df.columns:
        doc_df = doc_df.copy()
        doc_df["dup_group_id"] = (doc_df.get("press_release_id", pd.Series(range(N_docs))).astype(str))

    body_len = doc_df["full_text_clean"].astype(str).str.len().fillna(0).to_numpy()
    len_norm = body_len / (body_len.mean() + 1e-9)
    lang_conf = pd.to_numeric(doc_df.get("lang_conf", pd.Series(0.5, index=doc_df.index)), errors="coerce").fillna(0.5).to_numpy()
    w = np.clip(0.7 * lang_conf + 0.3 * len_norm, 0.1, None)

    groups = doc_df["dup_group_id"].astype(str).to_numpy()
    idx_df = pd.DataFrame({"g": groups, "i": np.arange(N_docs), "w": w})
    grouped = idx_df.groupby("g", sort=False)

    P_fused_g = np.empty_like(P_fused)

    for g, block in grouped:
        ii = block["i"].to_numpy()
        ww = block["w"].to_numpy()
        ww = ww / (ww.sum() + 1e-12)
        Pg = (P_fused[ii] * ww[:, None]).sum(axis=0)
        s = Pg.sum()
        if s > 0:
            Pg = Pg / s
        P_fused_g[ii, :] = Pg

    best_idx_g  = np.argmax(P_fused_g, axis=1)
    best_prob_g = P_fused_g[np.arange(N_docs), best_idx_g]
    second_g    = np.partition(P_fused_g, -2, axis=1)[:, -2] if N_c >= 2 else np.zeros_like(best_prob_g)
    margin_g    = best_prob_g - second_g
    entropy_fused_g = entropy_rows(P_fused_g)

    # group size + representative marker
    doc_df = doc_df.copy()
    doc_df["__body_len_tmp"] = body_len
    grp = doc_df.groupby("dup_group_id", dropna=False)
    doc_df["dup_group_size"] = grp["dup_group_id"].transform("size").astype(int)

    def _pick_rep(g):
        cols = ["__body_len_tmp"]; asc = [False]
        if "release_date" in g.columns:
            cols.append("release_date"); asc.append(True)
        return g.sort_values(cols, ascending=asc, kind="mergesort").index[0]
    rep_idx = grp.apply(_pick_rep)
    rep_idx = rep_idx.values if hasattr(rep_idx, "values") else rep_idx
    is_rep = pd.Series(False, index=doc_df.index)
    is_rep.loc[rep_idx] = True
    doc_df.drop(columns=["__body_len_tmp"], inplace=True)

    sample_weight = 1.0 / doc_df["dup_group_size"].clip(lower=1)

    return dict(
        P_fused_g=P_fused_g,
        best_idx_g=best_idx_g,
        best_prob_g=best_prob_g,
        margin_g=margin_g,
        entropy_fused_g=entropy_fused_g,
        dup_group_id=doc_df["dup_group_id"].to_numpy(),
        dup_group_size=doc_df["dup_group_size"].to_numpy(),
        is_group_rep=is_rep.values,
        sample_weight_dedup=np.round(sample_weight, 6),
    )