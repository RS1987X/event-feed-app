# near_dedup.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import os

from datetime import datetime, timezone  # add this
def write_near_dup_audit_csv(
    doc_df: pd.DataFrame,
    X: np.ndarray,
    *,
    path: str,
    run_id: str,
    sim_threshold: float,
    rep_strategy: str = "lang_conf",
    id_col: str = "press_release_id",
    title_col: str = "title_clean",
    ts_col: str = "release_date",
    url_col: str = "source_url",
    lang_col: str = "language",
    lang_conf_col: str = "lang_conf",
    body_col: str = "full_text_clean",
) -> None:
    """
    Writes one audit row per document in each multi-member duplicate group.
    Includes the representative with emb_cos_to_rep = 1.0.

    Expects doc_df to include:
      - dup_group_id (str)
      - is_dup_group_rep (bool)  # preferred but not strictly required
    """
    import os
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import normalize

    if "dup_group_id" not in doc_df.columns:
        print("[warn] write_near_dup_audit_csv: dup_group_id missing; nothing to write.")
        return

    # Positional index aligned with X rows
    df = doc_df.copy()
    df["__pos"] = np.arange(len(df))

    # Normalize embeddings for cosine similarity
    Xn = normalize(X, norm="l2", copy=True)

    # Safe accessors
    def _get(row: pd.Series, col: str, default=""):
        return row[col] if (col in row and pd.notna(row[col])) else default

    def _get_num(row: pd.Series, col: str, default: float = 0.0) -> float:
        val = _get(row, col, default)
        try:
            return float(val)
        except Exception:
            return float(default)

    rows = []

    # groupby keeps original order for each group (sort=False by default here)
    for gid, g in df.groupby("dup_group_id", dropna=False, sort=False):
        group_size = len(g)
        if group_size <= 1:
            # keep previous behavior: skip singletons; change to '>= 1' to include them
            continue

        # Pick representative: prefer flag; fallback = longest body
        if "is_dup_group_rep" in g.columns and g["is_dup_group_rep"].any():
            rep = g[g["is_dup_group_rep"]].iloc[[0]]
        else:
            # Fallback to longest body text if flag missing
            body_lengths = g[body_col].astype(str).str.len() if body_col in g.columns else pd.Series(0, index=g.index)
            rep_idx = body_lengths.idxmax()
            rep = g.loc[[rep_idx]]

        rep_pos = int(rep["__pos"].iloc[0])

        # Similarity of each group member to the representative
        grp_pos = g["__pos"].to_numpy()
        sims = Xn[grp_pos] @ Xn[rep_pos]  # shape: (group_size,)

        # Representative metadata (same for every row in this group)
        rep_row = rep.iloc[0]
        rep_meta = {
            "rep_row": rep_pos,
            "rep_id": _get(rep_row, id_col),
            "rep_title": _get(rep_row, title_col),
            "rep_lang": _get(rep_row, lang_col),
            "rep_lang_conf": _get_num(rep_row, lang_conf_col, 0.0),
            "rep_release_date": _get(rep_row, ts_col),
            "rep_source_url": _get(rep_row, url_col),
        }

        # Emit one row per document (including the representative)
        #rep_index_label = rep.index[0]
        rep_pos = int(rep["__pos"].iloc[0])
        for (_, r), sim in zip(g.iterrows(), sims):
            sim_val = 1.0 if int(r["__pos"]) == rep_pos else float(sim)

            rows.append({
                "run_id": run_id,
                "dup_group_id": gid,
                "dup_group_size": int(group_size),
                "rep_strategy": rep_strategy,
                "sim_threshold": float(sim_threshold),

                # The document in this row ("dup_*" fields, including the rep itself)
                "dup_row": int(r["__pos"]),
                "dup_id": _get(r, id_col),
                "dup_title": _get(r, title_col),
                "dup_lang": _get(r, lang_col),
                "dup_lang_conf": _get_num(r, lang_conf_col, 0.0),
                "dup_release_date": _get(r, ts_col),
                "dup_source_url": _get(r, url_col),

                "emb_cos_to_rep": sim_val,

                # The representative’s metadata
                **rep_meta,
            })

    if not rows:
        print("[info] write_near_dup_audit_csv: no multi-member groups; no audit rows.")
        return

    audit_df = pd.DataFrame(rows)
    audit_df.sort_values(
    ["dup_group_id", "emb_cos_to_rep"],
    ascending=[True, False],
    kind="mergesort",   # stable
    inplace=True,
)
    # Ensure parent dir exists
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    audit_df.to_csv(path, index=False)
    print(f"[ok] Near-dup audit written to {os.path.abspath(path)} with {len(audit_df)} rows.")

def _pick_rep(
    indices: np.ndarray,
    df: pd.DataFrame,
    *,
    rep_strategy: str = "length_ts",      # "length_ts" or "lang_conf"
    body_col: str = "full_text_clean",
    ts_col: Optional[str] = "release_date",
    lang_conf_col: str = "lang_conf",
) -> int:
    """
    Return a **positional** row index (0..N-1) for the representative.
    """
    indices = np.asarray(indices, dtype=int)
    g = df.iloc[indices].copy()
    g["__pos"] = indices  # carry original positions

    # features
    g["_len"] = g.get(body_col, "").astype(str).str.len() if body_col in g.columns else 0
    g["_ts"]  = pd.to_datetime(g[ts_col], errors="coerce") if ts_col in g.columns else pd.NaT
    g["_lc"]  = pd.to_numeric(g.get(lang_conf_col, -1.0), errors="coerce").fillna(-1.0)

    if rep_strategy == "lang_conf" and (g["_lc"] >= 0).any():
        g = g.sort_values(["_lc", "_len", "_ts"], ascending=[False, False, False], kind="mergesort")
    else:
        g = g.sort_values(["_len", "_ts"], ascending=[False, False], kind="mergesort")

    return int(g.iloc[0]["__pos"])  # <-- return position, not label

def near_dedup_embeddings(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    company_col: str = "company_name",
    ts_col: Optional[str] = "release_date",   # tie-break + blocking bucket
    title_col: str = "title_clean",
    body_col: str = "full_text_clean",
    date_bucket: str = "D",                   # 'D' for daily, 'W' weekly, etc.
    sim_threshold: float = 0.95,
    threshold: Optional[float] = None,        # alias
    drop: bool = True,                        # True = keep reps only; False = mark-only
    RUN_TS: str,
    rep_strategy: str = "length_ts",          # NEW: "length_ts" or "lang_conf"
    lang_conf_col: str = "lang_conf",         # used when rep_strategy="lang_conf"
    NEAR_DUP_AUDIT_CSV: Optional[str] = None,  # if set, write audit CSV
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int], np.ndarray]:
    """
    Cluster near-duplicates within (company, date_bucket) blocks using cosine similarity on X.
    Pick 1 representative per connected component at similarity >= threshold.

    Returns (depending on `drop`):
        drop=True  -> (df_kept, X_kept, stats, keep_idx)  # only representatives
        drop=False -> (df_marked_full, X_full, stats, keep_idx)  # all docs; keep_idx = reps

    Added columns in returned df:
        dup_group_id (str), dup_group_size (int), is_dup_group_rep (bool for reps)
    """
    if threshold is not None:
        sim_threshold = float(threshold)

    assert len(df) == X.shape[0], "df and X must align row-wise"

    df = df.reset_index(drop=True)

    N = len(df)
    dup_group_id = np.empty(N, dtype=object)
    keep_mask = np.ones(N, dtype=bool)
    rep_positions: List[int] = []

    # Blocking key: company + floored date
    comp = df[company_col].astype(str).str.lower() if company_col in df.columns else pd.Series([""] * N, index=df.index)
    if ts_col in df.columns:
        dt = pd.to_datetime(df[ts_col], errors="coerce")
        key = comp + " | " + dt.dt.floor(date_bucket).astype(str)
    else:
        key = comp

    # Normalize embeddings for cosine
    Xn = normalize(X, norm="l2", copy=True)

    # Group positional indices by blocking key (stable order)
    idx_df = pd.DataFrame({"k": key.values, "i": np.arange(N)}, index=df.index)
    groups = idx_df.groupby("k", sort=False)["i"].apply(lambda s: s.to_numpy())

    gid_counter = 0

    for _, block_pos in groups.items():
        m = block_pos.size
        if m == 0:
            continue

        if m == 1:
            dup_group_id[block_pos[0]] = f"g{gid_counter:08d}"
            rep_positions.append(block_pos[0])
            gid_counter += 1
            continue

        B = Xn[block_pos]                 # (m, d)
        S = B @ B.T                       # cosine sim (m x m)
        A = (S >= sim_threshold).astype(np.uint8)
        np.fill_diagonal(A, 0)

        graph = csr_matrix(A)
        n_comp, labels = connected_components(csgraph=graph, directed=False, connection='weak')

        for c in range(n_comp):
            comp_pos = block_pos[labels == c]
            if comp_pos.size == 0:
                continue
            gid = f"g{gid_counter:08d}"
            dup_group_id[comp_pos] = gid
            gid_counter += 1

            # Pick representative according to strategy
            rep = _pick_rep(
                comp_pos, df,
                rep_strategy=rep_strategy,
                body_col=body_col,
                ts_col=ts_col,
                lang_conf_col=lang_conf_col,
            )
            rep_positions.append(rep)
            if comp_pos.size > 1:
                drop_pos = np.setdiff1d(comp_pos, np.array([rep]), assume_unique=False)
                keep_mask[drop_pos] = False

    # Assign any unassigned (edge cases)
    unassigned = np.where(pd.isna(pd.Series(dup_group_id)))[0]
    for pos in unassigned:
        dup_group_id[pos] = f"g{gid_counter:08d}"
        rep_positions.append(int(pos))
        gid_counter += 1

    # Group sizes
    dup_group_id_series = pd.Series(dup_group_id, index=df.index, dtype="object")
    grp_sizes = dup_group_id_series.map(dup_group_id_series.value_counts())

    # Representative index positions (positional, validated)
    rep_positions = np.unique(np.asarray(rep_positions, dtype=int))
    assert rep_positions.size > 0, "No representatives selected."
    assert rep_positions.min() >= 0 and rep_positions.max() < len(df), \
        f"rep_positions out of bounds: max={rep_positions.max()} N={len(df)}"
    keep_idx = rep_positions  # representatives (used even when drop=False)

    stats: Dict[str, int] = {
        "total": int(N),
        "dropped_near_dupes": int((~keep_mask).sum()),
        "kept": int(keep_mask.sum()),
        "n_groups": int(dup_group_id_series.nunique()),
        "n_singletons": int((grp_sizes == 1).sum()),
        "n_multi_groups": int((grp_sizes > 1).sum()),
    }



            
    # ---- Build audit DataFrame with required columns ----
    df_audit = df.copy()
    df_audit["dup_group_id"]   = dup_group_id_series.values
    df_audit["dup_group_size"] = grp_sizes.astype(int).values
    is_rep = np.zeros(N, dtype=bool)
    is_rep[keep_idx] = True
    df_audit["is_dup_group_rep"] = is_rep

    # Optional: add language conf if it exists upstream (safe no-op otherwise)
    # df_audit["lang_conf"] = pd.to_numeric(df_audit.get("lang_conf", 0.0), errors="coerce").fillna(0.0)

    # ---- Write audit CSV (env default; arg overrides; "" disables) ----
    audit_path_env = os.getenv("NEAR_DUP_AUDIT_CSV", "")  # may be "", None->""
    audit_path = NEAR_DUP_AUDIT_CSV if NEAR_DUP_AUDIT_CSV is not None else audit_path_env

    if audit_path and audit_path.strip():
        try:
            os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
            write_near_dup_audit_csv(
                df_audit, X,
                path=audit_path,
                run_id=RUN_TS,
                sim_threshold=sim_threshold,
                rep_strategy=rep_strategy,
            )
            stats["audit_csv"] = os.path.abspath(audit_path)
        except Exception as e:
            stats["audit_csv_error"] = f"{type(e).__name__}: {e}"

    # ---- Return results ----
    if drop:
        kept_pos = keep_mask.nonzero()[0]
        df_out = df.iloc[kept_pos].copy()
        df_out["dup_group_id"]   = dup_group_id_series.iloc[kept_pos].values
        df_out["dup_group_size"] = grp_sizes.iloc[kept_pos].astype(int).values
        df_out["is_dup_group_rep"] = True
        return df_out, X[kept_pos], stats, keep_idx
    else:
        df_out = df.copy()
        df_out["dup_group_id"]   = dup_group_id_series.values
        df_out["dup_group_size"] = grp_sizes.astype(int).values
        df_out["is_dup_group_rep"] = False
        df_out.iloc[keep_idx, df_out.columns.get_loc("is_dup_group_rep")] = True
        return df_out, X, stats, keep_idx
