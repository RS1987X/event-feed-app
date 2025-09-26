import pandas as pd
from typing import Union, Tuple, Dict, Any

def filter_to_labeled(
    df_main: pd.DataFrame,
    labeling: Union[str, pd.DataFrame],
    *,
    id_col: str = "press_release_id",
    read_csv_kwargs: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keep only rows in df_main whose `id_col` exists in the labeling file.
    Returns (filtered_df, stats).
    """
    read_csv_kwargs = read_csv_kwargs or {}
    if isinstance(labeling, str):
        # we only need the id column; save RAM if the file is wide
        usecols = read_csv_kwargs.pop("usecols", None)
        if usecols is None:
            read_csv_kwargs["usecols"] = [id_col]
        df_lab = pd.read_csv(labeling, **read_csv_kwargs)
    else:
        df_lab = labeling.copy()

    if id_col not in df_main.columns:
        raise KeyError(f"`df_main` is missing ID column '{id_col}'.")
    if id_col not in df_lab.columns:
        raise KeyError(f"`labeling` is missing ID column '{id_col}'.")

    input_rows = len(df_main)
    label_ids = df_lab[id_col].dropna().astype(str).unique()
    keep_ids = set(label_ids)

    out = df_main[df_main[id_col].astype(str).isin(keep_ids)].copy()
    # Optional de-dup if df_main can contain duplicates per id
    out = out.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)

    kept_rows = len(out)
    stats = {
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "dropped_rows": input_rows - kept_rows,
        "keep_ratio": (kept_rows / input_rows) if input_rows else 0.0,
        "unique_label_ids": len(keep_ids),
        "id_col": id_col,
    }
    return out, stats
