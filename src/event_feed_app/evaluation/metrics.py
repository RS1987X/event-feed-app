import numpy as np
import pandas as pd

def report_rule_metrics(
    df: pd.DataFrame,
    assigned_col: str = "final_source",
    rule_id_col: str = "rule_id",
    rules_cat_col: str = "rule_pred_cid",
    model_cat_col: str = "ml_pred_cid",
    final_col: str = "final_cid",
):
    import logging, numpy as np, pandas as pd
    logger = logging.getLogger("cat.metrics")

    total = int(len(df))
    if total == 0:
        logger.info("report_rule_metrics: empty dataframe; skipping.")
        return

    # --- derive/sanitize assigned_by ---
    if assigned_col in df.columns:
        assigned = df[assigned_col]
    elif "final_source" in df.columns:
        assigned = df["final_source"].map({"rules": "rules", "ml": "model"}).fillna("none")
    else:
        rules_has = df.get(rules_cat_col)
        model_has = df.get(model_cat_col)
        assigned = np.where(
            (rules_has.notna() if rules_has is not None else False), "rules",
            np.where((model_has.notna() if model_has is not None else False), "model", "none")
        )

    def _scalarize(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray):
            if x.ndim == 0: return x.item()
            if x.size == 1: return x.reshape(-1)[0]
            return ",".join(map(str, x.tolist()))
        if isinstance(x, (list, tuple)):
            return x[0] if len(x) == 1 else ",".join(map(str, x))
        return x

    assigned_s = pd.Series(assigned).map(_scalarize).astype("string")

    # --- assignments by source ---
    by_source = (
        assigned_s.value_counts(dropna=False)
        .rename_axis("source").to_frame("n")
        .assign(pct=lambda s: (s["n"] / max(total, 1)).round(4))
        .reset_index()
    )
    logger.info("Assignments by source:\n%s", by_source.to_string(index=False))

    # --- rules coverage + top rules ---
    rules_mask = assigned_s.eq("rules")
    n_rules = int(rules_mask.sum())
    logger.info("Rules-assigned: %d/%d (%.2f%%)", n_rules, total, 100.0 * n_rules / max(total, 1))

    if rule_id_col in df.columns and n_rules:
        by_rule = (
            df.loc[rules_mask, rule_id_col]
              .astype("string")
              .value_counts()
              .rename_axis("rule_id")
              .to_frame("n")
        )
        by_rule["pct_of_rules"] = (by_rule["n"] / max(n_rules, 1)).round(4)
        logger.info("Top rules by coverage (up to 20):\n%s",
                    by_rule.head(20).to_string())

    # --- rules vs model conflicts ---
    if rules_cat_col in df.columns and model_cat_col in df.columns:
        rc = df[rules_cat_col].astype("string")
        mc = df[model_cat_col].astype("string")
        conflicts = rc.notna() & mc.notna() & rc.ne(mc)
        n_conflicts = int(conflicts.sum())
        logger.info("Rules vs Model conflicts: %d (%.2f%%)",
                    n_conflicts, 100.0 * n_conflicts / max(total, 1))

    # --- unassigned after final resolution ---
    if final_col in df.columns:
        unassigned = int(df[final_col].isna().sum())
        logger.info("Unassigned after resolution: %d (%.2f%%)",
                    unassigned, 100.0 * unassigned / max(total, 1))

    logger.info("End of rule metrics report.")