import pandas as pd, json, os

labels_csv = "data/labeling_pool_2025-09-01.csv"
assign_csv  = "./assignments_2025-09-02T151420Z.csv"   # or your latest

df_true = pd.read_csv(labels_csv)
df_pred = pd.read_csv(assign_csv)

# choose id and cid columns
id_true = "press_release_id" if "press_release_id" in df_true.columns else "id"
id_pred = "press_release_id" if "press_release_id" in df_pred.columns else "id"
cid_col = "cid" if "cid" in df_pred.columns else ("final_cid" if "final_cid" in df_pred.columns else None)

# build name→cid map from assignments
name2cid = {}
if cid_col and "category_name" in df_pred.columns:
    for _, r in (df_pred[["category_name", cid_col]]
                 .dropna().astype(str).drop_duplicates()).iterrows():
        name2cid[r["category_name"].strip().lower()] = r[cid_col]

allowed = set(df_pred[cid_col].dropna().astype(str)) if cid_col else set()

# simple normalizer (add aliases below if you need)
ALIAS_TO_CID = {
    "pdmr": "pdmr_managers_transactions",
    "credit rating": "credit_ratings",
    "listing": "admission_listing",
    "delisting": "admission_delisting",
    "bond issue": "debt_bond_issue",
    "orders/contracts": "orders_contracts",
    "product launch / partnership": "product_launch_partnership",
    "equity actions": "equity_actions_non_buyback",
    "agm/egm governance": "agm_egm_governance",
    "legal": "legal_regulatory_compliance",
    "personnel": "personnel_management_change",
    "other": "other_corporate_update",
}

def norm_truth(x):
    if pd.isna(x): return None
    s = str(x).strip()
    sl = s.lower()
    if s in allowed: return s
    if sl in ALIAS_TO_CID:
        return ALIAS_TO_CID[sl] if ALIAS_TO_CID[sl] in allowed else None
    if sl in name2cid: return name2cid[sl]
    for c in allowed:
        if c.lower() == sl: return c
    return None

col_true = "true_cid"  # change if your pool uses another name
df_true["__norm_true"] = df_true[col_true].apply(norm_truth)
bad = df_true[df_true["__norm_true"].isna()][[id_true, col_true]]

print("\nUnmappable truth rows (will be dropped):")
print(bad.to_string(index=False, max_rows=50))
print("\nCounts of unmappable label values:")
print(bad[col_true].astype(str).value_counts())

# optional: save for audit
os.makedirs("eval_outputs", exist_ok=True)
bad.to_csv("eval_outputs/unmappable_truth_rows.csv", index=False)
print("\n[ok] Wrote eval_outputs/unmappable_truth_rows.csv")
