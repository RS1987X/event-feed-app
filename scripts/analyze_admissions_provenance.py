#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit admissions (listing/delisting) provenance:
- Who produced the prediction (rules vs ML)?
- Which rule fired (and was it locked)?
- Do wrong predictions rely only on "soft cues"?
- Are ML admissions low-margin and missing exchange cues?

Usage:
  python analyze_admissions_provenance.py \
    --assign ./outputs/assignments_2025-09-22T113903Z.csv \
    --labels data/labeling/labeling_pool_2025-09-01.csv \
    --outdir ./outputs/eval
"""

import os
import re
import json
import argparse
import pandas as pd

# -----------------------------
# Config-ish bits (regex cues)
# -----------------------------

EXCHANGE_CUES_RX = re.compile(
    r"\b(nasdaq|first\s+north|euronext|london\s+stock\s+exchange|oslo\s+børs|"
    r"stockholmsbörsen|cboe|otc|børsen|ngm|premier\s+growth\s+market)\b",
    re.IGNORECASE
)

SOFT_CUES = [
    "trading halt", "suspension", "trading suspended",
    # Swedish/Danish/Norwegian synonyms
    "handelsstopp", "handelssuspendering", "handel ophørt", "avnotering",
]

ADMISSIONS = {"admission_listing", "admission_delisting"}

LABEL_COL_CANDIDATES = [
    "true_cid", "label_cid", "cid", "category", "label",
    "true_label", "final_label", "gold_cid", "gold"
]
ID_COL_CANDIDATES = ["press_release_id", "id", "press_id"]


# -----------------------------
# Helpers (mostly from eval)
# -----------------------------

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns are present: {candidates}")

def build_name2cid_from_assignments(assign_df: pd.DataFrame) -> dict:
    mapping = {}
    name_col = "final_category_name" if "final_category_name" in assign_df.columns else (
        "category_name" if "category_name" in assign_df.columns else None
    )
    cid_col = (
        "final_cid" if "final_cid" in assign_df.columns else
        ("cid" if "cid" in assign_df.columns else None)
    )
    if name_col and cid_col:
        pairs = (assign_df[[name_col, cid_col]]
                 .dropna()
                 .astype(str)
                 .drop_duplicates())
        for _, row in pairs.iterrows():
            mapping[row[name_col].strip().lower()] = row[cid_col]
    return mapping

def normalize_label(x, allowed_cids: set, local_name2cid: dict):
    import pandas as pd
    if pd.isna(x): return None
    s = str(x).strip()
    s_low = s.lower()
    if s in allowed_cids:
        return s
    for c in allowed_cids:
        if c.lower() == s_low:
            return c
    if s_low in local_name2cid:
        return local_name2cid[s_low]
    return None

def contains_exchange_cue(title: str, body: str) -> bool:
    t = f"{title or ''} {body or ''}"
    return bool(EXCHANGE_CUES_RX.search(t))

def contains_soft_cue(title: str, body: str) -> bool:
    t = f"{title or ''} {body or ''}".lower()
    return any(s in t for s in SOFT_CUES)


# -----------------------------
# Main audit
# -----------------------------

def main(assign_path: str, labels_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    assign = pd.read_csv(assign_path)
    labels = pd.read_csv(labels_path)

    # Pick join columns
    id_pred = pick_col(assign, ID_COL_CANDIDATES)
    id_true = pick_col(labels, ID_COL_CANDIDATES)

    # We’ll normalize truths to CIDs using assignment file for name→cid mapping
    pred_cid_col = "final_cid" if "final_cid" in assign.columns else (
        "cid" if "cid" in assign.columns else None
    )
    if pred_cid_col is None:
        raise KeyError("Predicted CID column not found in assignments ('final_cid' or 'cid').")

    true_col = pick_col(labels, LABEL_COL_CANDIDATES)

    # Allowed CIDs come from predictions present (simple & robust)
    present_cids = sorted(set(assign[pred_cid_col].dropna().astype(str)))
    name2cid = build_name2cid_from_assignments(assign)
    labels = labels.copy()
    labels["true_cid_norm"] = labels[true_col].apply(lambda v: normalize_label(v, set(present_cids), name2cid))

    labels_valid = labels[~labels["true_cid_norm"].isna()].copy()

    merged = labels_valid.merge(
        assign, left_on=id_true, right_on=id_pred, how="inner", suffixes=("_true", "_pred")
    )
    if merged.empty:
        raise RuntimeError("No overlapping IDs between assignments and labeling pool.")

    # Basic correctness
    merged["is_correct"] = merged["true_cid_norm"].astype(str) == merged[pred_cid_col].astype(str)

    # Convenience handle for text
    title_col = "title" if "title" in merged.columns else ("title_clean" if "title_clean" in merged.columns else None)
    body_col  = "snippet" if "snippet" in merged.columns else ("full_text_clean" if "full_text_clean" in merged.columns else None)

    if title_col is None or body_col is None:
        # best-effort: create blank columns
        merged["__title_tmp"] = ""
        merged["__body_tmp"]  = ""
        title_col = "__title_tmp"
        body_col  = "__body_tmp"

    # -----------------------------
    # High-level: who assigned admissions?
    # -----------------------------
    adm = merged[merged[pred_cid_col].isin(ADMISSIONS)].copy()

    # Source breakdown
    if "final_source" in adm.columns:
        src_ct = (adm.groupby("final_source")[pred_cid_col]
                    .value_counts()
                    .to_frame("n")
                    .sort_values("n", ascending=False))
    else:
        # fallback: infer presence of rule prediction columns
        src_series = ("rules" if "rule_pred_cid" in adm.columns else "ml")
        adm["__source"] = src_series
        src_ct = (adm.groupby("__source")[pred_cid_col]
                    .value_counts()
                    .to_frame("n")
                    .sort_values("n", ascending=False))

    print("\n=== Admissions predictions by source ===")
    print(src_ct.to_string())

    # -----------------------------
    # Errors: split rules vs ML
    # -----------------------------
    errs_adm = adm[~adm["is_correct"]].copy()

    print("\n=== Admissions errors by source ===")
    if "final_source" in errs_adm.columns:
        print((errs_adm.groupby("final_source")[pred_cid_col]
                .value_counts()
                .to_frame("n")).to_string())
    else:
        print("(No final_source column; cannot split cleanly)")

    # If rules: which rule + lock?
    if not errs_adm.empty and "final_source" in errs_adm.columns:
        rule_errs = errs_adm[errs_adm["final_source"].eq("rules")].copy()
        if not rule_errs.empty:
            print("\n=== Rule-assigned admissions errors: by rule_id and lock ===")
            cols = ["rule_id", "rule_lock"] if "rule_id" in rule_errs.columns else ["rule_pred_cid"]
            print((rule_errs.groupby(cols)[pred_cid_col]
                  .value_counts()
                  .to_frame("n")
                  .sort_values("n", ascending=False)).to_string())

    # -----------------------------
    # Cues analysis on admissions errors
    # -----------------------------
    if not errs_adm.empty:
        errs_adm["has_exchange_cue"] = errs_adm.apply(
            lambda r: contains_exchange_cue(str(r.get(title_col, "")), str(r.get(body_col, ""))), axis=1
        )
        errs_adm["has_soft_cue"] = errs_adm.apply(
            lambda r: contains_soft_cue(str(r.get(title_col, "")), str(r.get(body_col, ""))), axis=1
        )
        # ML confidence/margin if present
        for col in ["ml_conf_fused", "ml_margin_fused"]:
            if col not in errs_adm.columns:
                errs_adm[col] = pd.NA

        print("\n=== Admissions error cues summary ===")
        group_cols = ["final_source", "has_exchange_cue", "has_soft_cue"]
        print((errs_adm.groupby(group_cols)[pred_cid_col]
               .value_counts()
               .to_frame("n")
               .sort_values("n", ascending=False)).to_string())

        # Summary stats for ML-only errors
        ml_errs = errs_adm[errs_adm.get("final_source", "ml").eq("ml")].copy()
        if not ml_errs.empty:
            print("\n=== ML admissions errors: conf & margin describe ===")
            print(ml_errs[["ml_conf_fused", "ml_margin_fused"]].describe(include="all"))

        # Write sample for manual spot-check
        sample_cols = [c for c in [
            id_true, pred_cid_col, "true_cid_norm", "final_source",
            "rule_id", "rule_lock", "rule_conf",
            "ml_pred_cid", "ml_conf_fused", "ml_margin_fused",
            title_col, body_col, "has_exchange_cue", "has_soft_cue"
        ] if c in errs_adm.columns]
        out_errs_csv = os.path.join(outdir, "admissions_errors_sample.csv")
        errs_adm[sample_cols].head(300).to_csv(out_errs_csv, index=False)
        print(f"[ok] Wrote sample errors: {out_errs_csv}")

    # -----------------------------
    # Crosstab limited to admissions rows
    # -----------------------------
    sub = merged[
        merged["true_cid_norm"].isin(ADMISSIONS) | merged[pred_cid_col].isin(ADMISSIONS)
    ].copy()

    if not sub.empty:
        ct = pd.crosstab(sub["true_cid_norm"], sub[pred_cid_col], dropna=False)
        out_ct_csv = os.path.join(outdir, "admissions_confusion_submatrix.csv")
        ct.to_csv(out_ct_csv)
        print(f"[ok] Wrote admissions submatrix: {out_ct_csv}")
        print("\n=== Admissions-only confusion submatrix ===")
        print(ct.to_string())

    # -----------------------------
    # Optional: flag shaky ML admissions (report only)
    # -----------------------------
    if {"ml_pred_cid", "ml_margin_fused"}.issubset(assign.columns):
        ml_adm = assign[assign["ml_pred_cid"].isin(ADMISSIONS)].copy()
        if not ml_adm.empty:
            ml_adm["has_exchange_cue"] = ml_adm.apply(
                lambda r: contains_exchange_cue(str(r.get("title_clean","") or r.get("title","")),
                                                str(r.get("full_text_clean","") or r.get("snippet",""))), axis=1
            )
            shaky = ml_adm[(~ml_adm["has_exchange_cue"]) & (ml_adm["ml_margin_fused"] < 0.02)]
            out_shaky = os.path.join(outdir, "ml_admissions_shaky.csv")
            shaky_cols = [c for c in [
                id_pred, "ml_pred_cid", "ml_conf_fused", "ml_margin_fused",
                "final_source", "final_cid", "final_decision",
                "title_clean", "full_text_clean", "title", "snippet"
            ] if c in ml_adm.columns]
            shaky[shaky_cols].to_csv(out_shaky, index=False)
            print(f"[ok] Wrote shaky ML admissions (no exchange cue & low margin): {out_shaky}")

    # Summary JSON
    summary = {
        "assignments_file": os.path.abspath(assign_path),
        "labels_file": os.path.abspath(labels_path),
        "n_joined": int(len(merged)),
        "n_admissions_preds": int(len(adm)),
        "n_admissions_errors": int(len(errs_adm)),
    }
    summary_path = os.path.join(outdir, "admissions_provenance_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ok] Wrote summary: {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", required=True, help="Path to assignments CSV")
    ap.add_argument("--labels", required=True, help="Path to labeling pool CSV")
    ap.add_argument("--outdir", default="./", help="Output directory")
    args = ap.parse_args()
    main(args.assign, args.labels, args.outdir)
