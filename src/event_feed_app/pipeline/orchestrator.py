import os, json, logging, logging.config
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from event_feed_app.config import Settings

from event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter

from event_feed_app.preprocessing.dedup_utils import dedup_df
from event_feed_app.preprocessing.near_dedup import near_dedup_embeddings
from event_feed_app.preprocessing.company_name_remover import clean_df_from_company_names
from event_feed_app.preprocessing.subset import pick_diverse_subset

from event_feed_app.utils.io import load_from_gcs, append_row
#from event_feed_app.utils.text import build_doc_for_embedding
from event_feed_app.representation.prompts import (
    doc_prompt, cat_prompts_e5_query, PromptStyle
)
from event_feed_app.utils.lang import setup_langid, detect_language_batch, normalize_lang
from event_feed_app.utils.math import softmax_rows, entropy_rows, l2_normalize_rows, apply_abtt, fit_abtt,rowwise_standardize

from event_feed_app.models.embeddings import get_embedding_model, encode_texts
from event_feed_app.models.tfidf import multilingual_tfidf_cosine

# taxonomy adapter (single source of truth)
from event_feed_app.taxonomy.adapters import load as load_taxonomy
#from event_feed_app.taxonomy.adapters import texts_lsa, texts_emb, align_texts
from event_feed_app.taxonomy.ops import align_texts
from event_feed_app.representation.tfidf_texts import texts_lsa
#from event_feed_app.representation.prompts import cat_prompts_e5_query


from event_feed_app.pipeline.consensus import (
    decide, neighbor_consistency_scores, consensus_by_dup_group
)
from event_feed_app.pipeline.outputs import build_output_df, summarize_run

from event_feed_app.evaluation.metrics import report_rule_metrics

# rules
from event_feed_app.taxonomy.keyword import keywords
from event_feed_app.models.rules_classifier import classify_batch
from event_feed_app.gating.abstain_other_gate import apply_abstain_other_gate

def setup_logging(level: str):
    level = (level or "INFO").upper()
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}
        },
        "handlers": {
            "stderr": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "console",
                "stream": "ext://sys.stderr",
            }
        },
        "root": {"level": level, "handlers": ["stderr"]},
    })
    for noisy in ["urllib3", "gcsfs", "fsspec", "sentence_transformers", "transformers"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(cfg: Settings):
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    setup_logging(cfg.log_level)

    # 1) Load & housekeeping
    df = load_from_gcs(cfg.gcs_silver_root)
    df, stats = apply_housekeeping_filter(df)
    print(f"Housekeeping filter skipped {stats['skipped']} / {stats['total']} "
          f"({(stats['skipped']/max(stats['total'],1))*100:.1f}%).")

    df = clean_df_from_company_names(df)

    # 2) Hard dedup (id+content)
    df, dd_stats = dedup_df(
        df,
        id_col="press_release_id",
        title_col="title_clean",
        body_col="full_text_clean",
        ts_col="release_date",
    )
    print(dd_stats)

    # 3) Optional diverse subset
    base_df = pick_diverse_subset(
        df, "full_text_clean",
        cfg.subset_clusters, cfg.subset_per_cluster
    ) if cfg.use_subset else df
    base_df = base_df.reset_index(drop=True)

    # 3b) Rules-first (keywords + smart rules)
    base_df = classify_batch(
        base_df,
        text_cols=("title_clean", "full_text_clean"),
        keywords=keywords
    )

    # provenance from rules
    base_df["rule_id"]           = base_df["rule_pred_details"].map(lambda d: (d or {}).get("rule"))
    base_df["category_rules"]    = base_df["rule_pred_cid"]  # back-compat alias
    base_df["rule_conf"]         = base_df["rule_pred_details"].map(lambda d: float((d or {}).get("rule_conf", 0.0)))
    base_df["rule_lock"]         = base_df["rule_pred_details"].map(lambda d: bool((d or {}).get("lock", False)))
    base_df["rule_needs_review"] = base_df["rule_pred_details"].map(lambda d: bool((d or {}).get("needs_review_low_sim", False)))

    def is_rules_final(row) -> bool:
        det = row.get("rule_pred_details") or {}
        if det.get("lock"):
            return row.get("rule_pred_cid") != "other_corporate_update"
        if det.get("needs_review_low_sim", False):
            return False
        conf = float(det.get("rule_conf", 0.0))
        return (row.get("rule_pred_cid") not in (None, "", "other_corporate_update")) and (conf >= cfg.rule_conf_thr)

    mask_rules_final = base_df.apply(is_rules_final, axis=1)

    # 4) Taxonomy (adapter)
    tax_base  = load_taxonomy(cfg.taxonomy_version, "en")  # canonical order
    base_cids = tax_base.ordered_cids
    cid2name  = tax_base.cid2name

    # texts for models
    cat_texts_lsa_default = texts_lsa(tax_base)
    # cat_texts_emb         = cat_prompts_e5_query(tax_base, e5_prefix=cfg.e5_prefix)


    # 5) Embeddings
    docs_for_emb = [doc_prompt(t, b, style=PromptStyle.E5_PASSAGE)
                    for t, b in zip(base_df["title_clean"], base_df["full_text_clean"])]

    cat_texts_emb = cat_prompts_e5_query(tax_base)


    model = get_embedding_model(cfg.emb_model_name, cfg.max_seq_tokens)
    # docs_for_emb = [
    #     build_doc_for_embedding(t, b, cfg.e5_prefix)
    #     for t, b in zip(base_df["title_clean"].tolist(), base_df["full_text_clean"].tolist())
    # ]

    X = encode_texts(model, docs_for_emb, cfg.emb_batch_size)  # (N, d)
    Q = encode_texts(model, cat_texts_emb, cfg.emb_batch_size) # (C, d)

    # 1) debias geometry on document distribution (fit on X; you can also fit on np.vstack([X,Q]))
    mu, P = fit_abtt(X, k=getattr(cfg, "emb_abtt_k", 2))
    X_d = apply_abtt(X, mu, P)
    Q_d = apply_abtt(Q, mu, P)

    # 2) normalize after debiasing
    Xn = l2_normalize_rows(X_d)
    Qn = l2_normalize_rows(Q_d)

    # 3) cosine sims
    sims_emb = Xn @ Qn.T



    # # 4) (optional but recommended) row-wise standardize + temperature to restore contrast
    # if getattr(cfg, "emb_rowwise_standardize", True):
    #     sims_emb = sims_emb - sims_emb.mean(axis=1, keepdims=True)
    #     sims_emb = sims_emb / (sims_emb.std(axis=1, keepdims=True) + 1e-8)

    # tau = getattr(cfg, "emb_temperature", 0.35)  # 0.25–0.6 works well; tune
    # sims_emb = sims_emb / tau  # if you later softmax, this sharpens posteriors
    
    # 6) Language detection
    langid = setup_langid(cfg.lang_whitelist)
    langs, lang_confs = detect_language_batch(base_df["title_clean"], base_df["full_text_clean"], langid=langid)
    base_df["language"] = langs
    base_df["lang_conf"] = lang_confs

    # 7) Near-dedup (embedding space)
    base_df, X, near_dedupe_stats, rep_keep_idx = near_dedup_embeddings(
        base_df, X, threshold=0.98, drop=False, RUN_TS=run_id,
        rep_strategy="lang_conf", lang_conf_col="lang_conf",
        NEAR_DUP_AUDIT_CSV="near_dedup_audit.csv"
    )
    print(near_dedupe_stats)

    # 8) Per-language TF-IDF with localized category texts (adapter)
    langs_present = set(normalize_lang(x) for x in base_df["language"].astype(str).fillna("und"))
    lang_to_translated_cats = {}
    for lang in langs_present:
        try:
            localized = load_taxonomy(cfg.taxonomy_version, lang)
            lang_to_translated_cats[lang] = align_texts(tax_base, localized)
        except Exception:
            pass

    sims_lsa = multilingual_tfidf_cosine(
        doc_df=base_df,
        cat_texts_lsa_default=cat_texts_lsa_default,
        rep_keep_idx=rep_keep_idx,
        lang_to_translated_cats=lang_to_translated_cats,
    )

    # 1) Row-wise standardize similarities (per document)
    S_emb = rowwise_standardize(sims_emb)          # from cosine(Xn @ Qn.T)
    S_lsa = rowwise_standardize(sims_lsa)          # from multilingual_tfidf_cosine(...)


    # 9) Fusion
    P_emb = softmax_rows(S_emb, T=cfg.temp_emb)
    P_lsa = softmax_rows(S_lsa, T=cfg.temp_lsa)

    P_fused = cfg.fuse_alpha * P_emb + (1.0 - cfg.fuse_alpha) * P_lsa

    # 10) Decisions
    best_idx  = np.argmax(P_fused, axis=1)
    best_prob = P_fused[np.arange(len(P_fused)), best_idx]
    if P_fused.shape[1] >= 2:
        part = np.partition(P_fused, -2, axis=1)
        second_best = part[:, -2]
        margin = best_prob - second_best
    else:
        second_best = np.zeros_like(best_prob)
        margin = np.zeros_like(best_prob)

    # --- ML (fused) provenance + diagnostics ---
    assigned_cids = [base_cids[i] for i in best_idx]

    entropy_emb   = entropy_rows(P_emb)
    entropy_lsa   = entropy_rows(P_lsa)
    entropy_fused = entropy_rows(P_fused)

    p1_emb = P_emb[np.arange(len(P_emb)), np.argmax(P_emb, axis=1)]
    p1_lsa = P_lsa[np.arange(len(P_lsa)), np.argmax(P_lsa, axis=1)]

    top3_idx   = np.argsort(-P_fused, axis=1)[:, :3]
    top3_cids  = [[base_cids[j] for j in row] for row in top3_idx]
    top3_probs = [P_fused[i, top3_idx[i]].round(4).tolist() for i in range(len(P_fused))]

    base_df["ml_pred_cid"]          = assigned_cids
    base_df["ml_conf_fused"]        = best_prob
    base_df["ml_margin_fused"]      = margin
    base_df["ml_entropy_emb"]       = np.round(entropy_emb, 4)
    base_df["ml_entropy_lsa"]       = np.round(entropy_lsa, 4)
    base_df["ml_entropy_fused"]     = np.round(entropy_fused, 4)
    base_df["ml_p1_emb"]            = np.round(p1_emb, 4)
    base_df["ml_p1_lsa"]            = np.round(p1_lsa, 4)
    base_df["ml_agreement_emb_lsa"] = (np.argmax(P_emb, axis=1) == np.argmax(P_lsa, axis=1)).astype(bool)
    base_df["ml_top3_cids"]         = [json.dumps(v) for v in top3_cids]
    base_df["ml_top3_probs"]        = [json.dumps(v) for v in top3_probs]
    base_df["ml_version"]           = f"e5+tfidf:{cfg.taxonomy_version}|alpha={cfg.fuse_alpha}|T={cfg.temp_emb}/{cfg.temp_lsa}"

    # back-compat aliases
    base_df["category_model"] = base_df["ml_pred_cid"]
    base_df["model_conf"]     = base_df["ml_conf_fused"]
    base_df["model_version"]  = base_df["ml_version"]

    # Rules-first fusion
    final_cids, final_source = [], []
    for i in range(len(base_df)):
        if mask_rules_final.iloc[i]:
            final_cids.append(base_df.loc[i, "rule_pred_cid"])
            final_source.append("rules")
        else:
            final_cids.append(base_df.loc[i, "ml_pred_cid"])
            final_source.append("ml")
    base_df["final_cid"]    = final_cids
    base_df["final_source"] = final_source

    #Abstain → OTHER gate (only for ML-sourced, low-confidence cases) ----
    if cfg.abstain_enable:
        n_abstained = apply_abstain_other_gate(
            base_df,
            other_cid=cfg.abstain_other_cid,
            min_prob=cfg.abstain_min_prob,
            min_margin=cfg.abstain_min_margin,
            min_lang_conf=cfg.abstain_min_lang_conf,
            require_emb_lsa_agreement=cfg.abstain_require_agreement,
        )
        print(f"[info] Abstain gate routed {n_abstained} rows to {cfg.abstain_other_cid}")

    # Decisions (ML + final)
    base_df["ml_decision"] = [
        decide(float(p), float(m), cfg.cat_assign_min_sim, cfg.cat_low_margin)
        for p, m in zip(best_prob, margin)
    ]

    base_df["final_decision"] = np.select(
        [
            base_df["final_source"].eq("rules"),
            base_df["final_source"].eq("abstain_gate"),
        ],
        [
            "by_rules",
            "abstained",
        ],
        default=base_df["ml_decision"],
    )

    # Display name via adapter (must be AFTER the abstain gate)
    base_df["final_category_name"] = [cid2name[cid] for cid in base_df["final_cid"]]

    # Back-compat for evaluation
    base_df["assigned_by"] = base_df["final_source"].map({
        "rules": "rules",
        "ml": "model",
        "abstain_gate": "abstain",
    }).astype("string")
    base_df["category"] = base_df["final_cid"]

    report_rule_metrics(base_df)

    # 11) Diagnostics
    neighbor_consistency = neighbor_consistency_scores(X, best_idx, cfg.knn_k)

    # 12) Group consensus
    consensus_payload = consensus_by_dup_group(base_df, P_fused)
    consensus_payload["decision_group"] = [
        decide(float(p), float(m), cfg.cat_assign_min_sim, cfg.cat_low_margin)
        for p, m in zip(consensus_payload["best_prob_g"], consensus_payload["margin_g"])
    ]

    # 13) Output dataframe
    out = build_output_df(
        base_df=base_df,
        best_prob=best_prob,
        margin=margin,
        P_emb=P_emb,
        P_lsa=P_lsa,
        P_fused=P_fused,
        top3_idx=top3_idx,
        cfg=cfg,
        consensus_payload=consensus_payload,
        base_cids=base_cids,
        run_id=run_id,   # <-- pass the timestamp in
    )

    # 14) Write outputs
    output_csv = cfg.output_local_csv            # e.g., "outputs/assignments.csv"
    root, ext = os.path.splitext(output_csv)
    output_csv_ts = f"{root}_{run_id}{ext}"      # -> outputs/assignments_2025-09-03T...Z.csv
    os.makedirs(os.path.dirname(output_csv_ts) or ".", exist_ok=True)
    out.to_csv(output_csv_ts, index=False)
    print(f"[ok] Wrote {len(out)} assignments to {os.path.abspath(output_csv_ts)}")

    # 15) Run summary
    summary = summarize_run(out, neighbor_consistency, cfg, run_id=run_id)
    append_row(cfg.run_metrics_csv, summary)
    print(f"[ok] Appended run summary to {os.path.abspath(cfg.run_metrics_csv)}")


def main():
    cfg = Settings()
    run(cfg)


if __name__ == "__main__":
    main()
