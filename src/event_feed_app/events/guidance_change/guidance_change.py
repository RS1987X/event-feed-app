# SPDX-License-Identifier: MIT
# src/event_feed_app/events/jobs/guidance_change.py
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
import pandas as pd
import gcsfs

from event_feed_app.events.registry import get_plugin

from event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter
from event_feed_app.gating.newsletter_quotes_gate import apply_newsletter_gate

from event_feed_app.preprocessing.dedup_utils import dedup_df
try:
    from event_feed_app.preprocessing.filters import filter_to_labeled
except Exception:
    filter_to_labeled = None

from event_feed_app.taxonomy.rules_engine import classify_batch
import event_feed_app.taxonomy.rules.keywords_generic          # noqa: F401
import event_feed_app.taxonomy.rules.product_launch_partn  # noqa: F401
import event_feed_app.taxonomy.rules.admissions          # noqa: F401
import event_feed_app.taxonomy.rules.agm  # noqa: F401
import event_feed_app.taxonomy.rules.debt          # noqa: F401
import event_feed_app.taxonomy.rules.regulatory  # noqa: F401
import event_feed_app.taxonomy.rules.incidents          # noqa: F401
import event_feed_app.taxonomy.rules.invite  # noqa: F401
import event_feed_app.taxonomy.rules.mna          # noqa: F401
import event_feed_app.taxonomy.rules.orders  # noqa: F401
import event_feed_app.taxonomy.rules.pdmr          # noqa: F401
import event_feed_app.taxonomy.rules.personnel  # noqa: F401
import event_feed_app.taxonomy.rules.ratings          # noqa: F401
import event_feed_app.taxonomy.rules.rd_clinical_update          # noqa: F401
import event_feed_app.taxonomy.rules.earnings          # noqa: F401

from event_feed_app.configs.kwex.kwex_utils import load_kwex_snapshot
from event_feed_app.taxonomy.keywords.keyword import (
    keywords, SCHEMA_VERSION, LEXICON_VERSION, TAXONOMY_VERSION, SOURCE as KW_SOURCE
)


# ✅ use your unified GCS IO helpers in utils/
from event_feed_app.utils.gcs_io import (
    load_parquet_df,
    load_window_parquet_df,
    append_parquet_gcs,
    overwrite_parquet_gcs,
)

# Optional BigQuery support (only if you pass --input-kind bigquery)
try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None  # type: ignore

GUIDANCE_KEY = "guidance_change"
log = logging.getLogger("guidance_change_job")

import hashlib, json, os
from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_VERSION = "1"            # bump to invalidate all caches when logic changes
PIPELINE_TAG = "guidance_l1_v1"  # cache scope tag for this job
#Simple “latest” cache (coarse: any cached L1 set)
SIMPLE_L1_CACHE = CACHE_DIR / f"{PIPELINE_TAG}_latest.parquet"

# Minimal columns we expect in cached data (so downstream code works)
REQUIRED_BASE_COLS = {
    "company_id", "title", "body", "published_utc", "doc_id",
    "source_type", "source_url"
}

# Any of these means the dataframe is already L1-classified
L1_MARKER_COLS = {
    "rule_pred_cid", "kwex_tags", "kwex_topic",
    "earnings_hit", "earnings_report_hit"
}

# ---------------------- Config loading ----------------------
@dataclass
class Sinks:
    events_silver: str
    guidance_versions: str
    guidance_current: str

import os  # new

def _ensure_gs(uri: str) -> str:
    return uri if not uri or uri.startswith("gs://") else f"gs://{uri}"

def load_configs(config_dir: Path):
    """
    Supports:
      - monolith file named either 'significance_events.yaml' or 'significant_events.yaml'
      - monolith 'events' as dict OR list of items with 'event_key' or 'key'
      - per-event files under configs/events/*.yaml
    Returns: {"defaults": {...}, "events": {<event_key>: {...}}}
    """
    def _normalize_events(ev):
        if ev is None:
            return {}
        if isinstance(ev, dict):
            return ev
        if isinstance(ev, list):
            out = {}
            for item in ev:
                if not isinstance(item, dict):
                    continue
                k = item.get("event_key") or item.get("key")
                if not k:
                    continue
                out[k] = item
            return out
        return {}

    cfg_dir = Path(config_dir)

    # --- try monolithic files first ---
    for fname in ("significance_events.yaml", "significant_events.yaml"):
        p = cfg_dir / fname
        if p.exists():
            root = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            defaults = root.get("defaults", {})
            events = _normalize_events(root.get("events"))
            return {"defaults": defaults, "events": events}

    # --- fall back to per-event files ---
    defaults = {}
    dflt = cfg_dir / "defaults.yaml"
    if dflt.exists():
        defaults = yaml.safe_load(dflt.read_text(encoding="utf-8")) or {}

    events = {}
    events_dir = cfg_dir / "events"
    if events_dir.exists():
        for p in sorted(events_dir.glob("*.yaml")):
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            key = data.get("event_key") or data.get("key") or p.stem
            events[key] = data

    return {"defaults": defaults, "events": events}
def load_sinks(config_dir: Path) -> Sinks:
    for fname in ("event_sinks.yaml", "sinks.yaml"):  # support either
        p = config_dir / fname
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            g = data.get("gcs", {})
            return Sinks(
                events_silver=g["events_silver"],
                guidance_versions=g["guidance_versions"],
                guidance_current=g["guidance_current"],
            )
    raise FileNotFoundError(
        f"Missing sinks file. Expected one of: "
        f"{config_dir/'event_sinks.yaml'} or {config_dir/'sinks.yaml'}"
    )
# ---------------------- Input readers ----------------------
import re
def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "unknown"

def read_docs_window_gcs_parquet(
    gcs_path: str,
    start_utc: datetime,
    end_utc: datetime,
    max_docs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Reads normalized docs from a Parquet file/folder on GCS and adapts the schema
    to what the guidance plugin expects.
    Input columns seen today:
      press_release_id, company_name, category, release_date, ingested_at,
      title, full_text, source, source_url, ...
    Output dict keys expected by the plugin/orchestrator:
      company_id, title, body, published_utc, doc_id, source_type,
      source_url, cluster_id, l1_class, l1_conf
    """
    # 1) read full frame (we can’t project columns that aren’t there)
    df: pd.DataFrame = load_window_parquet_df(gcs_path, start_utc, end_utc)

    if df.empty:
        return []

    # 2) rename to target names where obvious
    rename_map = {
        "press_release_id": "doc_id",
        "full_text": "body",
        "release_date": "published_utc",
        "source": "source_type",
    }
    df = df.rename(columns=rename_map)

    # 3) ensure required columns exist (create from fallbacks or fill)
    if "company_id" not in df.columns:
        if "company_name" in df.columns:
            df["company_id"] = df["company_name"].fillna("").map(lambda x: f"name:{_slugify(str(x))}")
        else:
            df["company_id"] = "unknown"

    if "published_utc" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["published_utc"]):
            df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    else:
        # fall back to ingested_at if present
        if "ingested_at" in df.columns:
            df["published_utc"] = pd.to_datetime(df["ingested_at"], utc=True, errors="coerce")
        else:
            df["published_utc"] = pd.NaT

    # Optional / missing columns — add with None
    for col in ["cluster_id", "l1_class", "l1_conf"]:
        if col not in df.columns:
            df[col] = None

    # 4) keep only what downstream uses
    keep = [
        "company_id", "title", "body", "published_utc",
        "doc_id", "source_type", "source_url",
        "cluster_id", "l1_class", "l1_conf",
    ]
    # Some inputs may not have source_url; reindex will create it as NaN/None.
    df = df.reindex(columns=keep)

    # 5) trim & return
    if max_docs:
        df = df.head(max_docs)
    return df.to_dict("records")

def read_docs_window_bigquery(
    table_fqn: str,  # project.dataset.table
    start_utc: datetime,
    end_utc: datetime,
    max_docs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery not installed; use --input-kind gcs-parquet or install the lib.")
    client = bigquery.Client()
    q = f"""
    SELECT company_id, title, body, published_utc, doc_id, source_type, source_url, cluster_id, l1_class, l1_conf
    FROM `{table_fqn}`
    WHERE published_utc >= @start AND published_utc < @end
    ORDER BY published_utc DESC
    """
    job = client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "TIMESTAMP", start_utc),
                bigquery.ScalarQueryParameter("end", "TIMESTAMP", end_utc),
            ]
        ),
    )
    rows = [dict(r) for r in job.result(max_results=max_docs)]
    return rows

def _file_sig(path: Optional[str]) -> str:
    """Lightweight signature of a file for cache keys."""
    if not path:
        return "none"
    try:
        st = os.stat(path)
        return f"{Path(path).name}:{int(st.st_mtime)}:{st.st_size}"
    except Exception:
        return "missing"

def _l1_cache_path(df: pd.DataFrame, *, kwex_path: Optional[str]) -> Path:
    """Create a stable cache path for this input set + versions."""
    # use IDs (or doc_id fallback) to fingerprint the input set
    if "press_release_id" in df.columns:
        ids = df["press_release_id"].fillna("").astype(str).tolist()
    else:
        ids = df.get("doc_id", pd.Series([], dtype=str)).fillna("").astype(str).tolist()

    h = hashlib.sha1()
    for s in ids:
        h.update(s.encode("utf-8"))

    # add sizes + taxonomy/lexicon versions + kwex tuned file signature
    meta = {
        "v": CACHE_VERSION,
        "tag": PIPELINE_TAG,
        "n": len(ids),
        "schema": SCHEMA_VERSION,
        "lexicon": LEXICON_VERSION,
        "taxonomy": TAXONOMY_VERSION,
        "kw_source": KW_SOURCE,
        "kwex_sig": _file_sig(kwex_path),
    }
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))

    return CACHE_DIR / f"{PIPELINE_TAG}_{h.hexdigest()[:20]}.parquet"

# ---------------------- Dedup / authority ----------------------
def authority_rank(source_type: Optional[str]) -> int:
    return {"exchange": 3, "regulator": 3, "issuer_pr": 2, "wire": 1}.get((source_type or "wire"), 1)

def light_dedupe(docs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep a single canonical doc per story:
    - Prefer cluster_id if present, else normalized title as bucket key.
    - Within a bucket, keep the highest-authority source.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        key = d.get("cluster_id") or (d.get("title") or "").strip().lower()
        if not key:
            key = d.get("doc_id") or str(id(d))
        prev = buckets.get(key)
        if (prev is None) or (authority_rank(d.get("source_type")) > authority_rank(prev.get("source_type"))):
            buckets[key] = d
    return list(buckets.values())
import re
import ast
import pandas as pd
from typing import Mapping, Any, Optional
import re
import pandas as pd
from typing import Mapping, Any, Optional

def select_earnings(
    df: pd.DataFrame,
    *,
    pred_col: str = "rule_pred_cid",
    cid_to_label: Optional[Mapping[Any, str]] = None,
) -> pd.DataFrame:
    """
    Vectorized selection of earnings-related rows.
    - Handles boolean hits, exact single-label cols, multi-label cols (list or delimited string),
      and a prediction column (default: rule_pred_cid).
    - If `rule_pred_cid` (or any multi-label col) holds numeric CIDs, pass cid_to_label to map -> labels.
    """
    labels = ["earnings_report", "earnings", "results", "q1", "q2", "q3", "q4"]
    label_set = set(s.casefold() for s in labels)
    # word-boundary regex for delimited strings
    pat = re.compile(r"(?<!\w)(?:earnings_report|earnings|results|q[1-4])(?!\w)", re.I)

    mask = pd.Series(False, index=df.index)

    # 1) boolean hit columns (fast path)
    for col in ("earnings_report_hit", "earnings_hit"):
        if col in df.columns:
            mask |= df[col].fillna(False).astype(bool)

    # 2) exact-match single-label columns (fully vectorized)
    for col in ("kwex_topic", "kwex_label", "primary_topic", "rule_label", "class", "category"):
        if col in df.columns:
            s = df[col].astype("string", copy=False)
            mask |= s.str.casefold().isin(label_set)

    # helper to OR-in matches from a possibly mixed-type multi-label column
    def or_in_multilabel(col: str):
        nonlocal mask
        s = df[col]

        # a) string-like rows: regex search in delimited strings (vectorized)
        # avoid errors on non-strings by using astype('string') then .str.contains
        s_str = s.copy()
        s_str[~s_str.map(lambda v: isinstance(v, str))] = pd.NA
        mask_str = s_str.astype("string").str.contains(pat, na=False)

        # b) sequence-like rows: explode -> map -> lowercase -> isin -> groupby any
        is_seq = s.map(lambda v: isinstance(v, (list, tuple, set)))
        if is_seq.any():
            seq = s[is_seq].explode()
            # map CIDs -> labels if provided; otherwise keep as-is
            if cid_to_label is not None:
                seq = seq.map(lambda t: cid_to_label.get(t, t))
            seq = seq.astype("string").str.casefold()
            matched = seq.isin(label_set)
            mask_seq = matched.groupby(level=0).any().reindex(df.index, fill_value=False)
        else:
            mask_seq = False

        mask |= (mask_str | mask_seq)

    # 3) multi-label columns
    for col in ("labels", "topics", "kwex_tags"):
        if col in df.columns:
            or_in_multilabel(col)

    # 4) prediction column (treat as multi-label; handles scalars or lists, CIDs or strings)
    if pred_col in df.columns:
        or_in_multilabel(pred_col)

    return df.loc[mask]

# ---------------------- Runner core ----------------------
# ---------------------- Runner core ----------------------
def run_guidance_change_job(
    cfg: Dict[str, Any],
    sinks: Sinks,
    docs: Iterable[Dict[str, Any]],
    dry_run: bool = False,
    local_csv_path: Optional[str] = None,
    audit_csv_path: Optional[str] = None,
    *,
    input_is_l1: bool = False,   # docs already L1-classified?
) -> Dict[str, int]:
    plugin = get_plugin(GUIDANCE_KEY)

    # Load YAML config for this event and configure plugin (YAML = source of truth for triggers)
    event_cfg = cfg["events"].get(GUIDANCE_KEY)
    if not event_cfg:
        raise KeyError(
            f"Event config for '{GUIDANCE_KEY}' not found. "
            f"Loaded: {list(cfg['events'].keys())}"
        )
    plugin.configure(event_cfg)

    # Load current snapshot (tiny file)
    fs = gcsfs.GCSFileSystem()
    if fs.exists(sinks.guidance_current):
        current_df = load_parquet_df(sinks.guidance_current)
        current_rows = current_df.to_dict("records")
    else:
        current_rows = []
    current: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {
        (r["company_id"], r["period"], r["metric"], r["basis"]): r for r in current_rows
    }

    versions_buf: List[Dict[str, Any]] = []
    events_buf: List[Dict[str, Any]] = []
    audit_buf: List[Dict[str, Any]] = []
    

    if not docs:
        return {"versions": 0, "events": 0}

    df = pd.DataFrame(docs)

    keep_cols = [
        "company_id", "title", "body", "published_utc", "doc_id",
        "source_type", "source_url", "cluster_id", "l1_class", "l1_conf",
    ]

    if input_is_l1:
        # ---------- Fast path: already L1-classified ----------
        log.info("Input marked as L1-preclassified — skipping housekeeping/dedup/gates/classify.")
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None
        if "published_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["published_utc"]):
            df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
        
        df, hk_stats = apply_housekeeping_filter(df,body_col="body",title_col="title")
        log.info(
            "Housekeeping skipped %d / %d (%.1f%%).",
            hk_stats["skipped"], hk_stats["total"],
            (hk_stats["skipped"] / max(hk_stats["total"], 1)) * 100.0,
        )
        base_df = df.copy()

    else:
        # ---------- Normal path: clean → filter → dedup → gate ----------
        if "press_release_id" not in df.columns and "doc_id" in df.columns:
            df["press_release_id"] = df["doc_id"]

        if "release_date" not in df.columns:
            if "published_utc" in df.columns:
                df["release_date"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
            else:
                df["release_date"] = pd.NaT

        # def _cheap_clean(s: pd.Series) -> pd.Series:
        #     return (
        #         s.fillna("")
        #          .astype(str)
        #          .str.lower()
        #          .str.replace(r"http\S+|www\.\S+", " ", regex=True)
        #          .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        #          .str.replace(r"\s+", " ", regex=True)
        #          .str.strip()
        #     )

        # if "title_clean" not in df.columns and "title" in df.columns:
        #     df["title_clean"] = _cheap_clean(df["title"])
        # if "full_text_clean" not in df.columns and "body" in df.columns:
        #     df["full_text_clean"] = _cheap_clean(df["body"])

        # 1) Housekeeping
        df, hk_stats = apply_housekeeping_filter(df)
        log.info(
            "Housekeeping skipped %d / %d (%.1f%%).",
            hk_stats["skipped"], hk_stats["total"],
            (hk_stats["skipped"] / max(hk_stats["total"], 1)) * 100.0,
        )

        # 2) Optional labeled filter
        if filter_to_labeled and (os.getenv("GUIDANCE_FILTER_TO_LABELED", "").lower() in {"1","true","yes"}):
            labeling_path = os.getenv("GUIDANCE_LABELING_PATH", "data/labeling/pool.csv")
            id_col = os.getenv("GUIDANCE_LABELING_ID_COL", "press_release_id")
            try:
                df, stats = filter_to_labeled(df_main=df, labeling=labeling_path, id_col=id_col)
                log.info("filter_to_labeled: %s", stats)
            except FileNotFoundError as e:
                log.warning("filter_to_labeled skipped (missing file): %s", e)

        # 3) Hard dedup
        df, dd_stats = dedup_df(
            df,
            id_col="press_release_id",
            title_col="title_clean",
            body_col="full_text_clean",
            ts_col="release_date",
        )
        log.info("[dedup] %s", dd_stats)

        # 4) Newsletter / quotes gate
        df, nl_stats = apply_newsletter_gate(
            df,
            title_col="title_clean",
            body_col="full_text_clean",
            drop=True,
        )
        log.info("[gate] Newsletter/quote routed: %d / %d (kept=%d)",
                 nl_stats["gated"], nl_stats["total"], len(df))

        if df.empty:
            return {"versions": 0, "events": 0}

        # Convert back to plugin schema
        df = df.assign(
            doc_id=df["press_release_id"],
            published_utc=df["release_date"],
        )
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None

        log.info("Post-gating rows: %d", len(df))

        base_df = df[
            keep_cols + [c for c in ("title_clean", "full_text_clean") if c in df.columns]
        ].copy()

        # --- L1 classification if missing ---
        l1_marker_cols = {"rule_pred_cid", "kwex_tags", "kwex_topic", "earnings_hit", "earnings_report_hit"}
        already_l1 = any(col in base_df.columns for col in l1_marker_cols)

        if already_l1:
            log.info("Detected existing L1 markers; skipping classify_batch.")
        else:
            kwex_path = os.getenv("KWEX_TUNED_JSON", "src/event_feed_app/configs/kwex/kwex_tuned.json")
            try:
                load_kwex_snapshot(kwex_path)
                log.info("Loaded tuned KWEX knobs from %s", kwex_path)
            except Exception as e:
                log.warning("KWEX tuned knobs not loaded (%s). Using defaults.", e)

            cache_path = _l1_cache_path(base_df, kwex_path=kwex_path)
            if cache_path.exists():
                log.info("Using L1 hashed cache: %s", cache_path)
                base_df = pd.read_parquet(cache_path)
            else:
                log.info("L1 cache miss. Classifying %d rows…", len(base_df))
                base_df = classify_batch(
                    base_df,
                    text_cols=("title_clean", "full_text_clean") if "title_clean" in base_df.columns else ("title", "body"),
                    keywords=keywords
                )
                try:
                    cache_path.parent.mkdir(exist_ok=True)
                    base_df.to_parquet(cache_path, index=False)
                    log.info("Wrote L1 hashed cache (%d rows) → %s", len(base_df), cache_path)
                except Exception as e:
                    log.warning("Failed to write L1 hashed cache (%s): %s", cache_path, e)
                try:
                    SIMPLE_L1_CACHE.parent.mkdir(exist_ok=True)
                    base_df.to_parquet(SIMPLE_L1_CACHE, index=False)
                    log.info("Wrote L1 simple cache (%d rows) → %s", len(base_df), SIMPLE_L1_CACHE)
                except Exception as e:
                    log.warning("Failed to write L1 simple cache (%s): %s", SIMPLE_L1_CACHE, e)

    # ---- Common step: only keep earnings-related items before plugin detection ----
    base_df = select_earnings(base_df)
    log.info("Rules-first earnings candidates: %d / %d", len(base_df), len(df))
    if base_df.empty:
        log.info("No earnings candidates — exiting early.")
        return {"versions": 0, "events": 0}

    # Run plugin
    for doc in base_df.to_dict("records"):
        # now that triggers come from YAML, it's safe/useful to gate here
        if not plugin.gate(doc):
            continue

        title = doc.get("title_clean", "")
        body = doc.get("body_clean", "")

        for raw in plugin.detect(doc):
            cand = plugin.normalize({
                **raw,
                "company_id": doc["company_id"],
                "source_type": doc.get("source_type", "wire"),
                "_doc_title": title,
                "_doc_body": body,
                "_no_prior": False,
            })
            key = plugin.prior_key(cand)
            prior = current.get(key)
            if prior is None:
                cand["_no_prior"] = True

            comp = plugin.compare(cand, prior)
            is_sig, score, _ = plugin.decide_and_score(cand, comp, event_cfg)
            if not is_sig:
                continue

            vrow, erow, arow = plugin.build_rows(doc, cand, comp, is_sig, score)
            if vrow:
                versions_buf.append(vrow)
                current[key] = vrow
            if erow:
                events_buf.append(erow)
                if arow:
                    audit_buf.append(arow)

    if dry_run:
        log.info("[DRY RUN] Would write %d versions rows, %d events rows", len(versions_buf), len(events_buf))
        return {"versions": len(versions_buf), "events": len(events_buf)}

    if versions_buf:
        append_parquet_gcs(sinks.guidance_versions, versions_buf)
        overwrite_parquet_gcs(sinks.guidance_current, pd.DataFrame(list(current.values())))

    if events_buf:
        append_parquet_gcs(sinks.events_silver, events_buf)
       

    if local_csv_path:
        try:
            if events_buf:
                pd.DataFrame(events_buf).to_csv(local_csv_path, index=False, encoding="utf-8")
                log.info("Wrote local CSV with %d events to %s", len(events_buf), local_csv_path)
                pd.DataFrame(audit_buf).to_csv(audit_csv_path, index=False, encoding="utf-8")
                log.info("Wrote local CSV with %d events to %s", len(audit_buf), audit_csv_path)
            else:
                log.info("No events; skipped writing %s", local_csv_path)
        except Exception as e:
            log.warning("Failed to write OUTPUT_LOCAL_CSV (%s): %s", local_csv_path, e)

    return {"versions": len(versions_buf), "events": len(events_buf)}


    # # ---------- Earnings filter (works for both paths) ----------
    # base_df = select_earnings(base_df)
    # log.info("Rules-first earnings candidates: %d", len(base_df))
    # if base_df.empty:
    #     log.info("No earnings candidates — exiting early.")
    #     return {"versions": 0, "events": 0}

    # # Drive the guidance plugin
    # docs = base_df.to_dict("records")
    # for doc in docs:
    #     if not hits_trigger(doc, trigger_rx):
    #         continue
    #     title = doc.get("title", "")
    #     body = doc.get("body", "")
        
    #     for raw in plugin.detect(doc):
    #         cand = plugin.normalize({
    #             **raw,
    #             "company_id": doc["company_id"],
    #             "source_type": doc.get("source_type", "wire"),
    #             "_doc_title": title,
    #             "_doc_body": body,
    #             "_no_prior": False,
    #         })
    #         key = plugin.prior_key(cand)
    #         prior = current.get(key)
    #         if prior is None:
    #             cand["_no_prior"] = True

    #         comp = plugin.compare(cand, prior)
    #         is_sig, score, _ = plugin.decide_and_score(cand, comp, event_cfg)
    #         if not is_sig:
    #             continue

    #         vrow, erow = plugin.build_rows(doc, cand, comp, is_sig, score)
    #         if vrow:
    #             versions_buf.append(vrow)
    #             current[key] = vrow
    #         if erow:
    #             events_buf.append(erow)

    # if dry_run:
    #     log.info("[DRY RUN] Would write %d versions rows, %d events rows", len(versions_buf), len(events_buf))
    #     return {"versions": len(versions_buf), "events": len(events_buf)}

    # if versions_buf:
    #     append_parquet_gcs(sinks.guidance_versions, versions_buf)
    #     overwrite_parquet_gcs(sinks.guidance_current, pd.DataFrame(list(current.values())))

    # if events_buf:
    #     append_parquet_gcs(sinks.events_silver, events_buf)

    # if local_csv_path:
    #     try:
    #         if events_buf:
    #             pd.DataFrame(events_buf).to_csv(local_csv_path, index=False, encoding="utf-8")
    #             log.info("Wrote local CSV with %d events to %s", len(events_buf), local_csv_path)
    #         else:
    #             log.info("No events; skipped writing %s", local_csv_path)
    #     except Exception as e:
    #         log.warning("Failed to write OUTPUT_LOCAL_CSV (%s): %s", local_csv_path, e)

    # return {"versions": len(versions_buf), "events": len(events_buf)}



# ---------------------- CLI ----------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run guidance_change event job over a time window.")

    default_docs = _ensure_gs(os.getenv("GCS_SILVER_ROOT", ""))  # <— from your orchestrator
    ap.add_argument("--config-dir", type=Path, required=True,
                    help="Path to config dir (contains sinks.yaml and events configs).")
    ap.add_argument("--input-kind", choices=["gcs-parquet", "bigquery"], default="gcs-parquet")
    ap.add_argument("--docs-path", default=default_docs,  # <— not strictly required now
                    help="GCS path (parquet file/folder) OR BigQuery table FQN (project.dataset.table). "
                         "Defaults to $GCS_SILVER_ROOT.")
    ap.add_argument("--lookback-hours", type=int, default=48,
                    help="If --start-utc not given, read from now()-lookback to now().")
    ap.add_argument("--start-utc", type=str, default=None, help="ISO8601 UTC start, e.g. 2025-09-28T00:00:00Z")
    ap.add_argument("--end-utc", type=str, default=None, help="ISO8601 UTC end, default: now()")
    ap.add_argument("--max-docs", type=int, default=None, help="Limit docs for testing.")
    ap.add_argument("--dry-run", action="store_true", help="Process but do not write outputs.")
    ap.add_argument("--log-level", default="INFO")
    # parse_args()
    ap.add_argument("--debug-csv", type=Path, default=None,
                help="Optional: path to write emitted events as CSV for debugging.")
    ap.add_argument("--audit-csv", type=Path, default=None,
                help="Optional: path to write emitted events for audit as CSV for debugging.")
    
    return ap.parse_args()

def parse_iso8601_utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.docs_path:
        raise SystemExit(
            "docs-path not provided and $GCS_SILVER_ROOT is empty. "
            "Set --docs-path or export GCS_SILVER_ROOT."
        )

    cfg = load_configs(Path(args.config_dir))
    sinks = load_sinks(Path(args.config_dir))

    end_utc = parse_iso8601_utc(args.end_utc) or datetime.now(timezone.utc)
    start_utc = parse_iso8601_utc(args.start_utc) or (end_utc - timedelta(hours=args.lookback_hours))

    log.info("Reading docs from %s to %s", start_utc.isoformat(), end_utc.isoformat())

    # ---------- Prefer cached labeled/classified data ----------
    simple_l1_cache = globals().get("SIMPLE_L1_CACHE", CACHE_DIR / f"{PIPELINE_TAG}_latest.parquet")
    prefer_cache = os.getenv("GUIDANCE_PREFER_CACHE", "1").lower() in {"1", "true", "yes", "on"}

    docs: Optional[List[Dict[str, Any]]] = None
    input_is_l1 = False  # <- tells the runner to skip housekeeping/dedup/gates/classify

    if prefer_cache and simple_l1_cache.exists():
        try:
            cached_df = pd.read_parquet(simple_l1_cache)
            if not cached_df.empty:
                missing = REQUIRED_BASE_COLS - set(cached_df.columns)
                if missing:
                    log.warning(
                        "Cached L1 file is missing required columns %s; ignoring cache.",
                        sorted(missing),
                    )
                else:
                    log.info(
                        "Using cached L1 classified data from %s (rows=%d). Skipping input fetch.",
                        simple_l1_cache, len(cached_df)
                    )
                    docs = cached_df.to_dict("records")
                    input_is_l1 = True
            else:
                log.info("Simple L1 cache is empty; will fetch inputs.")
        except Exception as e:
            log.warning(
                "Failed to read simple L1 cache (%s). Will fetch inputs. Error: %s",
                simple_l1_cache, e
            )

    # Fallback: fetch inputs if no usable cache
    if docs is None:
        if args.input_kind == "gcs-parquet":
            docs = read_docs_window_gcs_parquet(args.docs_path, start_utc, end_utc, args.max_docs)
        else:
            docs = read_docs_window_bigquery(args.docs_path, start_utc, end_utc, args.max_docs)
        log.info("Loaded %d docs", len(docs))

    # -----------------------------------------------------------

    result = run_guidance_change_job(
        cfg,
        sinks,
        docs,
        dry_run=args.dry_run,
        local_csv_path=args.debug_csv,
        audit_csv_path=args.audit_csv,
        input_is_l1=input_is_l1,   # <- crucial: skip preproc/classify if cache used
    )
    log.info("Wrote versions=%d, events=%d", result["versions"], result["events"])


if __name__ == "__main__":
    main()
